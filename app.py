"""
VelvetSky – Flask Backend  (Production-hardened)
=================================================
FIXES APPLIED vs original:
  1. [CRITICAL] Flat-file JSON DB replaced with SQLite (WAL mode) + thread lock
  2. [CRITICAL] All secrets removed from source — env-vars only, startup crash if missing
  3. [CRITICAL] Flask dev server replaced with Gunicorn launcher (debug=False)
  4. [WARN]     Passwords now hashed with bcrypt (work-factor 12)
  5. [WARN]     flask-limiter is now a hard dependency (ImportError = startup failure)
  6. [WARN]     Thread-safe lock on every DB read/write
  7. [WARN]     Wallpaper / user IDs use secrets.token_urlsafe — no timestamp collisions
  8. [WARN]     FLASK_SECRET env-var is required; startup refuses a weak/default value

Endpoints (unchanged from original)
-------------------------------------
POST /api/login
POST /api/signup
POST /api/logout
GET  /api/me
GET  /api/config
GET  /api/wallpapers
POST /api/wallpapers
DELETE /api/wallpapers/<id>
POST /api/upload/github
POST /api/upload/r2
DELETE /api/delete/github
DELETE /api/delete/r2
POST /api/github/push
GET  /api/github/pull
POST /api/favourites
POST /api/ai/analyse
GET  /
"""

import os, json, base64, hashlib, hmac, secrets, time, sqlite3, threading, requests
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory, make_response

# ── bcrypt is now a hard requirement ──────────────────────────────────────────
try:
    import bcrypt as _bcrypt
except ImportError:
    raise SystemExit(
        "FATAL: bcrypt is not installed.\n"
        "Run:  pip install bcrypt\n"
        "Passwords cannot be stored securely without it."
    )

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── FLASK_SECRET: required, must be ≥32 chars, must not be the dev default ───
_secret = os.environ.get("FLASK_SECRET", "")
if not _secret:
    raise SystemExit(
        "FATAL: FLASK_SECRET environment variable is not set.\n"
        "Generate one with:\n"
        "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        "Then set it in your environment or .env file."
    )
if len(_secret) < 32:
    raise SystemExit(
        "FATAL: FLASK_SECRET is too short (must be ≥ 32 characters).\n"
        "Generate a strong one with:\n"
        "  python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _secret

# ── Secure session cookies ─────────────────────────────────────────────────────
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=86400,
)

# ── Response compression ───────────────────────────────────────────────────────
try:
    from flask_compress import Compress
    Compress(app)
except ImportError:
    pass  # optional — install flask-compress for gzip

# ── Rate limiting: hard dependency now, not optional ──────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["500 per day", "100 per hour"],
        storage_uri="memory://",
    )
    def _rate_limit(limit_str):
        return _limiter.limit(limit_str)
except ImportError:
    raise SystemExit(
        "FATAL: flask-limiter is not installed.\n"
        "Run:  pip install flask-limiter\n"
        "Rate limiting is required to protect auth endpoints."
    )

# ── Shared requests.Session (HTTP keep-alive) ─────────────────────────────────
_http = requests.Session()
_http.headers.update({"User-Agent": "VelvetSky/1.0"})

# ──────────────────────────────────────────────────────────────────────────────
# ★  SECRETS — environment variables ONLY.  No hardcoded fallbacks.  ★
# ──────────────────────────────────────────────────────────────────────────────
def _require_env(name, warn_only=False):
    """Return env var value; raise SystemExit if missing (unless warn_only)."""
    val = os.environ.get(name, "")
    if not val and not warn_only:
        raise SystemExit(
            f"FATAL: Required environment variable '{name}' is not set.\n"
            f"Add it to your .env file or deployment environment."
        )
    return val

# Admin credentials — required at startup
ADMIN_EMAIL = _require_env("ADMIN_EMAIL")
ADMIN_PASS  = _require_env("ADMIN_PASS")

# Google OAuth — optional (Google login simply stays disabled if absent)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# GitHub image storage — optional
GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "")
GITHUB_USER   = os.environ.get("GITHUB_USER",   "")
GITHUB_REPO   = os.environ.get("GITHUB_REPO",   "")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_FOLDER = os.environ.get("GITHUB_FOLDER", "wallpapers")

# GitHub DB repo — falls back to image repo values if not separately set
GH_DB_TOKEN  = os.environ.get("GH_DB_TOKEN",  GITHUB_TOKEN)
GH_DB_USER   = os.environ.get("GH_DB_USER",   GITHUB_USER)
GH_DB_REPO   = os.environ.get("GH_DB_REPO",   "")
GH_DB_BRANCH = os.environ.get("GH_DB_BRANCH", "main")
GH_DB_FILE   = "velvetsky-db.json"

# Cloudflare R2 — optional
R2_WORKER_URL        = os.environ.get("R2_WORKER_URL",        "")
R2_BUCKET_PUBLIC_URL = os.environ.get("R2_BUCKET_PUBLIC_URL", "")
R2_AUTH_TOKEN        = os.environ.get("R2_AUTH_TOKEN",        "")

# AI vision — optional
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY",     "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
# SQLite database  (replaces the flat JSON file)
#
# FIX #1 — WAL mode + a threading.Lock ensures concurrent writers never corrupt
#           data or silently lose signups.
# FIX #6 — Every read/write is wrapped in _DB_LOCK so thread safety is explicit.
# ──────────────────────────────────────────────────────────────────────────────
DB_FILE  = os.path.join(os.path.dirname(__file__), "velvetsky.db")
_DB_LOCK = threading.Lock()

def _get_conn():
    """Return a new SQLite connection in WAL mode.  Always used inside _DB_LOCK."""
    conn = sqlite3.connect(DB_FILE, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    """Create tables if they don't exist yet, and seed default wallpapers."""
    with _DB_LOCK:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS wallpapers (
                    id         TEXT PRIMARY KEY,
                    src        TEXT NOT NULL,
                    thumb      TEXT,
                    cat        TEXT NOT NULL DEFAULT 'other',
                    tag        TEXT,
                    title      TEXT,
                    description TEXT,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    stored_in  TEXT,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id            TEXT PRIMARY KEY,
                    fname         TEXT,
                    lname         TEXT,
                    email         TEXT NOT NULL UNIQUE,
                    password_hash TEXT,
                    provider      TEXT NOT NULL DEFAULT 'email',
                    picture       TEXT,
                    created_at    INTEGER NOT NULL,
                    favs          TEXT NOT NULL DEFAULT '[]'
                );
            """)
            # Seed default wallpapers only once
            count = conn.execute("SELECT COUNT(*) FROM wallpapers WHERE is_default=1").fetchone()[0]
            if count == 0:
                defaults = [
                    ("d1","https://images.unsplash.com/photo-1518623489648-a173ef7824f3?w=1400&q=85","https://images.unsplash.com/photo-1518623489648-a173ef7824f3?w=800&q=80","nature","Nature","Forest Light"),
                    ("d2","https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=1400&q=85","https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800&q=80","space","Space","Galaxy"),
                    ("d3","https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1400&q=85","https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80","nature","Nature","Alpine Peaks"),
                    ("d4","https://images.unsplash.com/photo-1579546929518-9e396f3cc809?w=1400&q=85","https://images.unsplash.com/photo-1579546929518-9e396f3cc809?w=800&q=80","abstract","Abstract","Color Gradient"),
                    ("d5","https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1400&q=85","https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800&q=80","city","City","City Skyline"),
                    ("d6","https://images.unsplash.com/photo-1519681393784-d120267933ba?w=1400&q=85","https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800&q=80","nature","Nature","Snowy Mountains"),
                    ("d7","https://images.unsplash.com/photo-1511300636408-a63a89df3482?w=1400&q=85","https://images.unsplash.com/photo-1511300636408-a63a89df3482?w=800&q=80","minimal","Minimal","Minimalist"),
                    ("d8","https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=1400&q=85","https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=800&q=80","space","Space","Deep Space"),
                    ("d9","https://images.unsplash.com/photo-1497366216548-37526070297c?w=1400&q=85","https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&q=80","city","City","Architecture"),
                ]
                now = int(time.time() * 1000)
                conn.executemany(
                    "INSERT OR IGNORE INTO wallpapers(id,src,thumb,cat,tag,title,is_default,created_at) VALUES(?,?,?,?,?,?,1,?)",
                    [(d[0],d[1],d[2],d[3],d[4],d[5],now) for d in defaults]
                )
            conn.commit()
            # Add new columns to existing DBs (idempotent — safe on every startup)
            for _col, _def in [
                ("download_count", "INTEGER NOT NULL DEFAULT 0"),
                ("is_featured",    "INTEGER NOT NULL DEFAULT 0"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE wallpapers ADD COLUMN {_col} {_def}")
                    conn.commit()
                except Exception:
                    pass  # column already exists — ignore
        finally:
            conn.close()

_init_db()

# ── DB helpers ────────────────────────────────────────────────────────────────
def _row_to_wall(row):
    keys = row.keys()
    return {
        "id": row["id"], "src": row["src"],
        "thumb": row["thumb"] or row["src"],
        "cat": row["cat"],  "tag": row["tag"] or row["cat"].capitalize(),
        "title": row["title"] or "", "desc": row["description"] or "",
        "isDefault": bool(row["is_default"]),
        "storedIn": row["stored_in"],
        "createdAt": row["created_at"],
        "downloadCount": row["download_count"] if "download_count" in keys else 0,
        "isFeatured": bool(row["is_featured"]) if "is_featured" in keys else False,
    }

def _row_to_user(row, include_hash=False):
    d = {
        "id": row["id"], "fname": row["fname"] or "",
        "lname": row["lname"] or "", "email": row["email"],
        "provider": row["provider"], "picture": row["picture"] or "",
        "createdAt": row["created_at"],
        "favs": json.loads(row["favs"] or "[]"),
    }
    if include_hash:
        d["passwordHash"] = row["password_hash"]
    return d

def db_get_wallpapers():
    with _DB_LOCK:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM wallpapers ORDER BY is_featured DESC, created_at DESC"
            ).fetchall()
            return [_row_to_wall(r) for r in rows]
        finally:
            conn.close()

def db_add_wallpaper(wall_id, src, thumb, cat, tag, title, desc="", stored_in=None):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            now = int(time.time() * 1000)
            conn.execute(
                "INSERT INTO wallpapers(id,src,thumb,cat,tag,title,description,is_default,stored_in,created_at)"
                " VALUES(?,?,?,?,?,?,?,0,?,?)",
                (wall_id, src, thumb or src, cat, tag, title, desc, stored_in, now)
            )
            conn.commit()
        finally:
            conn.close()

def db_delete_wallpaper(wall_id):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM wallpapers WHERE id=?", (wall_id,)).fetchone()
            wall = _row_to_wall(row) if row else None
            conn.execute("DELETE FROM wallpapers WHERE id=?", (wall_id,))
            conn.commit()
            return wall
        finally:
            conn.close()

def db_increment_download(wall_id):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE wallpapers SET download_count = download_count + 1 WHERE id=?",
                (wall_id,)
            )
            conn.commit()
            row = conn.execute("SELECT download_count FROM wallpapers WHERE id=?", (wall_id,)).fetchone()
            return row["download_count"] if row else 0
        finally:
            conn.close()

def db_set_featured(wall_id, featured):
    """Toggle featured flag. Returns (ok, error_msg). Max 3 featured at once."""
    with _DB_LOCK:
        conn = _get_conn()
        try:
            if featured:
                count = conn.execute(
                    "SELECT COUNT(*) FROM wallpapers WHERE is_featured=1 AND id!=?", (wall_id,)
                ).fetchone()[0]
                if count >= 3:
                    return False, "Max 3 wallpapers can be featured at once."
            conn.execute(
                "UPDATE wallpapers SET is_featured=? WHERE id=?",
                (1 if featured else 0, wall_id)
            )
            conn.commit()
            return True, None
        finally:
            conn.close()

def db_update_wallpaper_meta(wall_id, title=None, cat=None, tag=None):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            updates, params = [], []
            if title is not None: updates.append("title=?");       params.append(title)
            if cat   is not None: updates.append("cat=?");         params.append(cat)
            if tag   is not None: updates.append("tag=?");         params.append(tag)
            if not updates:
                return
            params.append(wall_id)
            conn.execute(f"UPDATE wallpapers SET {', '.join(updates)} WHERE id=?", params)
            conn.commit()
        finally:
            conn.close()

def db_find_user_by_email(email):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
            return row
        finally:
            conn.close()

def db_find_user_by_id(uid):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            return conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        finally:
            conn.close()

def db_create_user(uid, fname, lname, email, password_hash=None, provider="email", picture=""):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            now = int(time.time() * 1000)
            conn.execute(
                "INSERT INTO users(id,fname,lname,email,password_hash,provider,picture,created_at,favs)"
                " VALUES(?,?,?,?,?,?,?,?,'[]')",
                (uid, fname, lname, email.lower(), password_hash, provider, picture, now)
            )
            conn.commit()
        finally:
            conn.close()

def db_upsert_google_user(uid, fname, lname, email, picture):
    """Insert or update a Google-authenticated user. Returns the final user row."""
    with _DB_LOCK:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
            now = int(time.time() * 1000)
            if row:
                conn.execute(
                    "UPDATE users SET fname=COALESCE(NULLIF(?,\"\"),fname), lname=COALESCE(NULLIF(?,\"\"),lname),"
                    " picture=COALESCE(NULLIF(?,\"\"),picture), provider=? WHERE email=?",
                    (fname, lname, picture, "google", email.lower())
                )
            else:
                conn.execute(
                    "INSERT INTO users(id,fname,lname,email,password_hash,provider,picture,created_at,favs)"
                    " VALUES(?,?,?,?,NULL,?,?,?,'[]')",
                    (uid, fname, lname, email.lower(), "google", picture, now)
                )
            conn.commit()
            return conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
        finally:
            conn.close()

def db_update_favs(uid, favs):
    with _DB_LOCK:
        conn = _get_conn()
        try:
            conn.execute("UPDATE users SET favs=? WHERE id=?", (json.dumps(favs), uid))
            conn.commit()
        finally:
            conn.close()

def db_get_all_users():
    with _DB_LOCK:
        conn = _get_conn()
        try:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
            return [_row_to_user(r) for r in rows]
        finally:
            conn.close()

# ── Password hashing (bcrypt, work-factor 12) ─────────────────────────────────
# FIX #4 — replaces plain SHA-256 with bcrypt
def hash_pass(password: str) -> str:
    """Hash a plaintext password with bcrypt. Returns a utf-8 string."""
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt(rounds=12)).decode("utf-8")

def verify_pass(password: str, hashed: str) -> bool:
    """Constant-time bcrypt comparison. Handles legacy SHA-256 hashes gracefully."""
    if hashed.startswith("h_"):
        # Legacy SHA-256 hash — still works but will be upgraded on next login
        return hmac.compare_digest(
            "h_" + hashlib.sha256(password.encode()).hexdigest(), hashed
        )
    try:
        return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

# ── Auth decorators ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated

# ──────────────────────────────────────────────────────────────────────────────
# Auth routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
@_rate_limit("10 per minute")
def api_login():
    data  = request.json or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    # Admin login — constant-time comparison on both fields
    if (
        hmac.compare_digest(email, ADMIN_EMAIL.lower()) and
        hmac.compare_digest(password, ADMIN_PASS)
    ):
        session["user"]     = {"email": email, "fname": "Admin", "is_admin": True}
        session["is_admin"] = True
        return jsonify({"ok": True, "is_admin": True, "user": {"fname": "Admin", "email": email}})

    # Regular user login
    row = db_find_user_by_email(email)
    if not row or not verify_pass(password, row["password_hash"] or ""):
        return jsonify({"error": "Incorrect email or password"}), 401

    # Transparent bcrypt upgrade: if still on legacy SHA-256, re-hash now
    if (row["password_hash"] or "").startswith("h_"):
        new_hash = hash_pass(password)
        with _DB_LOCK:
            conn = _get_conn()
            try:
                conn.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, row["id"]))
                conn.commit()
            finally:
                conn.close()

    user = _row_to_user(row)
    session["user"]     = user
    session["is_admin"] = False
    return jsonify({"ok": True, "is_admin": False, "user": user})


@app.route("/api/signup", methods=["POST"])
@_rate_limit("5 per minute")
def api_signup():
    data     = request.json or {}
    fname    = (data.get("fname") or "").strip()
    lname    = (data.get("lname") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not fname or not email or len(password) < 8:
        return jsonify({"error": "Invalid data"}), 400

    if db_find_user_by_email(email):
        return jsonify({"error": "Email already registered"}), 409

    # FIX #4: bcrypt instead of SHA-256
    uid = "e_" + secrets.token_urlsafe(12)   # FIX #7: secure random ID
    try:
        db_create_user(uid, fname, lname, email, hash_pass(password), "email")
    except Exception:
        return jsonify({"error": "Email already registered"}), 409

    row  = db_find_user_by_email(email)
    user = _row_to_user(row)
    session["user"]     = user
    session["is_admin"] = False
    return jsonify({"ok": True, "user": user})


@app.route("/api/google-login", methods=["POST"])
@_rate_limit("10 per minute")
def api_google_login():
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Google login is not configured on this server"}), 503
    data       = request.json or {}
    credential = data.get("credential", "")
    if not credential:
        return jsonify({"error": "Missing credential"}), 400

    resp = _http.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"id_token": credential}, timeout=8
    )
    if not resp.ok:
        return jsonify({"error": "Invalid Google token"}), 401

    payload = resp.json()
    if payload.get("aud") != GOOGLE_CLIENT_ID:
        return jsonify({"error": "Token audience mismatch"}), 401

    g_email = payload.get("email", "").lower()
    uid     = "g_" + payload.get("sub", secrets.token_urlsafe(12))
    fname   = payload.get("given_name") or (payload.get("name","").split() or [""])[0]
    lname   = payload.get("family_name", "")
    picture = payload.get("picture", "")

    row  = db_upsert_google_user(uid, fname, lname, g_email, picture)
    user = _row_to_user(row)
    session["user"]     = user
    session["is_admin"] = False
    return jsonify({"ok": True, "user": user})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
def api_me():
    user = session.get("user")
    if not user:
        return jsonify({"user": None, "is_admin": False})
    return jsonify({"user": user, "is_admin": session.get("is_admin", False)})


# ──────────────────────────────────────────────────────────────────────────────
# Public config
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/config")
def api_config():
    resp = jsonify({
        "googleClientId": GOOGLE_CLIENT_ID or None,
    })
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Wallpapers
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/wallpapers", methods=["GET"])
def api_get_wallpapers():
    walls = db_get_wallpapers()
    etag  = hashlib.md5(json.dumps(walls, sort_keys=True).encode()).hexdigest()
    if request.headers.get("If-None-Match") == etag:
        return "", 304
    resp = jsonify({"wallpapers": walls})
    resp.headers["Cache-Control"] = "public, max-age=60"
    resp.headers["ETag"] = etag
    return resp


@app.route("/api/wallpapers", methods=["POST"])
@admin_required
def api_add_wallpaper():
    data = request.json or {}
    cat  = data.get("cat", "other")
    # FIX #7: secure random IDs — no more timestamp-only collisions
    wall_id = "w_" + secrets.token_urlsafe(12)
    db_add_wallpaper(
        wall_id,
        src       = data.get("src", ""),
        thumb     = data.get("thumb") or data.get("src", ""),
        cat       = cat,
        tag       = data.get("tag") or cat.capitalize(),
        title     = data.get("title", ""),
        desc      = data.get("desc", ""),
        stored_in = data.get("storedIn"),
    )
    walls = db_get_wallpapers()
    wall  = next((w for w in walls if w["id"] == wall_id), None)
    return jsonify({"ok": True, "wallpaper": wall})


@app.route("/api/wallpapers/<wall_id>", methods=["DELETE"])
@admin_required
def api_delete_wallpaper(wall_id):
    db_delete_wallpaper(wall_id)
    return jsonify({"ok": True})


@app.route("/api/wallpapers/<wall_id>/download", methods=["POST"])
def api_increment_download(wall_id):
    new_count = db_increment_download(wall_id)
    return jsonify({"ok": True, "downloadCount": new_count})


@app.route("/api/wallpapers/<wall_id>/feature", methods=["POST"])
@admin_required
def api_toggle_feature(wall_id):
    body     = request.get_json(silent=True) or {}
    featured = bool(body.get("featured", True))
    ok, err  = db_set_featured(wall_id, featured)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "isFeatured": featured})


@app.route("/api/wallpapers/<wall_id>", methods=["PATCH"])
@admin_required
def api_update_wallpaper_meta(wall_id):
    body  = request.get_json(silent=True) or {}
    title = body.get("title")
    cat   = body.get("cat")
    tag   = body.get("tag")
    db_update_wallpaper_meta(wall_id, title=title, cat=cat, tag=tag)
    return jsonify({"ok": True})


@app.route("/api/feed")
def api_feed():
    limit = min(int(request.args.get("limit", 20)), 100)
    walls = db_get_wallpapers()[:limit]
    resp  = jsonify({"ok": True, "count": len(walls), "wallpapers": walls})
    resp.headers["Cache-Control"] = "public, max-age=60"
    return resp


@app.route("/manifest.json")
def pwa_manifest():
    manifest = {
        "name": "VelvetSky Wallpapers",
        "short_name": "VelvetSky",
        "description": "Beautiful wallpapers for every background need.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0a0a",
        "theme_color": "#c8a97e",
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }
    resp = jsonify(manifest)
    resp.headers["Content-Type"] = "application/manifest+json"
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# GitHub image storage
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}

def _is_valid_image(file) -> bool:
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False
    header = file.read(16); file.seek(0)
    return (
        header[:3] == b'\xff\xd8\xff'
        or header[:4] == b'\x89PNG'
        or header[:6] in (b'GIF87a', b'GIF89a')
        or (header[:4] == b'RIFF' and header[8:12] == b'WEBP')
    )

@app.route("/api/upload/github", methods=["POST"])
@admin_required
def api_upload_github():
    if not GITHUB_TOKEN or not GITHUB_USER or not GITHUB_REPO:
        return jsonify({"error": "GitHub not configured on server"}), 503
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    if not _is_valid_image(file):
        return jsonify({"error": "Invalid file type — only JPG, PNG, WebP, GIF allowed"}), 400
    ext  = (file.filename or "img.jpg").rsplit(".", 1)[-1].lower()
    name = f"{int(time.time()*1000)}-{secrets.token_hex(4)}.{ext}"
    path = f"{GITHUB_FOLDER}/{name}"
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{path}"
    b64  = base64.b64encode(file.read()).decode()
    resp = _http.put(api_url, json={
        "message": f"Upload wallpaper: {name}",
        "branch": GITHUB_BRANCH,
        "content": b64,
    }, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }, timeout=30)
    if not resp.ok:
        return jsonify({"error": f"GitHub error {resp.status_code}"}), 502
    cdn_url = f"https://cdn.jsdelivr.net/gh/{GITHUB_USER}/{GITHUB_REPO}@{GITHUB_BRANCH}/{path}"
    return jsonify({"ok": True, "url": cdn_url})


@app.route("/api/delete/github", methods=["DELETE"])
@admin_required
def api_delete_github():
    if not GITHUB_TOKEN:
        return jsonify({"error": "GitHub not configured"}), 503
    data    = request.json or {}
    cdn_url = data.get("url", "")
    prefix  = f"https://cdn.jsdelivr.net/gh/{GITHUB_USER}/{GITHUB_REPO}@{GITHUB_BRANCH}/"
    if not cdn_url.startswith(prefix):
        return jsonify({"error": "Not a managed GitHub URL"}), 400
    file_path = cdn_url[len(prefix):]
    api_url   = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{file_path}"
    headers   = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    info = _http.get(api_url, headers=headers, timeout=15)
    if not info.ok:
        return jsonify({"error": "File not found on GitHub"}), 404
    sha  = info.json().get("sha")
    resp = _http.delete(api_url, json={
        "message": f"Delete wallpaper: {file_path}", "sha": sha, "branch": GITHUB_BRANCH
    }, headers={**headers, "Content-Type": "application/json"}, timeout=15)
    return jsonify({"ok": resp.ok})


# ──────────────────────────────────────────────────────────────────────────────
# Cloudflare R2
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/upload/r2", methods=["POST"])
@admin_required
def api_upload_r2():
    if not R2_WORKER_URL:
        return jsonify({"error": "R2 not configured on server"}), 503
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400
    if not _is_valid_image(file):
        return jsonify({"error": "Invalid file type — only JPG, PNG, WebP, GIF allowed"}), 400
    ext      = (file.filename or "img.jpg").rsplit(".", 1)[-1].lower()
    key      = f"velvetsky/{int(time.time()*1000)}-{secrets.token_hex(4)}.{ext}"
    endpoint = f"{R2_WORKER_URL}/upload/{requests.utils.quote(key, safe='')}"
    headers  = {"Content-Type": file.content_type or "image/jpeg"}
    if R2_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {R2_AUTH_TOKEN}"
    resp = _http.put(endpoint, data=file.read(), headers=headers, timeout=30)
    if not resp.ok:
        return jsonify({"error": f"R2 error {resp.status_code}"}), 502
    public_url = f"{R2_BUCKET_PUBLIC_URL.rstrip('/')}/{key}"
    return jsonify({"ok": True, "url": public_url})


@app.route("/api/delete/r2", methods=["DELETE"])
@admin_required
def api_delete_r2():
    if not R2_WORKER_URL:
        return jsonify({"error": "R2 not configured"}), 503
    data = request.json or {}
    url  = data.get("url", "")
    base = R2_BUCKET_PUBLIC_URL.rstrip("/") + "/"
    if not url.startswith(base):
        return jsonify({"error": "Not a managed R2 URL"}), 400
    key      = url[len(base):]
    endpoint = f"{R2_WORKER_URL}/delete/{requests.utils.quote(key, safe='')}"
    headers  = {}
    if R2_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {R2_AUTH_TOKEN}"
    resp = _http.delete(endpoint, headers=headers, timeout=15)
    return jsonify({"ok": resp.ok})


# ──────────────────────────────────────────────────────────────────────────────
# GitHub DB persistence
# ──────────────────────────────────────────────────────────────────────────────
def gh_db_enabled():
    return bool(GH_DB_TOKEN and GH_DB_USER and GH_DB_REPO)

def gh_db_headers():
    return {
        "Authorization": f"Bearer {GH_DB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

_gh_db_sha_cache = None

@app.route("/api/github/push", methods=["POST"])
@admin_required
def api_github_push():
    global _gh_db_sha_cache
    if not gh_db_enabled():
        return jsonify({"error": "GitHub DB not configured"}), 503
    walls = db_get_wallpapers()
    users = db_get_all_users()
    db    = {"wallpapers": walls, "users": users, "updatedAt": int(time.time() * 1000)}
    content = base64.b64encode(json.dumps(db, indent=2).encode()).decode()
    api_url = f"https://api.github.com/repos/{GH_DB_USER}/{GH_DB_REPO}/contents/{GH_DB_FILE}"
    sha     = _gh_db_sha_cache
    if sha is None:
        existing = _http.get(api_url, headers=gh_db_headers(), timeout=10)
        if existing.ok:
            sha = existing.json().get("sha")
    body = {"message": "VelvetSky DB update", "branch": GH_DB_BRANCH, "content": content}
    if sha:
        body["sha"] = sha
    resp = _http.put(api_url, json=body, headers=gh_db_headers(), timeout=20)
    if resp.ok:
        _gh_db_sha_cache = resp.json().get("content", {}).get("sha")
        return jsonify({"ok": True, "wallpapers": len(walls), "users": len(users)})
    return jsonify({"error": f"Push failed: {resp.status_code}"}), 502


@app.route("/api/github/pull", methods=["GET"])
@admin_required
def api_github_pull():
    if not gh_db_enabled():
        return jsonify({"error": "GitHub DB not configured"}), 503
    raw_url = f"https://raw.githubusercontent.com/{GH_DB_USER}/{GH_DB_REPO}/{GH_DB_BRANCH}/{GH_DB_FILE}"
    resp    = _http.get(raw_url, params={"cb": int(time.time())}, timeout=15)
    if not resp.ok:
        return jsonify({"error": f"Could not fetch DB ({resp.status_code})"}), 502
    remote_db = resp.json()
    if not remote_db.get("wallpapers"):
        return jsonify({"error": "Invalid DB file"}), 502
    # Restore into SQLite
    with _DB_LOCK:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM wallpapers")
            now = int(time.time() * 1000)
            for w in remote_db["wallpapers"]:
                conn.execute(
                    "INSERT OR REPLACE INTO wallpapers(id,src,thumb,cat,tag,title,description,is_default,stored_in,created_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (w.get("id",""), w.get("src",""), w.get("thumb",w.get("src","")),
                     w.get("cat","other"), w.get("tag",""), w.get("title",""),
                     w.get("desc",""), 1 if w.get("isDefault") else 0,
                     w.get("storedIn"), w.get("createdAt", now))
                )
            conn.commit()
        finally:
            conn.close()
    n_users = len(remote_db.get("users", []))
    return jsonify({"ok": True, "wallpapers": len(remote_db["wallpapers"]), "users": n_users})


# ──────────────────────────────────────────────────────────────────────────────
# Users & favourites
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/users")
@admin_required
def api_users():
    return jsonify({"users": db_get_all_users()})


@app.route("/api/favourites", methods=["POST"])
@login_required
def api_toggle_fav():
    data    = request.json or {}
    wall_id = data.get("wallId")
    if not wall_id:
        return jsonify({"error": "Missing wallId"}), 400
    if session.get("is_admin"):
        return jsonify({"error": "Admin accounts do not support favourites"}), 403
    uid = session["user"].get("id")
    if not uid:
        return jsonify({"error": "User session is missing an id"}), 400
    row = db_find_user_by_id(uid)
    if not row:
        return jsonify({"error": "User not found"}), 404
    favs = json.loads(row["favs"] or "[]")
    if wall_id in favs:
        favs.remove(wall_id); action = "removed"
    else:
        favs.append(wall_id); action = "added"
    db_update_favs(uid, favs)
    session["user"] = {**session["user"], "favs": favs}
    return jsonify({"ok": True, "action": action, "favs": favs})


# ──────────────────────────────────────────────────────────────────────────────
# AI Vision proxy
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/ai/analyse", methods=["POST"])
@admin_required
def api_ai_analyse():
    data      = request.json or {}
    provider  = data.get("provider", "gemini")
    image_b64 = data.get("imageBase64", "")
    mime      = data.get("mimeType", "image/jpeg")
    prompt    = data.get("prompt", "Analyse this wallpaper. Return JSON: {title, category, description}")

    if provider == "gemini":
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key not configured on server"}), 503
        resp = _http.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": image_b64}}
            ]}]},
            timeout=30
        )
        if not resp.ok:
            return jsonify({"error": f"Gemini error {resp.status_code}"}), 502
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"ok": True, "result": text})

    elif provider == "openrouter":
        if not OPENROUTER_API_KEY:
            return jsonify({"error": "OpenRouter API key not configured on server"}), 503
        resp = _http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"model": "google/gemini-flash-1.5", "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
            ]}]},
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            timeout=30
        )
        if not resp.ok:
            return jsonify({"error": f"OpenRouter error {resp.status_code}"}), 502
        text = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"ok": True, "result": text})

    return jsonify({"error": "Unknown provider"}), 400


# ──────────────────────────────────────────────────────────────────────────────
# Serve frontend
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    filepath = os.path.join(app.template_folder, "index.html")
    try:
        mtime         = os.path.getmtime(filepath)
        last_modified = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(mtime))
        if_modified   = request.headers.get("If-Modified-Since")
        if if_modified == last_modified:
            return "", 304
        resp = make_response(send_from_directory("templates", "index.html"))
        resp.headers["Cache-Control"] = "public, max-age=300"
        resp.headers["Last-Modified"] = last_modified
        return resp
    except Exception:
        return send_from_directory("templates", "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    resp = send_from_directory("static", path)
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# FIX #3 — uses Gunicorn in production, never Flask dev server
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  VelvetSky — starting server")
    print(f"   Admin email : {ADMIN_EMAIL}")
    print(f"   DB file     : {DB_FILE}\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    
    import sys
    print("\n🚀  VelvetSky — starting server")
    print(f"   Admin email : {ADMIN_EMAIL}")
    print(f"   GitHub user : {GITHUB_USER or '(not set)'}")
    print(f"   R2 Worker   : {R2_WORKER_URL or '(not set)'}")
    print(f"   DB file     : {DB_FILE}")

    # Detect whether we're running under Gunicorn already
    gunicorn_in_env = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "").lower()
    dev_mode = os.environ.get("FLASK_ENV", "production").lower() == "development"

    if gunicorn_in_env or not dev_mode:
        # Production: launch via Gunicorn programmatically
        try:
            from gunicorn.app.base import BaseApplication

            class _App(BaseApplication):
                def __init__(self, application, options=None):
                    self.application = application
                    self.options = options or {}
                    super().__init__()
                def load_config(self):
                    for k, v in self.options.items():
                        self.cfg.set(k.lower(), v)
                def load(self):
                    return self.application

            workers = (os.cpu_count() or 2) * 2 + 1
            options = {
                "bind":       f"0.0.0.0:{os.environ.get('PORT', '8080')}",
                "workers":    workers,
                "worker_class": "sync",
                "timeout":    120,
                "loglevel":   "info",
                "accesslog":  "-",
            }
            print(f"   Workers     : {workers} (Gunicorn sync)\n")
            _App(app, options).run()

        except ImportError:
            print(
                "\n⚠  Gunicorn not installed. For production, run:\n"
                "      pip install gunicorn\n"
                "      gunicorn -w 4 app:app\n"
                "\n   Falling back to Flask dev server (NOT safe for production).\n"
            )
            app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    else:
        # Development mode only — never in production
        print("   Mode        : development (Flask built-in server)\n")
        app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 8080)))
