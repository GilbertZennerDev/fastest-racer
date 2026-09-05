"""Minimal auth + subscription-tier tracking for the SaaS gate.

Deliberately simple: SQLite (no server to run), stdlib PBKDF2 password
hashing (no bcrypt native build to fight with), signed cookies via
itsdangerous instead of a full JWT library. This is enough for an MVP —
swap in a managed auth provider later if the product needs SSO, password
reset flows, etc.
"""
import hashlib
import os
import secrets
import sqlite3
import time
from pathlib import Path

from fastapi import Cookie, HTTPException
from itsdangerous import BadSignature, URLSafeTimedSerializer

DB_PATH = Path(os.environ.get("USERS_DB_PATH", str(Path(__file__).resolve().parent / "users.db")))
SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "dev-secret-change-in-production")
COOKIE_NAME = "session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

serializer = URLSafeTimedSerializer(SECRET_KEY, salt="fastest-racer-session")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'free',
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                created_at REAL NOT NULL
            )
        """)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def create_user(email: str, password: str) -> int:
    email = email.strip().lower()
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, salt, tier, created_at) VALUES (?, ?, ?, 'free', ?)",
                (email, password_hash, salt, time.time()),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        return cur.lastrowid


def verify_login(email: str, password: str):
    email = email.strip().lower()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if _hash_password(password, row["salt"]) != row["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return dict(row)


def get_user_by_id(user_id: int):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return dict(row) if row else None


def set_stripe_customer(user_id: int, customer_id: str):
    with _connect() as conn:
        conn.execute("UPDATE users SET stripe_customer_id = ? WHERE id = ?", (customer_id, user_id))


def set_subscription(user_id: int, tier: str, subscription_id: str | None):
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET tier = ?, stripe_subscription_id = ? WHERE id = ?",
            (tier, subscription_id, user_id),
        )


def get_user_by_stripe_customer(customer_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
    return dict(row) if row else None


def make_session_cookie(user_id: int) -> str:
    return serializer.dumps({"user_id": user_id})


def read_session_cookie(token: str):
    try:
        data = serializer.loads(token, max_age=COOKIE_MAX_AGE)
    except BadSignature:
        return None
    return data.get("user_id")


def get_current_user(session: str | None = Cookie(default=None)):
    """FastAPI dependency: returns the user dict, or None if not logged in."""
    if not session:
        return None
    user_id = read_session_cookie(session)
    if user_id is None:
        return None
    return get_user_by_id(user_id)


def require_user(session: str | None = Cookie(default=None)):
    user = get_current_user(session)
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user
