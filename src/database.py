import sqlite3
import os
import hashlib
import datetime
import logging

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.db")

def get_db_connection():
    """Create a database connection to SQLite"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the users table if it does not exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'Trader',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Database init error: {e}")

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = "newssense_salt_2026"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def register_or_login_user(email: str, password: str):
    """
    Register a new user or login existing user.
    Stores email and hashed password in SQLite database.
    Returns: (success: bool, user_data: dict, message: str)
    """
    if not email or not password:
        return False, None, "Email and password are required."

    email = email.strip().lower()
    hashed = hash_password(password)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if user already exists
    cursor.execute("SELECT id, email, password_hash, role, created_at FROM users WHERE email = ?", (email,))
    existing = cursor.fetchone()

    if existing:
        # User exists - verify password
        if existing["password_hash"] == hashed:
            conn.close()
            return True, {
                "id": existing["id"],
                "email": existing["email"],
                "role": existing["role"]
            }, "Login successful."
        else:
            conn.close()
            return False, None, "Invalid password for this email."
    else:
        # New user - register into database
        try:
            cursor.execute(
                "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, 'Trader', ?)",
                (email, hashed, datetime.datetime.now().isoformat())
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            logger.info(f"New user registered in database: {email} (ID: {user_id})")
            return True, {
                "id": user_id,
                "email": email,
                "role": "Trader"
            }, "Account registered successfully."
        except Exception as e:
            conn.close()
            logger.error(f"Error saving user to database: {e}")
            return False, None, f"Database error: {str(e)}"

# Auto-initialize database on import
init_db()
