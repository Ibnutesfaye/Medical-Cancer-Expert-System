"""
db/seed.py — Create tables and seed the default admin user.

Run once before starting the server:
    venv\\Scripts\\python.exe db/seed.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import create_tables, engine
from sqlalchemy.orm import Session
from services.user_service import create_user, get_user_by_username
from schemas.user import UserCreate
from dotenv import load_dotenv

load_dotenv()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")


def seed():
    print("Creating tables...")
    create_tables()
    print("Tables created.")

    with Session(engine) as db:
        if not get_user_by_username(db, ADMIN_USERNAME):
            create_user(db, UserCreate(
                username=ADMIN_USERNAME,
                password=ADMIN_PASSWORD,
                is_admin=True,
                full_name="System Admin",
            ))
            print(f"Admin user '{ADMIN_USERNAME}' created.")
        else:
            print(f"Admin user '{ADMIN_USERNAME}' already exists.")

    print("Seed complete.")


if __name__ == "__main__":
    seed()
