"""
Quick test script to verify MySQL connection.
Run this to check if your DATABASE_URL is correct.

Usage:
    venv\\Scripts\\python.exe test_mysql_connection.py
"""

import sys
import os
from dotenv import load_dotenv

print("=" * 60)
print("MySQL Connection Test")
print("=" * 60)
print()

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

print(f"DATABASE_URL: {DATABASE_URL}")
print()

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL not found in .env file")
    sys.exit(1)

# Test SQLAlchemy connection
print("Testing SQLAlchemy connection...")
try:
    from sqlalchemy import create_engine, text
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT VERSION()"))
        version = result.fetchone()[0]
        print(f"✅ Connected successfully!")
        print(f"   MySQL version: {version}")
        print()
        
        # Check if database exists
        result = conn.execute(text("SELECT DATABASE()"))
        db_name = result.fetchone()[0]
        print(f"✅ Current database: {db_name}")
        print()
        
        # List existing tables
        result = conn.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result.fetchall()]
        if tables:
            print(f"✅ Existing tables ({len(tables)}):")
            for table in tables:
                print(f"   - {table}")
        else:
            print("ℹ️  No tables yet (run db/seed.py to create them)")
        print()
        
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print()
    print("Common issues:")
    print("1. XAMPP MySQL not running - start it in XAMPP Control Panel")
    print("2. Database 'medical_chatbot' doesn't exist - create it in phpMyAdmin")
    print("3. Wrong password in DATABASE_URL - XAMPP default is no password (root:@)")
    print("4. MySQL dependencies not installed - run: pip install sqlalchemy pymysql")
    sys.exit(1)

print("=" * 60)
print("✅ All checks passed! MySQL is ready.")
print("=" * 60)
