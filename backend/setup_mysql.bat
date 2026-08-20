@echo off
echo ========================================
echo Medical Cancer Expert System - MySQL Setup
echo ========================================
echo.

echo [1/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv
    pause
    exit /b 1
)
echo OK
echo.

echo [2/5] Installing MySQL dependencies...
pip install --no-cache-dir sqlalchemy==2.0.36 pymysql==1.1.1 cryptography==43.0.3 passlib[bcrypt]==1.7.4
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo OK
echo.

echo [3/5] Testing MySQL connection...
python -c "from sqlalchemy import create_engine, text; import os; from dotenv import load_dotenv; load_dotenv(); url = os.getenv('DATABASE_URL'); print('Connecting to:', url); engine = create_engine(url); conn = engine.connect(); result = conn.execute(text('SELECT VERSION()')); print('Connected! MySQL version:', result.fetchone()[0])"
if errorlevel 1 (
    echo ERROR: Cannot connect to MySQL
    echo.
    echo Make sure:
    echo - XAMPP MySQL is running
    echo - DATABASE_URL in .env is correct: mysql+pymysql://root:@localhost:3306/medical_chatbot
    echo - Database 'medical_chatbot' exists in phpMyAdmin
    pause
    exit /b 1
)
echo OK
echo.

echo [4/5] Creating tables and seeding admin user...
python db\seed.py
if errorlevel 1 (
    echo ERROR: Failed to create tables
    pause
    exit /b 1
)
echo OK
echo.

echo [5/5] Starting backend server...
echo.
echo ========================================
echo Setup complete! Starting server...
echo Backend will run at: http://localhost:8000
echo Health check: http://localhost:8000/health
echo API docs: http://localhost:8000/docs
echo ========================================
echo.
python main_v2.py
