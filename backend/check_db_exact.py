import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
db_url = os.getenv('DATABASE_URL')
print(f"URL: {db_url}")

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        print("SUCCESS")
except Exception as e:
    print(f"FAILED EXACT ERROR: {str(e)}")
