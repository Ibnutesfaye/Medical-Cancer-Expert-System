import os
import sys
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
db_url = os.getenv('DATABASE_URL')

result = {"url": db_url}

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result["status"] = "SUCCESS"
except Exception as e:
    result["status"] = "FAILED"
    result["error"] = str(e)
    result["type"] = str(type(e))

with open("db_test_result.json", "w") as f:
    json.dump(result, f, indent=2)
