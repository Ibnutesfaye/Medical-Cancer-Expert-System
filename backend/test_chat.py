import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# We need a valid token. Let's just login first.
res = requests.post("http://localhost:8000/auth/login", json={"username": "admin", "password": "admin123"})
if not res.ok:
    print(f"Login failed: {res.status_code} {res.text}")
    exit(1)

token = res.json()["access_token"]
print("Logged in. Token:", token[:20], "...")

# Now test /chat
try:
    print("Sending POST to /chat...")
    res = requests.post(
        "http://localhost:8000/chat",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": "What is cancer?", "conversation_history": []},
        stream=True,
        timeout=10
    )
    print("Response Status:", res.status_code)
    for line in res.iter_lines():
        if line:
            print("Received:", line.decode("utf-8")[:100])
except Exception as e:
    print("Error:", str(e))
