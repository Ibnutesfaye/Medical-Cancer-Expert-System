"""Test the chat API endpoint directly."""

import requests
import json
import os

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30

def test_chat():
    print("=== Testing Chat API Endpoint ===\n")
    
    # 1. Login
    print("1. Logging in...")
    login_response = requests.post(
        f"{API_URL}/auth/login",
        json={
            "username": os.getenv("ADMIN_USERNAME", "admin"),
            "password": os.environ["ADMIN_PASSWORD"],
        },
        timeout=REQUEST_TIMEOUT,
    )
    
    if login_response.status_code != 200:
        print(f"✗ Login failed: {login_response.text}")
        return
    
    token = login_response.json()["access_token"]
    print(f"   ✓ Got token: {token[:20]}...")
    
    # 2. Send chat request
    print("\n2. Sending chat request...")
    query = "What is cancer?"
    
    response = requests.post(
        f"{API_URL}/chat",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "query": query,
            "conversation_history": []
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    
    if response.status_code != 200:
        print(f"✗ Chat failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print(f"   ✓ Status: {response.status_code}")
    print("\n3. Streaming response:")
    print("   ", end="", flush=True)
    
    # Read streaming response
    token_count = 0
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = line_str[6:]
                
                if data == '[DONE]':
                    print("\n\n   ✓ Stream complete")
                    break
                elif data.startswith('[CITATIONS]'):
                    citations = json.loads(data[11:])
                    print(f"\n\n   ✓ Citations: {len(citations)}")
                    for i, c in enumerate(citations[:2], 1):
                        print(f"      {i}. {c['document_name']}")
                elif data.startswith('[ERROR]'):
                    print(f"\n\n   ✗ Error: {data[7:]}")
                    break
                else:
                    # Unescape newlines
                    data = data.replace('\\n', '\n')
                    print(data, end="", flush=True)
                    token_count += 1
    
    print(f"\n   ✓ Received {token_count} tokens")

if __name__ == "__main__":
    try:
        test_chat()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
