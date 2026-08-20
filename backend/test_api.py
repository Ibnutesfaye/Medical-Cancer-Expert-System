"""Quick test script to verify API key and LLM service."""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check API key
groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY loaded: {'Yes' if groq_key else 'No'}")
if groq_key:
    print(f"Key starts with: {groq_key[:10]}...")

# Test Groq client
try:
    from groq import Groq
    client = Groq(api_key=groq_key)
    
    print("\nTesting Groq API...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say 'API working' in 2 words"}],
        temperature=0.2,
        max_tokens=50,
        stream=False
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print("✓ Groq API is working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
