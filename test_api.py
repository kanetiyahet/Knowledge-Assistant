# test_api.py - Test the API directly
import requests
import json

# Test the API
url = "http://localhost:8000/ask"
data = {"query": "Who is Gandhi?"}

print("📤 Sending:", data)

try:
    response = requests.post(url, json=data)
    print(f"📥 Status: {response.status_code}")
    print(f"📥 Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Answer: {result.get('answer', 'No answer')}")
except Exception as e:
    print(f"❌ Error: {e}")