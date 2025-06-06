# test_api.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_chat():
    """Test chat endpoint"""
    payload = {
        "message": "Hello! Can you help me create a Python function?",
        "user_id": "test_user"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Chat Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.json().get("session_id")

def test_session_info(session_id):
    """Test session info endpoint"""
    response = requests.get(f"{BASE_URL}/session/{session_id}")
    print(f"Session Info: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing Mortey Assistant API...")
    
    # Test health
    test_health()
    time.sleep(1)
    
    # Test chat
    session_id = test_chat()
    time.sleep(1)
    
    # Test session info
    if session_id:
        test_session_info(session_id)
