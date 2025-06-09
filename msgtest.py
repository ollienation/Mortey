# Quick test script
import asyncio
from core.assistant_core import AssistantCore

async def test_full_system():
    try:
        assistant = AssistantCore()
        await assistant.initialize()
        
        # Test message processing with persistence
        response = await assistant.process_message(
            "Hello! Can you tell me about yourself?",
            session_id="test_postgres_session6",
            user_id="test_user"
        )
        
        print(f"✅ Response: {response['response'][:100]}...")
        print(f"✅ Session ID: {response['session_id']}")
        
        # Test session persistence
        response2 = await assistant.process_message(
            "What did I just ask you?",
            session_id="test_postgres_session6",
            user_id="test_user"
        )
        
        print(f"✅ Follow-up: {response2['response'][:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_system())
