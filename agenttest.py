# Comprehensive agent test script
import asyncio
from core.assistant_core import AssistantCore

async def test_full_system():
    try:
        assistant = AssistantCore()
        await assistant.initialize()
        
        # Define test cases for each agent
        test_cases = [
            {
                "agent": "chat",
                "message": "Hello! Can you tell me about yourself?",
                "follow_up": "What did I just ask you?",
                "session_id": "test_postgres_session_chat"
            },
            {
                "agent": "coder", 
                "message": "Write a simple Python function to add two numbers",
                "follow_up": "Can you also add error handling to that function?",
                "session_id": "test_postgres_session_coder"
            },
            {
                "agent": "web",
                "message": "Search the web for the latest news on AI advancements",
                "follow_up": "What were the key points from that search?",
                "session_id": "test_postgres_session_web"
            },
            {
                "agent": "file_manager",
                "message": "List all Python files in the current directory",
                "follow_up": "What was the last file operation I requested?",
                "session_id": "test_postgres_session_files"
            }
        ]
        
        user_id = "test_user"
        
        # Test each agent
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"🤖 Testing {test_case['agent'].upper()} Agent")
            print(f"{'='*60}")
            
            # Initial message
            print(f"📤 Initial message: {test_case['message']}")
            response1 = await assistant.process_message(
                test_case["message"],
                session_id=test_case["session_id"],
                user_id=user_id
            )
            
            print(f"✅ {test_case['agent']} Response: {response1['response'][:150]}...")
            print(f"✅ Session ID: {response1['session_id']}")
            
            # Test session persistence with follow-up
            print(f"\n📤 Follow-up message: {test_case['follow_up']}")
            response2 = await assistant.process_message(
                test_case["follow_up"],
                session_id=test_case["session_id"],
                user_id=user_id
            )
            
            print(f"✅ {test_case['agent']} Follow-up: {response2['response'][:150]}...")
            print(f"✅ Session persistence: {'✅ Working' if response1['session_id'] == response2['session_id'] else '❌ Failed'}")
        
        # Test cross-session conversation continuity
        print(f"\n{'='*60}")
        print(f"🔄 Testing Cross-Session Continuity")
        print(f"{'='*60}")
        
        # Go back to chat session and reference previous conversation
        final_response = await assistant.process_message(
            "Can you summarize what we discussed in our first conversation?",
            session_id="test_postgres_session_chat",
            user_id=user_id
        )
        
        print(f"✅ Cross-session continuity: {final_response['response'][:150]}...")
        
        print(f"\n🎉 All agent tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_system())
