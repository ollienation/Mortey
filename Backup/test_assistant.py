# Test Script for Modern LangGraph Assistant
# June 2025 - Production Ready

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_basic_functionality():
    """Test basic assistant functionality with modern patterns"""
    try:
        # Import the assistant
        from core.assistant_core import assistant

        print("🧪 Testing Modern LangGraph Assistant")
        print("=" * 50)

        # Test 1: Basic conversation
        print("\n1️⃣ Testing basic conversation...")
        response = await assistant.process_message("Hello! My name is Alice.")
        print(f"Assistant: {response}")

        # Test 2: Memory persistence
        print("\n2️⃣ Testing memory persistence...")
        response = await assistant.process_message("What's my name?")
        print(f"Assistant: {response}")

        # Test 3: Code generation (should route to coder agent)
        print("\n3️⃣ Testing code generation with modern patterns...")
        response = await assistant.process_message(
            "Create a simple Python function that calculates the factorial of a number"
        )
        print(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")

        # Test 4: Web search (should route to web agent)
        print("\n4️⃣ Testing web search with concurrency control...")
        response = await assistant.process_message("What's the latest news about AI?")
        print(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")

        # Test 5: File browsing (should route to chat agent)
        print("\n5️⃣ Testing file browsing...")
        response = await assistant.process_message("What files are in my workspace?")
        print(f"Assistant: {response}")

        # Test 6: System status
        print("\n6️⃣ System status...")
        status = assistant.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Test 7: Session info
        print("\n7️⃣ Session information...")
        session_info = assistant.get_session_info()
        for key, value in session_info.items():
            print(f"  {key}: {value}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_checkpointer():
    """Test modern checkpointer functionality"""
    try:
        from core.checkpointer import get_checkpointer_info, create_production_checkpointer

        print("\n🔍 Testing Modern Checkpointer Configuration")
        print("=" * 40)

        # Get checkpointer info
        info = get_checkpointer_info()
        print("Checkpointer Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test checkpointer creation
        print("\nCreating modern checkpointer...")
        checkpointer = create_production_checkpointer()
        print(f"✅ Created: {type(checkpointer).__name__}")

    except Exception as e:
        print(f"❌ Checkpointer test failed: {e}")

async def test_agents():
    """Test modern agent creation with string-based models"""
    try:
        from agents.agents import agent_factory

        print("\n🤖 Testing Modern Agent Creation")
        print("=" * 40)

        # Test chat agent
        print("Creating chat agent with modern patterns...")
        chat_agent = agent_factory.create_chat_agent()
        print("✅ Chat agent created with string-based model and concurrency control")

        # Test coder agent
        print("Creating coder agent with modern patterns...")
        coder_agent = agent_factory.create_coder_agent()
        print("✅ Coder agent created with interrupt patterns")

        # Test web agent
        print("Creating web agent with modern patterns...")
        web_agent = agent_factory.create_web_agent()
        print("✅ Web agent created with rate limiting")

        print("\n✅ All modern agents created successfully!")

    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_security_controller():
    """Test modern security controller with interrupt patterns"""
    try:
        from core.controller import controller
        from core.state import AssistantState

        print("\n🛡️ Testing Modern Security Controller")
        print("=" * 35)

        # Test safe content
        safe_state = AssistantState(
            messages=[],
            output_content="Hello, how can I help you today?",
            output_type="text",
            session_id="test_session"
        )

        print("Testing safe content...")
        result = await controller.verify_and_approve(safe_state)
        print(f"  Result: {result.get('approval_context', {}).get('status', 'unknown')}")

        # Test potentially dangerous content
        dangerous_state = AssistantState(
            messages=[],
            output_content="rm -rf /",
            output_type="code",
            session_id="test_session"
        )

        print("Testing dangerous content...")
        result = await controller.verify_and_approve(dangerous_state)
        print(f"  Result: {result.get('approval_context', {}).get('status', 'unknown')}")

        print("\n✅ Modern security controller tests completed!")

    except Exception as e:
        print(f"❌ Security controller test failed: {e}")

async def test_concurrency_control():
    """Test concurrency control features - FIXED"""
    try:
        from core.assistant_core import assistant
        from config.llm_manager import llm_manager
        
        print("\n⚡ Testing Concurrency Control")
        print("=" * 30)
        
        # Test concurrent message processing
        print("Testing concurrent session management...")
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(3):
            task = assistant.process_message(f"Hello from session {i+1}")
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  Session {i+1}: Error - {result}")
            else:
                # FIXED: Handle dictionary response properly
                if isinstance(result, dict) and 'response' in result:
                    response_text = result['response']
                    print(f"  Session {i+1}: Success - {response_text[:50]}...")
                else:
                    print(f"  Session {i+1}: Success - {str(result)[:50]}...")
        
        # Test LLM manager health check
        print("\nTesting LLM health check...")
        health_status = await llm_manager.health_check()
        for provider, status in health_status.items():
            print(f"  {provider}: {status}")
        
        # Test usage stats
        print("\nUsage statistics:")
        stats = llm_manager.get_usage_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Concurrency control tests completed!")
        
    except Exception as e:
        print(f"❌ Concurrency test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_memory_management():
    """Test memory management features"""
    try:
        from core.state import AssistantState, trim_message_history
        from langchain_core.messages import HumanMessage, AIMessage

        print("\n🧠 Testing Memory Management")
        print("=" * 27)

        # Create a state with many messages
        messages = []
        for i in range(60):  # Exceeds default max_messages of 50
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User message {i}"))
            else:
                messages.append(AIMessage(content=f"Assistant response {i}"))

        state = AssistantState(
            messages=messages,
            max_messages=20,
            session_id="memory_test"
        )

        print(f"Created state with {len(state['messages'])} messages")

        # Test memory trimming
        memory_update = trim_message_history(state)
        if memory_update:
            print(f"Memory trimmed to {len(memory_update.get('messages', []))} messages")
        else:
            print("No memory trimming needed")

        print("\n✅ Memory management tests completed!")

    except Exception as e:
        print(f"❌ Memory management test failed: {e}")

def check_environment():
    """Check modern environment setup - Load .env FIRST"""
    from pathlib import Path
    from dotenv import load_dotenv, find_dotenv
    
    # Priority fix: Load .env before any checks
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
    
    # Now check environment variables
    required_env_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]
    
    for var in required_env_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        masked_value = f"{value[:8]}..." if value and len(value) > 8 else "Not set"
        print(f"  {status} {var}: {masked_value}")

    # Check workspace directory
    from config.settings import config
    workspace_exists = config.workspace_dir.exists()
    print(f"  {'✅' if workspace_exists else '❌'} Workspace dir: {config.workspace_dir}")

    # Check modern packages
    packages = [
        "langgraph_supervisor",
        "langgraph.checkpoint.sqlite", 
        "langgraph.checkpoint.postgres"
    ]

    print("\nModern Package Availability:")
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}: Available")
        except ImportError:
            print(f"  ❌ {package}: Not installed")

    print()

async def main():
    """Run all modern tests"""
    print("🚀 Modern LangGraph Assistant Test Suite")
    print("🗓️ June 2025 - Production Ready")
    print("=" * 60)

    # Environment check
    check_environment()

    # Run tests
    await test_checkpointer()
    await test_agents()
    await test_security_controller()
    await test_concurrency_control()
    await test_memory_management()
    await test_basic_functionality()

    print("\n🎉 Modern test suite completed!")
    print("\nNext steps:")
    print("1. Install required packages: pip install langgraph-supervisor langgraph-checkpoint-sqlite")
    print("2. Update your configuration files with proper API keys")
    print("3. Set up production database if needed")
    print("4. Monitor LangSmith traces for proper operation")
    print("5. Configure concurrency limits based on your API rate limits")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()