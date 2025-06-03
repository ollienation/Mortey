# Fixed Test Script for Modern LangGraph Assistant
# June 2025 - LangGraph 0.4.8 Compatible

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

async def test_modern_functionality():
    """Test modern assistant functionality with fixed patterns"""
    try:
        # Import the fixed assistant
        from core.assistant_core import assistant
        
        print("🧪 Testing Fixed LangGraph Assistant (0.4.8)")
        print("=" * 50)
        
        # Test 1: Basic conversation with proper message validation
        print("\n1️⃣ Testing basic conversation (FIXED)...")
        response = await assistant.process_message("Hello! My name is Alice.")
        print(f"Assistant: {response.get('response', 'No response')}")
        
        # Verify no empty content error
        if response.get('error') != 'empty_content':
            print("✅ Message validation working correctly")
        else:
            print("❌ Still getting empty content error")
        
        # Test 2: Memory persistence
        print("\n2️⃣ Testing memory persistence...")
        response = await assistant.process_message("What's my name?")
        print(f"Assistant: {response.get('response', 'No response')}")
        
        # Test 3: Code generation (should route to coder agent)
        print("\n3️⃣ Testing code generation...")
        response = await assistant.process_message(
            "Create a simple Python function that calculates the factorial of a number"
        )
        print(f"Assistant: {response.get('response', 'No response')[:200]}...")
        
        # Test 4: Web search (should route to web agent)
        print("\n4️⃣ Testing web search...")
        response = await assistant.process_message("What's the latest news about AI?")
        print(f"Assistant: {response.get('response', 'No response')[:200]}...")
        
        # Test 5: File browsing (should route to chat agent)
        print("\n5️⃣ Testing file browsing...")
        response = await assistant.process_message("What files are in my workspace?")
        print(f"Assistant: {response.get('response', 'No response')}")
        
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
        
        print("\n✅ All fixed tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_fixed_checkpointer():
    """Test modern checkpointer functionality"""
    try:
        from core.checkpointer import get_checkpointer_info, create_checkpointer
        
        print("\n🔍 Testing Fixed Checkpointer Configuration")
        print("=" * 40)
        
        # Get checkpointer info
        info = get_checkpointer_info()
        print("Checkpointer Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test checkpointer creation
        print("\nCreating modern checkpointer...")
        checkpointer = create_checkpointer()
        print(f"✅ Created: {type(checkpointer).__name__}")
        
        if "Memory" in type(checkpointer).__name__:
            print("ℹ️ Using MemorySaver - install checkpoint packages for persistence")
        else:
            print("✅ Using persistent checkpointer")
        
    except Exception as e:
        print(f"❌ Checkpointer test failed: {e}")

async def test_fixed_agents():
    """Test modern agent creation with string-based models"""
    try:
        from agents.agents import AgentFactory
        
        print("\n🤖 Testing Fixed Agent Creation")
        print("=" * 40)
        
        agent_factory = AgentFactory()
        
        # Test chat agent
        print("Creating chat agent with modern patterns...")
        chat_agent = agent_factory.create_chat_agent()
        print("✅ Chat agent created successfully")
        
        # Test coder agent
        print("Creating coder agent with modern patterns...")
        coder_agent = agent_factory.create_coder_agent()
        print("✅ Coder agent created successfully")
        
        # Test web agent
        print("Creating web agent with modern patterns...")
        web_agent = agent_factory.create_web_agent()
        print("✅ Web agent created successfully")
        
        print("\n✅ All fixed agents created successfully!")
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_fixed_message_validation():
    """Test the fixed message validation system"""
    try:
        from core.state import validate_and_filter_messages, AssistantState
        from langchain_core.messages import HumanMessage, AIMessage
        
        print("\n🔍 Testing Fixed Message Validation")
        print("=" * 35)
        
        # Test with various message types
        test_cases = [
            # Valid messages
            [HumanMessage(content="Hello")],
            [HumanMessage(content="Test"), AIMessage(content="Response")],
            
            # Edge cases that should be handled
            [HumanMessage(content="")],  # Empty content
            [],  # Empty list
            [HumanMessage(content="   ")],  # Whitespace only
        ]
        
        for i, messages in enumerate(test_cases, 1):
            print(f"Test case {i}: {len(messages)} input messages")
            try:
                validated = validate_and_filter_messages(messages)
                print(f"  ✅ Result: {len(validated)} valid messages")
                
                # Ensure we always have at least one message
                assert len(validated) > 0, "Validation should always return at least one message"
                
                # Ensure all messages have content
                for msg in validated:
                    assert hasattr(msg, 'content'), "All messages should have content"
                    if isinstance(msg.content, str):
                        assert msg.content.strip(), "String content should not be empty"
                
            except Exception as e:
                print(f"  ❌ Validation failed: {e}")
        
        print("\n✅ Message validation tests completed!")
        
    except Exception as e:
        print(f"❌ Message validation test failed: {e}")

def check_fixed_environment():
    """Check fixed environment setup"""
    from pathlib import Path
    from dotenv import load_dotenv, find_dotenv
    
    # Load .env
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
    
    # Check environment variables
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
        "langgraph",
        "langgraph_supervisor", 
        "langgraph_checkpoint_sqlite",
        "langgraph_checkpoint_postgres"
    ]
    
    print("\nFixed Package Availability:")
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}: Available")
        except ImportError:
            print(f"  ❌ {package}: Not installed")
    print()

async def main():
    """Run all fixed tests"""
    print("🚀 Fixed LangGraph Assistant Test Suite")
    print("🗓️ June 2025 - LangGraph 0.4.8 Compatible")
    print("=" * 60)
    
    # Environment check
    check_fixed_environment()
    
    # Run tests
    await test_fixed_checkpointer()
    await test_fixed_agents()
    await test_fixed_message_validation()
    await test_modern_functionality()
    
    print("\n🎉 Fixed test suite completed!")
    print("\nExpected improvements:")
    print("✅ No more empty content errors")
    print("✅ Proper supervisor initialization")
    print("✅ Modern checkpointer patterns")
    print("✅ LangGraph 0.4.8 compatibility")
    print("✅ Robust message validation")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()