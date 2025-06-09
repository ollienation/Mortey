# test-current-assistant.py - ✅ ACCURATE TEST SUITE FOR CURRENT CODEBASE
"""
Accurate Testing Suite for LangGraph 0.4.8 Assistant
Based on actual current implementation - June 2025

This test suite only tests components that actually exist in your codebase.
"""

import asyncio
import os
import sys
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging to match your patterns
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_suite")

class SimpleTestResult:
    """Simple test result tracking"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = time.time()
    
    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"✅ PASS: {test_name}")
    
    def record_fail(self, test_name: str, error: str):
        self.tests_run += 1
        self.tests_failed += 1
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {error}")
    
    def print_summary(self):
        duration = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Duration: {duration:.2f}s")
        if self.tests_failed == 0:
            print("🎉 ALL TESTS PASSED!")
        print(f"{'='*60}")

async def test_current_assistant():
    """Test the current assistant based on actual codebase"""
    results = SimpleTestResult()
    
    print("🧪 Testing Current LangGraph Assistant Implementation")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test 1: Core Imports
    try:
        print("\n1️⃣ Testing Core Imports...")
        from core.assistant_core import AssistantCore, AssistantSession
        from core.state import (
            AssistantState, create_optimized_state, StateValidator,
            validate_and_filter_messages_v3, safe_state_access
        )
        from agents.agents import AgentFactory
        from core.checkpointer import create_checkpointer, get_checkpointer_info
        from core.error_handling import ErrorHandler, ErrorType
        from config.settings import config
        from tools.file_tools import FileSystemTools
        results.record_pass("Core imports successful")
    except Exception as e:
        results.record_fail("Core imports", str(e))
        return results
    
    # Test 2: Configuration Loading
    try:
        print("\n2️⃣ Testing Configuration...")
        # Test config access
        workspace_exists = config.workspace_dir.exists()
        providers_count = len(config.providers)
        nodes_count = len(config.nodes)
        
        print(f"   Workspace: {config.workspace_dir} (exists: {workspace_exists})")
        print(f"   Providers: {providers_count}")
        print(f"   Nodes: {nodes_count}")
        
        if providers_count > 0 and nodes_count > 0:
            results.record_pass("Configuration loading")
        else:
            results.record_fail("Configuration loading", "No providers or nodes configured")
    except Exception as e:
        results.record_fail("Configuration loading", str(e))
    
    # Test 3: Checkpointer Creation
    try:
        print("\n3️⃣ Testing Checkpointer...")
        checkpointer_info = get_checkpointer_info()
        print(f"   Environment: {checkpointer_info['detected_environment']}")
        print(f"   Postgres available: {checkpointer_info['postgres_connection_sync_test_passed']}")
        
        checkpointer = await create_checkpointer(use_async=True)
        checkpointer_type = type(checkpointer).__name__
        print(f"   Created: {checkpointer_type}")
        results.record_pass(f"Checkpointer creation ({checkpointer_type})")
    except Exception as e:
        results.record_fail("Checkpointer creation", str(e))
    
    # Test 4: State Management
    try:
        print("\n4️⃣ Testing State Management...")
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Test state creation
        state = create_optimized_state(
            session_id="test_session",
            user_id="test_user"
        )
        
        # Test state validation
        is_valid, errors = StateValidator.validate_state(state)
        print(f"   State valid: {is_valid}")
        
        # Test message handling
        test_messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        
        filtered_messages = validate_and_filter_messages_v3(test_messages)
        print(f"   Message filtering: {len(test_messages)} -> {len(filtered_messages)}")
        
        # Test safe access
        session_id = safe_state_access(state, "session_id", "default")
        print(f"   Safe access: {session_id}")
        
        results.record_pass("State management")
    except Exception as e:
        results.record_fail("State management", str(e))
    
    # Test 5: Agent Factory
    try:
        print("\n5️⃣ Testing Agent Factory...")
        agent_factory = AgentFactory()
        
        # Test agent creation
        chat_agent = agent_factory.create_chat_agent()
        coder_agent = agent_factory.create_coder_agent()
        web_agent = agent_factory.create_web_agent()
        
        print(f"   Chat agent: {type(chat_agent).__name__}")
        print(f"   Coder agent: {type(coder_agent).__name__}")
        print(f"   Web agent: {type(web_agent).__name__}")
        
        # Test tool collection
        all_tools = agent_factory.get_all_tools()
        print(f"   Total tools: {len(all_tools)}")
        
        results.record_pass("Agent factory operations")
    except Exception as e:
        results.record_fail("Agent factory operations", str(e))
    
    # Test 6: File Tools
    try:
        print("\n6️⃣ Testing File Tools...")
        file_tools = FileSystemTools()
        tools = file_tools.get_tools()
        print(f"   File tools count: {len(tools)}")
        print(f"   Workspace: {file_tools.workspace_dir}")
        results.record_pass("File tools initialization")
    except Exception as e:
        results.record_fail("File tools initialization", str(e))
    
    # Test 7: Error Handling
    try:
        print("\n7️⃣ Testing Error Handling...")
        # Test error classification
        test_error = ConnectionError("Test connection error")
        error_response = ErrorHandler.handle_error(test_error, "test_context")
        
        print(f"   Error type: {error_response['error']}")
        print(f"   Retryable: {error_response['retryable']}")
        print(f"   Response: {error_response['response'][:50]}...")
        
        # Test async error handling
        async def failing_function():
            raise ValueError("Test async error")
        
        async_result = await ErrorHandler.with_error_handling(failing_function, context="async_test")
        print(f"   Async handling: {async_result.get('error', 'handled')}")
        
        results.record_pass("Error handling")
    except Exception as e:
        results.record_fail("Error handling", str(e))
    
    # Test 8: Supervisor
    try:
        print("\n8️⃣ Testing Simplified Supervisor...")
        from core.simplified_supervisor import SimplifiedSupervisor, SupervisorConfig
        
        supervisor = SimplifiedSupervisor()
        config_obj = SupervisorConfig()
        
        print(f"   Default agent: {config_obj.default_agent}")
        print(f"   Max replays: {config_obj.max_replays}")
        
        # Test keyword configuration
        keywords = config_obj.routing_keywords
        print(f"   Routing keywords: {len(keywords)} agent types")
        
        results.record_pass("Supervisor configuration")
    except Exception as e:
        results.record_fail("Supervisor configuration", str(e))
    
    # Test 9: Assistant Core Initialization
    try:
        print("\n9️⃣ Testing Assistant Core...")
        assistant = AssistantCore()
        
        # Test initialization
        await assistant.initialize()
        
        # Check components
        have_supervisor = assistant.supervisor is not None
        have_checkpointer = assistant.checkpointer is not None
        have_agents = all([
            assistant.chat_agent is not None,
            assistant.coder_agent is not None,
            assistant.web_agent is not None
        ])
        
        print(f"   Supervisor: {have_supervisor}")
        print(f"   Checkpointer: {have_checkpointer}")
        print(f"   Agents: {have_agents}")
        print(f"   Setup complete: {assistant._setup_complete}")
        
        if assistant._setup_complete:
            results.record_pass("Assistant core initialization")
        else:
            results.record_fail("Assistant core initialization", "Setup not complete")
    except Exception as e:
        results.record_fail("Assistant core initialization", str(e))
        assistant = None
    
    # Test 10: Message Processing (if assistant initialized)
    if assistant and assistant._setup_complete:
        try:
            print("\n🔟 Testing Message Processing...")
            test_session_id = f"test_{int(time.time())}"
            
            # Test simple message
            response1 = await assistant.process_message(
                "Hello, I'm testing the system!",
                thread_id=test_session_id,
                user_id="test_user"
            )
            
            print(f"   Response 1 success: {response1.get('success', False)}")
            print(f"   Response 1: {response1.get('response', 'No response')[:50]}...")
            
            # Test follow-up
            response2 = await assistant.process_message(
                "Can you list files in the workspace?",
                thread_id=test_session_id,
                user_id="test_user"
            )
            
            print(f"   Response 2 success: {response2.get('success', False)}")
            print(f"   Agent used: {response2.get('agent_used', 'unknown')}")
            
            # Test session info
            session_info = assistant.get_session_info()
            print(f"   Session ID: {session_info.get('session_id', 'none')}")
            print(f"   Message count: {session_info.get('message_count', 0)}")
            
            if response1.get('success') and response2.get('success'):
                results.record_pass("Message processing")
            else:
                results.record_fail("Message processing", "One or more messages failed")
        except Exception as e:
            results.record_fail("Message processing", str(e))
    else:
        results.record_fail("Message processing", "Assistant not initialized")
    
    # Test 11: Session Management
    if assistant:
        try:
            print("\n1️⃣1️⃣ Testing Session Management...")
            session_id = f"session_test_{int(time.time())}"
            
            # Create session
            session = await assistant._get_or_create_session(session_id, "test_user")
            
            print(f"   Session created: {session.session_id}")
            print(f"   User ID: {session.user_id}")
            print(f"   Start time: {session.start_time}")
            print(f"   Message count: {session.message_count}")
            
            # Test session summary
            summary = session.get_summary()
            print(f"   Summary keys: {list(summary.keys())}")
            
            results.record_pass("Session management")
        except Exception as e:
            results.record_fail("Session management", str(e))
    
    # Test 12: System Status
    if assistant:
        try:
            print("\n1️⃣2️⃣ Testing System Status...")
            status = assistant.get_system_status()
            
            print(f"   Setup complete: {status.get('setup_complete', False)}")
            print(f"   Checkpointer type: {status.get('checkpointer_type', 'unknown')}")
            print(f"   Session active: {status.get('session_active', False)}")
            print(f"   Total sessions: {status.get('total_sessions', 0)}")
            
            results.record_pass("System status")
        except Exception as e:
            results.record_fail("System status", str(e))
    
    return results

async def main():
    """Main test execution"""
    try:
        results = await test_current_assistant()
        results.print_summary()
        return 0 if results.tests_failed == 0 else 1
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
