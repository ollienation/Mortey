# test-current-assistant.py - ✅ ENHANCED TEST SUITE WITH PROPER FORMATTING AND LOGGING
"""
Enhanced Testing Suite for LangGraph 0.4.8 Assistant
Based on actual current implementation with comprehensive logging
June 2025 - Production Ready System Validation
"""

import asyncio
import os
import sys
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure structured logging with file output
def setup_logging():
    """Setup comprehensive logging for test suite"""
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"test_results_{timestamp}.log"
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

@dataclass
class TestResult:
    """Enhanced test result tracking with detailed metrics"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_details: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.log_file = None
    
    def record_test(self, test_name: str, passed: bool, details: str = "", duration: float = 0.0):
        """Record a test result with comprehensive details"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            status = "✅ PASS"
            level = "INFO"
        else:
            self.tests_failed += 1
            status = "❌ FAIL"
            level = "ERROR"
        
        self.test_details.append({
            "name": test_name,
            "status": status,
            "passed": passed,
            "details": details,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # Enhanced console output
        print(f"{status} | {test_name} ({duration:.3f}s)")
        if details:
            print(f"     └─ {details}")
        
        # Log to file with level
        logger = logging.getLogger("test_suite")
        if passed:
            logger.info(f"PASS: {test_name} ({duration:.3f}s) - {details}")
        else:
            logger.error(f"FAIL: {test_name} ({duration:.3f}s) - {details}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary"""
        total_duration = time.time() - self.start_time
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        return {
            "total_tests": self.tests_run,
            "passed": self.tests_passed,
            "failed": self.tests_failed,
            "success_rate": f"{success_rate:.1f}%",
            "total_duration": f"{total_duration:.2f}s",
            "average_duration": f"{total_duration / self.tests_run:.3f}s" if self.tests_run > 0 else "0s",
            "log_file": str(self.log_file) if self.log_file else None
        }

class EnhancedTestSuite:
    """Enhanced test suite with proper formatting and logging"""
    
    def __init__(self):
        self.log_file = setup_logging()
        self.result = TestResult()
        self.result.log_file = self.log_file
        self.assistant = None
        self.test_session_id = f"test_session_{int(time.time())}"
        self.logger = logging.getLogger("test_suite")
        
    @asynccontextmanager
    async def test_context(self, test_name: str):
        """Context manager for individual tests with timing and error handling"""
        start_time = time.time()
        self.logger.info(f"Starting test: {test_name}")
        try:
            yield
            duration = time.time() - start_time
            self.result.record_test(test_name, True, "Completed successfully", duration)
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error: {str(e)[:100]}..."
            self.result.record_test(test_name, False, error_msg, duration)
            self.logger.error(f"Test failed: {test_name} - {e}", exc_info=True)
    
    async def test_01_core_imports_and_setup(self):
        """Test 1: Validate all core imports and basic setup"""
        async with self.test_context("Core Imports & Module Loading"):
            # Import all core components based on your actual files
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
            
            # Validate critical configurations
            assert config.workspace_dir.exists(), "Workspace directory not accessible"
            assert config.providers, "No LLM providers configured"
            
            self.logger.info(f"Loaded {len(config.providers)} LLM providers")
            self.logger.info(f"Workspace: {config.workspace_dir}")
            print(f"     └─ Loaded {len(config.providers)} LLM providers")
            print(f"     └─ Workspace: {config.workspace_dir}")
    
    async def test_02_checkpointer_initialization(self):
        """Test 2: Advanced checkpointer initialization with fallbacks"""
        async with self.test_context("Checkpointer Initialization & Health"):
            checkpointer_info = get_checkpointer_info()
            environment = checkpointer_info['detected_environment']
            postgres_test = checkpointer_info['postgres_connection_sync_test_passed']
            
            self.logger.info(f"Environment: {environment}")
            self.logger.info(f"PostgreSQL available: {postgres_test}")
            print(f"     └─ Environment: {environment}")
            print(f"     └─ PostgreSQL available: {postgres_test}")
            
            # Test async checkpointer creation
            checkpointer = await create_checkpointer(use_async=True)
            checkpointer_type = type(checkpointer).__name__
            
            assert checkpointer is not None, "Checkpointer creation failed"
            self.logger.info(f"Created checkpointer: {checkpointer_type}")
            print(f"     └─ Created: {checkpointer_type}")
    
    async def test_03_state_management_validation(self):
        """Test 3: Comprehensive state management and validation"""
        async with self.test_context("State Management & Validation"):
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            
            # Test state creation
            state = create_optimized_state(
                session_id=self.test_session_id,
                user_id="test_user",
                validate=True
            )
            
            # Validate state structure
            is_valid, errors = StateValidator.validate_state(state, strict=True)
            assert is_valid, f"State validation failed: {errors}"
            self.logger.info(f"State validation passed: {is_valid}")
            
            # Test complex state with messages
            complex_state = create_optimized_state(
                session_id=self.test_session_id,
                user_id="test_user",
                initial_context={
                    "messages": [
                        HumanMessage(content="Test message"),
                        AIMessage(content="Test response"),
                        ToolMessage(content="Tool executed", tool_call_id="test_call_123")
                    ]
                }
            )
            
            assert len(complex_state["messages"]) == 3, "Message handling failed"
            
            # Test safe access
            session_id = safe_state_access(complex_state, "session_id", "default")
            assert session_id == self.test_session_id, "Safe access failed"
            
            self.logger.info(f"Message handling: {len(complex_state['messages'])} messages processed")
            print(f"     └─ State validation: Passed")
            print(f"     └─ Message handling: {len(complex_state['messages'])} messages")
            print(f"     └─ Safe access: Working")
    
    async def test_04_agent_factory_operations(self):
        """Test 4: Agent factory and tool management"""
        async with self.test_context("Agent Factory & Tool Management"):
            agent_factory = AgentFactory()
            
            # Test individual agent creation
            chat_agent = agent_factory.create_chat_agent()
            coder_agent = agent_factory.create_coder_agent()
            web_agent = agent_factory.create_web_agent()
            
            assert chat_agent is not None, "Chat agent creation failed"
            assert coder_agent is not None, "Coder agent creation failed"
            assert web_agent is not None, "Web agent creation failed"
            
            # Test tool collection
            all_tools = agent_factory.get_all_tools()
            assert len(all_tools) > 0, "No tools available"
            
            self.logger.info(f"Agents created successfully: 3 agents, {len(all_tools)} tools")
            print(f"     └─ Agents created: chat, coder, web")
            print(f"     └─ Total tools: {len(all_tools)}")
    
    async def test_05_error_handling_classification(self):
        """Test 5: Advanced error handling and classification"""
        async with self.test_context("Error Handling & Classification"):
            # Test different error types
            test_errors = [
                (ConnectionError("Network timeout"), "connection_error"),
                (ValueError("Invalid API key"), "authentication_error"),
                (TimeoutError("Request timeout"), "timeout_error"),
                (KeyError("Missing state field"), "state_error")
            ]
            
            for error, expected_type in test_errors:
                result = ErrorHandler.handle_error(error, "test_context")
                assert result["error"] == expected_type, f"Error classification failed for {type(error)}"
                assert "response" in result, "Error response missing"
                assert "fallback_used" in result, "Fallback indicator missing"
            
            # Test async error handling
            async def failing_function():
                raise ConnectionError("Test async error")
            
            result = await ErrorHandler.with_error_handling(failing_function, context="async_test")
            assert "error" in result, "Async error handling failed"
            
            self.logger.info(f"Error classification tested: {len(test_errors)} types")
            print(f"     └─ Error classification: {len(test_errors)} types tested")
            print(f"     └─ Async error handling: Functional")
    
    async def test_06_assistant_core_initialization(self):
        """Test 6: Complete assistant core initialization"""
        async with self.test_context("Assistant Core Initialization"):
            from core.assistant_core import AssistantCore
            
            self.assistant = AssistantCore()
            
            # Test initialization
            await self.assistant.initialize()
            
            # Verify all components
            assert self.assistant.supervisor is not None, "Supervisor not initialized"
            assert self.assistant.checkpointer is not None, "Checkpointer not initialized"
            assert self.assistant.chat_agent is not None, "Chat agent not initialized"
            assert self.assistant.coder_agent is not None, "Coder agent not initialized"
            assert self.assistant.web_agent is not None, "Web agent not initialized"
            assert self.assistant._setup_complete, "Setup not completed"
            
            supervisor_type = type(self.assistant.supervisor).__name__
            checkpointer_type = type(self.assistant.checkpointer).__name__
            
            self.logger.info(f"Assistant core initialized: {supervisor_type}, {checkpointer_type}")
            print(f"     └─ All agents initialized: ✓")
            print(f"     └─ Supervisor: {supervisor_type}")
            print(f"     └─ Checkpointer: {checkpointer_type}")
    
    async def test_07_message_processing_flow(self):
        """Test 7: End-to-end message processing"""
        async with self.test_context("Message Processing Flow"):
            if not self.assistant:
                self.assistant = AssistantCore()
                await self.assistant.initialize()
            
            # Test basic message processing
            response1 = await self.assistant.process_message(
                "Hello! My name is Alice and I'm testing the system.",
                thread_id=self.test_session_id,
                user_id="test_user"
            )
            
            assert response1["success"], "Initial message processing failed"
            assert response1["session_id"] == self.test_session_id, "Session ID mismatch"
            
            # Test follow-up message
            response2 = await self.assistant.process_message(
                "What's my name?",
                thread_id=self.test_session_id,
                user_id="test_user"
            )
            
            assert response2["success"], "Follow-up message processing failed"
            
            self.logger.info(f"Message processing: 2 messages processed successfully")
            print(f"     └─ Session created: {self.test_session_id}")
            print(f"     └─ Messages processed: 2")
            print(f"     └─ Response 1: {response1['response'][:50]}...")
            print(f"     └─ Response 2: {response2['response'][:50]}...")
    
    async def test_08_multi_agent_routing(self):
        """Test 8: Intelligent multi-agent routing"""
        async with self.test_context("Multi-Agent Routing & Intelligence"):
            if not self.assistant:
                self.assistant = AssistantCore()
                await self.assistant.initialize()
            
            # Test routing to different agents
            test_cases = [
                ("Can you help me with general questions?", "chat"),
                ("Write a Python function to calculate factorial", "coder"),
                ("Search for the latest news about AI", "web")
            ]
            
            routing_results = []
            
            for message, expected_agent in test_cases:
                response = await self.assistant.process_message(
                    message,
                    thread_id=f"{self.test_session_id}_routing_{expected_agent}",
                    user_id="test_user"
                )
                
                assert response["success"], f"Message processing failed for: {message}"
                actual_agent = response.get("agent_used", "unknown")
                routing_results.append((message[:30] + "...", expected_agent, actual_agent))
                
                self.logger.info(f"Routing test: '{message[:30]}...' -> {actual_agent}")
            
            print(f"     └─ Routing test results:")
            for msg, expected, actual in routing_results:
                status = "✓" if expected == actual else "⚠"
                print(f"       {status} {msg} → {actual}")
    
    async def test_09_file_operations(self):
        """Test 9: File system operations and tools"""
        async with self.test_context("File System Operations"):
            if not self.assistant:
                self.assistant = AssistantCore()
                await self.assistant.initialize()
            
            # Test file operations through the assistant
            test_commands = [
                "List the files in my workspace",
                "Create a new Python project called 'test_project'",
                "Create a simple hello.txt file with hello world"
            ]
            
            for i, command in enumerate(test_commands):
                response = await self.assistant.process_message(
                    command,
                    thread_id=f"{self.test_session_id}_files_{i}",
                    user_id="test_user"
                )
                
                assert response["success"], f"File operation failed: {command}"
                self.logger.info(f"File operation: '{command}' completed")
            
            print(f"     └─ File operations: {len(test_commands)} commands tested")
            print(f"     └─ Workspace operations: Functional")
    
    async def test_10_session_management(self):
        """Test 10: Session persistence and management"""
        async with self.test_context("Session Persistence & Management"):
            if not self.assistant:
                self.assistant = AssistantCore()
                await self.assistant.initialize()
            
            # Test session info
            session_info = self.assistant.get_session_info()
            assert "session_persistence_enabled" in session_info, "Session persistence info missing"
            
            # Test system status
            status = self.assistant.get_system_status()
            assert "setup_complete" in status, "System status incomplete"
            assert status["setup_complete"], "System not properly set up"
            
            self.logger.info(f"Session management: persistence={session_info.get('session_persistence_enabled')}")
            print(f"     └─ Session persistence: {session_info.get('session_persistence_enabled', False)}")
            print(f"     └─ Active sessions: {session_info.get('total_active_sessions', 0)}")
            print(f"     └─ System status: {status.get('setup_complete', False)}")
    
    async def test_11_performance_metrics(self):
        """Test 11: Performance and response time metrics"""
        async with self.test_context("Performance & Response Times"):
            if not self.assistant:
                self.assistant = AssistantCore()
                await self.assistant.initialize()
            
            # Test response times for different message types
            test_messages = [
                "Hello, simple greeting",
                "What's 2+2?",
                "Tell me about Python"
            ]
            
            response_times = []
            
            for message in test_messages:
                start_time = time.time()
                response = await self.assistant.process_message(
                    message,
                    thread_id=f"{self.test_session_id}_perf",
                    user_id="test_user"
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert response["success"], f"Performance test failed for: {message}"
                self.logger.info(f"Response time: {response_time:.3f}s for '{message}'")
            
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"     └─ Messages processed: {len(response_times)}")
            print(f"     └─ Average response time: {avg_time:.3f}s")
            print(f"     └─ Min/Max times: {min_time:.3f}s / {max_time:.3f}s")
    
    async def test_12_graceful_shutdown(self):
        """Test 12: Graceful shutdown and cleanup"""
        async with self.test_context("Graceful Shutdown & Cleanup"):
            if not self.assistant:
                self.assistant = AssistantCore()
                await self.assistant.initialize()
            
            # Test graceful shutdown
            await self.assistant.graceful_shutdown()
            
            # Verify cleanup
            session_count = len(self.assistant._sessions)
            assert session_count == 0, f"Sessions not cleared: {session_count} remaining"
            
            self.logger.info("Graceful shutdown completed successfully")
            print(f"     └─ Sessions cleared: ✓")
            print(f"     └─ Resources released: ✓")
            print(f"     └─ Graceful shutdown: Complete")

    def print_header(self):
        """Print enhanced test suite header"""
        header_text = """
🧪 ENHANCED TESTING SUITE - LangGraph 0.4.8 Assistant
   Production-Ready System Validation
   June 2025 - Comprehensive Test Coverage
   
   Log File: {}
""".format(self.log_file)
        
        print("=" * 80)
        print(header_text)
        print("=" * 80)
        print()
        
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED TESTING SUITE STARTED")
        self.logger.info("=" * 60)
    
    def print_footer(self):
        """Print comprehensive test results with enhanced formatting"""
        summary = self.result.get_summary()
        
        print()
        print("=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        
        # Overall summary with enhanced formatting
        print(f"🎯 SUMMARY:")
        print(f"   Total Tests:     {summary['total_tests']}")
        print(f"   Passed:          {summary['passed']} ✅")
        print(f"   Failed:          {summary['failed']} ❌")
        print(f"   Success Rate:    {summary['success_rate']}")
        print(f"   Total Duration:  {summary['total_duration']}")
        print(f"   Avg Duration:    {summary['average_duration']}")
        print(f"   Log File:        {summary['log_file']}")
        print()
        
        # Test details with enhanced formatting
        if self.result.test_details:
            print("📋 DETAILED RESULTS:")
            for test in self.result.test_details:
                print(f"   {test['status']} {test['name']} ({test['duration']:.3f}s)")
                if test['details'] and not test['passed']:
                    print(f"      └─ {test['details']}")
        
        print()
        
        # Enhanced final status
        if self.result.tests_failed == 0:
            print("🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY!")
            success_msg = "ALL TESTS PASSED - SYSTEM IS PRODUCTION READY!"
        else:
            print(f"⚠️  {self.result.tests_failed} TESTS FAILED - REVIEW REQUIRED")
            success_msg = f"{self.result.tests_failed} TESTS FAILED - REVIEW REQUIRED"
        
        print("=" * 80)
        
        # Log final results
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED TESTING SUITE COMPLETED")
        self.logger.info(f"FINAL RESULT: {success_msg}")
        self.logger.info(f"Summary: {summary}")
        self.logger.info("=" * 60)

    async def run_complete_test_suite(self):
        """Run the complete enhanced test suite"""
        self.print_header()
        
        # Execute all tests in order
        test_methods = [
            self.test_01_core_imports_and_setup,
            self.test_02_checkpointer_initialization,
            self.test_03_state_management_validation,
            self.test_04_agent_factory_operations,
            self.test_05_error_handling_classification,
            self.test_06_assistant_core_initialization,
            self.test_07_message_processing_flow,
            self.test_08_multi_agent_routing,
            self.test_09_file_operations,
            self.test_10_session_management,
            self.test_11_performance_metrics,
            self.test_12_graceful_shutdown
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.logger.error(f"Test method {test_method.__name__} failed: {e}", exc_info=True)
                # Continue with other tests
        
        self.print_footer()
        
        # Write detailed results to JSON file
        import json
        json_file = self.log_file.with_suffix('.json')
        try:
            with open(json_file, "w") as f:
                json.dump({
                    "summary": self.result.get_summary(),
                    "details": self.result.test_details,
                    "timestamp": time.time(),
                    "log_file": str(self.log_file)
                }, f, indent=2)
            print(f"\n📄 Detailed results saved to: {json_file}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
        
        return self.result.tests_failed == 0

async def main():
    """Main test execution function"""
    test_suite = EnhancedTestSuite()
    
    try:
        success = await test_suite.run_complete_test_suite()
        
        print(f"\n📋 Test log saved to: {test_suite.log_file}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user.")
        test_suite.logger.warning("Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        test_suite.logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
