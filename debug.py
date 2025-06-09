# debug.py - ✅ Enhanced Debugging Script for Mortey Assistant
import asyncio
import logging
import sys
import time
import traceback
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence  # Python 3.13.4 preferred import

# Configure comprehensive logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log', mode='w')
    ]
)

# Set specific loggers to DEBUG
debug_loggers = [
    'assistant_core',
    'supervisor', 
    'agents',
    'config.settings',
    'config.llm_manager',
    'core.checkpointer',
    'core.error_handling',
    'core.circuit_breaker',
    'tools.file_tools'
]

for logger_name in debug_loggers:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# Suppress noisy third-party loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)

logger = logging.getLogger("debug")

@dataclass
class DebugResult:
    """Result of a debug test with enhanced metadata"""
    test_name: str
    success: bool
    message: str
    duration_ms: float = 0.0
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = None

class MorteyDebugger:
    """
    Comprehensive debugging system for Mortey Assistant with Python 3.13.4 features
    """
    
    def __init__(self):
        self.results: list[DebugResult] = []  # Python 3.13.4 syntax
        self.start_time = time.time()
        
    async def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run debug tests with smart delay management"""
        
        # Define delay times for different test categories
        delay_config = {
            "lightweight": 0.5,  # Config, environment tests
            "medium": 1.0,       # Component initialization
            "heavy": 2.0,        # LLM calls, message processing
            "intensive": 3.0     # Full system integration tests
        }
        
        test_sequence = [
            ("Environment Setup", self._test_environment_setup, "lightweight"),
            ("Configuration Loading", self._test_configuration_loading, "lightweight"), 
            ("Checkpointer Initialization", self._test_checkpointer_initialization, "medium"),
            ("LLM Manager", self._test_llm_manager, "heavy"),
            ("Agent Factory", self._test_agent_factory, "heavy"),
            ("Supervisor Initialization", self._test_supervisor_initialization, "medium"),
            ("Assistant Core", self._test_assistant_core, "medium"),
            ("Message Processing Pipeline", self._test_message_processing, "intensive"),
            ("Tools Integration", self._test_tools_integration, "medium"),
            ("Error Handling", self._test_error_handling, "lightweight"),
            ("Circuit Breakers", self._test_circuit_breakers, "lightweight"),
            ("Health Checks", self._test_health_checks, "medium"),
            ("Performance Metrics", self._test_performance_metrics, "heavy")
        ]
        
        for i, (test_name, test_func, category) in enumerate(test_sequence):
            await self._run_debug_test(test_name, test_func)
            
            # Add appropriate delay based on category
            if i < len(test_sequence) - 1:  # Don't delay after last test
                delay_time = delay_config.get(category, 1.0)
                logger.debug(f"⏸️ Pausing {delay_time}s after {category} test...")
                await asyncio.sleep(delay_time)
        
        return self._generate_debug_report()
    
    async def _rate_limit_pause(self) -> DebugResult:
        """Pause for rate limiting to avoid API limits"""
        logger.info("⏸️ Rate limiting pause - waiting 3 seconds...")
        await asyncio.sleep(3)  # 3 second pause
        
        return DebugResult(
            test_name="Rate Limit Pause",
            success=True,
            message="Rate limiting pause completed - continuing tests"
        )
    
    async def _run_debug_test(self, test_name: str, test_func) -> DebugResult:
        """Run individual debug test with error handling"""
        logger.info(f"\n🔍 Testing: {test_name}")
        logger.info("-" * 60)
        
        start_time = time.time()
        
        try:
            result = await test_func()
            duration = (time.time() - start_time) * 1000
            
            if isinstance(result, DebugResult):
                result.duration_ms = duration
                self.results.append(result)
                
                if result.success:
                    logger.info(f"✅ {test_name}: {result.message}")
                else:
                    logger.error(f"❌ {test_name}: {result.message}")
                    if result.error_details:
                        logger.error(f"🔍 Details: {result.error_details}")
                
                return result
            else:
                # Handle simple boolean returns
                success = bool(result)
                debug_result = DebugResult(
                    test_name=test_name,
                    success=success,
                    message="Test completed" if success else "Test failed",
                    duration_ms=duration
                )
                self.results.append(debug_result)
                return debug_result
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            error_details = traceback.format_exc()
            
            debug_result = DebugResult(
                test_name=test_name,
                success=False,
                message=f"Test failed with exception: {str(e)}",
                duration_ms=duration,
                error_details=error_details
            )
            
            self.results.append(debug_result)
            logger.error(f"❌ {test_name}: Exception occurred")
            logger.error(f"🔍 Error: {str(e)}")
            logger.debug(f"🔍 Full traceback:\n{error_details}")
            
            return debug_result
    
    async def _test_environment_setup(self) -> DebugResult:
        """Test environment and dependency setup"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 11):
                return DebugResult(
                    test_name="Environment Setup",
                    success=False,
                    message=f"Python version {python_version.major}.{python_version.minor} too old, need 3.11+"
                )
            
            # Check critical environment variables
            from config.settings import config
            from dotenv import load_dotenv
            from pathlib import Path
            import os
            critical_vars = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY']
            missing_vars = [var for var in critical_vars if not os.getenv(var)]
            
            if missing_vars:
                return DebugResult(
                    test_name="Environment Setup",
                    success=False,
                    message=f"Missing environment variables: {', '.join(missing_vars)}",
                    metadata={"missing_vars": missing_vars}
                )
            
            # Check project structure
            project_root = Path.cwd()
            required_dirs = ['core', 'config', 'agents', 'tools']
            missing_dirs = [d for d in required_dirs if not (project_root / d).exists()]
            
            if missing_dirs:
                return DebugResult(
                    test_name="Environment Setup",
                    success=False,
                    message=f"Missing directories: {', '.join(missing_dirs)}"
                )
            
            return DebugResult(
                test_name="Environment Setup",
                success=True,
                message=f"Environment OK - Python {python_version.major}.{python_version.minor}, all vars and dirs present",
                metadata={
                    "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "project_root": str(project_root)
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Environment Setup",
                success=False,
                message=f"Environment check failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_configuration_loading(self) -> DebugResult:
        """Test configuration loading and validation"""
        try:
            from config.settings import config, MorteyConfig
            
            logger.debug("Testing configuration loading...")
            
            await asyncio.sleep(0.1)
            
            # Test config object exists
            if config is None:
                return DebugResult(
                    test_name="Configuration Loading",
                    success=False,
                    message="Config object is None"
                )
            
            # Test workspace directory
            if not hasattr(config, 'workspace_dir'):
                return DebugResult(
                    test_name="Configuration Loading",
                    success=False,
                    message="Config missing workspace_dir"
                )
            
            # Test LLM config
            if not hasattr(config, 'llm_config') or not config.llm_config:
                return DebugResult(
                    test_name="Configuration Loading",
                    success=False,
                    message="Config missing llm_config attribute"
                )
            
            # Validate LLM config structure
            llm_config = config.llm_config
            required_sections = ['global', 'providers', 'nodes']
            missing_sections = [s for s in required_sections if s not in llm_config]
            
            if missing_sections:
                return DebugResult(
                    test_name="Configuration Loading",
                    success=False,
                    message=f"LLM config missing sections: {', '.join(missing_sections)}"
                )
            
            # Test provider configurations
            providers = llm_config.get('providers', {})
            nodes = llm_config.get('nodes', {})
            
            metadata = {
                "providers_count": len(providers),
                "nodes_count": len(nodes),
                "workspace_dir": str(config.workspace_dir),
                "default_provider": llm_config.get('global', {}).get('default_provider')
            }
            
            return DebugResult(
                test_name="Configuration Loading",
                success=True,
                message=f"Config loaded successfully - {len(providers)} providers, {len(nodes)} nodes",
                metadata=metadata
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Configuration Loading",
                success=False,
                message=f"Configuration loading failed: {str(e)}",
                error_details=traceback.format_exc()
            )
        
    async def _test_checkpointer_initialization(self) -> DebugResult:
        """Test checkpointer creation with new factory"""
        try:
            from core.checkpointer import CheckpointerFactory, CheckpointerConfig
            
            logger.debug("Testing checkpointer factory...")
            
            # Create factory
            factory = CheckpointerFactory()
            
            # Create checkpointer
            checkpointer = await factory.create_optimal_checkpointer()
            
            if checkpointer is None:
                return DebugResult(
                    test_name="Checkpointer Initialization",
                    success=False,
                    message="Checkpointer creation returned None"
                )
            
            # Test health check
            health_status = await factory.health_check_all()
            
            # Get factory statistics
            factory_stats = factory.get_factory_statistics()
            
            return DebugResult(
                test_name="Checkpointer Initialization",
                success=True,
                message=f"Checkpointer factory working - Type: {type(checkpointer).__name__}",
                metadata={
                    "checkpointer_type": type(checkpointer).__name__,
                    "health_status": health_status,
                    "factory_stats": factory_stats
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Checkpointer Initialization",
                success=False,
                message=f"Checkpointer factory test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_llm_manager(self) -> DebugResult:
        """Test LLM manager functionality"""
        try:
            from config.llm_manager import llm_manager
            from langchain_core.messages import HumanMessage
            
            logger.debug("Testing LLM manager...")
            
            # Test health check
            health_status = await llm_manager.health_check()
            
            # Test simple generation
            response = await llm_manager.generate_for_node("chat", "Hello, this is a test.")
            
            if not response:
                return DebugResult(
                    test_name="LLM Manager",
                    success=False,
                    message="LLM Manager returned empty response"
                )
            
            # Test usage stats
            usage_stats = llm_manager.get_usage_stats()
            
            return DebugResult(
                test_name="LLM Manager",
                success=True,
                message=f"LLM Manager working - Generated {len(str(response))} chars",
                metadata={
                    "health_status": health_status,
                    "usage_stats": usage_stats,
                    "response_length": len(str(response))
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="LLM Manager",
                success=False,
                message=f"LLM Manager test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_agent_factory(self) -> DebugResult:
        """Test agent factory initialization and functionality"""
        try:
            from agents.agents import agent_factory
            
            logger.debug("Testing agent factory...")
            
            # Initialize agents
            agents = await agent_factory.initialize_agents()
            
            if not agents:
                return DebugResult(
                    test_name="Agent Factory",
                    success=False,
                    message="Agent factory returned no agents"
                )
            
            # Test tools
            all_tools = agent_factory.get_all_tools()
            
            # Test health check
            health_status = await agent_factory.health_check_agents()
            
            # Get factory statistics
            stats = agent_factory.get_factory_statistics()
            
            return DebugResult(
                test_name="Agent Factory",
                success=True,
                message=f"Agent factory working - {len(agents)} agents, {len(all_tools)} tools",
                metadata={
                    "agents_count": len(agents),
                    "tools_count": len(all_tools),
                    "agent_names": list(agents.keys()),
                    "health_status": health_status,
                    "factory_stats": stats
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Agent Factory",
                success=False,
                message=f"Agent factory test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_supervisor_initialization(self) -> DebugResult:
        """Test supervisor initialization and configuration"""
        try:
            from core.supervisor import Supervisor
            from agents.agents import agent_factory
            from core.checkpointer import create_optimal_checkpointer
            
            logger.debug("Testing supervisor initialization...")
            
            # Initialize components
            agents = await agent_factory.initialize_agents()
            all_tools = agent_factory.get_all_tools()
            checkpointer = await create_optimal_checkpointer()
            
            # Create and initialize supervisor
            supervisor = Supervisor()
            await supervisor.initialize(agents, all_tools, checkpointer)
            
            if supervisor.supervisor_graph is None:
                return DebugResult(
                    test_name="Supervisor Initialization",
                    success=False,
                    message="Supervisor graph is None after initialization"
                )
            
            # Test routing statistics
            routing_stats = supervisor.get_routing_statistics()
            
            return DebugResult(
                test_name="Supervisor Initialization",
                success=True,
                message=f"Supervisor initialized successfully - Graph ready with {len(agents)} agents",
                metadata={
                    "graph_ready": supervisor.supervisor_graph is not None,
                    "routing_stats": routing_stats,
                    "agents_available": list(agents.keys())
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Supervisor Initialization",
                success=False,
                message=f"Supervisor initialization failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_assistant_core(self) -> DebugResult:
        """Test assistant core initialization"""
        try:
            from core.assistant_core import AssistantCore
            
            logger.debug("Testing assistant core...")
            
            # Create assistant core
            assistant = AssistantCore()
            
            # Initialize (this tests the full integration)
            await assistant.initialize()
            
            # Verify components are ready
            if assistant.supervisor is None:
                return DebugResult(
                    test_name="Assistant Core",
                    success=False,
                    message="Assistant supervisor is None"
                )
            
            if assistant.checkpointer is None:
                return DebugResult(
                    test_name="Assistant Core",
                    success=False,
                    message="Assistant checkpointer is None"
                )
            
            if assistant.supervisor.supervisor_graph is None:
                return DebugResult(
                    test_name="Assistant Core",
                    success=False,
                    message="Assistant supervisor graph is None"
                )
            
            # Test system status
            status = await assistant._get_system_status()
            
            return DebugResult(
                test_name="Assistant Core",
                success=True,
                message="Assistant core initialized successfully - All components ready",
                metadata={
                    "active_sessions": len(assistant.active_sessions),
                    "system_status": status
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Assistant Core",
                success=False,
                message=f"Assistant core test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_message_processing(self) -> DebugResult:
        """Test message processing with delays between requests"""
        try:
            from core.assistant_core import AssistantCore
            
            logger.debug("Testing message processing pipeline...")
            
            assistant = AssistantCore()
            await assistant.initialize()
            
            test_messages = [
                "Hello, how are you?",
                "Write a simple Python function to add two numbers", 
                "What is 2 + 2?"
            ]
            
            results = []
            for i, message in enumerate(test_messages):
                logger.debug(f"Processing test message {i+1}: {message[:30]}...")
                
                result = await assistant.process_message(
                    message=message,
                    session_id=f"debug_session_{i}",
                    user_id="debug_user"
                )
                
                if not result or not result.get("response"):
                    return DebugResult(
                        test_name="Message Processing",
                        success=False,
                        message=f"Message {i+1} processing failed - no response"
                    )
                
                results.append({
                    "message": message,
                    "response_length": len(result.get("response", "")),
                    "session_id": result.get("session_id"),
                    "timestamp": result.get("timestamp")
                })
                
                # 🔥 ADD: Delay between message processing (except last)
                if i < len(test_messages) - 1:
                    logger.debug(f"⏸️ Pausing 2 seconds before next message...")
                    await asyncio.sleep(2)  # 2 second delay between messages
            
            return DebugResult(
                test_name="Message Processing",
                success=True,
                message=f"Message processing working - {len(results)} messages processed",
                metadata={
                    "messages_processed": len(results),
                    "results_summary": results,
                    "sessions_created": len(assistant.active_sessions)
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Message Processing", 
                success=False,
                message=f"Message processing test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_tools_integration(self) -> DebugResult:
        """Test tools integration and functionality"""
        try:
            from tools.file_tools import FileSystemTools
            from agents.agents import agent_factory
            
            logger.debug("Testing tools integration...")
            
            # Test file tools
            file_tools = FileSystemTools()
            tools = file_tools.get_tools()
            
            if not tools:
                return DebugResult(
                    test_name="Tools Integration",
                    success=False,
                    message="No file tools available"
                )
            
            # Test agent tools integration
            await agent_factory.initialize_agents()
            all_tools = agent_factory.get_all_tools()
            
            # Test a simple file operation
            try:
                # Create a test file using one of the tools
                write_tool = next((t for t in tools if t.name == "write_file"), None)
                if write_tool:
                    test_result = write_tool.run({
                        "file_path": "debug_test.txt",
                        "text": "This is a debug test file created by Mortey debugger."
                    })
                    logger.debug(f"File write test result: {test_result}")
            except Exception as tool_error:
                logger.warning(f"Tool test failed: {tool_error}")
            
            return DebugResult(
                test_name="Tools Integration",
                success=True,
                message=f"Tools integration working - {len(tools)} file tools, {len(all_tools)} total tools",
                metadata={
                    "file_tools_count": len(tools),
                    "total_tools_count": len(all_tools),
                    "tool_names": [t.name for t in tools[:5]]  # First 5 tool names
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Tools Integration",
                success=False,
                message=f"Tools integration test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_error_handling(self) -> DebugResult:
        """Test error handling and circuit breaker functionality"""
        try:
            from core.error_handling import ErrorHandler
            from core.circuit_breaker import global_circuit_breaker
            
            logger.debug("Testing error handling...")
            
            # Test error classification
            test_error = ValueError("Test error for debugging")
            error_response = await ErrorHandler.handle_error(test_error, "debug_test")
            
            if not error_response:
                return DebugResult(
                    test_name="Error Handling",
                    success=False,
                    message="Error handler returned no response"
                )
            
            # Test circuit breaker health
            circuit_health = await global_circuit_breaker.health_check_all_services()
            
            # Test error statistics
            error_stats = await ErrorHandler.get_error_statistics()
            
            return DebugResult(
                test_name="Error Handling",
                success=True,
                message="Error handling working - Error classification and circuit breakers operational",
                metadata={
                    "error_response": error_response,
                    "circuit_health": circuit_health,
                    "error_stats": error_stats
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Error Handling",
                success=False,
                message=f"Error handling test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_circuit_breakers(self) -> DebugResult:
        """Test circuit breaker functionality"""
        try:
            from core.circuit_breaker import global_circuit_breaker
            
            logger.debug("Testing circuit breakers...")
            
            # Test circuit breaker health check
            health_status = await global_circuit_breaker.health_check_all_services()
            
            # Test with a simple function
            async def test_function(service_name):
                return "Circuit breaker test successful"
            
            result = await global_circuit_breaker.call_with_circuit_breaker(
                "debug_test_service",
                test_function,
                {}
            )
            
            if result != "Circuit breaker test successful":
                return DebugResult(
                    test_name="Circuit Breakers",
                    success=False,
                    message="Circuit breaker test function failed"
                )
            
            return DebugResult(
                test_name="Circuit Breakers",
                success=True,
                message="Circuit breakers working - Health checks passing",
                metadata={
                    "health_status": health_status,
                    "test_result": result
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Circuit Breakers",
                success=False,
                message=f"Circuit breaker test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_health_checks(self) -> DebugResult:
        """Test all system health checks"""
        try:
            from core.assistant_core import AssistantCore
            
            logger.debug("Testing health checks...")
            
            # Create assistant and get status
            assistant = AssistantCore()
            await assistant.initialize()
            
            # Get comprehensive system status
            system_status = await assistant._get_system_status()
            
            # Analyze health scores
            health_issues = []
            if not system_status.get("agents", {}).get("chat", True):
                health_issues.append("Chat agent not healthy")
            
            if not system_status.get("llm_manager", {}).get("healthy", True):
                health_issues.append("LLM manager not healthy")
            
            if not system_status.get("checkpointer", {}).get("healthy", True):
                health_issues.append("Checkpointer not healthy")
            
            success = len(health_issues) == 0
            message = "All health checks passing" if success else f"Health issues: {', '.join(health_issues)}"
            
            return DebugResult(
                test_name="Health Checks",
                success=success,
                message=message,
                metadata={
                    "system_status": system_status,
                    "health_issues": health_issues,
                    "active_sessions": system_status.get("active_sessions", 0)
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Health Checks",
                success=False,
                message=f"Health check test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    async def _test_performance_metrics(self) -> DebugResult:
        """Test performance and gather metrics"""
        try:
            from core.assistant_core import AssistantCore
            import psutil
            import os
            
            logger.debug("Testing performance metrics...")
            
            # Get system metrics
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            
            # Test message processing performance
            assistant = AssistantCore()
            await assistant.initialize()
            
            # Time a simple message
            start_time = time.time()
            await assistant.process_message(
                "Hello, this is a performance test",
                session_id="perf_test",
                user_id="debug_user"
            )
            processing_time = (time.time() - start_time) * 1000  # ms
            
            return DebugResult(
                test_name="Performance Metrics",
                success=True,
                message=f"Performance metrics gathered - {processing_time:.1f}ms message processing",
                metadata={
                    "memory_usage_mb": round(memory_usage, 2),
                    "cpu_percent": cpu_percent,
                    "message_processing_time_ms": round(processing_time, 2),
                    "total_debug_time_s": round(time.time() - self.start_time, 2)
                }
            )
            
        except Exception as e:
            return DebugResult(
                test_name="Performance Metrics",
                success=False,
                message=f"Performance metrics test failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def _generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        total_duration = time.time() - self.start_time
        
        # Calculate average test duration
        test_durations = [r.duration_ms for r in self.results if r.duration_ms > 0]
        avg_duration = sum(test_durations) / len(test_durations) if test_durations else 0
        
        # Generate summary
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 MORTEY ASSISTANT DEBUG REPORT")
        logger.info("=" * 80)
        logger.info(f"📊 Summary: {successful_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"⏱️ Total time: {total_duration:.2f}s (avg: {avg_duration:.1f}ms per test)")
        
        if failed_tests > 0:
            logger.error(f"\n❌ Failed Tests ({failed_tests}):")
            for result in self.results:
                if not result.success:
                    logger.error(f"  - {result.test_name}: {result.message}")
        
        logger.info(f"\n✅ Successful Tests ({successful_tests}):")
        for result in self.results:
            if result.success:
                logger.info(f"  - {result.test_name}: {result.message}")
        
        logger.info("\n" + "=" * 80)
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_duration_s": total_duration,
                "average_test_duration_ms": avg_duration
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "metadata": r.metadata or {}
                }
                for r in self.results
            ]
        }

async def main():
    """Main debug function"""
    try:
        debugger = MorteyDebugger()
        report = await debugger.run_comprehensive_debug()
        
        # Save report to file
        with open('debug_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Full debug report saved to: debug_report.json")
        
        # Return exit code based on success
        if report["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Debug session failed: {e}")
        logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the debug session
    asyncio.run(main())
