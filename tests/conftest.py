# tests/conftest.py - ✅ COMPREHENSIVE TEST CONFIGURATION WITH PYTHON 3.13.4 ENHANCEMENTS
"""
Global pytest configuration and fixtures for AI Assistant testing.

This module provides comprehensive testing infrastructure including:
- Async testing support with TaskGroup patterns
- LLM response mocking and validation
- Database fixtures for SQLite and PostgreSQL
- Circuit breaker testing utilities
- State management fixtures
- Performance testing tools
"""

import asyncio
import os
import pytest
import pytest_asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, AsyncGenerator, Generator, Any
from collections.abc import Mapping  # Python 3.13.4 preferred import
from unittest.mock import AsyncMock, MagicMock, patch
from asyncio import TaskGroup  # Python 3.13.4 TaskGroup

# Test-specific imports
import psycopg
import aiosqlite
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, drop_database, database_exists

# Import application components for testing
from core.state import AssistantState, create_optimized_state, StateValidator
from core.checkpointer import create_optimal_checkpointer, CheckpointerConfig, Environment
from core.circuit_breaker import AdvancedCircuitBreaker, ServiceConfig
from core.error_handling import ErrorHandler, ErrorClassifier
from config.settings import MorteyConfig
from config.llm_manager import LLMManager

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_framework")

# Test configuration constants
TEST_DB_NAME = "test_assistant"
TEST_SESSION_TIMEOUT = 30.0  # seconds
MAX_CONCURRENT_TESTS = 5

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests (> 5 seconds)")
    config.addinivalue_line("markers", "database: Tests requiring database")
    config.addinivalue_line("markers", "llm: Tests involving LLM calls")
    config.addinivalue_line("markers", "circuit_breaker: Circuit breaker tests")
    config.addinivalue_line("markers", "performance: Performance tests")

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and characteristics."""
    for item in items:
        # Mark tests based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Mark slow tests (those with specific fixtures or names)
        if any(fixture in item.fixturenames for fixture in ["postgresql_db", "performance_monitor"]):
            item.add_marker(pytest.mark.slow)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_config() -> MorteyConfig:
    """Create test configuration with isolated settings."""
    # Create temporary directories for testing
    temp_dir = Path(tempfile.mkdtemp(prefix="mortey_test_"))
    
    # Override environment variables for testing
    test_env = {
        "MORTEY_WORKSPACE_DIR": str(temp_dir / "workspace"),
        "ENVIRONMENT": "testing",
        "LANGSMITH_TRACING": "false",  # Disable tracing in tests
        "DATABASE_URL": f"sqlite:///{temp_dir}/test.db",
    }
    
    with patch.dict(os.environ, test_env):
        config = MorteyConfig.from_environment()
        # Ensure directories exist
        config.workspace_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        
        yield config
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def isolated_workspace(test_config: MorteyConfig) -> Path:
    """Create isolated workspace for each test."""
    test_workspace = test_config.workspace_dir / f"test_{uuid.uuid4().hex[:8]}"
    test_workspace.mkdir(parents=True, exist_ok=True)
    
    yield test_workspace
    
    # Cleanup
    import shutil
    shutil.rmtree(test_workspace, ignore_errors=True)

# ============================================================================
# STATE MANAGEMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_state() -> AssistantState:
    """Create sample state for testing."""
    return create_optimized_state(
        session_id=f"test_session_{uuid.uuid4().hex[:8]}",
        user_id="test_user",
        initial_context={"current_agent": "chat"}
    )

@pytest.fixture
def state_validator() -> StateValidator:
    """Create state validator for testing."""
    return StateValidator()

@pytest.fixture
async def complex_state() -> AssistantState:
    """Create complex state with multiple messages for testing."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    state = create_optimized_state(
        session_id=f"complex_session_{uuid.uuid4().hex[:8]}",
        user_id="test_user"
    )
    
    # Add realistic conversation history
    state["messages"] = [
        HumanMessage(content="Hello, can you help me with file management?"),
        AIMessage(content="I'd be happy to help with file management. What would you like to do?"),
        HumanMessage(content="Create a new project structure for a Python web app"),
        AIMessage(
            content="I'll create a Python web project structure for you.",
            tool_calls=[{
                "name": "create_project",
                "args": {"project_name": "my_web_app", "project_type": "web"},
                "id": "call_123"
            }]
        ),
        ToolMessage(
            content="Project created successfully",
            tool_call_id="call_123"
        )
    ]
    
    return state

# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
async def postgresql_test_db():
    """Create PostgreSQL test database for the session."""
    # Check if PostgreSQL is available
    postgres_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not postgres_url or not postgres_url.startswith("postgresql://"):
        pytest.skip("PostgreSQL not available for testing")
    
    # Extract connection details
    import urllib.parse
    parsed = urllib.parse.urlparse(postgres_url)
    base_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}"
    test_db_name = f"test_assistant_{int(time.time())}"
    test_db_url = f"{base_url}/{test_db_name}"
    
    # Create test database
    try:
        async with await psycopg.AsyncConnection.connect(base_url, autocommit=True) as conn:
            await conn.execute(f'CREATE DATABASE "{test_db_name}"')
        
        logger.info(f"Created PostgreSQL test database: {test_db_name}")
        yield test_db_url
        
        # Cleanup
        async with await psycopg.AsyncConnection.connect(base_url, autocommit=True) as conn:
            await conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}" WITH (FORCE)')
        
        logger.info(f"Cleaned up PostgreSQL test database: {test_db_name}")
        
    except Exception as e:
        logger.warning(f"PostgreSQL test database setup failed: {e}")
        pytest.skip("PostgreSQL test database setup failed")

@pytest.fixture
async def sqlite_test_db(test_config: MorteyConfig):
    """Create SQLite test database for individual tests."""
    db_path = test_config.workspace_dir / f"test_{uuid.uuid4().hex[:8]}.db"
    db_url = f"sqlite:///{db_path}"
    
    yield db_url
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()

@pytest.fixture
async def checkpointer_sqlite(sqlite_test_db: str):
    """Create SQLite checkpointer for testing."""
    config = CheckpointerConfig(
        environment=Environment.TESTING,
        prefer_async=True
    )
    
    # Override database URL
    with patch.dict(os.environ, {"DATABASE_URL": sqlite_test_db}):
        checkpointer = await create_optimal_checkpointer(config)
        yield checkpointer
        
        # Cleanup
        if hasattr(checkpointer, 'cleanup_connections'):
            await checkpointer.cleanup_connections()

@pytest.fixture
async def checkpointer_postgresql(postgresql_test_db: str):
    """Create PostgreSQL checkpointer for testing."""
    config = CheckpointerConfig(
        environment=Environment.TESTING,
        prefer_async=True
    )
    
    # Override database URL
    with patch.dict(os.environ, {"POSTGRES_URL": postgresql_test_db}):
        checkpointer = await create_optimal_checkpointer(config)
        yield checkpointer
        
        # Cleanup
        if hasattr(checkpointer, 'cleanup_connections'):
            await checkpointer.cleanup_connections()

# ============================================================================
# LLM MOCKING FIXTURES
# ============================================================================

@pytest.fixture
def mock_llm_responses():
    """Predefined LLM responses for consistent testing."""
    return {
        "chat_response": "Hello! I'm a test assistant ready to help you.",
        "coder_response": """Here's a Python function that solves your problem:

```python
def solve_problem():
    return "Problem solved!"
```

This function demonstrates the solution approach.""",
        "web_response": "I found some information about your query. Here are the key points...",
        "error_response": "I apologize, but I encountered an issue processing your request.",
        "tool_call_response": {
            "content": "I'll help you with that file operation.",
            "tool_calls": [{
                "name": "create_file",
                "args": {"filename": "test.txt", "content": "Test content"},
                "id": "test_call_123"
            }]
        }
    }

@pytest.fixture
def mock_llm_manager(mock_llm_responses):
    """Create mocked LLM manager for testing."""
    mock_manager = AsyncMock(spec=LLMManager)
    
    async def mock_generate(node_name: str, prompt: str, **kwargs):
        """Mock generation based on node type."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Return appropriate response based on node
        if "chat" in node_name:
            return mock_llm_responses["chat_response"]
        elif "coder" in node_name:
            return mock_llm_responses["coder_response"]
        elif "web" in node_name:
            return mock_llm_responses["web_response"]
        else:
            return mock_llm_responses["chat_response"]
    
    mock_manager.generate_for_node = mock_generate
    mock_manager.health_check.return_value = {"status": "healthy"}
    mock_manager.get_cache_info.return_value = {"total_models": 0}
    
    return mock_manager

# ============================================================================
# CIRCUIT BREAKER FIXTURES
# ============================================================================

@pytest.fixture
def circuit_breaker() -> AdvancedCircuitBreaker:
    """Create circuit breaker for testing."""
    breaker = AdvancedCircuitBreaker()
    
    # Configure with test-friendly settings
    test_config = ServiceConfig(
        failure_threshold=3,
        recovery_timeout=1.0,  # Fast recovery for tests
        call_timeout=0.5,
        min_throughput=1
    )
    
    breaker.default_configs["test_service"] = test_config
    return breaker

@pytest.fixture
async def failing_service():
    """Create service that simulates failures."""
    failure_count = 0
    
    async def service_call(should_fail: bool = False):
        nonlocal failure_count
        if should_fail:
            failure_count += 1
            raise ConnectionError(f"Service failure #{failure_count}")
        return f"Success after {failure_count} failures"
    
    return service_call

# ============================================================================
# ERROR HANDLING FIXTURES
# ============================================================================

@pytest.fixture
def error_classifier() -> ErrorClassifier:
    """Create error classifier for testing."""
    return ErrorClassifier()

@pytest.fixture
def error_handler() -> ErrorHandler:
    """Create error handler for testing."""
    return ErrorHandler()

# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    start_time = time.time()
    metrics = {
        "start_time": start_time,
        "memory_usage": [],
        "execution_times": []
    }
    
    yield metrics
    
    # Calculate final metrics
    metrics["total_time"] = time.time() - start_time
    logger.info(f"Test completed in {metrics['total_time']:.3f}s")

@pytest.fixture
async def concurrent_executor():
    """Execute multiple tasks concurrently using TaskGroup."""
    async def execute_concurrent(tasks, max_concurrent=MAX_CONCURRENT_TESTS):
        """Execute tasks with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        # Use TaskGroup for concurrent execution (Python 3.13.4)
        async with TaskGroup() as tg:
            task_list = [tg.create_task(limited_task(task)) for task in tasks]
        
        return [task.result() for task in task_list]
    
    return execute_concurrent

# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================

@pytest.fixture
async def assistant_core_mock():
    """Create mocked assistant core for integration testing."""
    from core.assistant_core import AssistantCore
    
    mock_core = AsyncMock(spec=AssistantCore)
    mock_core.initialize.return_value = None
    mock_core.process_message.return_value = {
        "response": "Test response",
        "session_id": "test_session",
        "timestamp": time.time()
    }
    
    return mock_core

@pytest.fixture
async def supervisor_mock():
    """Create mocked supervisor for testing."""
    from core.supervisor import Supervisor
    
    mock_supervisor = AsyncMock(spec=Supervisor)
    mock_supervisor.initialize.return_value = None
    mock_supervisor.process.return_value = {
        "messages": [],
        "session_id": "test_session",
        "user_id": "test_user",
        "current_agent": "chat"
    }
    
    return mock_supervisor

# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def test_file_content():
    """Provide test file content for file operation tests."""
    return {
        "python_code": '''#!/usr/bin/env python3
"""Test Python file."""

def hello_world():
    """Simple test function."""
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
''',
        "json_data": '{"test": true, "value": 42, "nested": {"key": "data"}}',
        "markdown": '''# Test Document

This is a test markdown document.

## Features

- Feature 1
- Feature 2
- Feature 3

## Code Example

```python
print("Hello, World!")
```
''',
        "csv_data": '''name,age,city
Alice,25,New York
Bob,30,San Francisco
Charlie,35,Chicago
'''
    }

@pytest.fixture
def async_context_manager_mock():
    """Create async context manager mock for testing."""
    class AsyncContextManagerMock:
        def __init__(self, mock_object=None):
            self.mock = mock_object or AsyncMock()
        
        async def __aenter__(self):
            return self.mock
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False
    
    return AsyncContextManagerMock

# ============================================================================
# CLEANUP AND TEARDOWN
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_async_resources():
    """Automatically cleanup async resources after each test."""
    yield
    
    # Cancel any remaining tasks
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    if tasks:
        for task in tasks:
            task.cancel()
        
        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Failed to cancel {len(tasks)} async tasks within timeout")

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton objects between tests."""
    yield
    
    # Reset any global state that might affect other tests
    from core.circuit_breaker import global_circuit_breaker
    from config.llm_manager import llm_manager
    
    # Clear caches and reset state
    if hasattr(llm_manager, 'clear_cache'):
        llm_manager.clear_cache()
    
    # Reset circuit breaker state
    global_circuit_breaker.circuits.clear()

# ============================================================================
# ASYNC TEST CONFIGURATION
# ============================================================================

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

# Set default async test scope
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for testing."""
    return asyncio.DefaultEventLoopPolicy()

# Mark all async test functions
def pytest_collection_modifyitems(config, items):
    """Automatically mark async test functions."""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)