# tests/integration/test_database_backends.py
import pytest
import asyncio
from typing import Dict, Any, List
from asyncio import TaskGroup

class TestDatabaseBackends:
    """Test compatibility across PostgreSQL and SQLite backends."""
    
    @pytest.mark.parametrize("backend", ["postgresql", "sqlite"])
    @pytest.mark.asyncio
    async def test_checkpointer_compatibility(
        self, 
        backend: str, 
        checkpointer_factory
    ):
        """Test checkpointer functionality across database backends."""
        checkpointer = await checkpointer_factory(backend)
        
        # Test basic operations
        state = self._create_test_state()
        config = {"configurable": {"thread_id": "compatibility_test"}}
        
        # Save state
        await checkpointer.aput(config, state, {})
        
        # Retrieve state
        retrieved = await checkpointer.aget(config)
        
        assert retrieved is not None
        assert retrieved["session_id"] == state["session_id"]
        assert len(retrieved["messages"]) == len(state["messages"])
    
    async def test_concurrent_database_access(
        self, 
        postgresql_checkpointer,
        concurrent_executor
    ):
        """Test concurrent database access patterns."""
        sessions = [f"concurrent_session_{i}" for i in range(20)]
        
        async def save_and_retrieve(session_id: str):
            state = self._create_test_state(session_id=session_id)
            config = {"configurable": {"thread_id": session_id}}
            
            await postgresql_checkpointer.aput(config, state, {})
            retrieved = await postgresql_checkpointer.aget(config)
            
            return retrieved["session_id"] == session_id
        
        tasks = [save_and_retrieve(session) for session in sessions]
        results = await concurrent_executor(tasks)
        
        assert all(results), "Concurrent database access failed"
    
    @pytest.mark.stress
    async def test_large_state_persistence(
        self, 
        postgresql_checkpointer
    ):
        """Test persistence of large conversation states."""
        # Create state with large conversation history
        large_state = self._create_large_state(message_count=1000)
        config = {"configurable": {"thread_id": "large_state_test"}}
        
        start_time = time.time()
        await postgresql_checkpointer.aput(config, large_state, {})
        save_time = time.time() - start_time
        
        start_time = time.time()
        retrieved = await postgresql_checkpointer.aget(config)
        load_time = time.time() - start_time
        
        assert save_time < 5.0, f"Save time {save_time:.2f}s too slow"
        assert load_time < 2.0, f"Load time {load_time:.2f}s too slow"
        assert len(retrieved["messages"]) == 1000
