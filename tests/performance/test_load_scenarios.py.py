# tests/performance/test_load_scenarios.py
import pytest
import asyncio
import time
from typing import List, Dict, Any
from asyncio import TaskGroup

class TestLoadScenarios:
    """Comprehensive load testing for production readiness."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_concurrent_user_sessions(
        self, 
        assistant_core, 
        performance_monitor
    ):
        """Test assistant performance under concurrent user load."""
        concurrent_users = 50
        messages_per_user = 10
        
        async def simulate_user_session(user_id: int):
            session_id = f"load_test_user_{user_id}"
            response_times = []
            
            for i in range(messages_per_user):
                start_time = time.time()
                
                response = await assistant_core.process_message(
                    f"Test message {i} from user {user_id}",
                    session_id=session_id,
                    user_id=f"user_{user_id}"
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                assert response is not None
                assert "response" in response
                
                # Small delay between messages
                await asyncio.sleep(0.1)
            
            return {
                "user_id": user_id,
                "avg_response_time": sum(response_times) / len(response_times),
                "max_response_time": max(response_times),
                "total_messages": len(response_times)
            }
        
        # Execute concurrent user sessions using TaskGroup
        async with TaskGroup() as tg:
            tasks = [
                tg.create_task(simulate_user_session(user_id))
                for user_id in range(concurrent_users)
            ]
        
        results = [task.result() for task in tasks]
        
        # Analyze performance metrics
        avg_response_times = [r["avg_response_time"] for r in results]
        overall_avg = sum(avg_response_times) / len(avg_response_times)
        
        performance_monitor["concurrent_users"] = concurrent_users
        performance_monitor["overall_avg_response_time"] = overall_avg
        
        # Performance assertions
        assert overall_avg < 2.0, f"Average response time {overall_avg:.2f}s too high"
        assert all(r["total_messages"] == messages_per_user for r in results)
    
    @pytest.mark.performance
    async def test_memory_usage_stability(
        self, 
        assistant_core, 
        memory_profiler
    ):
        """Test memory usage stability over extended operations."""
        initial_memory = memory_profiler.get_current_usage()
        
        # Simulate extended usage
        for batch in range(10):
            async with TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        assistant_core.process_message(
                            f"Batch {batch} message {i}",
                            session_id=f"memory_test_{i}",
                            user_id="memory_test_user"
                        )
                    )
                    for i in range(20)
                ]
            
            current_memory = memory_profiler.get_current_usage()
            memory_growth = current_memory - initial_memory
            
            # Check for memory leaks
            assert memory_growth < 100 * 1024 * 1024, f"Memory growth {memory_growth} too high"
