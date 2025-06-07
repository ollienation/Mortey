# tests/integration/test_langgraph_workflows.py
import pytest
import asyncio
from typing import Dict, Any
from asyncio import TaskGroup

from core.supervisor import Supervisor, SupervisorConfig
from core.state import AssistantState
from langchain_core.messages import HumanMessage, AIMessage

class TestLangGraphWorkflows:
    """Test LangGraph workflow execution and routing logic."""
    
    @pytest.mark.asyncio
    async def test_supervisor_routing_accuracy(
        self, 
        supervisor_with_mocked_agents, 
        routing_test_cases
    ):
        """Test routing accuracy across different input types."""
        routing_results = []
        
        async with TaskGroup() as tg:
            tasks = [
                tg.create_task(self._test_single_routing(
                    supervisor_with_mocked_agents,
                    test_case
                ))
                for test_case in routing_test_cases
            ]
        
        results = [task.result() for task in tasks]
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        
        assert accuracy >= 0.9, f"Routing accuracy {accuracy:.2f} below threshold"
    
    async def test_workflow_state_transitions(self, supervisor, sample_workflow):
        """Test state transitions through workflow execution."""
        state_history = []
        
        async def state_recorder(state: AssistantState):
            state_history.append(state.copy())
            return state
        
        # Inject state recording into workflow
        modified_workflow = self._inject_state_recording(
            sample_workflow, 
            state_recorder
        )
        
        result = await supervisor.process(
            sample_workflow["initial_state"],
            {"configurable": {"thread_id": "test_workflow"}}
        )
        
        # Validate state progression
        assert len(state_history) >= 2, "Insufficient state transitions recorded"
        assert self._validate_state_progression(state_history)
    
    @pytest.mark.performance
    async def test_concurrent_workflow_execution(
        self, 
        supervisor, 
        concurrent_executor
    ):
        """Test concurrent workflow execution without state interference."""
        workflows = [
            self._create_test_workflow(f"session_{i}")
            for i in range(10)
        ]
        
        results = await concurrent_executor([
            supervisor.process(
                workflow["state"],
                {"configurable": {"thread_id": workflow["session_id"]}}
            )
            for workflow in workflows
        ])
        
        # Verify no state interference
        for i, result in enumerate(results):
            expected_session = f"session_{i}"
            assert result["session_id"] == expected_session
