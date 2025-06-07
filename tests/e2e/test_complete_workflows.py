# tests/e2e/test_complete_workflows.py
import pytest
import asyncio
from typing import Dict, Any, List

class TestCompleteWorkflows:
    """End-to-end testing of complete assistant workflows."""
    
    @pytest.mark.e2e
    async def test_coding_assistance_workflow(
        self, 
        assistant_core_live,
        isolated_workspace
    ):
        """Test complete coding assistance workflow."""
        session_id = "e2e_coding_test"
        
        # Step 1: Request code creation
        response1 = await assistant_core_live.process_message(
            "Create a Python function to calculate fibonacci numbers",
            session_id=session_id,
            user_id="e2e_test_user"
        )
        
        assert "fibonacci" in response1["response"].lower()
        assert "def" in response1["response"]  # Should contain function definition
        
        # Step 2: Request file creation
        response2 = await assistant_core_live.process_message(
            "Save this function to a file called fibonacci.py",
            session_id=session_id,
            user_id="e2e_test_user"
        )
        
        # Verify file was created
        fibonacci_file = isolated_workspace / "fibonacci.py"
        assert fibonacci_file.exists()
        
        content = fibonacci_file.read_text()
        assert "fibonacci" in content.lower()
        assert "def" in content
        
        # Step 3: Request code modification
        response3 = await assistant_core_live.process_message(
            "Add error handling to the fibonacci function",
            session_id=session_id,
            user_id="e2e_test_user"
        )
        
        # Verify modification
        updated_content = fibonacci_file.read_text()
        assert any(keyword in updated_content for keyword in ["try", "except", "raise"])
    
    @pytest.mark.e2e
    async def test_research_and_file_workflow(
        self, 
        assistant_core_live,
        isolated_workspace
    ):
        """Test research followed by file organization workflow."""
        session_id = "e2e_research_test"
        
        # Step 1: Research request
        response1 = await assistant_core_live.process_message(
            "Search for information about Python async programming best practices",
            session_id=session_id,
            user_id="e2e_test_user"
        )
        
        assert len(response1["response"]) > 100  # Should have substantial content
        
        # Step 2: Save research results
        response2 = await assistant_core_live.process_message(
            "Save this research to a markdown file",
            session_id=session_id,
            user_id="e2e_test_user"
        )
        
        # Find created markdown file
        md_files = list(isolated_workspace.glob("*.md"))
        assert len(md_files) >= 1
        
        # Step 3: Organize files
        response3 = await assistant_core_live.process_message(
            "Create a research folder and organize these files",
            session_id=session_id,
            user_id="e2e_test_user"
        )
        
        # Verify organization
        research_dir = isolated_workspace / "research"
        assert research_dir.exists() and research_dir.is_dir()
