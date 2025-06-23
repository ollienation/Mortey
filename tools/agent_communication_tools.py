# tools/agent_communication_tools.py - Tools for Cross-Agent Communication

import logging
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

from core.enhanced_state import (
    EnhancedAssistantState, AgentCommunicationManager, ScratchpadManager,
    AgentMessage, MessageType, MessagePriority
)

logger = logging.getLogger("agent_communication_tools")

class AgentCommunicationTools:
    """Tools for enabling cross-agent communication"""
    
    @staticmethod
    def get_tools():
        """Get all agent communication tools"""
        
        @tool
        def send_message_to_agent(
            target_agent: str,
            message_content: str,
            message_type: str = "notification",
            priority: str = "normal",
            data: Optional[str] = None
        ) -> str:
            """
            Send a message to another agent in the system.
            
            Args:
                target_agent: Name of the agent to send message to (chat, coder, web, file_manager, project_management)
                message_content: Content of the message to send
                message_type: Type of message (request, response, notification, data_share)
                priority: Priority level (low, normal, high, urgent)
                data: Optional JSON data to include with message
            """
            try:
                # This would be called within agent execution context
                # For now, return instruction for implementation
                return f"Message queued for {target_agent}: {message_content}"
                
            except Exception as e:
                return f"Failed to send message: {str(e)}"
        
        @tool
        def request_agent_assistance(
            target_agent: str,
            task_description: str,
            expected_response_format: str = "text",
            timeout_seconds: int = 60
        ) -> str:
            """
            Request assistance from another agent with a specific task.
            
            Args:
                target_agent: Agent to request assistance from
                task_description: Description of the task needing assistance
                expected_response_format: Expected format of response (text, json, code, data)
                timeout_seconds: Maximum time to wait for response
            """
            try:
                return f"Assistance requested from {target_agent} for: {task_description}"
                
            except Exception as e:
                return f"Failed to request assistance: {str(e)}"
        
        @tool
        def share_data_with_agents(
            data_key: str,
            data_value: str,
            target_agents: str,
            description: str = "",
            expires_in_minutes: int = 60
        ) -> str:
            """
            Share data with other agents through the scratchpad system.
            
            Args:
                data_key: Unique key for the shared data
                data_value: Data to share (as string, will be parsed if JSON)
                target_agents: Comma-separated list of agent names to share with
                description: Description of the shared data
                expires_in_minutes: How long the data should remain available
            """
            try:
                agents = [agent.strip() for agent in target_agents.split(",")]
                
                # Parse JSON data if possible
                import json
                try:
                    parsed_data = json.loads(data_value)
                except json.JSONDecodeError:
                    parsed_data = data_value
                
                return f"Data shared with agents {agents}: {data_key}"
                
            except Exception as e:
                return f"Failed to share data: {str(e)}"
        
        @tool
        def get_shared_data(
            data_key: str,
            requesting_agent: str = ""
        ) -> str:
            """
            Retrieve shared data from the scratchpad system.
            
            Args:
                data_key: Key of the data to retrieve
                requesting_agent: Name of the requesting agent
            """
            try:
                # This would access the scratchpad in the actual implementation
                return f"Retrieved shared data for key: {data_key}"
                
            except Exception as e:
                return f"Failed to retrieve data: {str(e)}"
        
        @tool
        def check_agent_messages(
            agent_name: str = ""
        ) -> str:
            """
            Check for pending messages for the current agent.
            
            Args:
                agent_name: Name of the agent to check messages for
            """
            try:
                # This would check the current state for messages
                return f"Checked messages for agent: {agent_name}"
                
            except Exception as e:
                return f"Failed to check messages: {str(e)}"
        
        @tool
        def broadcast_status_update(
            status_message: str,
            status_type: str = "info",
            include_data: str = ""
        ) -> str:
            """
            Broadcast a status update to all other agents.
            
            Args:
                status_message: Status message to broadcast
                status_type: Type of status (info, warning, error, completion)
                include_data: Optional data to include with status
            """
            try:
                return f"Status broadcast: {status_message}"
                
            except Exception as e:
                return f"Failed to broadcast status: {str(e)}"
        
        return [
            send_message_to_agent,
            request_agent_assistance,
            share_data_with_agents,
            get_shared_data,
            check_agent_messages,
            broadcast_status_update
        ]

# Global instance
agent_communication_tools = AgentCommunicationTools()
