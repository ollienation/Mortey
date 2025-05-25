from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    ROUTER = "router"
    CODER = "coder" 
    WEB = "web"
    VISION = "vision"
    IMAGE = "image"
    CONTROLLER = "controller"

@dataclass
class ThinkingState:
    """Real-time thinking state for GUI display"""
    active_agent: AgentType
    current_task: str
    progress: float
    details: str

class MorteyState(TypedDict):
    """Central state for all agents"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    current_agent: AgentType
    thinking_state: ThinkingState
    
    # Memory and context
    session_id: str
    conversation_history: List[Dict[str, Any]]
    relevant_context: List[Dict[str, Any]]
    
    # Agent-specific data
    code_context: Dict[str, Any]
    web_results: List[Dict[str, Any]]
    vision_data: Dict[str, Any]
    image_data: Dict[str, Any]
    
    # Control and safety
    verification_required: bool
    loop_count: int
    max_loops: int
    
    # File and output management
    sandbox_files: List[str]
    output_content: str
    output_type: str
