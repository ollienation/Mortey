# LangGraph State Management with MessagesState
# June 2025 - Production Ready with Modern Patterns

from typing import Annotated, TypedDict, List, Dict, Any, Optional
import operator
from langgraph.graph.message import add_messages, MessagesState
from langchain_core.messages import BaseMessage, trim_messages
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    """Available agent types in the system"""
    CHAT = "chat"
    CODER = "coder"
    WEB = "web"
    SUPERVISOR = "supervisor"

@dataclass
class ThinkingState:
    """Real-time thinking state for GUI display"""
    active_agent: AgentType
    current_task: str
    progress: float
    details: str

class AssistantState(MessagesState):
    """
    Modern LangGraph state using MessagesState pattern with proper reducers.
    
    Key improvements from June 2025:
    - Uses add_messages reducer for concurrent-safe message handling
    - Implements proper reducers for all list fields
    - Built-in memory management support
    - Optimized for production use
    """
    
    # Messages with automatic handling via add_messages reducer
    # This provides concurrent-safe message updates and deduplication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Current context with proper state management
    current_agent: str = ""
    thinking_state: Optional[ThinkingState] = None
    
    # Session management
    session_id: str = ""
    user_id: str = "default_user"
    
    # Agent-specific data with concurrent-safe reducers
    code_context: Dict[str, Any] = {}
    
    # Web results with proper reducer for concurrent appending
    web_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # Human-in-the-loop control (NEW: uses interrupt patterns)
    requires_approval: bool = False
    approval_context: Dict[str, Any] = {}
    pending_human_input: Optional[str] = None
    
    # File management with concurrent-safe list handling
    temp_files: Annotated[List[str], operator.add]
    saved_files: Annotated[List[str], operator.add]
    
    # Output handling
    output_content: str = ""
    output_type: str = "text"
    
    # Voice/GUI integration
    voice_mode: bool = False
    gui_callback: Optional[Any] = None
    
    # Memory management (NEW in June 2025)
    max_messages: int = 50  # Maximum messages before trimming
    memory_strategy: str = "trim"  # "trim", "summarize", or "delete"
    
    # Performance tracking
    token_usage: Dict[str, int] = {}
    processing_time: float = 0.0

# Memory management helper functions
def trim_message_history(state: AssistantState) -> Dict[str, Any]:
    """
    Trim message history to prevent context window overflow.
    Uses LangGraph's built-in trim_messages function.
    """
    if len(state["messages"]) > state.get("max_messages", 50):
        # Use LangGraph's trim_messages for intelligent trimming
        trimmed_messages = trim_messages(
            state["messages"],
            max_tokens=4000,  # Keep within reasonable context
            strategy="last",  # Keep most recent messages
            allow_partial=False
        )
        
        return {
            "messages": trimmed_messages,
            "memory_strategy": "trimmed"
        }
    
    return {}

def summarize_old_messages(state: AssistantState) -> Dict[str, Any]:
    """
    Summarize older messages to preserve context while reducing tokens.
    This would be implemented with an LLM call to create summaries.
    """
    # Implementation would use LLM to summarize older messages
    # and replace them with a summary message
    pass

# Reducer helper for complex state updates
def update_agent_context(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Custom reducer for agent context that merges dictionaries safely"""
    if not left:
        return right
    if not right:
        return left
    
    result = left.copy()
    result.update(right)
    return result

# Enhanced state for specific use cases
class CodeAssistantState(AssistantState):
    """Specialized state for coding tasks"""
    
    # Code-specific context with custom reducer
    code_context: Annotated[Dict[str, Any], update_agent_context]
    
    # Active file being edited
    active_file: str = ""
    
    # Code execution results
    execution_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # Linting and analysis results
    code_analysis: Dict[str, Any] = {}

class WebAssistantState(AssistantState):
    """Specialized state for web research tasks"""
    
    # Search query history
    search_queries: Annotated[List[str], operator.add]
    
    # Web results with metadata
    web_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # Sources for citation tracking
    sources: Annotated[List[str], operator.add]
    
    # Research session context
    research_context: Dict[str, Any] = {}

# Legacy compatibility
MorteyState = AssistantState

# State validation helpers
def validate_state(state: AssistantState) -> bool:
    """Validate state structure and contents"""
    required_fields = ["messages", "session_id", "user_id"]
    
    for field in required_fields:
        if field not in state:
            return False
    
    # Validate message types
    if not all(isinstance(msg, BaseMessage) for msg in state["messages"]):
        return False
    
    return True

def get_conversation_summary(state: AssistantState) -> str:
    """Get a summary of the current conversation state"""
    message_count = len(state["messages"])
    current_agent = state.get("current_agent", "none")
    session_duration = state.get("processing_time", 0)
    
    return f"Session {state['session_id']}: {message_count} messages, agent: {current_agent}, duration: {session_duration:.2f}s"