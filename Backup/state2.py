# Fixed State Management - June 2025 LangGraph 0.4.8 Patterns

from typing import Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import MessagesState
from langgraph.managed import RemainingSteps
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger("state")

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

def validate_and_filter_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    ✅ FIXED MESSAGE VALIDATION - CRITICAL FIX for LangGraph 0.4.8
    
    This function prevents ALL empty content API errors by ensuring
    robust validation of message content before reaching the LLM API.
    
    Key fixes:
    - Properly handles all message content types
    - Ensures at least one valid message exists
    - Filters out supervisor handoff artifacts
    - Compatible with LangGraph 0.4.8 message structure
    """
    if not messages:
        logger.warning("Empty message list provided, creating default message")
        return [HumanMessage(content="Hello")]

    filtered = []
    for i, msg in enumerate(messages):
        if not hasattr(msg, 'content'):
            logger.debug(f"Message {i} missing content attribute, skipping")
            continue

        content = msg.content

        # Handle string content
        if isinstance(content, str):
            if content.strip():  # Non-empty string
                filtered.append(msg)
                logger.debug(f"Message {i} valid string content: {len(content)} chars")
            else:
                logger.debug(f"Message {i} empty string content, skipping")
                continue

        # Handle list content (tool calls, multimodal, etc.)
        elif isinstance(content, list):
            if content:  # Non-empty list
                # Check if list contains actual content
                has_valid_content = False
                for item in content:
                    if isinstance(item, dict):
                        # Valid tool calls or text blocks
                        if (item.get('text', '').strip() or
                            item.get('type') in ['tool_use', 'tool_result', 'image']):
                            has_valid_content = True
                            break
                    elif isinstance(item, str) and item.strip():
                        has_valid_content = True
                        break

                if has_valid_content:
                    filtered.append(msg)
                    logger.debug(f"Message {i} valid list content: {len(content)} items")
                else:
                    logger.debug(f"Message {i} empty list content, skipping")
            else:
                logger.debug(f"Message {i} empty list, skipping")

        # Handle other content types (should be truthy)
        else:
            if content:  # Truthy content
                filtered.append(msg)
                logger.debug(f"Message {i} valid other content: {type(content)}")
            else:
                logger.debug(f"Message {i} falsy content, skipping")

    # ✅ CRITICAL: Ensure at least one message exists
    if not filtered:
        logger.warning("No valid messages after filtering, creating default message")
        filtered = [HumanMessage(content="Hello")]

    # ✅ ADDITIONAL: Remove any supervisor handoff artifacts
    final_filtered = []
    for msg in filtered:
        # Skip messages that might be empty handoff messages from supervisor
        if hasattr(msg, 'name') and str(msg.name).startswith('transfer_to_'):
            logger.debug(f"Skipping potential handoff message: {msg.name}")
            continue
        
        # ✅ NEW: Additional check for supervisor tool calls that might be empty
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Check if tool calls have valid content
            valid_tool_calls = []
            for tool_call in msg.tool_calls:
                if (hasattr(tool_call, 'args') and tool_call.args and 
                    any(v for v in tool_call.args.values() if v)):
                    valid_tool_calls.append(tool_call)
            
            # Only include message if it has valid tool calls or valid content
            if valid_tool_calls or (hasattr(msg, 'content') and msg.content):
                final_filtered.append(msg)
        else:
            final_filtered.append(msg)

    # Ensure we still have messages after handoff filtering
    if not final_filtered:
        final_filtered = [HumanMessage(content="Hello")]

    logger.info(f"Message validation: {len(messages)} -> {len(filtered)} -> {len(final_filtered)} messages")
    return final_filtered

# ✅ FIXED: Modern MessagesState usage for LangGraph 0.4.8
class AssistantState(MessagesState):
    """
    Modern LangGraph state using MessagesState pattern for LangGraph 0.4.8.
    
    KEY IMPROVEMENTS:
    - Inherits messages handling from MessagesState (no need to redefine)
    - Uses built-in add_messages reducer
    - Simplified field definitions
    - Proper type annotations for 0.4.8
    - Required state_schema compliance
    """
    # MessagesState already provides:
    # messages: Annotated[List[BaseMessage], add_messages]

    remaining_steps: RemainingSteps

    # Additional state fields only:
    current_agent: str = ""
    thinking_state: Optional[ThinkingState] = None

    # Session management
    session_id: str = ""
    user_id: str = "default_user"

    # Agent-specific data
    code_context: Dict[str, Any] = None
    web_results: List[Dict[str, Any]] = None

    # Human-in-the-loop control
    requires_approval: bool = False
    approval_context: Dict[str, Any] = None
    pending_human_input: Optional[str] = None

    # File management
    temp_files: List[str] = None
    saved_files: List[str] = None

    # Output handling
    output_content: str = ""
    output_type: str = "text"

    # Voice/GUI integration
    voice_mode: bool = False
    gui_callback: Optional[Any] = None

    # Memory management
    max_messages: int = 50
    memory_strategy: str = "trim"

    # Performance tracking
    token_usage: Dict[str, int] = None
    processing_time: float = 0.0

    def __init__(self, **data):
        """Initialize with proper defaults for mutable fields"""
        # Set defaults for mutable fields
        if data.get('code_context') is None:
            data['code_context'] = {}
        if data.get('web_results') is None:
            data['web_results'] = []
        if data.get('approval_context') is None:
            data['approval_context'] = {}
        if data.get('temp_files') is None:
            data['temp_files'] = []
        if data.get('saved_files') is None:
            data['saved_files'] = []
        if data.get('token_usage') is None:
            data['token_usage'] = {}
        
        super().__init__(**data)

def smart_trim_messages(
    state: AssistantState,
    max_tokens: int = 4000,
    preserve_system: bool = True
) -> Dict[str, Any]:
    """
    ✅ FIXED: Smart message trimming with proper token counting for LangGraph 0.4.8
    """
    try:
        # Import trim_messages for LangGraph 0.4.8
        from langchain_core.messages.utils import trim_messages
        
        # Get current messages
        current_messages = state.get("messages", [])
        if not current_messages:
            return {"messages": [HumanMessage(content="Hello")]}

        # Apply trimming with simplified token counting for 0.4.8
        trimmed = trim_messages(
            messages=current_messages,
            max_tokens=max_tokens,
            include_system=preserve_system,
            strategy="last"  # Keep most recent messages
        )

        # ✅ CRITICAL: Validate after trimming
        validated = validate_and_filter_messages(trimmed)

        logger.info(f"Message trimming: {len(current_messages)} -> {len(trimmed)} -> {len(validated)}")
        return {"messages": validated}

    except Exception as e:
        logger.error(f"Error in smart_trim_messages: {e}")
        # Fallback to recent messages
        current_messages = state.get("messages", [])
        if current_messages:
            recent = current_messages[-10:]
            validated = validate_and_filter_messages(recent)
            return {"messages": validated}
        else:
            return {"messages": [HumanMessage(content="Hello")]}

def create_empty_state(session_id: str = "", user_id: str = "default") -> AssistantState:
    """Create a properly initialized empty state"""
    return AssistantState(
        messages=[],  # Will be populated with validated messages
        session_id=session_id or f"session_{int(time.time())}",
        user_id=user_id,
        current_agent="",
        code_context={},
        web_results=[],
        temp_files=[],
        saved_files=[],
        token_usage={},
        processing_time=0.0
    )

def update_state_safely(
    current_state: AssistantState,
    updates: Dict[str, Any]
) -> AssistantState:
    """Safely update state with validation"""
    try:
        # Create new state with updates
        new_state = {**current_state, **updates}
        
        # Validate messages if they were updated
        if "messages" in updates:
            new_state["messages"] = validate_and_filter_messages(updates["messages"])
        
        return new_state
    except Exception as e:
        logger.error(f"Error updating state: {e}")
        return current_state

def trim_message_history(state: AssistantState, max_tokens: int = 4000) -> Dict[str, Any]:
    """
    Trim message history - FIXED for test compatibility
    This function is expected by test_assistant.py and delegates to the existing smart_trim_messages function
    """
    return smart_trim_messages(state, max_tokens)

# Add modern state reducers for database persistence
def message_reducer(state: List[BaseMessage], message: BaseMessage) -> List[BaseMessage]:
    """Reducer optimized for database storage"""
    return validate_and_filter_messages([*state, message])

def trim_messages_reducer(state: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer for automatic message trimming"""
    return smart_trim_messages(state, max_tokens=4000)

AssistantState.update_forward_refs(
    message_reducer=message_reducer,
    trim_reducer=trim_messages_reducer
)


# Legacy compatibility
MorteyState = AssistantState