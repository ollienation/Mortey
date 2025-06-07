# core/state.py - ✅ ENHANCED WITH COMPREHENSIVE STATE VALIDATION
from typing import Annotated, Dict, Any, Optional, List, Union, Sequence, TypedDict, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
import logging
import time
import uuid

logger = logging.getLogger("state")

# ✅ ENHANCED: More robust TypedDict with optional fields for flexibility
class AssistantState(TypedDict):
    """
    The state of the assistant. This is a TypedDict, so all access
    should be through dictionary keys, e.g., state['messages'].
    """
    # LangGraph will automatically manage this field
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Required fields for basic operation
    session_id: str
    user_id: str
    current_agent: str

class StateValidationError(Exception):
    """Custom exception for state validation errors"""
    pass

class StateValidator:
    """
    ✅ NEW: Comprehensive state validation and sanitization utilities.
    
    This class provides robust validation, sanitization, and migration
    capabilities for AssistantState objects.
    """
    
    @staticmethod
    def validate_state(state: Any, strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate an AssistantState object comprehensively.
        
        Args:
            state: The state object to validate
            strict: If True, enforces strict validation rules
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if state is a dictionary
        if not isinstance(state, dict):
            errors.append(f"State must be a dictionary, got {type(state)}")
            return False, errors
        
        # Check required fields
        required_fields = ["messages", "session_id", "user_id", "current_agent"]
        for field in required_fields:
            if field not in state:
                errors.append(f"Missing required field: {field}")
            elif state[field] is None:
                errors.append(f"Field {field} cannot be None")
        
        # Validate messages field
        if "messages" in state:
            messages_valid, message_errors = StateValidator._validate_messages(
                state["messages"], strict
            )
            if not messages_valid:
                errors.extend(message_errors)
        
        # Validate string fields
        string_fields = ["session_id", "user_id", "current_agent"]
        for field in string_fields:
            if field in state and not isinstance(state[field], str):
                errors.append(f"Field {field} must be a string, got {type(state[field])}")
            elif field in state and not state[field].strip():
                if strict:
                    errors.append(f"Field {field} cannot be empty string")
        
        # Check for unknown fields in strict mode
        if strict:
            known_fields = set(required_fields)
            unknown_fields = set(state.keys()) - known_fields
            if unknown_fields:
                errors.append(f"Unknown fields in strict mode: {unknown_fields}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_messages(messages: Any, strict: bool = False) -> Tuple[bool, List[str]]:
        """Validate the messages field specifically"""
        errors = []
        
        if not isinstance(messages, (list, tuple)):
            errors.append(f"Messages must be a list or tuple, got {type(messages)}")
            return False, errors
        
        valid_message_types = (HumanMessage, AIMessage, ToolMessage, SystemMessage)
        
        for i, message in enumerate(messages):
            if not isinstance(message, BaseMessage):
                errors.append(f"Message {i} is not a BaseMessage instance: {type(message)}")
                continue
            
            # Check message content
            if hasattr(message, 'content'):
                content = getattr(message, 'content', '')
                if strict and isinstance(message, HumanMessage) and not content.strip():
                    errors.append(f"HumanMessage {i} has empty content in strict mode")
                elif isinstance(message, AIMessage):
                    # AIMessage can have empty content if it has tool calls
                    has_tool_calls = (hasattr(message, 'tool_calls') and 
                                    message.tool_calls and 
                                    len(message.tool_calls) > 0)
                    if not content.strip() and not has_tool_calls:
                        errors.append(f"AIMessage {i} has no content and no tool calls")
            
            # Validate tool calls for AIMessage
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls'):
                tool_calls = getattr(message, 'tool_calls', [])
                if tool_calls:
                    for j, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, dict):
                            errors.append(f"Tool call {j} in message {i} must be a dict")
                            continue
                        
                        required_tool_fields = ["name", "args", "id"]
                        for field in required_tool_fields:
                            if field not in tool_call:
                                errors.append(f"Tool call {j} missing field: {field}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_state(state: Dict[str, Any]) -> AssistantState:
        """
        Sanitize and normalize a state dictionary to ensure it's valid.
        
        This method attempts to fix common issues with state objects.
        """
        try:
            # Create a clean state dictionary
            sanitized: Dict[str, Any] = {}
            
            # Handle messages field
            if "messages" in state:
                sanitized["messages"] = StateValidator._sanitize_messages(state["messages"])
            else:
                sanitized["messages"] = []
            
            # Handle string fields with defaults
            sanitized["session_id"] = str(state.get("session_id", f"session_{int(time.time())}")).strip()
            sanitized["user_id"] = str(state.get("user_id", "default_user")).strip()
            sanitized["current_agent"] = str(state.get("current_agent", "")).strip()
            
            # Ensure no empty strings
            if not sanitized["session_id"]:
                sanitized["session_id"] = f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            if not sanitized["user_id"]:
                sanitized["user_id"] = "default_user"
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing state: {e}")
            # Return minimal valid state as fallback
            return {
                "messages": [],
                "session_id": f"fallback_session_{int(time.time())}",
                "user_id": "default_user",
                "current_agent": ""
            }
    
    @staticmethod
    def _sanitize_messages(messages: Any) -> List[BaseMessage]:
        """Sanitize messages list"""
        if not isinstance(messages, (list, tuple)):
            logger.warning(f"Messages field is not a list/tuple: {type(messages)}")
            return []
        
        sanitized_messages = []
        for i, message in enumerate(messages):
            try:
                if isinstance(message, BaseMessage):
                    # Validate the message content
                    if isinstance(message, AIMessage):
                        # Check if AIMessage has content or tool calls
                        content = getattr(message, 'content', '')
                        tool_calls = getattr(message, 'tool_calls', [])
                        
                        if content.strip() or (tool_calls and len(tool_calls) > 0):
                            sanitized_messages.append(message)
                        else:
                            logger.debug(f"Skipping empty AIMessage {i}")
                    else:
                        # For other message types, just check content
                        content = getattr(message, 'content', '')
                        if content.strip():
                            sanitized_messages.append(message)
                        else:
                            logger.debug(f"Skipping empty message {i}: {type(message)}")
                else:
                    logger.warning(f"Skipping non-BaseMessage object {i}: {type(message)}")
            except Exception as e:
                logger.error(f"Error processing message {i}: {e}")
                continue
        
        return sanitized_messages

def create_optimized_state(
    session_id: str = "",
    user_id: str = "default",
    initial_context: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> AssistantState:
    """
    ✅ ENHANCED: Creates a state dictionary that conforms to the AssistantState TypedDict.
    
    Args:
        session_id: Session identifier
        user_id: User identifier  
        initial_context: Initial context to apply to the state
        validate: Whether to validate the resulting state
    """
    
    # Base state data that matches the TypedDict definition
    state_data: Dict[str, Any] = {
        "session_id": session_id or f"session_{int(time.time())}",
        "user_id": user_id,
        "messages": [],
        "current_agent": "",
    }
    
    # Apply initial context
    if initial_context:
        if "messages" in initial_context:
            validated_messages = validate_and_filter_messages_v3(initial_context["messages"])
            state_data["messages"] = validated_messages
        if "current_agent" in initial_context:
            state_data["current_agent"] = str(initial_context["current_agent"])
    
    # Sanitize the state if validation is requested
    if validate:
        state_data = StateValidator.sanitize_state(state_data)
        
        # Perform final validation
        is_valid, errors = StateValidator.validate_state(state_data, strict=False)
        if not is_valid:
            logger.warning(f"State validation warnings: {errors}")
    
    return state_data

def validate_and_filter_messages_v3(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    ✅ ENHANCED: Validates messages with comprehensive error checking.
    
    This version provides more robust validation and better error recovery.
    """
    if not messages:
        return []
    
    return StateValidator._sanitize_messages(messages)

def safe_state_access(state: AssistantState, key: str, default: Any = None) -> Any:
    """
    ✅ NEW: Safely access state fields with fallback values.
    
    This utility function provides safe access to state fields with
    proper error handling and logging.
    """
    try:
        if not isinstance(state, dict):
            logger.error(f"State is not a dictionary: {type(state)}")
            return default
        
        value = state.get(key, default)
        
        # Type validation for known fields
        if key == "messages" and value is not None:
            if not isinstance(value, (list, tuple)):
                logger.warning(f"Messages field is not a list: {type(value)}")
                return default if default is not None else []
        elif key in ["session_id", "user_id", "current_agent"] and value is not None:
            if not isinstance(value, str):
                logger.warning(f"{key} field is not a string: {type(value)}")
                return str(value) if value is not None else default
        
        return value
        
    except Exception as e:
        logger.error(f"Error accessing state key '{key}': {e}")
        return default

def migrate_legacy_state(old_state: Dict[str, Any]) -> AssistantState:
    """
    ✅ NEW: Migrate legacy state formats to current TypedDict format.
    
    This function handles backward compatibility with older state formats.
    """
    try:
        # Handle different legacy formats
        migrated_state = {}
        
        # Basic field migration
        migrated_state["session_id"] = str(old_state.get(
            "session_id", 
            old_state.get("thread_id", f"migrated_session_{int(time.time())}")
        ))
        migrated_state["user_id"] = str(old_state.get("user_id", "migrated_user"))
        migrated_state["current_agent"] = str(old_state.get("current_agent", ""))
        
        # Messages migration
        messages = old_state.get("messages", [])
        migrated_state["messages"] = validate_and_filter_messages_v3(messages)
        
        # Remove any legacy fields that might cause issues
        legacy_fields = ["thread_id", "remaining_steps", "context", "metadata"]
        
        logger.info(f"Migrated legacy state with {len(messages)} messages")
        return StateValidator.sanitize_state(migrated_state)
        
    except Exception as e:
        logger.error(f"Error migrating legacy state: {e}")
        # Return minimal valid state
        return create_optimized_state()

def get_state_summary(state: AssistantState) -> Dict[str, Any]:
    """
    ✅ NEW: Get a summary of the current state for debugging and monitoring.
    """
    try:
        messages = safe_state_access(state, "messages", [])
        
        message_types = {}
        for msg in messages:
            msg_type = type(msg).__name__
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        return {
            "session_id": safe_state_access(state, "session_id", "unknown"),
            "user_id": safe_state_access(state, "user_id", "unknown"),
            "current_agent": safe_state_access(state, "current_agent", "none"),
            "message_count": len(messages),
            "message_types": message_types,
            "is_valid": StateValidator.validate_state(state)[0],
            "state_size_bytes": len(str(state))
        }
    except Exception as e:
        logger.error(f"Error creating state summary: {e}")
        return {"error": str(e)}

# Aliases for backward compatibility
create_assistant_state = create_optimized_state
smart_trim_messages_v2 = lambda state, max_messages=15, **kwargs: {
    "messages": safe_state_access(state, "messages", [])[-max_messages:]
}

# ✅ NEW: State performance optimization utilities
def optimize_state_for_processing(state: AssistantState, max_messages: int = 20) -> AssistantState:
    """
    Optimize state for processing by limiting message history and cleaning up data.
    """
    try:
        optimized_state = dict(state)  # Create a copy
        
        # Limit message history to prevent context overflow
        messages = safe_state_access(state, "messages", [])
        if len(messages) > max_messages:
            # Keep the most recent messages, but always keep the first message if it exists
            if messages:
                first_message = messages[0]
                recent_messages = messages[-(max_messages-1):]
                optimized_state["messages"] = [first_message] + recent_messages
                logger.debug(f"Trimmed messages from {len(messages)} to {len(optimized_state['messages'])}")
        
        return optimized_state
        
    except Exception as e:
        logger.error(f"Error optimizing state: {e}")
        return state
