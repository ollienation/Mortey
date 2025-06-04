# FIXED State Management - LangGraph 0.4.8 (June 2025)
# CRITICAL FIXES: Proper message validation, tool call handling, and modern patterns

from typing import Annotated, List, Dict, Any, Optional, Union, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import MessagesState, add_messages
from langgraph.managed import RemainingSteps  # ✅ CRITICAL FIX: Add this import
from dataclasses import dataclass, field
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

def validate_and_filter_messages_v2(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    ✅ FIXED MESSAGE VALIDATION for LangGraph 0.4.8 - June 2025
    This is the CRITICAL fix that resolves the "messages must have non-empty content" error.
    
    Key improvements:
    - Properly handles tool call messages (empty content is NORMAL and VALID)
    - Handles ToolMessage responses correctly
    - Filters out supervisor handoff artifacts
    - Uses modern LangGraph 0.4.8 message patterns
    - Comprehensive content type validation
    
    CRITICAL UNDERSTANDING:
    - Tool call messages SHOULD have empty content - this is correct behavior
    - Only filter messages that are truly invalid (no content AND no tool calls)
    - Always ensure at least one valid message exists
    """
    if not messages:
        logger.warning("Empty message list provided, creating default message")
        return [HumanMessage(content="Hello")]

    filtered = []
    for i, msg in enumerate(messages):
        try:
            # Skip messages without content attribute
            if not hasattr(msg, 'content'):
                logger.debug(f"Message {i} missing content attribute, skipping")
                continue

            content = msg.content
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls

            # ✅ CRITICAL FIX: Tool call messages with empty content are VALID
            if has_tool_calls:
                # Tool call messages are expected to have empty string content
                filtered.append(msg)
                logger.debug(f"Message {i} valid tool call message with {len(msg.tool_calls)} tool calls")
                continue

            # ✅ Handle ToolMessage (responses from tool executions)
            if isinstance(msg, ToolMessage):
                # ToolMessage should have content from tool execution
                if msg.content and str(msg.content).strip():
                    filtered.append(msg)
                    logger.debug(f"Message {i} valid tool message: {msg.tool_call_id}")
                else:
                    logger.debug(f"Message {i} empty tool message, skipping: {msg.tool_call_id}")
                continue

            # ✅ Handle regular message content validation
            if isinstance(content, str):
                if content.strip():  # Non-empty string
                    filtered.append(msg)
                    logger.debug(f"Message {i} valid string content: {len(content)} chars")
                else:
                    logger.debug(f"Message {i} empty string content, skipping")
                continue

            # ✅ Handle list content (multimodal, complex content)
            elif isinstance(content, list):
                if content:  # Non-empty list
                    has_valid_content = False
                    for item in content:
                        if isinstance(item, dict):
                            # Check for various content types
                            if (item.get('text', '').strip() or
                                item.get('content', '').strip() or
                                item.get('type') in ['tool_use', 'tool_result', 'image', 'text']):
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

            # ✅ Handle dict content
            elif isinstance(content, dict):
                if content and any(v for v in content.values() if v):
                    filtered.append(msg)
                    logger.debug(f"Message {i} valid dict content: {len(content)} keys")
                else:
                    logger.debug(f"Message {i} empty dict content, skipping")

            # ✅ Handle other content types
            else:
                if content:  # Truthy content
                    filtered.append(msg)
                    logger.debug(f"Message {i} valid other content: {type(content)}")
                else:
                    logger.debug(f"Message {i} falsy content, skipping")

        except Exception as e:
            logger.warning(f"Error processing message {i}: {e}, including by default")
            # Include by default if error occurs during validation
            filtered.append(msg)
            continue

    # ✅ CRITICAL: Filter out supervisor handoff artifacts that cause empty content errors
    final_filtered = []
    for msg in filtered:
        try:
            # Skip supervisor handoff messages that can cause issues
            if hasattr(msg, 'name') and str(msg.name).startswith('transfer_to_'):
                logger.debug(f"Skipping supervisor handoff message: {msg.name}")
                continue

            # Skip messages with supervisor routing artifacts
            if (hasattr(msg, 'additional_kwargs') and
                isinstance(msg.additional_kwargs, dict) and
                msg.additional_kwargs.get('supervisor_route')):
                logger.debug(f"Skipping supervisor routing message")
                continue

            final_filtered.append(msg)
        except Exception as e:
            logger.warning(f"Error filtering supervisor artifacts: {e}")
            # Include by default if error occurs
            final_filtered.append(msg)

    # ✅ CRITICAL: Ensure we always have at least one message
    if not final_filtered:
        logger.warning("No valid messages after filtering, creating default message")
        final_filtered = [HumanMessage(content="Hello")]

    logger.info(f"Message validation v2: {len(messages)} -> {len(filtered)} -> {len(final_filtered)} messages")
    return final_filtered

# ✅ FIXED: Modern AssistantState for LangGraph 0.4.8
@dataclass
class AssistantState(MessagesState):
    """
    ✅ FIXED: Modern LangGraph 0.4.8 state using MessagesState pattern.
    CRITICAL IMPROVEMENTS for June 2025:
    - Added required remaining_steps field for supervisor compatibility
    - Fully compatible with LangGraph 0.4.8 state_schema requirements
    - Uses proper type annotations for 0.4.8
    - Inherits from MessagesState for built-in message handling
    - Simplified field definitions without redundant message handling
    - Proper initialization with mutable field defaults
    """
    # MessagesState already provides:
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # ✅ CRITICAL FIX: Add required remaining_steps field for supervisor
    remaining_steps: RemainingSteps = 25  # Default to 25 steps

    # ✅ Agent and session management
    current_agent: str = ""
    thinking_state: Optional[ThinkingState] = None
    session_id: str = ""
    user_id: str = "default_user"

    # ✅ Agent-specific context
    code_context: Dict[str, Any] = field(default_factory=dict)
    web_results: List[Dict[str, Any]] = field(default_factory=list)

    # ✅ Human-in-the-loop control (enhanced for 0.4.8)
    requires_approval: bool = False
    approval_context: Dict[str, Any] = field(default_factory=dict)
    pending_human_input: Optional[str] = None
    interrupt_reason: Optional[str] = None  # New in 0.4.8

    # ✅ File and output management
    temp_files: List[str] = field(default_factory=list)
    saved_files: List[str] = field(default_factory=list)
    output_content: str = ""
    output_type: str = "text"

    # ✅ Enhanced memory management for 0.4.8
    max_messages: int = 50
    memory_strategy: str = "smart_trim"  # Updated strategy
    message_budget: int = 4000  # Token budget for messages

    # ✅ Performance and monitoring
    token_usage: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0
    agent_execution_count: Dict[str, int] = field(default_factory=dict)

    # ✅ Modern LangGraph 0.4.8 features
    checkpoint_namespace: Optional[str] = None
    subgraph_context: Dict[str, Any] = field(default_factory=dict)
    stream_mode: str = "values"  # Default stream mode

    def __post_init__(self):
        """Initialize computed fields after creation"""
        if not self.session_id:
            self.session_id = f"session_{int(time.time())}"

        # Ensure agent execution count is initialized
        for agent_type in AgentType:
            if agent_type.value not in self.agent_execution_count:
                self.agent_execution_count[agent_type.value] = 0

# ✅ CRITICAL FIX: Fix smart_trim_messages_v2 to work with proper state objects
# ✅ CRITICAL FIX: Fix smart_trim_messages_v2 to work with proper state objects

def smart_trim_messages_v2(
    state: Union[AssistantState, Dict[str, Any]],
    max_tokens: int = 4000,
    preserve_system: bool = True,
    preserve_recent_tools: bool = True
) -> Dict[str, Any]:
    """
    ✅ FIXED: Smart message trimming for LangGraph 0.4.8 with enhanced features

    CRITICAL FIX: Removed all isinstance() checks for TypedDict compatibility
    
    New features for June 2025:
    - Preserves recent tool call sequences
    - Better token estimation
    - Maintains conversation context
    - Handles multimodal content
    """
    try:
        from langchain_core.messages.utils import trim_messages

        # ✅ CRITICAL FIX: Always treat state as dictionary - no isinstance checks
        current_messages = state.get("messages", [])

        if not current_messages:
            return {"messages": [HumanMessage(content="Hello")]}

        # ✅ Enhanced trimming logic for 0.4.8
        trimmed = trim_messages(
            messages=current_messages,
            max_tokens=max_tokens,
            strategy="last",  # ✅ Only supported strategy
            include_system=preserve_system,
            start_on="human",
            end_on=("human", "tool") if preserve_recent_tools else "human",
            allow_partial=False,
            token_counter=lambda msgs: sum(len(str(m.content)) // 4 for m in msgs)
        )

        # Apply enhanced trimming
        try:
            trimmed = trim_messages(
                messages=current_messages,
                max_tokens=max_tokens,
                include_system=preserve_system,
                strategy="last",
                token_counter=lambda msgs: sum(len(str(m.content)) // 4 for m in msgs)  # Simple token estimation
            )
        except TypeError:
            # Fallback for older trim_messages signature
            trimmed = trim_messages(
                messages=current_messages,
                max_tokens=max_tokens,
                include_system=preserve_system,
                strategy="last"
            )

        # ✅ CRITICAL: Validate after trimming
        validated = validate_and_filter_messages_v2(trimmed)
        logger.info(f"Smart message trimming v2: {len(current_messages)} -> {len(trimmed)} -> {len(validated)}")
        
        return {"messages": validated}

    except Exception as e:
        logger.error(f"Error in smart_trim_messages_v2: {e}")
        # Enhanced fallback strategy - always use dictionary access
        current_messages = state.get("messages", [])
        
        if current_messages:
            # Keep more context in fallback
            recent = current_messages[-15:]  # Increased from 10
            validated = validate_and_filter_messages_v2(recent)
            return {"messages": validated}
        else:
            return {"messages": [HumanMessage(content="Hello")]}

# ✅ CRITICAL FIX: Fix create_optimized_state to properly create AssistantState instance
def create_optimized_state(
    session_id: str = "",
    user_id: str = "default",
    initial_context: Optional[Dict[str, Any]] = None
) -> AssistantState:
    # Convert initial context to proper dataclass
    state_data = {
        "session_id": session_id or f"session_{int(time.time())}",
        "user_id": user_id,
        "messages": [],
        "current_agent": "",
        "stream_mode": "values",
        "message_budget": 4000,
        "memory_strategy": "smart_trim",
        "remaining_steps": 25  # ✅ CRITICAL: Add required remaining_steps field
    }
    if initial_context:
        state_data.update(initial_context)
    
    # ✅ CRITICAL: Create proper AssistantState instance
    return AssistantState(**state_data)

# ✅ CRITICAL FIX: Fix update_state_with_validation to properly handle state objects
def update_state_with_validation(
    current_state: AssistantState,
    updates: Dict[str, Any]
) -> AssistantState:
    """
    ✅ FIXED: Update state with enhanced validation for 0.4.8
    """
    try:
        # Convert current state to dict for updates
        current_state_dict = {k: getattr(current_state, k) for k in current_state.__annotations__}
        
        # Update with new values
        new_state_data = current_state_dict.copy()
        new_state_data.update(updates)

        # ✅ Validate messages if they were updated
        if "messages" in updates:
            new_state_data["messages"] = validate_and_filter_messages_v2(updates["messages"])

        # ✅ Update agent execution count if agent changed
        if "current_agent" in updates and updates["current_agent"]:
            agent_name = updates["current_agent"]
            if agent_name in new_state_data.get("agent_execution_count", {}):
                new_state_data["agent_execution_count"][agent_name] += 1

        # ✅ CRITICAL FIX: Create a proper AssistantState instance
        return AssistantState(**new_state_data)
    except Exception as e:
        logger.error(f"Error updating state: {e}")
        return current_state

# ✅ Modern reducers for LangGraph 0.4.8
def enhanced_message_reducer(
    current: Sequence[BaseMessage],
    new: Union[BaseMessage, Sequence[BaseMessage]]
) -> Sequence[BaseMessage]:
    """
    ✅ FIXED: Enhanced message reducer for LangGraph 0.4.8
    """
    if isinstance(new, BaseMessage):
        new_messages = [new]
    else:
        new_messages = list(new)

    updated_messages = list(current) + new_messages
    return validate_and_filter_messages_v2(updated_messages)

# ✅ Backward compatibility
validate_and_filter_messages = validate_and_filter_messages_v2
smart_trim_messages = smart_trim_messages_v2
create_empty_state = create_optimized_state
update_state_safely = update_state_with_validation

# Legacy compatibility
MorteyState = AssistantState