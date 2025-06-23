# core/enhanced_state.py - Enhanced State with Scratchpad and Communication

from typing import Annotated, Optional, Union, TypedDict, Self, Any, List, Dict
from collections.abc import Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
import logging
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("enhanced_state")

class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    DATA_SHARE = "data_share"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.NOTIFICATION
    priority: MessagePriority = MessagePriority.NORMAL
    content: str = ""
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    processed: bool = False
    response_required: bool = False
    correlation_id: Optional[str] = None

@dataclass
class ScratchpadEntry:
    """Shared scratchpad entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = ""
    value: Any = None
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[float] = None
    access_count: int = 0

@dataclass
class FileProcessingStatus:
    """File processing pipeline status"""
    file_id: str = ""
    filename: str = ""
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

class EnhancedAssistantState(TypedDict):
    """Enhanced state with communication and scratchpad capabilities"""
    # Original fields
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    user_id: str
    current_agent: str
    
    # P6-specific state fields
    p6_session_active: Optional[bool]
    current_project_id: Optional[str]
    current_project_name: Optional[str]
    p6_user_preferences: Optional[Dict[str, Any]]
    
    # NEW: Cross-agent communication
    agent_messages: List[AgentMessage]
    agent_communication_history: List[AgentMessage]
    
    # NEW: Shared scratchpad
    scratchpad: Dict[str, ScratchpadEntry]
    scratchpad_history: List[ScratchpadEntry]
    
    # NEW: File processing pipeline
    file_processing_queue: List[FileProcessingStatus]
    file_processing_history: List[FileProcessingStatus]
    
    # NEW: Supervisor verification
    pending_verification: Optional[Dict[str, Any]]
    verification_history: List[Dict[str, Any]]
    
    # NEW: Streaming status
    streaming_enabled: bool
    current_stream_id: Optional[str]
    stream_events: List[Dict[str, Any]]

class AgentCommunicationManager:
    """Manages inter-agent communication"""
    
    @staticmethod
    def send_message(state: EnhancedAssistantState, message: AgentMessage) -> EnhancedAssistantState:
        """Send message between agents"""
        new_state = dict(state)
        
        # Initialize agent_messages if not present
        if 'agent_messages' not in new_state:
            new_state['agent_messages'] = []
        if 'agent_communication_history' not in new_state:
            new_state['agent_communication_history'] = []
        
        # Add to current messages
        new_state['agent_messages'].append(message)
        
        # Add to history
        new_state['agent_communication_history'].append(message)
        
        logger.info(f"Agent message sent: {message.from_agent} -> {message.to_agent}: {message.content[:50]}...")
        
        return new_state
    
    @staticmethod
    def get_messages_for_agent(state: EnhancedAssistantState, agent_name: str) -> List[AgentMessage]:
        """Get unprocessed messages for specific agent"""
        messages = state.get('agent_messages', [])
        return [msg for msg in messages if msg.to_agent == agent_name and not msg.processed]
    
    @staticmethod
    def mark_message_processed(state: EnhancedAssistantState, message_id: str) -> EnhancedAssistantState:
        """Mark message as processed"""
        new_state = dict(state)
        
        messages = new_state.get('agent_messages', [])
        for msg in messages:
            if msg.id == message_id:
                msg.processed = True
                break
        
        return new_state
    
    @staticmethod
    def create_response_message(original: AgentMessage, content: str, data: Optional[Dict] = None) -> AgentMessage:
        """Create response to an agent message"""
        return AgentMessage(
            from_agent=original.to_agent,
            to_agent=original.from_agent,
            message_type=MessageType.RESPONSE,
            content=content,
            data=data,
            correlation_id=original.id
        )

class ScratchpadManager:
    """Manages shared scratchpad for agent collaboration"""
    
    @staticmethod
    def set_data(state: EnhancedAssistantState, key: str, value: Any, agent_name: str, 
                 tags: Optional[List[str]] = None, expires_in: Optional[float] = None) -> EnhancedAssistantState:
        """Set data in scratchpad"""
        new_state = dict(state)
        
        # Initialize scratchpad if not present
        if 'scratchpad' not in new_state:
            new_state['scratchpad'] = {}
        if 'scratchpad_history' not in new_state:
            new_state['scratchpad_history'] = []
        
        # Create entry
        entry = ScratchpadEntry(
            key=key,
            value=value,
            created_by=agent_name,
            tags=tags or [],
            expires_at=time.time() + expires_in if expires_in else None
        )
        
        # Store in scratchpad
        new_state['scratchpad'][key] = entry
        
        # Add to history
        new_state['scratchpad_history'].append(entry)
        
        logger.info(f"Scratchpad updated by {agent_name}: {key}")
        
        return new_state
    
    @staticmethod
    def get_data(state: EnhancedAssistantState, key: str) -> Optional[Any]:
        """Get data from scratchpad"""
        scratchpad = state.get('scratchpad', {})
        entry = scratchpad.get(key)
        
        if not entry:
            return None
        
        # Check if expired
        if entry.expires_at and time.time() > entry.expires_at:
            return None
        
        # Increment access count
        entry.access_count += 1
        
        return entry.value
    
    @staticmethod
    def search_by_tags(state: EnhancedAssistantState, tags: List[str]) -> Dict[str, Any]:
        """Search scratchpad entries by tags"""
        scratchpad = state.get('scratchpad', {})
        results = {}
        
        for key, entry in scratchpad.items():
            if any(tag in entry.tags for tag in tags):
                # Check if not expired
                if not entry.expires_at or time.time() <= entry.expires_at:
                    results[key] = entry.value
        
        return results
    
    @staticmethod
    def clean_expired(state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Remove expired entries"""
        new_state = dict(state)
        scratchpad = new_state.get('scratchpad', {})
        
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in scratchpad.items():
            if entry.expires_at and current_time > entry.expires_at:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del scratchpad[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned {len(keys_to_remove)} expired scratchpad entries")
        
        return new_state

class FileProcessingManager:
    """Manages file processing pipeline status"""
    
    @staticmethod
    def add_file_to_queue(state: EnhancedAssistantState, filename: str, total_steps: int = 5) -> tuple[EnhancedAssistantState, str]:
        """Add file to processing queue"""
        new_state = dict(state)
        
        # Initialize queue if not present
        if 'file_processing_queue' not in new_state:
            new_state['file_processing_queue'] = []
        if 'file_processing_history' not in new_state:
            new_state['file_processing_history'] = []
        
        # Create file processing status
        file_status = FileProcessingStatus(
            file_id=str(uuid.uuid4()),
            filename=filename,
            total_steps=total_steps
        )
        
        # Add to queue
        new_state['file_processing_queue'].append(file_status)
        
        logger.info(f"File added to processing queue: {filename}")
        
        return new_state, file_status.file_id
    
    @staticmethod
    def update_file_progress(state: EnhancedAssistantState, file_id: str, 
                           progress: float, current_step: str, results: Optional[Dict] = None) -> EnhancedAssistantState:
        """Update file processing progress"""
        new_state = dict(state)
        queue = new_state.get('file_processing_queue', [])
        
        for file_status in queue:
            if file_status.file_id == file_id:
                file_status.progress = progress
                file_status.current_step = current_step
                file_status.status = "processing"
                
                if results:
                    file_status.results.update(results)
                
                if progress >= 100.0:
                    file_status.status = "completed"
                    file_status.completed_at = time.time()
                
                break
        
        return new_state
    
    @staticmethod
    def get_file_status(state: EnhancedAssistantState, file_id: str) -> Optional[FileProcessingStatus]:
        """Get file processing status"""
        queue = state.get('file_processing_queue', [])
        for file_status in queue:
            if file_status.file_id == file_id:
                return file_status
        return None

class StreamingManager:
    """Manages real-time streaming updates"""
    
    @staticmethod
    def start_stream(state: EnhancedAssistantState, stream_id: str) -> EnhancedAssistantState:
        """Start streaming session"""
        new_state = dict(state)
        new_state['streaming_enabled'] = True
        new_state['current_stream_id'] = stream_id
        
        if 'stream_events' not in new_state:
            new_state['stream_events'] = []
        
        # Add start event
        new_state['stream_events'].append({
            'type': 'stream_start',
            'stream_id': stream_id,
            'timestamp': time.time()
        })
        
        return new_state
    
    @staticmethod
    def add_stream_event(state: EnhancedAssistantState, event_type: str, data: Dict[str, Any]) -> EnhancedAssistantState:
        """Add streaming event"""
        new_state = dict(state)
        
        if not new_state.get('streaming_enabled', False):
            return new_state
        
        if 'stream_events' not in new_state:
            new_state['stream_events'] = []
        
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time(),
            'stream_id': new_state.get('current_stream_id')
        }
        
        new_state['stream_events'].append(event)
        
        return new_state

# Enhanced state creation function
def create_enhanced_state(
    session_id: str = "",
    user_id: str = "default",
    initial_context: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> EnhancedAssistantState:
    """Create enhanced state with new capabilities"""
    
    state_data = {
        # Original fields
        "session_id": session_id or f"session_{int(time.time())}",
        "user_id": user_id,
        "messages": [],
        "current_agent": "",
        
        # P6 fields
        "p6_session_active": False,
        "current_project_id": None,
        "current_project_name": None,
        "p6_user_preferences": {},
        
        # NEW: Communication fields
        "agent_messages": [],
        "agent_communication_history": [],
        
        # NEW: Scratchpad fields
        "scratchpad": {},
        "scratchpad_history": [],
        
        # NEW: File processing fields
        "file_processing_queue": [],
        "file_processing_history": [],
        
        # NEW: Verification fields
        "pending_verification": None,
        "verification_history": [],
        
        # NEW: Streaming fields
        "streaming_enabled": False,
        "current_stream_id": None,
        "stream_events": []
    }
    
    # Apply initial context
    if initial_context:
        state_data.update(initial_context)
    
    return state_data

# Convenience functions for backward compatibility
def migrate_to_enhanced_state(old_state: Dict[str, Any]) -> EnhancedAssistantState:
    """Migrate old state to enhanced state"""
    enhanced = create_enhanced_state(
        session_id=old_state.get("session_id", ""),
        user_id=old_state.get("user_id", "default")
    )
    
    # Copy over existing fields
    for key in ["messages", "current_agent", "p6_session_active", 
                "current_project_id", "current_project_name", "p6_user_preferences"]:
        if key in old_state:
            enhanced[key] = old_state[key]
    
    return enhanced