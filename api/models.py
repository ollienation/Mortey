# api/models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., description="User message", min_length=1)
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    user_id: str = Field("default_user", description="User identifier")
    stream: bool = Field(False, description="Enable streaming response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session identifier")
    message_count: int = Field(..., description="Total messages in conversation")
    agent_used: Optional[str] = Field(None, description="Agent that handled the request")
    processing_time: float = Field(..., description="Processing time in seconds")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    user_id: str
    start_time: float
    message_count: int
    last_interaction: float
    duration_minutes: float

class SystemStatus(BaseModel):
    """System status model"""
    status: Literal["healthy", "degraded", "unhealthy"]
    supervisor_initialized: bool
    checkpointer_type: str
    setup_complete: bool
    agents_available: List[str]
    session_active: bool
    langsmith_enabled: bool
    modern_patterns: str
    timestamp: float

class ConversationHistory(BaseModel):
    """Conversation history model"""
    messages: List[Dict[str, Any]]
    total_messages: int
    session_id: str
    user_id: str
