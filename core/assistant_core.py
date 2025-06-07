# core/assistant_core.py - ✅ ENHANCED WITH DATABASE PERSISTENCE
import asyncio
import os
import time
import uuid
import json
import pickle
from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import logging

# ✅ UPDATED: Modern LangGraph imports for 0.4.8
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable

# ✅ UPDATED: Import the simplified supervisor class
from core.simplified_supervisor import SimplifiedSupervisor

# Core components with corrected imports
from core.state import (
    AssistantState,
    create_optimized_state,
    StateValidator,
    safe_state_access,
    optimize_state_for_processing
)
from agents.agents import AgentFactory
from core.checkpointer import create_checkpointer
from core.error_handling import ErrorHandler
from config.settings import config
from config.llm_manager import llm_manager

logger = logging.getLogger("assistant")

@dataclass
class AssistantSession:
    """Enhanced session management for assistant conversations with persistence"""
    session_id: str
    user_id: str
    start_time: float
    message_count: int = 0
    last_interaction: float = None
    total_tokens_used: int = 0
    agent_usage: Dict[str, int] = None
    conversation_topics: List[str] = None
    session_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize additional fields"""
        if self.agent_usage is None:
            self.agent_usage = {"chat": 0, "coder": 0, "web": 0}
        if self.conversation_topics is None:
            self.conversation_topics = []
        if self.session_metadata is None:
            self.session_metadata = {}
        if self.last_interaction is None:
            self.last_interaction = self.start_time
    
    def update_metrics(self, token_count: int = 0, agent_used: str = "", topic: str = ""):
        """Update session metrics with enhanced tracking"""
        self.message_count += 1
        self.last_interaction = time.time()
        self.total_tokens_used += token_count
        
        # Track agent usage
        if agent_used and agent_used in self.agent_usage:
            self.agent_usage[agent_used] += 1
        
        # Track conversation topics
        if topic and topic not in self.conversation_topics:
            self.conversation_topics.append(topic)
            # Keep only recent topics (last 10)
            self.conversation_topics = self.conversation_topics[-10:]
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """Check if session has expired"""
        if not self.last_interaction:
            return False
        return (time.time() - self.last_interaction) > (timeout_minutes * 60)
    
    def get_duration_minutes(self) -> float:
        """Get session duration in minutes"""
        end_time = self.last_interaction or time.time()
        return (end_time - self.start_time) / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssistantSession':
        """Create session from dictionary"""
        return cls(**data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary for monitoring"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "duration_minutes": self.get_duration_minutes(),
            "message_count": self.message_count,
            "total_tokens": self.total_tokens_used,
            "primary_agent": max(self.agent_usage.items(), key=lambda x: x[1])[0] if self.agent_usage else "none",
            "topics_discussed": len(self.conversation_topics),
            "is_expired": self.is_expired(),
            "last_seen": datetime.fromtimestamp(self.last_interaction).isoformat()
        }

class SessionPersistenceManager:
    """✅ NEW: Manages session persistence using database storage"""
    
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.session_table = "assistant_sessions"
        
    async def save_session(self, session: AssistantSession) -> bool:
        """Save session to persistent storage"""
        try:
            session_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "session_json": json.dumps(session.to_dict()),
                "last_interaction": session.last_interaction,
                "created_at": session.start_time,
                "message_count": session.message_count,
                "total_tokens": session.total_tokens_used
            }
            
            # Use checkpointer's connection if available
            if hasattr(self.checkpointer, 'conn'):
                await self._save_to_database(session_data)
            else:
                # Fallback to file-based storage
                await self._save_to_file(session_data)
            
            logger.debug(f"✅ Session {session.session_id} saved to persistent storage")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save session {session.session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[AssistantSession]:
        """Load session from persistent storage"""
        try:
            if hasattr(self.checkpointer, 'conn'):
                session_data = await self._load_from_database(session_id)
            else:
                session_data = await self._load_from_file(session_id)
            
            if session_data:
                session_dict = json.loads(session_data["session_json"])
                session = AssistantSession.from_dict(session_dict)
                logger.debug(f"✅ Session {session_id} loaded from persistent storage")
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from persistent storage"""
        try:
            if hasattr(self.checkpointer, 'conn'):
                await self._delete_from_database(session_id)
            else:
                await self._delete_from_file(session_id)
            
            logger.debug(f"✅ Session {session_id} deleted from persistent storage")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete session {session_id}: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions for a user"""
        try:
            if hasattr(self.checkpointer, 'conn'):
                return await self._get_user_sessions_from_database(user_id, limit)
            else:
                return await self._get_user_sessions_from_files(user_id, limit)
        except Exception as e:
            logger.error(f"❌ Failed to get user sessions for {user_id}: {e}")
            return []
    
    async def cleanup_expired_sessions(self, expiry_hours: int = 24) -> int:
        """Clean up expired sessions from storage"""
        try:
            cutoff_time = time.time() - (expiry_hours * 3600)
            
            if hasattr(self.checkpointer, 'conn'):
                count = await self._cleanup_database_sessions(cutoff_time)
            else:
                count = await self._cleanup_file_sessions(cutoff_time)
            
            if count > 0:
                logger.info(f"✅ Cleaned up {count} expired sessions")
            
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired sessions: {e}")
            return 0
    
    async def _save_to_database(self, session_data: Dict[str, Any]):
        """Save session using database connection"""
        # This would be implemented based on the specific database type
        # For now, we'll use the checkpointer's method if available
        if hasattr(self.checkpointer, '_execute_query'):
            query = """
                INSERT OR REPLACE INTO assistant_sessions 
                (session_id, user_id, session_json, last_interaction, created_at, message_count, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                session_data["session_id"],
                session_data["user_id"], 
                session_data["session_json"],
                session_data["last_interaction"],
                session_data["created_at"],
                session_data["message_count"],
                session_data["total_tokens"]
            )
            await self.checkpointer._execute_query(query, params)
    
    async def _save_to_file(self, session_data: Dict[str, Any]):
        """Fallback file-based session storage"""
        sessions_dir = config.workspace_dir / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        
        session_file = sessions_dir / f"{session_data['session_id']}.json"
        
        # Use asyncio to write file
        import aiofiles
        try:
            async with aiofiles.open(session_file, 'w') as f:
                await f.write(json.dumps(session_data, indent=2))
        except ImportError:
            # Fallback to sync file writing
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
    
    async def _load_from_database(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from database"""
        if hasattr(self.checkpointer, '_fetch_one'):
            query = "SELECT * FROM assistant_sessions WHERE session_id = ?"
            return await self.checkpointer._fetch_one(query, (session_id,))
        return None
    
    async def _load_from_file(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from file"""
        sessions_dir = config.workspace_dir / "sessions"
        session_file = sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            import aiofiles
            async with aiofiles.open(session_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except ImportError:
            with open(session_file, 'r') as f:
                return json.load(f)
    
    async def _delete_from_database(self, session_id: str):
        """Delete session from database"""
        if hasattr(self.checkpointer, '_execute_query'):
            query = "DELETE FROM assistant_sessions WHERE session_id = ?"
            await self.checkpointer._execute_query(query, (session_id,))
    
    async def _delete_from_file(self, session_id: str):
        """Delete session file"""
        sessions_dir = config.workspace_dir / "sessions"
        session_file = sessions_dir / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
    
    async def _get_user_sessions_from_database(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get user sessions from database"""
        if hasattr(self.checkpointer, '_fetch_all'):
            query = """
                SELECT session_id, last_interaction, created_at, message_count, total_tokens
                FROM assistant_sessions 
                WHERE user_id = ? 
                ORDER BY last_interaction DESC 
                LIMIT ?
            """
            return await self.checkpointer._fetch_all(query, (user_id, limit))
        return []
    
    async def _get_user_sessions_from_files(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get user sessions from files"""
        sessions_dir = config.workspace_dir / "sessions"
        if not sessions_dir.exists():
            return []
        
        user_sessions = []
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    if session_data.get("user_id") == user_id:
                        user_sessions.append({
                            "session_id": session_data["session_id"],
                            "last_interaction": session_data["last_interaction"],
                            "created_at": session_data["created_at"],
                            "message_count": session_data["message_count"],
                            "total_tokens": session_data["total_tokens"]
                        })
            except Exception:
                continue
        
        # Sort by last interaction and limit
        user_sessions.sort(key=lambda x: x["last_interaction"], reverse=True)
        return user_sessions[:limit]
    
    async def _cleanup_database_sessions(self, cutoff_time: float) -> int:
        """Cleanup expired sessions from database"""
        if hasattr(self.checkpointer, '_execute_query'):
            # First count the sessions to be deleted
            count_query = "SELECT COUNT(*) FROM assistant_sessions WHERE last_interaction < ?"
            count_result = await self.checkpointer._fetch_one(count_query, (cutoff_time,))
            count = count_result[0] if count_result else 0
            
            # Delete expired sessions
            delete_query = "DELETE FROM assistant_sessions WHERE last_interaction < ?"
            await self.checkpointer._execute_query(delete_query, (cutoff_time,))
            
            return count
        return 0
    
    async def _cleanup_file_sessions(self, cutoff_time: float) -> int:
        """Cleanup expired session files"""
        sessions_dir = config.workspace_dir / "sessions"
        if not sessions_dir.exists():
            return 0
        
        count = 0
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    if session_data.get("last_interaction", 0) < cutoff_time:
                        session_file.unlink()
                        count += 1
            except Exception:
                continue
        
        return count

class AssistantCore:
    """
    ✅ ENHANCED: Complete assistant core implementation with persistent session management
    """
    
    def __init__(self):
        self.supervisor = None
        self.checkpointer = None
        self.current_session: Optional[AssistantSession] = None
        self._setup_complete = False
        
        # Agent instances
        self.chat_agent = None
        self.coder_agent = None
        self.web_agent = None
        
        # Concurrency control
        self._session_semaphore = asyncio.Semaphore(3)
        
        # ✅ ENHANCED: Session management with persistence
        self._sessions: Dict[str, AssistantSession] = {}
        self._session_persistence: Optional[SessionPersistenceManager] = None
        self._session_cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()
        self._session_expiry_hours = 24  # Sessions expire after 24 hours
        
    async def initialize(self):
        """Initialize assistant core components with enhanced session management"""
        try:
            logger.info("🚀 Initializing Assistant Core...")
            
            # Initialize checkpointer
            self.checkpointer = await create_checkpointer(use_async=True)
            logger.info("✅ Checkpointer initialized")
            
            # ✅ NEW: Initialize session persistence
            self._session_persistence = SessionPersistenceManager(self.checkpointer)
            await self._initialize_session_storage()
            
            # Initialize agents
            self.agent_factory = AgentFactory()
            self.chat_agent = self.agent_factory.create_chat_agent()
            self.coder_agent = self.agent_factory.create_coder_agent()
            self.web_agent = self.agent_factory.create_web_agent()
            logger.info("✅ Agents initialized")
            
            # Initialize supervisor
            await self._initialize_supervisor()
            
            self._setup_complete = True
            logger.info("✅ Assistant Core initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Assistant initialization failed: {e}")
            # Attempt graceful degradation
            await self._attempt_fallback_initialization()
            raise
    
    async def _initialize_session_storage(self):
        """✅ NEW: Initialize session storage table if using database"""
        try:
            if hasattr(self.checkpointer, 'conn') and hasattr(self.checkpointer, '_execute_query'):
                # Create sessions table if it doesn't exist
                create_table_query = """
                    CREATE TABLE IF NOT EXISTS assistant_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_json TEXT NOT NULL,
                        last_interaction REAL NOT NULL,
                        created_at REAL NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        INDEX idx_user_id (user_id),
                        INDEX idx_last_interaction (last_interaction)
                    )
                """
                await self.checkpointer._execute_query(create_table_query)
                logger.info("✅ Session storage table initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize session storage table: {e}")
    
    async def _get_or_create_session(self, thread_id: str, user_id: str) -> AssistantSession:
        """✅ ENHANCED: Session management with database persistence and recovery"""
        session_id = thread_id or str(uuid.uuid4())
        
        # Check if session exists in memory
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if not session.is_expired():
                self.current_session = session
                return session
            else:
                # Remove expired session from memory
                del self._sessions[session_id]
                logger.info(f"Removed expired session from memory: {session_id}")
        
        # Try to load session from persistent storage
        if self._session_persistence:
            persisted_session = await self._session_persistence.load_session(session_id)
            if persisted_session and not persisted_session.is_expired():
                # Restore session to memory
                self._sessions[session_id] = persisted_session
                self.current_session = persisted_session
                logger.info(f"Restored session from storage: {session_id}")
                return persisted_session
            elif persisted_session and persisted_session.is_expired():
                # Clean up expired session from storage
                await self._session_persistence.delete_session(session_id)
                logger.info(f"Cleaned up expired session from storage: {session_id}")
        
        # Create new session
        session = AssistantSession(
            session_id=session_id,
            user_id=user_id,
            start_time=time.time()
        )
        
        # Add to memory and save to persistent storage
        self._sessions[session_id] = session
        self.current_session = session
        
        if self._session_persistence:
            await self._session_persistence.save_session(session)
        
        logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session
    
    async def _cleanup_expired_sessions(self):
        """✅ ENHANCED: Clean up expired sessions from both memory and storage"""
        try:
            current_time = time.time()
            
            # Only run cleanup periodically
            if current_time - self._last_cleanup < self._session_cleanup_interval:
                return
            
            # Clean up memory sessions
            expired_sessions = [
                session_id for session_id, session in self._sessions.items()
                if session.is_expired()
            ]
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions from memory")
            
            # Clean up persistent storage
            if self._session_persistence:
                storage_cleanup_count = await self._session_persistence.cleanup_expired_sessions(
                    self._session_expiry_hours
                )
                if storage_cleanup_count > 0:
                    logger.info(f"Cleaned up {storage_cleanup_count} expired sessions from storage")
            
            self._last_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    def _extract_response_comprehensively(self, result: Dict[str, Any], session: AssistantSession) -> Dict[str, Any]:
        """✅ ENHANCED: Comprehensive response extraction with session persistence"""
        try:
            response_content = ""
            agent_used = safe_state_access(result, 'current_agent', 'unknown')
            tokens_used = 0
            
            # Strategy 1: Extract from messages in result
            messages = safe_state_access(result, "messages", [])
            
            if messages:
                # Look for the last AI message with content
                for message in reversed(messages):
                    if isinstance(message, AIMessage):
                        content = getattr(message, 'content', '').strip()
                        if content:
                            response_content = content
                            break
                        
                        # Check for tool calls if no content
                        tool_calls = getattr(message, 'tool_calls', [])
                        if tool_calls and not response_content:
                            response_content = f"I'm processing your request using {len(tool_calls)} tool(s)..."
                            break
            
            # Strategy 2: Fallback to any string content in result
            if not response_content:
                for key in ["response", "content", "output", "result"]:
                    if key in result and isinstance(result[key], str):
                        potential_content = result[key].strip()
                        if potential_content:
                            response_content = potential_content
                            break
            
            # Strategy 3: Generate contextual fallback response
            if not response_content:
                response_content = self._generate_contextual_fallback(agent_used, session)
            
            # Strategy 4: Absolute fallback
            if not response_content:
                response_content = "I'm ready to help you with your next request."
            
            # Extract token usage if available
            try:
                usage_info = result.get("usage", {})
                tokens_used = usage_info.get("total_tokens", 0)
            except:
                tokens_used = 0
            
            # ✅ ENHANCED: Update session with agent usage and save to storage
            if session:
                # Determine topic from response content (simple keyword extraction) 
                topic = self._extract_topic_from_response(response_content)
                session.update_metrics(tokens_used, agent_used, topic)
                
                # Save updated session to persistent storage
                if self._session_persistence:
                    asyncio.create_task(self._session_persistence.save_session(session))
            
            return {
                "response": response_content,
                "session_id": session.session_id if session else "unknown",
                "message_count": session.message_count if session else 0,
                "agent_used": agent_used,
                "tokens_used": tokens_used,
                "processing_time": time.time() - session.last_interaction if session and session.last_interaction else 0,
                "success": True,
                "session_summary": session.get_summary() if session else {}
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive response extraction: {e}")
            return {
                "response": "I encountered an issue processing your request, but I'm ready to help with your next question.",
                "session_id": session.session_id if session else "unknown",
                "message_count": session.message_count if session else 0,
                "agent_used": "error_handler",
                "tokens_used": 0,
                "success": False,
                "error": str(e)
            }
    
    def _extract_topic_from_response(self, response: str) -> str:
        """✅ NEW: Simple topic extraction from response content"""
        response_lower = response.lower()
        
        # Simple keyword-based topic detection
        topics = {
            "coding": ["code", "function", "programming", "python", "javascript", "html", "css"],
            "files": ["file", "directory", "folder", "document", "create", "write", "read"],
            "search": ["search", "find", "look", "google", "web", "information"],
            "help": ["help", "assist", "guide", "how", "what", "explain"],
            "conversation": ["hello", "hi", "thanks", "thank you", "goodbye", "bye"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in response_lower for keyword in keywords):
                return topic
        
        return "general"
    
    async def get_user_session_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """✅ NEW: Get session history for a user"""
        if not self._session_persistence:
            return []
        
        try:
            return await self._session_persistence.get_user_sessions(user_id, limit)
        except Exception as e:
            logger.error(f"Error getting user session history: {e}")
            return []
    
    async def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """✅ NEW: Restore a previous session"""
        if not self._session_persistence:
            return None
        
        try:
            session = await self._session_persistence.load_session(session_id)
            if session and not session.is_expired():
                self._sessions[session_id] = session
                self.current_session = session
                logger.info(f"Session restored: {session_id}")
                return session.get_summary()
            return None
        except Exception as e:
            logger.error(f"Error restoring session {session_id}: {e}")
            return None
    
    def get_session_info(self) -> Dict[str, Any]:
        """✅ ENHANCED: Get comprehensive session information with persistence data"""
        base_info = {
            "status": "no_active_session",
            "total_active_sessions": len(self._sessions),
            "session_persistence_enabled": self._session_persistence is not None
        }
        
        if not self.current_session:
            return base_info
            
        session_info = self.current_session.get_summary()
        session_info.update({
            "total_active_sessions": len(self._sessions),
            "session_persistence_enabled": self._session_persistence is not None,
            "agent_usage_breakdown": self.current_session.agent_usage,
            "conversation_topics": self.current_session.conversation_topics,
            "session_metadata": self.current_session.session_metadata
        })
        
        return session_info
    
    async def graceful_shutdown(self):
        """✅ ENHANCED: Graceful shutdown with session persistence"""
        try:
            logger.info("🔄 Initiating graceful shutdown...")
            
            # Save all active sessions to persistent storage
            if self._session_persistence:
                save_tasks = []
                for session in self._sessions.values():
                    save_tasks.append(self._session_persistence.save_session(session))
                
                if save_tasks:
                    await asyncio.gather(*save_tasks, return_exceptions=True)
                    logger.info(f"✅ Saved {len(save_tasks)} active sessions to storage")
            
            # Save active sessions if possible
            if self.checkpointer and hasattr(self.checkpointer, 'close'):
                await self.checkpointer.close()
            
            # Clear sessions
            self._sessions.clear()
            
            # Clear models cache
            if hasattr(llm_manager, 'clear_cache'):
                llm_manager.clear_cache()
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")

    # [Previous methods like _attempt_fallback_initialization, _initialize_supervisor, 
    #  process_message, _sanitize_user_input, etc. remain the same as in your uploaded file]
