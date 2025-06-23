# core/enhanced_assistant_core.py - Enhanced Assistant Core with All Capabilities

import asyncio
import logging
import time
import uuid
import traceback
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from core.enhanced_state import (
    EnhancedAssistantState, create_enhanced_state, migrate_to_enhanced_state,
    AgentCommunicationManager, ScratchpadManager, FileProcessingManager,
    StreamingManager
)
from core.enhanced_supervisor import EnhancedSupervisor, VerificationConfig, StreamingConfig
from core.p6_integration_manager import P6IntegrationManager, P6QueryRequest, QueryType
from core.file_processing_pipeline import FileProcessingPipeline
from tools.agent_communication_tools import agent_communication_tools
from agents.agents import agent_factory, get_agent
from core.checkpointer import CheckpointerFactory, CheckpointerConfig, Checkpointer
from core.error_handling import ErrorHandler, handle_error
from core.circuit_breaker import global_circuit_breaker
from config.llm_manager import llm_manager
from config.settings import config

logger = logging.getLogger("enhanced_assistant_core")

class EnhancedAssistantCore:
    """Enhanced assistant core with all new capabilities"""
    
    def __init__(self):
        logger.info("🚀 Initializing Enhanced AssistantCore...")
        self.app = FastAPI(lifespan=self.lifespan)
        
        # Enhanced components
        self.supervisor = EnhancedSupervisor(
            verification_config=VerificationConfig(enabled=True),
            streaming_config=StreamingConfig(enabled=True)
        )
        
        # Integration managers
        self.p6_manager = P6IntegrationManager()
        self.file_pipeline = FileProcessingPipeline()
        
        # Legacy components
        self.checkpointer_factory = CheckpointerFactory()
        self.checkpointer: Optional[Checkpointer] = None
        self.active_sessions: Dict[str, EnhancedAssistantState] = {}
        self._initialized: bool = False
        
        # Streaming connections
        self.streaming_connections: Dict[str, WebSocket] = {}
        
        self._setup_enhanced_routes()
        logger.debug("✅ Enhanced AssistantCore instance created")
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Enhanced async context manager for component lifecycle"""
        logger.info("🔄 Starting enhanced application lifespan...")
        await self.initialize()
        yield
        logger.info("🔄 Shutting down enhanced application lifespan...")
        await self.graceful_shutdown()
    
    async def initialize(self):
        """Enhanced initialization with all new capabilities"""
        if self._initialized:
            logger.debug("🔄 Enhanced AssistantCore already initialised – skipping")
            return
        
        try:
            logger.info("🚀 Starting Enhanced Assistant Core Initialization")
            
            # Initialize checkpointer
            logger.info("📂 Initializing checkpointer...")
            self.checkpointer = await self.checkpointer_factory.create_optimal_checkpointer()
            logger.info(f"✅ Checkpointer initialized: {type(self.checkpointer).__name__}")
            
            # Pre-initialize LLM models
            logger.info("🔥 Pre-initializing LLM models...")
            await llm_manager.initialize_models()
            
            # Initialize agents with communication tools
            logger.info("🤖 Initializing enhanced agents...")
            self.agents = await agent_factory.initialize_agents()
            
            # Add communication tools to all agents
            communication_tools = agent_communication_tools.get_tools()
            agent_factory.tools["agent_communication"] = communication_tools
            
            logger.info(f"✅ Enhanced agents initialized: {len(self.agents)} agents")
            
            # Get all tools including communication tools
            all_tools = agent_factory.get_all_tools()
            logger.info(f"✅ Tools gathered: {len(all_tools)} tools")
            
            # Initialize enhanced supervisor
            logger.info("🎯 Initializing enhanced supervisor...")
            await self.supervisor.initialize(self.agents, all_tools, self.checkpointer)
            logger.info("✅ Enhanced supervisor initialized successfully")
            
            # Initialize P6 integration if credentials available
            logger.info("🔌 Initializing P6 integration...")
            p6_username = config.get_env_var("P6_USERNAME")
            p6_password = config.get_env_var("P6_PASSWORD")
            p6_database = config.get_env_var("P6_DATABASE", "PMDB")
            
            if p6_username and p6_password:
                p6_success = await self.p6_manager.initialize_connection(
                    p6_username, p6_password, p6_database
                )
                if p6_success:
                    logger.info("✅ P6 integration initialized")
                else:
                    logger.warning("⚠️ P6 integration failed - continuing without P6")
            else:
                logger.info("ℹ️ P6 credentials not found - P6 integration disabled")
            
            # Start background tasks
            logger.info("⚙️ Starting enhanced background tasks...")
            asyncio.create_task(self._session_cleanup_task())
            asyncio.create_task(self._health_monitoring_task())
            asyncio.create_task(self._scratchpad_cleanup_task())
            logger.debug("✅ Enhanced background tasks started")
            
            logger.info("✅ Enhanced Assistant Core Initialization Complete")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Enhanced initialization failed")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error message: {str(e)}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            await self.graceful_shutdown()
            raise
    
    def _setup_enhanced_routes(self):
        """Configure enhanced FastAPI routes"""
        logger.debug("🛣️ Setting up enhanced FastAPI routes...")
        
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @self.app.get("/", response_class=HTMLResponse)
        async def web_interface():
            return self._get_web_interface()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_enhanced_websocket(websocket)
        
        @self.app.websocket("/stream/{session_id}")
        async def streaming_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_streaming_websocket(websocket, session_id)
        
        @self.app.post("/api/message")
        async def handle_message(message: dict):
            return await self._handle_enhanced_api_message(message)
        
        @self.app.post("/api/upload")
        async def upload_file(file: UploadFile = File(...), session_id: str = "default"):
            return await self._handle_file_upload(file, session_id)
        
        @self.app.post("/api/p6/query")
        async def p6_query(query_request: dict):
            return await self._handle_p6_query(query_request)
        
        @self.app.get("/api/status/enhanced")
        async def get_enhanced_system_status():
            return await self._get_enhanced_system_status()
        
        @self.app.get("/api/scratchpad/{session_id}")
        async def get_scratchpad_data(session_id: str):
            return await self._get_scratchpad_data(session_id)
        
        @self.app.post("/api/agent/communicate")
        async def agent_communicate(communication_request: dict):
            return await self._handle_agent_communication(communication_request)
        
        logger.debug("✅ Enhanced FastAPI routes configured")
    
    async def _handle_enhanced_websocket(self, websocket: WebSocket):
        """Handle enhanced WebSocket connections with streaming"""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        logger.info(f"🔌 Enhanced WebSocket connected: {session_id}")
        
        try:
            while True:
                message = await websocket.receive_json()
                logger.debug(f"📨 Enhanced WebSocket message received: {session_id}")
                
                # Enable streaming for this session
                message["enable_streaming"] = True
                message["stream_session_id"] = session_id
                
                response = await self.process_enhanced_message(
                    message.get("content", ""),
                    session_id=session_id,
                    user_id=message.get("user_id", "anonymous"),
                    enable_streaming=True
                )
                
                await websocket.send_json(response)
                logger.debug(f"📤 Enhanced WebSocket response sent: {session_id}")
                
        except WebSocketDisconnect:
            logger.info(f"🔌 Enhanced WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Enhanced WebSocket error for {session_id}: {e}")
            error_response = await handle_error(e, "enhanced_websocket_handler")
            await websocket.send_json(error_response)
    
    async def _handle_streaming_websocket(self, websocket: WebSocket, session_id: str):
        """Handle dedicated streaming WebSocket"""
        await websocket.accept()
        self.streaming_connections[session_id] = websocket
        logger.info(f"📡 Streaming WebSocket connected: {session_id}")
        
        try:
            # Keep connection alive and send ping/pong
            while True:
                await websocket.receive_text()  # Wait for ping
                await websocket.send_text("pong")
                
        except WebSocketDisconnect:
            if session_id in self.streaming_connections:
                del self.streaming_connections[session_id]
            logger.info(f"📡 Streaming WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ Streaming WebSocket error: {e}")
    
    async def _handle_enhanced_api_message(self, message: dict):
        """Handle enhanced API message requests"""
        logger.debug(f"📨 Enhanced API message received")
        
        try:
            response = await self.process_enhanced_message(
                message.get("content", ""),
                session_id=message.get("session_id"),
                user_id=message.get("user_id", "anonymous"),
                enable_streaming=message.get("enable_streaming", False)
            )
            
            logger.debug("📤 Enhanced API response prepared")
            return response
            
        except Exception as e:
            logger.error(f"❌ Enhanced API message handling error: {e}")
            return await handle_error(e, "enhanced_api_message_handler")
    
    async def _handle_file_upload(self, file: UploadFile, session_id: str):
        """Handle file upload with processing pipeline"""
        try:
            logger.info(f"📁 File upload received: {file.filename}")
            
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Get session state
            state = await self._get_enhanced_session_state(session_id, "file_upload_user")
            
            # Process through file pipeline
            state, file_id, results = await self.file_pipeline.process_file_upload(
                state, tmp_file_path, "file_manager", ["chat", "coder"]
            )
            
            # Update session
            self.active_sessions[session_id] = state
            
            # Clean up temp file
            import os
            os.unlink(tmp_file_path)
            
            return {
                "success": True,
                "file_id": file_id,
                "filename": file.filename,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_p6_query(self, query_request: dict):
        """Handle P6 database query"""
        try:
            # Create P6 query request
            request = P6QueryRequest(
                query_type=QueryType(query_request.get("query_type", "project_search")),
                natural_language_query=query_request.get("query", ""),
                structured_filters=query_request.get("filters"),
                requested_fields=query_request.get("fields"),
                order_by=query_request.get("order_by"),
                limit=query_request.get("limit", 100),
                requesting_agent=query_request.get("agent", "api")
            )
            
            # Get session state
            session_id = query_request.get("session_id", "p6_query")
            state = await self._get_enhanced_session_state(session_id, "p6_user")
            
            # Execute query
            state, response = await self.p6_manager.execute_structured_query(state, request)
            
            # Update session
            self.active_sessions[session_id] = state
            
            return {
                "success": response.success,
                "data": response.data,
                "total_count": response.total_count,
                "execution_time_ms": response.execution_time_ms,
                "filter_applied": response.filter_applied,
                "error": response.error_message
            }
            
        except Exception as e:
            logger.error(f"❌ P6 query failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_agent_communication(self, communication_request: dict):
        """Handle agent-to-agent communication"""
        try:
            session_id = communication_request.get("session_id", "communication")
            state = await self._get_enhanced_session_state(session_id, "system")
            
            # Send message between agents
            from core.enhanced_state import AgentMessage, MessageType
            
            message = AgentMessage(
                from_agent=communication_request.get("from_agent"),
                to_agent=communication_request.get("to_agent"),
                content=communication_request.get("content"),
                message_type=MessageType(communication_request.get("type", "notification"))
            )
            
            state = AgentCommunicationManager.send_message(state, message)
            self.active_sessions[session_id] = state
            
            return {"success": True, "message_id": message.id}
            
        except Exception as e:
            logger.error(f"❌ Agent communication failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_enhanced_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: str = "anonymous",
        enable_streaming: bool = False
    ) -> Dict[str, Any]:
        """Enhanced message processing with all new capabilities"""
        logger.info(f"🔄 Starting enhanced message processing")
        
        try:
            if not self._initialized:
                logger.warning("Enhanced AssistantCore not initialised – performing lazy start-up")
                await self.initialize()
            
            # Get or create enhanced session state
            state = await self._get_enhanced_session_state(session_id, user_id)
            
            # Create new conversation detection
            if self._is_new_conversation(message):
                logger.info(f"🔄 Creating new enhanced session for conversation starter")
                session_id = str(uuid.uuid4())
                state = await self._get_enhanced_session_state(session_id, user_id)
            
            # Add message to state
            from langchain_core.messages import HumanMessage
            human_msg = HumanMessage(content=message)
            human_msg.additional_kwargs = {
                'session_id': state["session_id"],
                'timestamp': time.time()
            }
            
            new_messages = list(state.get("messages", [])) + [human_msg]
            state["messages"] = new_messages
            
            # Enable streaming if requested
            if enable_streaming:
                stream_id = f"stream_{session_id}_{int(time.time())}"
                state = StreamingManager.start_stream(state, stream_id)
            
            # Clean expired scratchpad entries
            state = ScratchpadManager.clean_expired(state)
            
            # Process through enhanced supervisor
            unique_thread_id = f"{state['user_id']}:{state['session_id']}"
            config_dict = {"configurable": {"thread_id": unique_thread_id}}
            
            if enable_streaming and session_id in self.streaming_connections:
                # Create streaming queue
                stream_queue = asyncio.Queue()
                
                # Start streaming task
                streaming_task = asyncio.create_task(
                    self._stream_to_websocket(stream_queue, session_id)
                )
                
                # Process with streaming
                result = await self.supervisor.process_with_streaming(
                    state, config_dict, stream_queue
                )
                
                # Clean up streaming
                streaming_task.cancel()
            else:
                # Process normally
                result = await self.supervisor.process_with_streaming(
                    state, config_dict
                )
            
            # Update session state
            self.active_sessions[state["session_id"]] = result
            
            # Extract response
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            else:
                response_content = str(result)
            
            # Clean up response formatting
            if response_content.startswith('content="') and response_content.endswith('"'):
                response_content = response_content[9:-1]
            elif response_content.startswith("content='") and response_content.endswith("'"):
                response_content = response_content[9:-1]
            
            logger.info(f"✅ Enhanced message processing completed")
            
            # Build enhanced response
            enhanced_response = {
                "response": response_content,
                "session_id": state["session_id"],
                "timestamp": time.time(),
                "verification_status": result.get("pending_verification"),
                "agent_messages": len(result.get("agent_messages", [])),
                "scratchpad_entries": len(result.get("scratchpad", {})),
                "file_processing_queue": len(result.get("file_processing_queue", [])),
                "streaming_enabled": result.get("streaming_enabled", False)
            }
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Enhanced message processing failed: {e}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            return await handle_error(e, "enhanced_message_processing")
    
    async def _stream_to_websocket(self, stream_queue: asyncio.Queue, session_id: str):
        """Stream events to WebSocket connection"""
        try:
            websocket = self.streaming_connections.get(session_id)
            if not websocket:
                return
            
            while True:
                try:
                    event = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                    await websocket.send_json(event)
                except asyncio.TimeoutError:
                    # Send keep-alive
                    await websocket.send_json({"type": "keep_alive", "timestamp": time.time()})
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Stream to WebSocket failed: {e}")
    
    async def _get_enhanced_session_state(
        self,
        session_id: Optional[str],
        user_id: str
    ) -> EnhancedAssistantState:
        """Get or create enhanced session state"""
        if not session_id or session_id not in self.active_sessions:
            new_session_id = session_id or str(uuid.uuid4())
            
            if self.checkpointer and session_id:
                try:
                    # Use namespaced thread ID for retrieval
                    namespaced_id = f"{user_id}:{session_id}"
                    config = {"configurable": {"thread_id": namespaced_id}}
                    logger.debug(f"🔍 Retrieving enhanced session: {namespaced_id}")
                    
                    raw_state = await self.checkpointer.aget(config)
                    
                    if raw_state and self._validate_session_state(raw_state, session_id, user_id):
                        # Migrate to enhanced state if needed
                        if "agent_messages" not in raw_state:
                            enhanced_state = migrate_to_enhanced_state(raw_state)
                        else:
                            enhanced_state = raw_state
                        
                        self.active_sessions[session_id] = enhanced_state
                        return enhanced_state
                        
                except Exception as e:
                    logger.debug(f"Error loading enhanced session: {e}")
            
            # Create new enhanced state
            new_state = create_enhanced_state(
                session_id=new_session_id,
                user_id=user_id
            )
            
            self.active_sessions[new_session_id] = new_state
            return new_state
        
        return self.active_sessions[session_id]
    
    async def _get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status"""
        try:
            base_status = await self._get_system_status()
            
            # Add enhanced status information
            enhanced_status = {
                **base_status,
                "enhanced_features": {
                    "supervisor_verification": self.supervisor.verification_config.enabled,
                    "streaming_enabled": self.supervisor.streaming_config.enabled,
                    "p6_integration": self.p6_manager.client is not None,
                    "file_processing": True,
                    "agent_communication": True,
                    "scratchpad_system": True
                },
                "active_streams": len(self.streaming_connections),
                "p6_stats": self.p6_manager.get_connection_stats(),
                "file_processing_stats": self.file_pipeline.get_processing_stats(),
                "supervisor_stats": self.supervisor.get_enhanced_statistics()
            }
            
            return enhanced_status
            
        except Exception as e:
            logger.error(f"❌ Enhanced system status failed: {e}")
            return {"error": str(e)}
    
    async def _get_scratchpad_data(self, session_id: str) -> Dict[str, Any]:
        """Get scratchpad data for session"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            state = self.active_sessions[session_id]
            scratchpad = state.get("scratchpad", {})
            
            # Convert to serializable format
            scratchpad_data = {}
            for key, entry in scratchpad.items():
                scratchpad_data[key] = {
                    "value": entry.value,
                    "created_by": entry.created_by,
                    "created_at": entry.created_at,
                    "tags": entry.tags,
                    "access_count": entry.access_count
                }
            
            return {
                "session_id": session_id,
                "scratchpad": scratchpad_data,
                "entry_count": len(scratchpad_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Get scratchpad data failed: {e}")
            return {"error": str(e)}
    
    async def _scratchpad_cleanup_task(self):
        """Background task to clean up expired scratchpad entries"""
        logger.debug("🧹 Starting scratchpad cleanup task...")
        
        while True:
            try:
                logger.debug("🔍 Checking for expired scratchpad entries...")
                
                for session_id, state in list(self.active_sessions.items()):
                    cleaned_state = ScratchpadManager.clean_expired(state)
                    if cleaned_state != state:
                        self.active_sessions[session_id] = cleaned_state
                
                await asyncio.sleep(600)  # Clean every 10 minutes
                
            except Exception as e:
                logger.error(f"❌ Scratchpad cleanup error: {e}")
                await asyncio.sleep(60)

# Global enhanced assistant instance
enhanced_assistant = EnhancedAssistantCore()
