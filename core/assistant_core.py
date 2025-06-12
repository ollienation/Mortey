# core/assistant_core.py - ✅ Enhanced with Comprehensive Logging
import asyncio
import signal
import uuid
import logging
import time
import traceback
from typing import Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from core.state import AssistantState
from core.supervisor import Supervisor, SupervisorConfig
from agents.agents import agent_factory, get_agent
from core.checkpointer import CheckpointerFactory, CheckpointerConfig, Checkpointer
from core.error_handling import ErrorHandler, handle_error
from core.circuit_breaker import global_circuit_breaker
from config.llm_manager import llm_manager
from config.settings import config

logger = logging.getLogger("assistant_core")

class AssistantCore:
    """Main assistant class integrating all components with enhanced logging"""
    
    def __init__(self):
        logger.info("🚀 Initializing AssistantCore...")
        self.app = FastAPI(lifespan=self.lifespan)
        self.supervisor = Supervisor()
        self.checkpointer_factory = CheckpointerFactory()
        self.checkpointer: Optional[Checkpointer] = None
        self.active_sessions: dict[str, AssistantState] = {}
        self._initialized: bool = False   
        self._setup_routes()
        logger.debug("✅ AssistantCore instance created")
        
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Async context manager for component lifecycle"""
        logger.info("🔄 Starting application lifespan...")
        await self.initialize()
        yield
        logger.info("🔄 Shutting down application lifespan...")
        await self.graceful_shutdown()
    
    async def initialize(self):
        # ------------------------------------------------------------------  
        # Guard: skip when another coroutine (e.g. lifespan) already ran init
        # ------------------------------------------------------------------
        if self._initialized:
            logger.debug("🔄 AssistantCore already initialised – skipping")
            return
        try:
            logger.info("🚀 Starting Assistant Core Initialization")
            
            # Log configuration state
            logger.debug("🔧 Configuration check:")
            logger.debug(f"  - Config object: {config is not None}")
            logger.debug(f"  - Workspace dir: {config.workspace_dir}")
            logger.debug(f"  - LLM config loaded: {hasattr(config, 'llm_config')}")
            
            # Initialize checkpointer with detailed logging
            logger.info("📂 Initializing checkpointer...")
            self.checkpointer = await self.checkpointer_factory.create_optimal_checkpointer()
            logger.info(f"✅ Checkpointer initialized: {type(self.checkpointer).__name__}")

            # Pre-initialize all models before starting
            logger.info("🔥 Pre-initializing LLM models...")
            await llm_manager.initialize_models()
            
            # Initialize agents first with detailed logging
            logger.info("🤖 Initializing agents...")
            self.agents = await agent_factory.initialize_agents()
            logger.info(f"✅ Agents initialized: {len(self.agents)} agents")
            logger.debug(f"📊 Available agents: {list(self.agents.keys())}")
            
            # Get all tools with logging
            logger.info("🔧 Gathering tools...")
            all_tools = agent_factory.get_all_tools()
            logger.info(f"✅ Tools gathered: {len(all_tools)} tools")
            
            # Initialize supervisor with required arguments and detailed logging
            logger.info("🎯 Initializing supervisor...")
            logger.debug(f"🔧 Supervisor initialization inputs:")
            logger.debug(f"  - Agents count: {len(self.agents)}")
            logger.debug(f"  - Tools count: {len(all_tools)}")
            logger.debug(f"  - Checkpointer: {self.checkpointer is not None}")
            
            supervisor_config = SupervisorConfig(
                default_agent="chat",
                enable_routing_logs=True,
                enable_context_aware_routing=True,
                conversation_history_limit=6,
                continuity_bonus_weight=1.0,  # Reduced from 1.5
                keyword_match_weight=3.0      # Increased from 2.0
            )
            await self.supervisor.initialize(self.agents, all_tools, self.checkpointer)
            logger.info("✅ Supervisor initialized successfully")
            
            # Verify supervisor graph is ready
            if self.supervisor.supervisor_graph is None:
                logger.error("❌ CRITICAL: Supervisor graph is None after initialization")
                raise RuntimeError("Supervisor graph failed to initialize")
            else:
                logger.debug("✅ Supervisor graph verified as ready")
            
            # Start background tasks with logging
            logger.info("⚙️ Starting background tasks...")
            asyncio.create_task(self._session_cleanup_task())
            asyncio.create_task(self._health_monitoring_task())
            logger.debug("✅ Background tasks started")
            
            logger.info("✅ Assistant Core Initialization Complete")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Initialization failed")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error message: {str(e)}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            
            # Log component states at failure
            logger.error("🔍 Component state at failure:")
            logger.error(f"  - Checkpointer: {self.checkpointer is not None}")
            logger.error(f"  - Agent factory: {hasattr(agent_factory, 'agents')}")
            logger.error(f"  - Supervisor: {hasattr(self.supervisor, 'supervisor_graph')}")
            
            await self.graceful_shutdown()
            raise
    
    async def graceful_shutdown(self):
        """Shutdown all components gracefully"""
        logger.info("🔌 Starting graceful shutdown...")
        
        try:
            # Close checkpointer with error handling
            if self.checkpointer_factory:
                logger.debug("📂 Cleaning up checkpointer connections...")
                
                # ✅ ADD SAFETY CHECK:
                if hasattr(self.checkpointer_factory, 'cleanup_connections'):
                    await self.checkpointer_factory.cleanup_connections()
                    logger.debug("✅ Checkpointer connections cleaned up")
                else:
                    logger.warning("⚠️ CheckpointerFactory missing cleanup_connections method")
                    # Manual cleanup fallback
                    if hasattr(self.checkpointer_factory, '_connection_cache'):
                        self.checkpointer_factory._connection_cache.clear()
                        logger.debug("✅ Manual checkpointer cache cleared")
            
            # Close LLM manager
            if hasattr(llm_manager, 'close'):
                logger.debug("🧠 Closing LLM manager...")
                await llm_manager.close()
                logger.debug("✅ LLM manager closed")
            
            # Close circuit breakers
            logger.debug("⚡ Stopping circuit breakers...")
            await global_circuit_breaker.stop_monitoring()
            logger.debug("✅ Circuit breakers stopped")
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    
    def _setup_routes(self):
        """Configure FastAPI routes"""
        logger.debug("🛣️ Setting up FastAPI routes...")
        
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @self.app.get("/", response_class=HTMLResponse)
        async def web_interface():
            return self._get_web_interface()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
        
        @self.app.post("/api/message")
        async def handle_message(message: dict):
            return await self._handle_api_message(message)
        
        @self.app.get("/api/status")
        async def get_system_status():
            return await self._get_system_status()
        
        logger.debug("✅ FastAPI routes configured")
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections"""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        logger.info(f"🔌 WebSocket connected: {session_id}")
        
        try:
            while True:
                message = await websocket.receive_json()
                logger.debug(f"📨 WebSocket message received: {session_id}")
                
                response = await self.process_message(
                    message.get("content", ""),
                    session_id=session_id,
                    user_id=message.get("user_id", "anonymous")
                )
                
                await websocket.send_json(response)
                logger.debug(f"📤 WebSocket response sent: {session_id}")
                
        except WebSocketDisconnect:
            logger.info(f"🔌 WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"❌ WebSocket error for {session_id}: {e}")
            error_response = await handle_error(e, "websocket_handler")
            await websocket.send_json(error_response)
    
    async def _handle_api_message(self, message: dict):
        """Handle API message requests"""
        logger.debug(f"📨 API message received: {message.get('content', '')[:50]}...")
        
        try:
            response = await self.process_message(
                message.get("content", ""),
                session_id=message.get("session_id"),
                user_id=message.get("user_id", "anonymous")
            )
            logger.debug("📤 API response prepared")
            return response
            
        except Exception as e:
            logger.error(f"❌ API message handling error: {e}")
            return await handle_error(e, "api_message_handler")

    async def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> dict[str, Any]:
        """Main message processing pipeline with comprehensive logging"""
        logger.info(f"🔄 Starting message processing - Session: {session_id}, User: {user_id}")
        logger.debug(f"📝 Message content: {message[:100]}...")
        
        try:
            # Log component status at start
            logger.debug(f"🔧 Component status check:")
            logger.debug(f"  - Supervisor initialized: {self.supervisor.supervisor_graph is not None}")
            logger.debug(f"  - Checkpointer available: {self.checkpointer is not None}")
            logger.debug(f"  - Active sessions: {len(self.active_sessions)}")

            if not self._initialized:
                logger.warning("AssistantCore not initialised – performing lazy start-up")
                await self.initialize()
            
            # Get or create session state with detailed logging
            logger.debug("📂 Retrieving session state...")
            session_state = await self._get_session_state(session_id, user_id)
            logger.debug(f"✅ Session state retrieved: {session_state['session_id']}")
            
            if self._is_new_conversation(message):
                logger.info(f"🔄 Creating new session for conversation starter")
                session_id = str(uuid.uuid4())  # Generate new session ID
                session_state = await self._get_session_state(session_id, user_id)
            
            # Add new message to state
            logger.debug("💬 Adding message to state...")
            from langchain_core.messages import HumanMessage
            human_msg = HumanMessage(content=message)
            human_msg.additional_kwargs = {'session_id': session_state["session_id"], 'timestamp': time.time()}
            new_messages = list(session_state.get("messages", [])) + [human_msg]
            
            updated_state = {
                "messages": new_messages,
                "session_id": session_state["session_id"],
                "user_id": session_state["user_id"],
                "current_agent": session_state.get("current_agent", "chat"),
            }
            logger.debug(f"📊 State updated - Messages: {len(new_messages)}, Agent: {updated_state['current_agent']}")
            
            # Process through supervisor with proper LangGraph config
            logger.debug("🎯 Preparing supervisor invocation...")
            # In process_message method
            unique_thread_id = f"{session_state['user_id']}:{session_state['session_id']}"
            config_dict     = {"configurable": {"thread_id": unique_thread_id}}
            logger.debug(f"🔧 Using thread ID: {unique_thread_id}")

            # Fail fast if the graph never compiled
            if self.supervisor.supervisor_graph is None:
                raise RuntimeError("Supervisor graph not initialised – startup must succeed before use")

            # Always go through the Supervisor façade
            result = await self.supervisor.process(updated_state, config_dict)
            
            # Log result structure
            logger.debug(f"📋 Result type: {type(result)}")
            if isinstance(result, dict):
                logger.debug(f"📋 Result keys: {list(result.keys())}")
                if "messages" in result:
                    logger.debug(f"📋 Result messages count: {len(result['messages'])}")
            
            # Update session state
            self.active_sessions[session_state["session_id"]] = result
            logger.debug("💾 Session state updated")
            
            # Extract response from result
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                
                # Extract clean content
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
                
                # Clean up any formatting artifacts
                if response_content.startswith('content="') and response_content.endswith('"'):
                    response_content = response_content[9:-1]  # Remove content=" wrapper
                elif response_content.startswith("content='") and response_content.endswith("'"):
                    response_content = response_content[9:-1]  # Remove content=' wrapper
            else:
                response_content = str(result)
            
            logger.info(f"✅ Message processing completed - Response length: {len(response_content)}")
            
            return {
                "response": response_content,
                "session_id": session_state["session_id"],
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Message processing failed at step: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            logger.error(f"❌ Exception args: {e.args}")
            
            # Log component states at failure
            logger.error("🔍 Component state at failure:")
            logger.error(f"  - Supervisor graph: {hasattr(self.supervisor, 'supervisor_graph') and self.supervisor.supervisor_graph is not None}")
            logger.error(f"  - Checkpointer: {self.checkpointer is not None}")
            logger.error(f"  - Session ID: {session_id}")
            logger.error(f"  - Updated state keys: {list(updated_state.keys()) if 'updated_state' in locals() else 'Not created'}")
            
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            
            return await handle_error(e, "message_processing")

    def _tag_messages_with_session(self, messages: list, session_id: str) -> list:
        """Add session metadata to messages for better tracking"""
        tagged_messages = []
        for msg in messages:
            # Clone the message to avoid modifying original
            if not hasattr(msg, 'additional_kwargs'):
                msg.additional_kwargs = {}
            msg.additional_kwargs['session_id'] = session_id
            msg.additional_kwargs['timestamp'] = time.time()
            tagged_messages.append(msg)
        return tagged_messages
    
    async def _get_session_state(
        self,
        session_id: Optional[str], 
        user_id: str
    ) -> AssistantState:
        """Get or create session state with checkpointer integration"""
        if not session_id or session_id not in self.active_sessions:
            new_session_id = session_id or str(uuid.uuid4())
            
            if self.checkpointer and session_id:
                try:
                    # Use namespaced thread ID for retrieval
                    namespaced_id = f"{user_id}:{session_id}"
                    config = {"configurable": {"thread_id": namespaced_id}}
                    logger.debug(f"🔍 Retrieving session with namespaced ID: {namespaced_id}")
                    raw_state = await self.checkpointer.aget(config)
                    
                    # Validate state integrity before using
                    if not self._validate_session_state(raw_state, session_id, user_id):
                        logger.warning(f"⚠️ Session validation failed, creating new state")
                        raise ValueError("Session validation failed")
                    
                    # Normalize to AssistantState format
                    state = {
                        "session_id": session_id,
                        "user_id": user_id,
                        "messages": raw_state.get("messages", []),
                        "current_agent": raw_state.get("current_agent", "chat")
                    }
                    
                    self.active_sessions[session_id] = state
                    return state
                except Exception as e:
                    logger.debug(f"Error loading session: {e}")
            
            # Create new state with required fields
            new_state = AssistantState(
                session_id=new_session_id,
                user_id=user_id,
                messages=[],
                current_agent="chat"
            )
            self.active_sessions[new_session_id] = new_state
            return new_state
        
        return self.active_sessions[session_id]

    def _validate_session_state(self, state: dict, expected_session_id: str, user_id: str) -> bool:
        """Validate that the retrieved state belongs to the expected session"""
        if not isinstance(state, dict):
            logger.warning(f"Invalid state type: {type(state)}")
            return False
            
        if state.get("session_id") != expected_session_id:
            logger.warning(f"Session ID mismatch: expected {expected_session_id}, got {state.get('session_id')}")
            return False
            
        if state.get("user_id") != user_id:
            logger.warning(f"User ID mismatch: expected {user_id}, got {state.get('user_id')}")
            return False

        if state is None:
            return False
            
        return True

    def _is_new_conversation(self, message: str) -> bool:
        """Detect if this is a new conversation based on message content"""
        # Check for conversation starters
        starters = ["hello", "hi", "start", "reset", "new conversation", "let's talk about"]
        message_lower = message.lower()
        
        for starter in starters:
            if message_lower.startswith(starter):
                logger.info(f"🔄 New conversation detected: '{message[:20]}...'")
                return True
                
        # Check if no conversation in last 30 minutes
        # (implement timestamp check)
        
        return False

    async def _get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status with TaskGroup - FIXED"""
        logger.debug("📊 Gathering system status...")
        
        # Default status structure
        status = {
            "agents": {},
            "circuit_breakers": {},
            "checkpointer": {},
            "llm_manager": {},
            "active_sessions": len(self.active_sessions),
            "versions": {
                "python": "3.13.4",
                "langgraph": "0.4.8",
                "postgres": "17"
            }
        }
        
        try:
            tasks = {}
            
            async with asyncio.TaskGroup() as tg:
                tasks['agents'] = tg.create_task(self._safe_agent_status())
                tasks['circuit_breakers'] = tg.create_task(self._safe_circuit_status())
                tasks['checkpointer'] = tg.create_task(self._safe_checkpointer_status())
                tasks['llm_manager'] = tg.create_task(self._safe_llm_status())
            
            # Collect results
            for key, task in tasks.items():
                try:
                    status[key] = task.result()
                except Exception as e:
                    logger.warning(f"⚠️ Status collection failed for {key}: {e}")
                    status[key] = {"error": str(e)}
            
        except* Exception as eg:  # 🔥 FIX: Handle exception group without return
            logger.error(f"❌ System status TaskGroup failed with {len(eg.exceptions)} exceptions")
            # Status already has default values, no return needed here
        
        logger.debug("✅ System status gathered successfully")
        return status

    # 🔥 NEW: Safe wrapper methods
    async def _safe_agent_status(self):
        """Safe agent status check"""
        try:
            return await agent_factory.health_check_agents()
        except Exception as e:
            logger.debug(f"Agent status check failed: {e}")
            return {"error": str(e)}

    async def _safe_circuit_status(self):
        """Safe circuit breaker status check"""
        try:
            return await global_circuit_breaker.health_check_all_services()
        except Exception as e:
            logger.debug(f"Circuit breaker status check failed: {e}")
            return {"error": str(e)}

    async def _safe_checkpointer_status(self):
        """Safe checkpointer status check"""
        try:
            health_status = await self.checkpointer_factory.health_check_all()
            factory_stats = self.checkpointer_factory.get_factory_statistics()
            
            return {
                "health_status": health_status,
                "factory_stats": factory_stats,
                "active_connections": len(self.checkpointer_factory._connection_cache),
                "checkpointer_type": type(self.checkpointer).__name__ if self.checkpointer else "None"
            }
        except Exception as e:
            logger.debug(f"Checkpointer status check failed: {e}")
            return {"error": str(e)}

    async def _safe_llm_status(self):
        """Safe LLM manager status check"""
        try:
            return await llm_manager.health_check()
        except Exception as e:
            logger.debug(f"LLM manager status check failed: {e}")
            return {"error": str(e)}
    
    async def _session_cleanup_task(self):
        """Background task to clean up expired sessions"""
        logger.debug("🧹 Starting session cleanup task...")
        
        while True:
            try:
                logger.debug("🔍 Checking for expired sessions...")
                
                expired = []
                for session_id, state in list(self.active_sessions.items()):
                    # Simple expiration check - sessions older than 1 hour
                    if hasattr(state, 'get') and 'timestamp' in state:
                        if time.time() - state.get('timestamp', 0) > 3600:
                            expired.append(session_id)
                
                for session_id in expired:
                    logger.debug(f"🗑️ Cleaning up expired session: {session_id}")
                    del self.active_sessions[session_id]
                    if self.checkpointer and hasattr(self.checkpointer, "adelete"):
                        cached_state  = self.active_sessions.get(session_id, {})
                        namespaced_id = f"{cached_state.get('user_id','')}:{session_id}"
                        await self.checkpointer.adelete({"configurable": {"thread_id": namespaced_id}})
                
                if expired:
                    logger.info(f"🧹 Cleaned up {len(expired)} expired sessions")
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Session cleanup error: {e}")
                await asyncio.sleep(60)
    
    # core/assistant_core.py - OPTIMIZED health monitoring
    async def _health_monitoring_task(self):
        """Optimized health monitoring with configurable intervals"""
        logger.debug("❤️ Starting optimized health monitoring...")
        
        # ✅ MUCH longer intervals and smarter triggers
        base_interval = 180  # 3 minutes instead of 30 seconds
        degraded_interval = 90  # 1.5 minutes when degraded
        critical_interval = 60  # 1 minute when critical
        
        consecutive_failures = 0
        last_full_check = 0
        
        while True:
            try:
                current_time = time.time()
                
                # ✅ SMART: Skip full checks if recent
                if current_time - last_full_check < 60:  # Minimum 1 minute between full checks
                    await asyncio.sleep(30)
                    continue
                
                logger.debug("🔍 Performing lightweight health check...")
                
                # ✅ LIGHTWEIGHT: Basic connectivity check only
                health_score = await self._lightweight_health_check()
                
                # ✅ ADAPTIVE: Adjust interval based on health
                if health_score >= 0.8:
                    next_interval = base_interval
                    consecutive_failures = 0
                elif health_score >= 0.5:
                    next_interval = degraded_interval
                    consecutive_failures = 0
                else:
                    next_interval = critical_interval
                    consecutive_failures += 1
                    
                    # ✅ RECOVERY: Only trigger recovery after multiple failures
                    if consecutive_failures >= 3:
                        logger.warning(f"⚠️ Health critical for {consecutive_failures} checks - triggering recovery")
                        await self._trigger_health_recovery()
                        consecutive_failures = 0
                        last_full_check = current_time
                
                logger.debug(f"✅ Health check complete - Score: {health_score:.2f}, Next check in {next_interval}s")
                await asyncio.sleep(next_interval)
                
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(60)  # Fallback interval

    def initialize_sync(self) -> None:
        """Blocking wrapper that guarantees the assistant is ready.

        If called inside a running event loop, it schedules `initialize()`
        as a background task; otherwise it runs it synchronously.
        """
        if self._initialized:
            return

        try:
            loop = asyncio.get_running_loop()
            # Inside an async context → schedule and return immediately
            loop.create_task(self.initialize())
        except RuntimeError:
            # No running loop → we can safely block
            asyncio.run(self.initialize())


    async def _lightweight_health_check(self) -> float:
        """Lightweight health check without expensive LLM calls"""
        try:
            # ✅ FAST: Check component availability only
            checks = {
                "supervisor": self.supervisor.supervisor_graph is not None,
                "checkpointer": self.checkpointer is not None,
                "agents": len(self.agents) > 0,
                "llm_manager": hasattr(llm_manager, 'models') and len(llm_manager.models) > 0
            }
            
            healthy_components = sum(checks.values())
            total_components = len(checks)
            
            return healthy_components / total_components if total_components > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Lightweight health check failed: {e}")
            return 0.0

    # ADD THIS METHOD for better error handling:
    async def get_system_status(self) -> dict[str, Any]:
        """Public method for system status with better error handling"""
        try:
            return await self._get_system_status()
        except Exception as e:
            logger.error(f"❌ System status check failed: {e}")
            return {
                "error": str(e),
                "agents": {"chat": False},
                "circuit_breakers": {},
                "checkpointer": {"error": "status_check_failed"},
                "llm_manager": {"healthy": False, "error": str(e)},
                "active_sessions": len(self.active_sessions),
                "versions": {"python": "3.13.4", "langgraph": "0.4.8", "postgres": "17"}
            }
    
    async def _trigger_health_recovery(self):
        """Attempt system health recovery"""
        logger.info("🔄 Starting health recovery...")
        
        try:
            # Cycle critical components
            logger.debug("⚡ Stopping circuit breakers...")
            await global_circuit_breaker.stop_monitoring()
            
            logger.debug("🤖 Reinitializing agents...")
            await agent_factory.initialize_agents()
            
            logger.debug("⚡ Starting circuit breakers...")
            await global_circuit_breaker.start_monitoring()
            
            # Reset LLM manager cache
            if hasattr(llm_manager, 'clear_cache'):
                logger.debug("🧠 Clearing LLM manager cache...")
                llm_manager.clear_cache()
            
            logger.info("✅ Completed health recovery cycle")
            
        except Exception as e:
            logger.error(f"❌ Health recovery failed: {e}")
    
    def _get_web_interface(self) -> str:
        """Return simple web interface HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mortey Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                #chat-container { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px; }
                #message-input { width: 70%; padding: 5px; }
                button { padding: 5px 10px; }
                .message { margin: 5px 0; padding: 5px; background: #f0f0f0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Mortey Assistant</h1>
            <div id="chat-container"></div>
            <input type="text" id="message-input" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
            <script>
                const ws = new WebSocket('ws://' + window.location.host + '/ws');
                
                function sendMessage() {
                    const input = document.getElementById('message-input');
                    if (input.value.trim()) {
                        ws.send(JSON.stringify({content: input.value}));
                        addMessage('You: ' + input.value, 'user');
                        input.value = '';
                    }
                }
                
                function addMessage(text, sender) {
                    const chatDiv = document.getElementById('chat-container');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + sender;
                    messageDiv.textContent = text;
                    chatDiv.appendChild(messageDiv);
                    chatDiv.scrollTop = chatDiv.scrollHeight;
                }
                
                ws.onmessage = function(event) {
                    const response = JSON.parse(event.data);
                    addMessage('Mortey: ' + response.response, 'assistant');
                };
                
                document.getElementById('message-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            </script>
        </body>
        </html>
        """

# Singleton instance
assistant = AssistantCore()
