# Fixed LangGraph Assistant Core - June 2025 LangGraph 0.4.8 Patterns

import asyncio
import os
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Modern LangGraph imports for June 2025 (0.4.8)
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, trim_messages, BaseMessage
from langsmith import traceable

# ✅ FIXED: Modern supervisor import
from langgraph_supervisor import create_supervisor

# Core components
from core.state import AssistantState, validate_and_filter_messages
from agents.agents import AgentFactory
from core.checkpointer import create_checkpointer
from config.settings import config
from config.llm_manager import llm_manager

logger = logging.getLogger("assistant")

@dataclass
class AssistantSession:
    """Session tracking for the assistant"""
    session_id: str
    user_id: str
    start_time: float
    message_count: int = 0
    last_interaction: float = 0

class AssistantCore:
    """
    Modern LangGraph assistant using 2025 best practices for LangGraph 0.4.8.
    
    KEY IMPROVEMENTS:
    - Uses modern langgraph-supervisor package
    - Robust message validation preventing empty content errors
    - Proper MessagesState usage with built-in reducers
    - Modern checkpointer patterns with separate packages
    - Comprehensive error handling with fallbacks
    - StateGraph with required state_schema for 0.4.8
    """

    def __init__(self):
        self.supervisor_graph = None
        self.checkpointer = None
        self.current_session: Optional[AssistantSession] = None
        self.gui_callback = None
        
        # Concurrency control
        self.MAX_CONCURRENT_SESSIONS = 10
        self._session_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_SESSIONS)
        self._setup_complete = False

    async def initialize(self):
        """Initialize all components with proper error handling"""
        try:
            # Initialize in correct order
            await self._initialize_checkpointer()
            await self._initialize_agents()
            await self._initialize_supervisor()
            self._setup_langsmith()
            
            self._setup_complete = True
            logger.info("🤖 Modern Assistant Core initialized successfully")
        except Exception as e:
            logger.error(f"❌ Assistant initialization failed: {e}")
            raise

    async def _initialize_checkpointer(self):
        """Initialize modern checkpointer"""
        try:
            self.checkpointer = create_checkpointer()
            logger.info("✅ Modern checkpointer initialized")
        except Exception as e:
            logger.error(f"❌ Checkpointer initialization failed: {e}")
            raise

    async def _initialize_agents(self):
        """Initialize agents using modern create_react_agent patterns"""
        try:
            self.agent_factory = AgentFactory()
            
            # ✅ FIXED: create_react_agent returns CompiledStateGraph (synchronous)
            self.chat_agent = self.agent_factory.create_chat_agent()
            self.coder_agent = self.agent_factory.create_coder_agent()
            self.web_agent = self.agent_factory.create_web_agent()
            
            logger.info("✅ Modern agents initialized")
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise

    async def _initialize_supervisor(self):
        """
        ✅ FIXED: Initialize supervisor using modern langgraph-supervisor package
        Compatible with LangGraph 0.4.8 and latest supervisor patterns
        """
        try:
            # Get supervisor model using modern pattern
            supervisor_model = llm_manager._get_model("router")

            # ✅ FIXED: Use modern langgraph-supervisor package correctly
            self.supervisor_graph = create_supervisor(
                agents=[self.chat_agent, self.coder_agent, self.web_agent],
                model=supervisor_model,
                prompt=(
                    "You are a supervisor managing three specialized agents:\n"
                    "\n"
                    "**chat_agent**: Handles general conversation, greetings, and file browsing\n"
                    "**coder_agent**: Handles code generation, programming tasks, and file creation\n"
                    "**web_agent**: Handles web searches, current information, and research\n"
                    "\n"
                    "Route user requests to the most appropriate agent based on their primary intent.\n"
                    "Always choose exactly one agent.\n"
                ),
                state_schema=AssistantState,  # ✅ Required for LangGraph 0.4.8
                output_mode="last_message",
                add_handoff_messages=False,  # ✅ Prevents empty handoff messages
                parallel_tool_calls=False,
                supervisor_name="supervisor"
            )
            
            logger.info("✅ Modern supervisor initialized with langgraph-supervisor")
        except Exception as e:
            logger.error(f"❌ Supervisor initialization failed: {e}")
            raise

    def _setup_langsmith(self):
        """Setup LangSmith tracing"""
        if config.langsmith_tracing and config.langsmith_api_key:
            try:
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
                os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
                logger.info(f"✅ LangSmith tracing enabled: {config.langsmith_project}")
            except Exception as e:
                logger.warning(f"⚠️ LangSmith setup failed: {e}")

    @traceable(name="process_message", run_type="chain")
    async def process_message(
        self,
        message: str,
        thread_id: str = None,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Process user message with robust error handling and modern patterns.
        
        ✅ FIXED: Comprehensive message validation
        ✅ FIXED: Proper error handling and fallbacks
        ✅ FIXED: Modern supervisor usage with LangGraph 0.4.8
        """
        if not self._setup_complete:
            await self.initialize()

        async with self._session_semaphore:
            try:
                # Create or update session
                if not self.current_session or (thread_id and thread_id != self.current_session.session_id):
                    self.current_session = AssistantSession(
                        session_id=thread_id or str(uuid.uuid4()),
                        user_id=user_id,
                        start_time=time.time()
                    )

                self.current_session.message_count += 1
                self.current_session.last_interaction = time.time()

                # ✅ CRITICAL FIX: Validate message content before processing
                if not message or not message.strip():
                    return {
                        "response": "I received an empty message. Please try again with your question.",
                        "error": "empty_message_content"
                    }

                # Create initial state with validated messages
                initial_messages = [HumanMessage(content=message.strip())]
                validated_messages = validate_and_filter_messages(initial_messages)

                # ✅ FIXED: Proper state initialization for LangGraph 0.4.8
                initial_state = AssistantState(
                    messages=validated_messages,
                    session_id=self.current_session.session_id,
                    user_id=user_id
                )

                # Configure thread for persistence
                config_dict = {
                    "configurable": {
                        "thread_id": self.current_session.session_id,
                        "user_id": user_id
                    }
                }

                logger.info(f"🎯 Processing message with thread_id: {self.current_session.session_id}")

                # ✅ FIXED: Use modern supervisor with proper compilation for LangGraph 0.4.8
                compiled_supervisor = self.supervisor_graph.compile(
                    checkpointer=self.checkpointer
                )

                # Process with the modern supervisor
                result = await asyncio.to_thread(
                    compiled_supervisor.invoke,
                    initial_state,
                    config_dict
                )

                # ✅ FIXED: Robust response extraction with validation
                response_content = self._extract_and_validate_response(result)

                # Update GUI if callback exists
                if self.gui_callback:
                    try:
                        self.gui_callback("Assistant", response_content)
                    except Exception as e:
                        logger.error(f"❌ GUI callback error: {e}")

                return {
                    "response": response_content,
                    "session_id": self.current_session.session_id,
                    "message_count": self.current_session.message_count
                }

            except Exception as e:
                logger.error(f"❌ Message processing error: {e}")
                return await self._handle_processing_error(message, str(e))

    def _extract_and_validate_response(self, result: Dict[str, Any]) -> str:
        """
        Extract and validate response from supervisor result.
        ✅ FIXED: Robust response extraction with fallbacks for LangGraph 0.4.8
        """
        try:
            if result and 'messages' in result and result['messages']:
                # Get the last message
                last_message = result['messages'][-1]
                
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    
                    # Handle string content
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    
                    # Handle list content (tool calls, etc.)
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get('text'):
                                text_parts.append(item['text'])
                            elif isinstance(item, str):
                                text_parts.append(item)
                        if text_parts:
                            return " ".join(text_parts).strip()
                    
                    # Fallback: convert to string
                    return str(last_message).strip() or "I processed your request successfully."
                
                return "I processed your request, but didn't generate a response."
        except Exception as e:
            logger.error(f"❌ Response extraction error: {e}")
            return "I encountered an issue while processing your request."

    async def _handle_processing_error(self, original_message: str, error: str) -> Dict[str, Any]:
        """
        Handle processing errors with graceful recovery.
        ✅ FIXED: Specific error handling for common issues
        """
        try:
            # Handle specific error types
            if "empty content" in error.lower():
                return {
                    "response": "I detected an issue with message content. Please try rephrasing your request.",
                    "error": "empty_content",
                    "fallback_used": True
                }
            elif "rate limit" in error.lower():
                return {
                    "response": "I'm currently experiencing high demand. Please try again in a moment.",
                    "error": "rate_limit",
                    "fallback_used": True
                }
            elif "400" in error and "anthropic" in error.lower():
                return {
                    "response": "I encountered an API issue. Let me try a different approach.",
                    "error": "api_error",
                    "fallback_used": True
                }
            
            # Generate fallback response using LLM manager
            fallback_prompt = f"""
            The user sent: "{original_message}"
            There was a system error: {error}
            
            Please provide a helpful response acknowledging the issue and offering to help differently.
            Keep it brief and friendly.
            """
            
            response = await llm_manager.generate_for_node(
                "chat",
                fallback_prompt,
                override_max_tokens=150
            )
            
            return {
                "response": response,
                "error": "system_error",
                "fallback_used": True
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback response generation failed: {e}")
            return {
                "response": "I'm experiencing technical difficulties. Please try again.",
                "error": "fallback_failed",
                "fallback_used": True
            }

    async def get_conversation_history(self, thread_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history with proper error handling"""
        try:
            if not thread_id and self.current_session:
                thread_id = self.current_session.session_id

            if not thread_id:
                return []

            config_dict = {"configurable": {"thread_id": thread_id}}

            # Get state from checkpointer
            state = self.supervisor_graph.get_state(config_dict)
            if state and state.values and 'messages' in state.values:
                history = []
                for msg in state.values['messages']:
                    history.append({
                        'role': 'user' if msg.type == 'human' else 'assistant',
                        'content': getattr(msg, 'content', str(msg)),
                        'timestamp': getattr(msg, 'timestamp', time.time())
                    })
                return history
            return []
        except Exception as e:
            logger.error(f"❌ Error retrieving conversation history: {e}")
            return []

    def set_gui_callback(self, callback):
        """Set GUI callback for message updates"""
        self.gui_callback = callback

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if not self.current_session:
            return {"status": "no_active_session"}

        return {
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "start_time": self.current_session.start_time,
            "message_count": self.current_session.message_count,
            "last_interaction": self.current_session.last_interaction,
            "duration_minutes": (time.time() - self.current_session.start_time) / 60
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information"""
        return {
            "supervisor_initialized": self.supervisor_graph is not None,
            "checkpointer_type": type(self.checkpointer).__name__ if self.checkpointer else None,
            "setup_complete": self._setup_complete,
            "agents_available": ["chat_agent", "coder_agent", "web_agent"],
            "session_active": self.current_session is not None,
            "langsmith_enabled": config.langsmith_tracing and config.langsmith_api_key,
            "modern_patterns": "LangGraph 0.4.8",
            "timestamp": time.time()
        }

# Global instance
assistant = AssistantCore()