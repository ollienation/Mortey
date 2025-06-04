# FIXED Assistant Core - LangGraph 0.4.8 (June 2025)
# CRITICAL FIXES: Message validation, supervisor patterns, tool call handling

import asyncio
import os
import time
import uuid
from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass

import logging

# ✅ FIXED: Modern LangGraph imports for 0.4.8
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langsmith import traceable

# ✅ FIXED: Modern supervisor import (separate package)
from langgraph_supervisor import create_supervisor

# Core components
from core.state import AssistantState, validate_and_filter_messages_v2, smart_trim_messages_v2
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
    
    def update_metrics(self):
        """Update session metrics"""
        self.message_count += 1
        self.last_interaction = time.time()
        self.duration_minutes = (time.time() - self.start_time) / 60

class AssistantCore:
    """
    ✅ FIXED LangGraph assistant using modern June 2025 patterns for LangGraph 0.4.8.
    
    CRITICAL IMPROVEMENTS:
    - Uses separate langgraph-supervisor package (required for 0.4.8)
    - Required state_schema specification (mandatory in 0.4.8)
    - Fixed response extraction with proper tool call handling
    - Enhanced message validation preventing empty content errors
    - Modern checkpointer patterns with improved fallback
    - Comprehensive error handling with graceful degradation
    - Added human-in-the-loop capabilities
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
            logger.info("🚀 Initializing Modern LangGraph Assistant Core (0.4.8)")
            
            # Initialize in correct order
            await self._initialize_checkpointer()
            await self._initialize_agents()
            await self._initialize_supervisor()
            self._setup_langsmith()
            
            self._setup_complete = True
            logger.info("✅ Modern Assistant Core initialized successfully")
        except Exception as e:
            logger.error(f"❌ Assistant initialization failed: {e}")
            raise

    async def _initialize_checkpointer(self):
        """Initialize modern checkpointer with fallback handling"""
        try:
            self.checkpointer = await create_checkpointer()
            logger.info("✅ Modern checkpointer initialized")
        except Exception as e:
            logger.error(f"❌ Checkpointer initialization failed: {e}")
            
            # Create fallback memory checkpointer
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
            logger.warning("⚠️ Using fallback MemorySaver - no persistence")

    async def _initialize_agents(self):
        """Initialize agents using modern create_react_agent patterns"""
        try:
            self.agent_factory = AgentFactory()
            
            # ✅ FIXED: Modern create_react_agent 0.4.8 usage
            self.chat_agent = self.agent_factory.create_chat_agent()
            self.coder_agent = self.agent_factory.create_coder_agent() 
            self.web_agent = self.agent_factory.create_web_agent()
            
            logger.info("✅ Modern agents initialized successfully")
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise
            
    async def _initialize_supervisor(self):
        """
        ✅ FIXED: Initialize supervisor using separate langgraph-supervisor package
        
        CRITICAL FIXES for June 2025:
        - Uses langgraph-supervisor package (required for 0.4.8)
        - Properly specifies state_schema (required for StateGraph)
        - Modern supervisor patterns with proper configuration
        - Enhanced supervisor prompts
        """
        try:
            # Get supervisor model using modern pattern
            supervisor_model = llm_manager._get_model("router")
            
            # ✅ FIXED: Enhanced supervisor prompt for 2025
            supervisor_prompt = """
            You are a supervisor managing three specialized agents:

            **chat_agent**: Handles general conversation, greetings, file browsing, and everyday queries
            **coder_agent**: Handles code generation, programming tasks, file creation, and technical questions
            **web_agent**: Handles web searches, current information, research, and fact-checking

            Route user requests to the most appropriate agent based on their primary intent.
            Always choose exactly one agent for each request based on the following criteria:

            - Route to chat_agent for:
              * Casual conversation and greetings
              * Personal questions about preferences or opinions
              * Questions about the system or how to use it
              * File browsing and local system interactions
              * General information without time-sensitivity

            - Route to coder_agent for:
              * Any request involving code generation
              * Technical questions about programming
              * File creation with specific formats
              * Debugging or analyzing existing code
              * Questions about algorithms, data structures, or programming concepts

            - Route to web_agent for:
              * Current events and news
              * Real-time information that might be outdated in your knowledge
              * Fact-checking and research
              * Questions about specific websites or web content
              * Searches for information not readily available in your training

            Be decisive and route quickly to avoid delays.
            If a request potentially spans multiple agents, select the agent that handles the primary intent.
            """
            
            # ✅ FIXED: Use langgraph-supervisor with state_schema and modern configuration
            self.supervisor_graph = create_supervisor(
                agents=[self.chat_agent, self.coder_agent, self.web_agent],
                model=supervisor_model,
                prompt=supervisor_prompt,
                # ✅ CRITICAL: Required state_schema for LangGraph 0.4.8
                state_schema=AssistantState,
                # ✅ FIXED: Modern supervisor configuration
                output_mode="last_message",  # Return only the final message
                add_handoff_messages=False,  # Prevents empty handoff messages
                parallel_tool_calls=False,   # Ensures sequential agent calls
                supervisor_name="supervisor"
            )
            
            logger.info("✅ Modern supervisor initialized with langgraph-supervisor")
        except ImportError as e:
            logger.error(f"❌ langgraph-supervisor package not installed: {e}")
            logger.info("Install with: pip install langgraph-supervisor>=0.0.27")
            raise
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
        ✅ FIXED: Process user message with comprehensive error handling
        
        CRITICAL FIXES for June 2025:
        - Robust message validation preventing empty content errors
        - Proper state initialization with required state_schema
        - Modern supervisor usage with error handling
        - Fixed response extraction for tool call messages
        - Graceful fallback responses for all error conditions
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
                
                self.current_session.update_metrics()
                
                # ✅ CRITICAL FIX: Validate message content before processing
                if not message or not message.strip():
                    return {
                        "response": "I received an empty message. Please try again with your question.",
                        "error": "empty_message_content",
                        "session_id": self.current_session.session_id
                    }
                
                # Create initial state with validated messages
                initial_messages = [HumanMessage(content=message.strip())]
                validated_messages = validate_and_filter_messages_v2(initial_messages)
                
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
                
                # ✅ FIXED: Modern supervisor compilation with checkpointer
                compiled_supervisor = self.supervisor_graph.compile(
                    checkpointer=self.checkpointer
                )
                
                # Process with the modern supervisor
                result = await asyncio.to_thread(
                    compiled_supervisor.invoke,
                    initial_state,
                    config_dict
                )
                
                # ✅ FIXED: Robust response extraction with tool call handling
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
        ✅ FIXED: Extract and validate response from supervisor result
        
        CRITICAL IMPROVEMENTS for June 2025:
        - Properly handles tool call messages (empty content is normal)
        - Extracts full responses (fixes truncation issues)
        - Handles complex message structures
        - Provides meaningful fallbacks for all scenarios
        """
        try:
            if result and 'messages' in result and result['messages']:
                # Get the last message
                last_message = result['messages'][-1]
                
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
                    
                    # ✅ CRITICAL FIX: Handle tool call messages properly
                    if has_tool_calls:
                        # This is a tool call message - extract tool call information
                        tool_descriptions = []
                        for tool_call in last_message.tool_calls:
                            if hasattr(tool_call, 'name'):
                                tool_name = tool_call.name
                                tool_args = getattr(tool_call, 'args', {})
                                tool_descriptions.append(f"Using {tool_name} with parameters: {tool_args}")
                        
                        if tool_descriptions:
                            return f"I'm processing your request using: {', '.join(tool_descriptions)}"
                    
                    # ✅ Handle string content - fixes truncation issue
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    
                    # ✅ Handle list content (tool calls, multimodal, etc.)
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('text'):
                                    text_parts.append(item['text'])
                                elif item.get('type') == 'text' and item.get('content'):
                                    text_parts.append(item['content'])
                            elif isinstance(item, str) and item.strip():
                                text_parts.append(item)
                        
                        if text_parts:
                            return " ".join(text_parts).strip()
                    
                    # ✅ Handle other content types
                    elif content:
                        content_str = str(content).strip()
                        # Filter out raw message metadata
                        if not any(keyword in content_str.lower() for keyword in 
                                  ['content=[]', 'additional_kwargs', 'response_metadata', 'tool_calls=[]']):
                            return content_str
                
                # ✅ Fallback: Look for previous messages with actual content
                for msg in reversed(result['messages'][:-1]):
                    if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                        # Don't return user messages as assistant responses
                        if not isinstance(msg, HumanMessage):
                            return msg.content.strip()
                
                # ✅ Fallback: Generic success message
                return "I've processed your request successfully."
            
            return "I processed your request, but didn't generate a response. Can I help with something else?"
            
        except Exception as e:
            logger.error(f"❌ Response extraction error: {e}")
            return "I encountered an issue while processing your request, but I'm ready to help with your next question."
            
    async def _handle_processing_error(self, original_message: str, error: str) -> Dict[str, Any]:
        """
        ✅ FIXED: Handle processing errors with graceful recovery
        
        IMPROVEMENTS for June 2025:
        - Specific error handling for common issues
        - Intelligent fallback responses
        - Error categorization for debugging
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
            elif "connection" in error.lower():
                return {
                    "response": "I'm having trouble connecting to my services. Please try again.",
                    "error": "connection_error",
                    "fallback_used": True
                }
            elif "langgraph-supervisor" in error.lower():
                return {
                    "response": "I'm experiencing a configuration issue. Please contact support.",
                    "error": "supervisor_error",
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
            
            if state and hasattr(state, 'values') and 'messages' in state.values:
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
            "modern_patterns": "LangGraph 0.4.8 + langgraph-supervisor",
            "timestamp": time.time()
        }
        
    # ✅ NEW: Human-in-the-loop methods for June 2025
    async def provide_human_feedback(
        self, 
        thread_id: str,
        feedback: str,
        feedback_type: Literal["approve", "reject", "modify"] = "modify"
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Provide human feedback to continue execution
        
        This feature enables human-in-the-loop capabilities
        with LangGraph 0.4.8's improved interrupt patterns.
        """
        try:
            if not thread_id:
                return {"error": "thread_id_required"}
                
            config_dict = {"configurable": {"thread_id": thread_id}}
            
            # Resume interrupted thread with feedback
            if feedback_type == "approve":
                # Simple approval
                resume_result = await self.supervisor_graph.aresume_interruption(
                    config_dict,
                    {"approved": True}
                )
            elif feedback_type == "reject":
                # Rejection with reason
                resume_result = await self.supervisor_graph.aresume_interruption(
                    config_dict,
                    {"approved": False, "reason": feedback}
                )
            else:
                # Modification with new content
                resume_result = await self.supervisor_graph.aresume_interruption(
                    config_dict,
                    {"modified_content": feedback}
                )
                
            # Extract response from resumed execution
            response_content = self._extract_and_validate_response(resume_result)
            
            return {
                "response": response_content,
                "session_id": thread_id,
                "feedback_applied": True
            }
            
        except Exception as e:
            logger.error(f"❌ Human feedback application failed: {e}")
            return {
                "error": "feedback_application_failed",
                "details": str(e)
            }

# Global instance
assistant = AssistantCore()