# Modern Supervisor-Based Assistant Core
# June 2025 - Production Ready

import asyncio
import os
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Modern LangGraph supervisor imports
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langsmith import traceable

# Core components
from Core.state import AssistantState, AgentType, ThinkingState, trim_message_history
from agents.agents import agent_factory
from Core.controller import controller
from Core.checkpointer import create_production_checkpointer
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
    Modern LangGraph assistant using supervisor pattern.
    
    Key improvements for June 2025:
    - Uses langgraph-supervisor package for proper supervisor pattern
    - String-based model references with modern initialization
    - Proper interrupt patterns for human-in-the-loop
    - Built-in memory management with automatic trimming
    - Semaphore-based concurrency control
    - Production-ready persistence and error handling
    """

    def __init__(self):
        self.supervisor = None
        self.checkpointer = None
        self.current_session: Optional[AssistantSession] = None
        self.gui_callback = None
        
        # Concurrency control
        self.MAX_CONCURRENT_SESSIONS = 10
        self._session_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_SESSIONS)
        
        # Initialize core components
        self._initialize_checkpointer()
        self._initialize_agents()
        self._initialize_supervisor()
        self._setup_langsmith()
        
        logger.info("🤖 Assistant Core initialized with modern supervisor pattern")

    def _initialize_checkpointer(self):
        """Initialize production-ready checkpointer"""
        try:
            self.checkpointer = create_production_checkpointer()
            logger.info("✅ Production checkpointer initialized")
        except Exception as e:
            logger.error(f"❌ Checkpointer initialization failed: {e}")
            raise

    def _initialize_agents(self):
        """Initialize individual agents using modern create_react_agent"""
        try:
            self.chat_agent = agent_factory.create_chat_agent()
            self.coder_agent = agent_factory.create_coder_agent()
            self.web_agent = agent_factory.create_web_agent()
            
            logger.info("✅ Individual agents initialized with modern patterns")
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise

    def _initialize_supervisor(self):
        """Initialize supervisor using modern langgraph-supervisor package"""
        try:
            # Get model for supervisor using modern string-based pattern
            from langchain.chat_models import init_chat_model
            
            # Use router config or fallback to chat config
            node_config = config.get_node_config("router") or config.get_node_config("chat")
            if node_config:
                provider_config = config.get_provider_config(node_config.provider)
                model_config = config.get_model_config(node_config.provider, node_config.model)
                
                if provider_config and model_config and provider_config.api_key:
                    # Set API key
                    os.environ[provider_config.api_key_env] = provider_config.api_key
                    
                    # Initialize supervisor model with modern pattern
                    model_string = f"{node_config.provider}:{model_config.model_id}"
                    supervisor_model = init_chat_model(
                        model_string,
                        temperature=node_config.temperature,
                        max_tokens=node_config.max_tokens
                    )
                else:
                    # Fallback model
                    supervisor_model = init_chat_model("anthropic:claude-3-haiku-20240307")
            else:
                # Default fallback
                supervisor_model = init_chat_model("anthropic:claude-3-haiku-20240307")

            # Create supervisor with modern patterns
            self.supervisor = create_supervisor(
                agents=[self.chat_agent, self.coder_agent, self.web_agent],
                model=supervisor_model,
                prompt=(
                    "You are a supervisor managing three specialized agents:\n"
                    "\n"
                    "**chat_agent**: Handles general conversation, greetings, file browsing, and questions\n"
                    "**coder_agent**: Handles code generation, programming tasks, file creation, and technical implementation\n"
                    "**web_agent**: Handles web searches, current information, news, and research\n"
                    "\n"
                    "Route user requests to the most appropriate agent:\n"
                    "- For programming, coding, or file creation tasks → use coder_agent\n"
                    "- For web searches, current events, or research → use web_agent\n"
                    "- For general chat, greetings, or file browsing → use chat_agent\n"
                    "\n"
                    "Always choose exactly one agent. Consider the user's primary intent.\n"
                    "Keep track of conversation history to maintain context."
                ),
                state_schema=AssistantState,
                supervisor_name="supervisor",
                output_mode="last_message",  # Only show final response
                add_handoff_messages=True,   # Include handoff context
                parallel_tool_calls=False,   # Sequential processing for control
            )
            
            logger.info("✅ Modern supervisor initialized with agent delegation")
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
                os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
                logger.info(f"✅ LangSmith tracing enabled: {config.langsmith_project}")
            except Exception as e:
                logger.warning(f"⚠️ LangSmith setup failed: {e}")
        else:
            logger.info("📊 LangSmith tracing disabled")

    @traceable(name="process_message", run_type="chain", project_name="mortey-assistant")
    async def process_message(self, message: str, user_id: str = "default_user") -> str:
        """
        Process user message using modern supervisor pattern.
        
        Key improvements:
        - Concurrent session management with semaphores
        - Automatic memory management with trimming
        - Proper interrupt patterns for security
        - Built-in error recovery and retry logic
        """
        # Apply semaphore for concurrent session control
        async with self._session_semaphore:
            try:
                # Create or update session
                if not self.current_session:
                    self.current_session = AssistantSession(
                        session_id=str(uuid.uuid4()),
                        user_id=user_id,
                        start_time=time.time()
                    )

                self.current_session.message_count += 1
                self.current_session.last_interaction = time.time()

                # Create initial state with proper MessagesState structure
                initial_state = AssistantState(
                    messages=[HumanMessage(content=message)],
                    session_id=self.current_session.session_id,
                    user_id=user_id,
                    current_agent="",
                    thinking_state=ThinkingState(
                        active_agent=AgentType.SUPERVISOR,
                        current_task="Processing request",
                        progress=0.1,
                        details=f"Analyzing: {message[:50]}..."
                    )
                )

                # Configure thread for persistence
                config_dict = {
                    "configurable": {
                        "thread_id": self.current_session.session_id,
                        "user_id": user_id
                    }
                }

                logger.info(f"🎯 Processing message with thread_id: {self.current_session.session_id}")

                # Compile supervisor with checkpointer and invoke
                compiled_supervisor = self.supervisor.compile(checkpointer=self.checkpointer)
                
                # Use supervisor to process the message
                result = await asyncio.to_thread(
                    compiled_supervisor.invoke,
                    initial_state,
                    config_dict
                )

                # Apply memory management if needed
                if len(result.get("messages", [])) > result.get("max_messages", 50):
                    memory_update = trim_message_history(result)
                    if memory_update:
                        # Update state with trimmed messages
                        result = {**result, **memory_update}

                # Extract response from result
                response_content = self._extract_response_content(result)

                # Apply security verification if needed
                if self._requires_security_check(response_content):
                    verification_state = AssistantState(
                        **result,
                        output_content=response_content,
                        output_type=self._detect_output_type(response_content)
                    )
                    
                    verified_state = await controller.verify_and_approve(verification_state)
                    response_content = verified_state.get('output_content', response_content)

                # Update GUI if callback exists
                if self.gui_callback:
                    try:
                        self.gui_callback("Assistant", response_content)
                    except Exception as e:
                        logger.error(f"❌ GUI callback error: {e}")

                return response_content

            except Exception as e:
                logger.error(f"❌ Message processing error: {e}")
                
                # Attempt recovery with simplified response
                try:
                    return await self._handle_processing_error(message, str(e))
                except Exception as recovery_error:
                    logger.error(f"❌ Recovery also failed: {recovery_error}")
                    return f"I encountered an error processing your request: {str(e)}"

    def _extract_response_content(self, result: Dict[str, Any]) -> str:
        """Extract response content from supervisor result"""
        if result and 'messages' in result and result['messages']:
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return str(last_message)
        else:
            return "I processed your request, but didn't generate a response."

    async def _handle_processing_error(self, original_message: str, error: str) -> str:
        """Handle processing errors with graceful recovery"""
        try:
            # Create a simple fallback response using LLM manager
            fallback_prompt = f"""
            The user sent this message: "{original_message}"
            
            There was a system error: {error}
            
            Please provide a helpful response acknowledging the issue and offering to help in a different way.
            Keep the response brief and friendly.
            """
            
            response = await llm_manager.generate_for_node("chat", fallback_prompt, override_max_tokens=150)
            return response
            
        except Exception as e:
            logger.error(f"❌ Fallback response generation failed: {e}")
            return "I'm experiencing some technical difficulties. Please try rephrasing your request or try again in a moment."

    def _requires_security_check(self, content: str) -> bool:
        """Determine if content requires security verification"""
        security_indicators = [
            'rm ', 'del ', 'sudo', 'chmod', 'exec', 'eval',
            'system', 'shell', 'subprocess', 'os.system'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in security_indicators)

    def _detect_output_type(self, content: str) -> str:
        """Detect the type of output content"""
        if '```python' in content or 'def ' in content or 'import ' in content:
            return "code"
        elif 'http' in content or 'www.' in content:
            return "web_results"
        else:
            return "text"

    async def get_conversation_history(self, thread_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history from persistent storage"""
        try:
            if not thread_id and self.current_session:
                thread_id = self.current_session.session_id

            if not thread_id:
                return []

            config_dict = {"configurable": {"thread_id": thread_id}}

            # Get checkpoint history
            history = []
            try:
                checkpoints = self.checkpointer.list(config_dict)
                for checkpoint in checkpoints:
                    if 'channel_values' in checkpoint and 'messages' in checkpoint['channel_values']:
                        messages = checkpoint['channel_values']['messages']
                        for msg in messages:
                            history.append({
                                'role': 'user' if msg.type == 'human' else 'assistant',
                                'content': msg.content,
                                'timestamp': getattr(msg, 'timestamp', time.time())
                            })
            except Exception as e:
                logger.error(f"Error retrieving history: {e}")

            return history
        except Exception as e:
            logger.error(f"❌ Error retrieving conversation history: {e}")
            return []

    async def clear_conversation_memory(self, thread_id: str = None):
        """Clear conversation memory for a thread"""
        try:
            if not thread_id and self.current_session:
                thread_id = self.current_session.session_id

            if thread_id:
                # Note: Actual memory clearing depends on checkpointer implementation
                logger.info(f"🧠 Memory clearing requested for thread: {thread_id}")
        except Exception as e:
            logger.error(f"❌ Error clearing memory: {e}")

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
            "supervisor_initialized": self.supervisor is not None,
            "checkpointer_type": type(self.checkpointer).__name__ if self.checkpointer else None,
            "agents_available": ["chat_agent", "coder_agent", "web_agent"],
            "session_active": self.current_session is not None,
            "langsmith_enabled": config.langsmith_tracing and config.langsmith_api_key,
            "memory_management": "enabled",
            "concurrency_control": "enabled",
            "timestamp": time.time()
        }

# Global instance for backwards compatibility
assistant = AssistantCore()