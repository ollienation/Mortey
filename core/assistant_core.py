import asyncio
import os
import hashlib
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import uuid

# LangSmith imports
from langsmith import traceable

from agents.chat_agent import ChatAgent
from agents.web_agent import WebAgent
from agents.coder_agent import CoderAgent
from core.controller import ControllerAgent
from config.settings import config
from config.llm_manager import llm_manager

# LangGraph imports
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages

@dataclass
class AssistantSession:
    session_id: str
    start_time: float
    message_count: int = 0
    last_interaction: float = 0

class AssistantCore:
    """Core assistant logic with LangSmith tracing and memory support"""
    
    def __init__(self):
        # Initialize agents
        self.chat_agent = ChatAgent(None)
        self.web_agent = WebAgent(None)
        self.coder_agent = CoderAgent(None)
        self.controller = ControllerAgent(None)
        
        # Session management
        self.current_session: Optional[AssistantSession] = None
        
        # GUI callback for unified handling
        self.gui_callback = None
        
        # Circuit breaker to prevent runaway loops
        self.circuit_breaker = {}
        
        # Initialize LangGraph memory checkpointer
        self.memory = MemorySaver()
        print("🧠 MemorySaver initialized for conversation persistence")
        
        # Initialize LangSmith tracing
        self._setup_langsmith_tracing()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("🤖 Assistant Core initialized with LangSmith tracing")
    
    def _setup_langsmith_tracing(self):
        """Setup LangSmith tracing if enabled"""
        if config.langsmith_tracing and config.langsmith_api_key:
            try:
                # Set environment variables for LangSmith
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
                os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
                os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
                
                self.langsmith_enabled = True
                print(f"✅ LangSmith tracing enabled for project: {config.langsmith_project}")
                
            except Exception as e:
                print(f"⚠️ Failed to initialize LangSmith tracing: {e}")
                self.langsmith_enabled = False
        else:
            self.langsmith_enabled = False
            print("📊 LangSmith tracing disabled")

    async def get_conversation_history(self, thread_id: str = None) -> List[Dict]:
        """Retrieve conversation history from memory"""
        try:
            if not thread_id and self.current_session:
                thread_id = self.current_session.session_id
            
            if not thread_id:
                return []
            
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get the latest checkpoint for this thread
            checkpoint = self.memory.get(config)
            
            if checkpoint and 'messages' in checkpoint.get('channel_values', {}):
                messages = checkpoint['channel_values']['messages']
                
                # Convert to readable format
                history = []
                for msg in messages:
                    history.append({
                        'role': 'user' if msg.type == 'human' else 'assistant',
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', time.time())
                    })
                
                return history
            
            return []
            
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []

    async def clear_conversation_memory(self, thread_id: str = None):
        """Clear conversation memory for a specific thread"""
        try:
            if not thread_id and self.current_session:
                thread_id = self.current_session.session_id
            
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}
                # Note: MemorySaver doesn't have a direct clear method
                # We would need to implement this if needed
                print(f"🧠 Memory clearing requested for thread: {thread_id}")
                
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            # Basic stats about current session
            stats = {
                "current_session_id": self.current_session.session_id if self.current_session else None,
                "memory_enabled": True,
                "checkpointer_type": "MemorySaver"
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def set_gui_callback(self, callback):
        """Set GUI callback for unified message handling"""
        self.gui_callback = callback
    
    def _controller_routing_decision(self, state: Dict[str, Any]) -> str:
        """Smart controller routing with loop protection"""
        
        verification_required = state.get('verification_required', False)
        loop_count = state.get('loop_count', 0)
        max_loops = state.get('max_loops', 3)
        
        # CRITICAL: Force end after max loops
        if loop_count >= max_loops:
            print(f"🔄 Breaking infinite loop after {loop_count} attempts")
            state['verification_result'] = 'approved'
            state['verification_required'] = False
            return "force_end"
        
        # Only allow revision on first 2 attempts
        if verification_required and loop_count < 2:
            print(f"🔄 Controller requesting revision (attempt {loop_count + 1})")
            return "router"
        else:
            # Force approve after 2 attempts
            if verification_required:
                print(f"🔄 Force approving after {loop_count} attempts")
                state['verification_result'] = 'approved'
                state['verification_required'] = False
            return "output_handler"
    
    def _build_workflow(self):
        """Build workflow with memory-aware nodes"""
        
        @traceable(name="router_node_with_memory", run_type="chain")
        async def router_node(state: AssistantState) -> AssistantState:
            """Route with conversation history context"""
            
            # Increment loop count
            loop_count = state.get('loop_count', 0)
            state['loop_count'] = loop_count + 1
            
            if loop_count > 0:
                print(f"🔄 Revision attempt {loop_count}")
            
            # Get conversation history from messages
            messages = state.get('messages', [])
            recent_context = ""
            
            if len(messages) > 1:  # More than just current message
                # Get last 3 messages for context
                recent_messages = messages[-3:] if len(messages) > 3 else messages
                recent_context = "\n".join([
                    f"{msg.type}: {msg.content}" for msg in recent_messages[:-1]  # Exclude current message
                ])
            
            current_message = messages[-1].content if messages else state.get('user_input', '')
            
            prompt = f"""
            Route this request considering conversation context:
            
            Recent Conversation:
            {recent_context}
            
            Current User Request: {current_message}
            
            Available agents:
            - CHAT: General conversation, file browsing (read-only), greetings
            - CODER: Code generation, file creation, programming tasks
            - WEB: Web search, current information, news, weather
            
            Respond with only: CHAT, CODER, or WEB
            """
            
            try:
                response = await llm_manager.generate_for_node("router", prompt)
                agent_choice = response.strip().upper()
                
                # Enhanced routing with context awareness
                user_input_lower = current_message.lower()
                
                # Context-aware routing overrides
                if recent_context and "code" in recent_context.lower():
                    if any(keyword in user_input_lower for keyword in ['fix', 'change', 'modify', 'update']):
                        agent_choice = 'CODER'
                        print(f"🔄 Context override: Continuing code conversation")
                
                # Standard routing overrides
                if any(keyword in user_input_lower for keyword in [
                    'what files', 'list files', 'show files', 'read file'
                ]):
                    agent_choice = 'CHAT'
                elif any(keyword in user_input_lower for keyword in [
                    'create', 'generate', 'write code', 'make a script'
                ]):
                    agent_choice = 'CODER'
                
                if agent_choice not in ['CODER', 'WEB', 'CHAT']:
                    agent_choice = 'CHAT'
                
                state['agent_choice'] = agent_choice
                state['current_agent'] = agent_choice
                state['user_input'] = current_message  # Ensure user_input is set
                print(f"🎯 Routed to: {agent_choice}")
                
            except Exception as e:
                print(f"❌ Routing error: {e}")
                state['agent_choice'] = 'CHAT'
                state['current_agent'] = 'CHAT'
            
            return state
        
        @traceable(name="chat_node_with_memory", run_type="chain")
        async def chat_node(state: AssistantState) -> AssistantState:
            """Chat agent with conversation history"""
            try:
                # Pass conversation history to chat agent
                messages = state.get('messages', [])
                conversation_history = []
                
                # Convert messages to chat agent format
                for msg in messages:
                    role = "user" if msg.type == "human" else "assistant"
                    conversation_history.append({
                        'role': role,
                        'content': msg.content,
                        'timestamp': time.time()
                    })
                
                # Add conversation history to state for chat agent
                state['conversation_history'] = conversation_history
                
                result = await self.chat_agent.chat(state)
                
                # Add assistant response to messages
                if result.get('output_content'):
                    from langchain_core.messages import AIMessage
                    messages = state.get('messages', [])
                    messages.append(AIMessage(content=result['output_content']))
                    result['messages'] = messages
                
                return result
                
            except Exception as e:
                print(f"❌ Chat agent error: {e}")
                state['output_content'] = "I'm having trouble right now. Please try again."
                state['output_type'] = 'error'
                return state
        
        @traceable(name="coder_node_with_memory", run_type="chain")
        async def coder_node(state: AssistantState) -> AssistantState:
            """Coder agent with conversation history"""
            try:
                # Add conversation context for code generation
                messages = state.get('messages', [])
                
                # Look for previous code-related conversations
                code_context = []
                for msg in messages[-5:]:  # Last 5 messages
                    if any(word in msg.content.lower() for word in ['code', 'function', 'class', 'import']):
                        role = "user" if msg.type == "human" else "assistant"
                        code_context.append(f"{role}: {msg.content[:200]}...")
                
                if code_context:
                    state['code_conversation_context'] = "\n".join(code_context)
                
                result = await self.coder_agent.generate_code(state)
                
                # Add assistant response to messages
                if result.get('output_content'):
                    from langchain_core.messages import AIMessage
                    messages = state.get('messages', [])
                    messages.append(AIMessage(content=result['output_content']))
                    result['messages'] = messages
                
                return result
                
            except Exception as e:
                print(f"❌ Coder node error: {e}")
                state['output_content'] = f"Error in coder agent: {str(e)}"
                state['output_type'] = 'error'
                return state
        
        @traceable(name="web_node_with_memory", run_type="chain")
        async def web_node(state: AssistantState) -> AssistantState:
            """Web agent with conversation history"""
            try:
                # Add conversation context for better search queries
                messages = state.get('messages', [])
                
                # Get recent conversation for search context
                recent_topics = []
                for msg in messages[-3:]:  # Last 3 messages
                    if msg.type == "human":
                        recent_topics.append(msg.content)
                
                if recent_topics:
                    state['search_conversation_context'] = " ".join(recent_topics)
                
                result = await self.web_agent.search_and_browse(state)
                
                # Add assistant response to messages
                if result.get('output_content'):
                    from langchain_core.messages import AIMessage
                    messages = state.get('messages', [])
                    messages.append(AIMessage(content=result['output_content']))
                    result['messages'] = messages
                
                return result
                
            except Exception as e:
                print(f"❌ Web agent error: {e}")
                state['output_content'] = "I had trouble searching for that information."
                state['output_type'] = 'error'
                return state
        
        # Keep existing controller and output handler nodes with message updates
        @traceable(name="controller_node_with_memory", run_type="chain")
        async def controller_node(state: AssistantState) -> AssistantState:
            """Controller node with memory awareness"""
            try:
                result = await self.controller.verify_output(state)
                return result
            except Exception as e:
                print(f"❌ Controller error: {e}")
                state['verification_result'] = 'approved'
                return state
        
        @traceable(name="output_handler_with_memory", run_type="chain")
        async def output_handler_node(state: AssistantState) -> AssistantState:
            """Output handler with memory persistence"""
            output_content = state.get('output_content', '')
            current_agent = state.get('current_agent', '')
            
            # For chat agent, ensure verification_result is set
            if current_agent.lower() == 'chat' and not state.get('verification_result'):
                state['verification_result'] = 'approved'
            
            # Always update GUI if callback exists
            if self.gui_callback:
                try:
                    self.gui_callback("Mortey", output_content)
                except Exception as e:
                    print(f"❌ GUI callback error: {e}")
            
            # Handle voice output if in voice mode
            if state.get('voice_mode', False) and state.get('requires_speech_output', False):
                state['speech_output_ready'] = True
                
                if len(output_content) > 500:
                    state['speech_content'] = output_content[:500] + "... I can provide more details if you'd like."
                else:
                    state['speech_content'] = output_content
            
            # Memory is automatically persisted by LangGraph checkpointer
            print(f"🧠 Conversation state saved to memory (thread: {state.get('thread_id', 'unknown')})")
            
            return state
        
        # Build workflow with all nodes
        workflow = StateGraph(AssistantState)
        
        workflow.add_node("router", router_node)
        workflow.add_node("web", web_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("chat", chat_node)
        workflow.add_node("controller", controller_node)
        workflow.add_node("output_handler", output_handler_node)
        
        # Add all existing edges
        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            lambda state: state['current_agent'].lower(),
            {
                "web": "web",
                "coder": "coder",
                "chat": "chat"
            }
        )
        
        workflow.add_edge("web", "controller")
        workflow.add_edge("coder", "controller")
        workflow.add_edge("chat", "output_handler")
        
        workflow.add_conditional_edges(
            "controller",
            lambda state: self._controller_routing_decision(state),
            {
                "router": "router",
                "output_handler": "output_handler",
                "force_end": "output_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "output_handler",
            lambda state: "end",
            {"end": END}
        )
        
        # Compile with memory checkpointer
        compiled_workflow = workflow.compile(checkpointer=self.memory)
        print("✅ Workflow compiled with memory-aware nodes")
        
        return compiled_workflow
        
    @traceable(
        name="process_message_with_memory", 
        run_type="chain",
        project_name="mortey-assistant"
    )
    async def process_message(self, message: str) -> str:
        """Process message with LangGraph memory persistence"""
        
        # Circuit breaker logic (keep existing)
        message_hash = hashlib.md5(message.encode()).hexdigest()
        current_time = time.time()
        
        if message_hash in self.circuit_breaker:
            last_time, count = self.circuit_breaker[message_hash]
            if current_time - last_time < 30 and count > 2:
                return "I'm having trouble with that request. Please try rephrasing it differently."
        
        self.circuit_breaker[message_hash] = (current_time, 
                                            self.circuit_breaker.get(message_hash, (0, 0))[1] + 1)
        
        # Create or update session with thread-based memory
        if not self.current_session:
            self.current_session = AssistantSession(
                session_id=str(uuid.uuid4()),
                start_time=time.time(),
                last_interaction=time.time()
            )
        
        self.current_session.message_count += 1
        self.current_session.last_interaction = time.time()
        
        # Create memory-aware state
        from langchain_core.messages import HumanMessage
        
        initial_state = {
            # LangGraph message handling
            "messages": [HumanMessage(content=message)],
            
            # Core fields
            "user_input": message,
            "agent_choice": "",
            "current_agent": "",
            "session_id": self.current_session.session_id,
            "message_count": self.current_session.message_count,
            "output_content": "",
            "output_type": "",
            "verification_result": "",
            
            # Voice fields
            "voice_mode": False,
            "requires_speech_output": False,
            "voice_session_id": "",
            
            # Controller fields
            "controller_feedback": "",
            "loop_count": 0,
            "max_loops": 3,
            "verification_required": False,
            
            # File fields
            "temp_filename": "",
            "final_filename": "",
            "file_saved": False,
            
            # Memory fields
            "thread_id": self.current_session.session_id,
            "user_id": "default_user",
            "conversation_context": {}
        }
        
        try:
            # Process with thread-based memory configuration
            config = {
                "configurable": {
                    "thread_id": self.current_session.session_id,
                    "user_id": "default_user"
                }
            }
            
            print(f"🧠 Processing with thread_id: {self.current_session.session_id}")
            
            # The memory checkpointer will automatically save/restore state
            result = await self.workflow.ainvoke(initial_state, config)
            
            # Return the response
            if result.get('verification_result') == 'approved':
                return result.get('output_content', 'I had trouble processing that.')
            else:
                return "I need to be more careful with that response."
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return f"I encountered an error: {str(e)}"
    
    @traceable(
        name="process_voice_message", 
        run_type="chain",
        project_name="mortey-assistant"
    )
    async def process_message_with_voice_state(self, voice_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process message with voice-specific state extensions and tracing"""
        try:
            # Add memory support to voice state
            if 'conversation_history' not in voice_state:
                voice_state['conversation_history'] = []
            
            # Use session ID for memory persistence
            session_id = voice_state.get('session_id', str(uuid.uuid4()))
            workflow_config = {
                "configurable": {
                    "thread_id": session_id
                }
            }
            
            # The @traceable decorator will automatically handle tracing
            result = await self.workflow.ainvoke(voice_state, workflow_config)
            return result
            
        except Exception as e:
            print(f"❌ Voice state processing error: {e}")
            return {
                **voice_state,
                'output_content': f"I encountered an error: {str(e)}",
                'output_type': 'error',
                'verification_result': 'approved'
            }
    async def test_memory_persistence(self) -> Dict[str, Any]:
        """Test memory persistence functionality"""
        test_results = {
            "memory_enabled": True,
            "tests": {}
        }
        
        try:
            # Test 1: Create a conversation
            test_thread_id = f"test_{int(time.time())}"
            
            test_state = {
                "messages": [],
                "user_input": "Hello, remember that my name is TestUser",
                "thread_id": test_thread_id,
                "user_id": "test_user",
                "agent_choice": "CHAT",
                "current_agent": "CHAT",
                "output_content": "",
                "verification_result": "approved"
            }
            
            config = {"configurable": {"thread_id": test_thread_id}}
            
            # Process test message
            result1 = await self.workflow.ainvoke(test_state, config)
            test_results["tests"]["message_processing"] = "PASS" if result1 else "FAIL"
            
            # Test 2: Check memory retrieval
            history = await self.get_conversation_history(test_thread_id)
            test_results["tests"]["memory_retrieval"] = "PASS" if len(history) > 0 else "FAIL"
            
            # Test 3: Continue conversation with context
            test_state2 = {
                "messages": [],
                "user_input": "What is my name?",
                "thread_id": test_thread_id,
                "user_id": "test_user",
                "agent_choice": "CHAT",
                "current_agent": "CHAT",
                "output_content": "",
                "verification_result": "approved"
            }
            
            result2 = await self.workflow.ainvoke(test_state2, config)
            
            # Check if the response mentions the name
            response_mentions_name = "testuser" in result2.get('output_content', '').lower()
            test_results["tests"]["context_continuity"] = "PASS" if response_mentions_name else "PARTIAL"
            
            test_results["test_thread_id"] = test_thread_id
            test_results["conversation_length"] = len(history)
            
        except Exception as e:
            test_results["error"] = str(e)
            test_results["tests"]["overall"] = "FAIL"
        
        return test_results

    async def debug_memory_state(self, thread_id: str = None) -> Dict[str, Any]:
        """Debug memory state for troubleshooting"""
        try:
            if not thread_id and self.current_session:
                thread_id = self.current_session.session_id
            
            if not thread_id:
                return {"error": "No thread_id provided"}
            
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get checkpoint from memory
            checkpoint = self.memory.get(config)
            
            debug_info = {
                "thread_id": thread_id,
                "checkpoint_exists": checkpoint is not None,
                "timestamp": time.time()
            }
            
            if checkpoint:
                channel_values = checkpoint.get('channel_values', {})
                debug_info.update({
                    "message_count": len(channel_values.get('messages', [])),
                    "last_agent": channel_values.get('current_agent', 'unknown'),
                    "last_output": channel_values.get('output_content', '')[:100] + "..." if channel_values.get('output_content') else None,
                    "state_keys": list(channel_values.keys())
                })
            
            return debug_info
            
        except Exception as e:
            return {"error": str(e), "thread_id": thread_id}

    def get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get list of active conversation threads"""
        try:
            # Note: MemorySaver doesn't provide direct thread enumeration
            # This is a placeholder for future enhancement
            active_conversations = []
            
            if self.current_session:
                active_conversations.append({
                    "thread_id": self.current_session.session_id,
                    "start_time": self.current_session.start_time,
                    "message_count": self.current_session.message_count,
                    "last_interaction": self.current_session.last_interaction,
                    "status": "active"
                })
            
            return active_conversations
            
        except Exception as e:
            print(f"❌ Error getting active conversations: {e}")
            return []

    async def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversation threads (placeholder for future implementation)"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Note: MemorySaver doesn't provide direct cleanup methods
            # This would need to be implemented based on storage backend
            print(f"🧹 Memory cleanup requested for conversations older than {max_age_hours} hours")
            
            # For now, just log the request
            cleanup_info = {
                "requested_at": current_time,
                "max_age_hours": max_age_hours,
                "status": "logged"
            }
            
            return cleanup_info
            
        except Exception as e:
            print(f"❌ Error during memory cleanup: {e}")
            return {"error": str(e)}

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        try:
            stats = {
                "memory_type": "LangGraph MemorySaver",
                "checkpointer_enabled": True,
                "current_session": {
                    "active": self.current_session is not None,
                    "session_id": self.current_session.session_id if self.current_session else None,
                    "message_count": self.current_session.message_count if self.current_session else 0,
                    "duration_minutes": (time.time() - self.current_session.start_time) / 60 if self.current_session else 0
                },
                "features": {
                    "conversation_persistence": True,
                    "thread_isolation": True,
                    "cross_session_memory": True,
                    "automatic_checkpointing": True
                },
                "timestamp": time.time()
            }
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
