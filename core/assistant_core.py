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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

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
        
        # Initialize memory
        self.memory = MemorySaver()
        
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
        """Build the LangGraph workflow with LangSmith tracing"""
        
        class AssistantState(dict):
            user_input: str
            agent_choice: str
            current_agent: str
            session_id: str
            message_count: int
            output_content: str
            output_type: str
            verification_result: str
            
            # Voice-specific extensions
            voice_mode: bool = False
            requires_speech_output: bool = False
            voice_session_id: str = ""
            
            # Controller feedback with loop protection
            controller_feedback: str = ""
            loop_count: int = 0
            max_loops: int = 3
            verification_required: bool = False
            
            # File handling
            temp_filename: str = ""
            final_filename: str = ""
            file_saved: bool = False
            
            # Memory support
            conversation_history: list = []
        
        @traceable(name="router_node", run_type="chain")
        async def router_node(state: AssistantState) -> AssistantState:
            """Route using node-specific configuration with loop protection"""
            
            # Increment loop count to track revisions
            loop_count = state.get('loop_count', 0)
            state['loop_count'] = loop_count + 1
            
            if loop_count > 0:
                print(f"🔄 Revision attempt {loop_count}")
            
            prompt = f"""
            Route this request to the appropriate agent:
            User: {state['user_input']}
            
            Available agents:
            - CHAT: General conversation, file browsing (read-only)
            - CODER: Code generation, file creation, programming tasks
            - WEB: Web search, current information, news, weather
            
            Respond with only: CHAT, CODER, or WEB
            """
            
            try:
                response = await llm_manager.generate_for_node("router", prompt)
                agent_choice = response.strip().upper()
                
                # Enhanced routing overrides
                user_input_lower = state['user_input'].lower()
                
                if any(keyword in user_input_lower for keyword in [
                    'what files', 'which files', 'list files', 'show files',
                    'read file', 'file contents', 'workspace files', 'browse files'
                ]):
                    agent_choice = 'CHAT'
                    print(f"🔄 File browsing override: Routed to CHAT")
                
                elif any(keyword in user_input_lower for keyword in [
                    'create', 'generate', 'write code', 'make a script', 'build'
                ]):
                    agent_choice = 'CODER'
                    print(f"🔄 Code generation override: Routed to CODER")
                
                if agent_choice not in ['CODER', 'WEB', 'CHAT']:
                    agent_choice = 'CHAT'
                
                state['agent_choice'] = agent_choice
                state['current_agent'] = agent_choice
                print(f"🎯 Routed to: {agent_choice}")
                
            except Exception as e:
                print(f"❌ Routing error: {e}")
                state['agent_choice'] = 'CHAT'
                state['current_agent'] = 'CHAT'
            
            return state
        
        @traceable(name="web_node", run_type="chain")
        async def web_node(state: AssistantState) -> AssistantState:
            """Web agent node"""
            try:
                result = await self.web_agent.search_and_browse(state)
                return result
            except Exception as e:
                print(f"❌ Web agent error: {e}")
                state['output_content'] = "I had trouble searching for that information."
                state['output_type'] = 'error'
                return state
        
        @traceable(name="coder_node", run_type="chain")
        async def coder_node(state: AssistantState) -> AssistantState:
            """Coder agent node"""
            try:
                result = await self.coder_agent.generate_code(state)
                return result
            except Exception as e:
                print(f"❌ Coder node error: {e}")
                state['output_content'] = f"Error in coder agent: {str(e)}"
                state['output_type'] = 'error'
                return state
        
        @traceable(name="chat_node", run_type="chain")
        async def chat_node(state: AssistantState) -> AssistantState:
            """Chat agent node"""
            try:
                result = await self.chat_agent.chat(state)
                return result
            except Exception as e:
                print(f"❌ Chat agent error: {e}")
                state['output_content'] = "I'm having trouble right now. Please try again."
                state['output_type'] = 'error'
                return state
        
        @traceable(name="controller_node", run_type="chain")
        async def controller_node(state: AssistantState) -> AssistantState:
            """Controller node"""
            try:
                result = await self.controller.verify_output(state)
                return result
            except Exception as e:
                print(f"❌ Controller error: {e}")
                state['verification_result'] = 'approved'
                return state
        
        @traceable(name="output_handler_node", run_type="chain")
        async def output_handler_node(state: AssistantState) -> AssistantState:
            """Handle output based on interaction mode"""
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
            
            return state
        
        # Build workflow
        workflow = StateGraph(AssistantState)
        
        workflow.add_node("router", router_node)
        workflow.add_node("web", web_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("chat", chat_node)
        workflow.add_node("controller", controller_node)
        workflow.add_node("output_handler", output_handler_node)
        
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
        
        # Different paths: chat bypasses controller, others go through it
        workflow.add_edge("web", "controller")
        workflow.add_edge("coder", "controller")
        workflow.add_edge("chat", "output_handler")  # Chat bypasses controller
        
        # Controller routing with loop protection
        workflow.add_conditional_edges(
            "controller",
            lambda state: self._controller_routing_decision(state),
            {
                "router": "router",  # Revision loop
                "output_handler": "output_handler",
                "force_end": "output_handler"  # Emergency exit
            }
        )
        
        # End from output handler
        workflow.add_conditional_edges(
            "output_handler",
            lambda state: "end",
            {"end": END}
        )
        
        # Compile with memory checkpointer
        return workflow.compile(checkpointer=self.memory)
    
    @traceable(
        name="process_message", 
        run_type="chain",
        project_name="mortey-assistant"
    )
    async def process_message(self, message: str) -> str:
        """Process a text message with LangSmith tracing and circuit breaker protection"""
        
        # Circuit breaker logic
        message_hash = hashlib.md5(message.encode()).hexdigest()
        current_time = time.time()
        
        if message_hash in self.circuit_breaker:
            last_time, count = self.circuit_breaker[message_hash]
            if current_time - last_time < 30 and count > 2:
                return "I'm having trouble with that request. Please try rephrasing it differently."
        
        self.circuit_breaker[message_hash] = (current_time, 
                                            self.circuit_breaker.get(message_hash, (0, 0))[1] + 1)
        
        # Create or update session
        if not self.current_session:
            self.current_session = AssistantSession(
                session_id=str(uuid.uuid4()),
                start_time=time.time(),
                last_interaction=time.time()
            )
        
        self.current_session.message_count += 1
        self.current_session.last_interaction = time.time()
        
        # Create state for processing
        initial_state = {
            'user_input': message,
            'agent_choice': '',
            'current_agent': '',
            'session_id': self.current_session.session_id,
            'message_count': self.current_session.message_count,
            'output_content': '',
            'output_type': '',
            'verification_result': '',
            'voice_mode': False,
            'requires_speech_output': False,
            'loop_count': 0,
            'max_loops': 3,
            'conversation_history': []
        }
        
        try:
            # Process through workflow with memory
            workflow_config = {
                "configurable": {
                    "thread_id": self.current_session.session_id
                }
            }
            
            # The @traceable decorator will automatically handle tracing
            result = await self.workflow.ainvoke(initial_state, workflow_config)
            
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
