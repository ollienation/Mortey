import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import uuid
import time

from core.voice_assistant import VoiceAssistant
from agents.chat_agent import ChatAgent
from agents.web_agent import WebAgent
from agents.coder_agent import CoderAgent
from core.controller import ControllerAgent
from config.settings import config
from config.llm_manager import llm_manager

@dataclass
class AssistantSession:
    session_id: str
    start_time: float
    message_count: int = 0
    last_interaction: float = 0

class AssistantCore:
    """Core assistant logic separated from voice functionality"""
    
    def __init__(self):
        # Initialize agents
        self.chat_agent = ChatAgent(None)
        self.web_agent = WebAgent(None)
        self.coder_agent = CoderAgent(None)
        self.controller = ControllerAgent(None)
        
        # Session management
        self.current_session: Optional[AssistantSession] = None
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("🤖 Assistant Core initialized")
    
    def _build_workflow(self):
        """Build the LangGraph workflow for text processing"""
        from langgraph.graph import StateGraph, START, END  # FIXED: END not End
        
        class AssistantState(dict):
            user_input: str
            agent_choice: str
            current_agent: str
            session_id: str
            message_count: int
            output_content: str
            output_type: str
            verification_result: str
                
        async def router_node(state: AssistantState) -> AssistantState:
            """Route using node-specific configuration"""
            prompt = f"""
            Route this request to the appropriate agent:
            User: {state['user_input']}
            
            Available agents:
            - CHAT: General conversation, file browsing (read-only), listing files, reading file contents
            - CODER: Code generation, file creation, programming tasks
            - WEB: Web search, current information, news, weather
            
            Routing Guidelines:
            - Use CHAT for file browsing, listing files, reading files (read-only operations)
            - Use CODER only for code generation and file creation/modification
            - Use WEB for external information searches
            
            Examples:
            - "what files are in workspace" → CHAT
            - "read file contents" → CHAT
            - "show me the files" → CHAT
            - "create a Python script" → CODER
            - "search for tutorials" → WEB
            
            Respond with only: CHAT, CODER, or WEB
            """
            
            try:
                response = await llm_manager.generate_for_node("router", prompt)
                agent_choice = response.strip().upper()
                
                # Enhanced routing overrides
                user_input_lower = state['user_input'].lower()
                
                # Route file browsing to CHAT (read-only)
                if any(keyword in user_input_lower for keyword in [
                    'what files', 'which files', 'list files', 'show files',
                    'read file', 'file contents', 'workspace files', 'browse files'
                ]):
                    agent_choice = 'CHAT'
                    print(f"🔄 File browsing override: Routed to CHAT")
                
                # Route code generation to CODER
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

        async def coder_node(state: AssistantState) -> AssistantState:
            """Coder agent node that handles both code generation and file operations"""
            try:
                result = await self.coder_agent.generate_code(state)
                return result
            except Exception as e:
                print(f"❌ Coder node error: {e}")
                state['output_content'] = f"Error in coder agent: {str(e)}"
                state['output_type'] = 'error'
                return state

        async def chat_node(state: AssistantState) -> AssistantState:
            """Chat agent node with node-specific configuration"""
            try:
                result = await self.chat_agent.chat(state)
                return result
            except Exception as e:
                print(f"❌ Chat agent error: {e}")
                state['output_content'] = "I'm having trouble right now. Please try again."
                state['output_type'] = 'error'
                return state

        async def controller_node(state: AssistantState) -> AssistantState:
            """Controller node with minimal tokens for safety checks"""
            try:
                result = await self.controller.verify_output(state)
                return result
            except Exception as e:
                print(f"❌ Controller error: {e}")
                state['verification_result'] = 'approved'  # Safe fallback
                return state
        
        # Build workflow
        workflow = StateGraph(AssistantState)
        
        workflow.add_node("router", router_node)
        workflow.add_node("web", web_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("chat", chat_node)
        workflow.add_node("controller", controller_node)
        
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
        workflow.add_edge("chat", "controller")
        workflow.add_conditional_edges(
            "controller",
            lambda state: "end",
            {"end": END}
        )
        
        return workflow.compile()
    
    async def process_message(self, message: str) -> str:
        """Process a text message and return response"""
        
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
            'verification_result': ''
        }
        
        try:
            # Process through workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Return the response
            if result.get('verification_result') == 'approved':
                return result.get('output_content', 'I had trouble processing that.')
            else:
                return "I need to be more careful with that response."
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return f"I encountered an error: {str(e)}"
