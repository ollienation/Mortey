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
        
        # Build workflow (same as voice assistant but without speech)
        self.workflow = self._build_workflow()
        
        print("🤖 Assistant Core initialized")
    
    def _build_workflow(self):
        """Build the same LangGraph workflow but for text processing"""
        from langgraph.graph import StateGraph, START, End
        
        # Same workflow as voice assistant
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
            """Route using configured provider and model"""
            prompt = f"""
            Route this request to the appropriate agent:
            User: {state['user_input']}
            
            Available agents:
            - WEB: Information lookup, repository browsing, current data, research
            - CODER: Code generation, technical implementation, debugging
            - CHAT: General conversation, explanations, greetings
            
            Respond with only: WEB, CODER, or CHAT
            """
            
            try:
                # Use LLM manager with routing task
                response = await llm_manager.generate(
                    task="routing",
                    prompt=prompt,
                    max_tokens=10
                )
                
                agent_choice = response.strip().upper()
                
                # Keyword-based overrides
                user_input_lower = state['user_input'].lower()
                if any(keyword in user_input_lower for keyword in [
                    'look at', 'check', 'find', 'search', 'browse', 'repository', 'repo'
                ]):
                    agent_choice = 'WEB'
                
                if agent_choice not in ['CODER', 'WEB', 'CHAT']:
                    agent_choice = 'CHAT'
                
                state['agent_choice'] = agent_choice
                state['current_agent'] = agent_choice
                
                # Get current provider info for logging
                provider = config.get_default_provider_for_task("routing")
                model = config.get_provider_config(provider).model
                print(f"🎯 Routed to: {agent_choice} (using {provider}/{model})")
                
            except Exception as e:
                print(f"❌ Routing error: {e}")
                # Smart fallback logic
                user_input_lower = state['user_input'].lower()
                if any(keyword in user_input_lower for keyword in ['look', 'find', 'search', 'repo']):
                    agent_choice = 'WEB'
                elif any(keyword in user_input_lower for keyword in ['code', 'function', 'script']):
                    agent_choice = 'CODER'
                else:
                    agent_choice = 'CHAT'
                
                state['agent_choice'] = agent_choice
                state['current_agent'] = agent_choice
            
            return state
        
        async def chat_node(state: AssistantState) -> AssistantState:
            """Chat agent node"""
            state['assistant_name'] = "Mortey"
            result = await self.chat_agent.chat(state)
            return result
        
        async def controller_node(state: AssistantState) -> AssistantState:
            """Controller node"""
            try:
                result = await self.controller.verify_output(state)
                return result
            except Exception as e:
                print(f"❌ Controller error: {e}")
                state['verification_result'] = 'approved'
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
