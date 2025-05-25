import asyncio
import os
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

from langgraph.graph import StateGraph, START, END
import ollama
from anthropic import Anthropic

from services.speech.speech_manager import SpeechManager, VoiceSettings
from agents.web_agent import WebAgent  
from core.controller import ControllerAgent
from agents.coder_agent import CoderAgent
from agents.chat_agent import ChatAgent

from dotenv import load_dotenv
load_dotenv()
from config.settings import config

class AssistantState(Enum):
    SLEEPING = "sleeping"
    LISTENING = "listening" 
    PROCESSING = "processing"
    RESPONDING = "responding"

@dataclass
class ConversationSession:
    session_id: str
    start_time: float
    conversation_count: int = 0
    last_interaction: float = 0

# Your existing LangGraph state structure
class VoiceAssistantState(dict):
    """State for voice assistant using your existing LangGraph structure"""
    user_input: str
    agent_choice: str
    current_agent: str
    session_id: str
    conversation_count: int
    loop_count: int
    max_loops: int
    verification_required: bool
    verification_result: str
    
    # Agent outputs
    web_results: list
    output_content: str
    output_type: str
    
    # Thinking state for GUI (future)
    thinking_state: dict

class VoiceAssistant:
    """Complete voice assistant using LangGraph workflow"""
    
    def __init__(self, 
                assistant_name: str = "Assistant",
                wake_word: str = "bumblebee", 
                voice_settings: VoiceSettings = None):
        
        self.assistant_name = assistant_name
        self.wake_word = wake_word
        self.voice_settings = voice_settings or VoiceSettings(rate=180, volume=0.9)
        
        # Initialize components
        self.speech_manager = SpeechManager(self.voice_settings)
        self.web_agent = WebAgent(None)
        self.coder_agent = CoderAgent(None)
        self.chat_agent = ChatAgent(None) 
        self.controller = ControllerAgent(None)

        # No gui
        self.gui_callback = None  
        
        # DEBUG: Add debug prints for workflow initialization
        print("🔧 Building LangGraph workflow...")
        try:
            self.workflow = self._build_langgraph_workflow()
            print(f"✅ Workflow initialized: {type(self.workflow)}")
            print(f"✅ Workflow is None: {self.workflow is None}")
        except Exception as e:
            print(f"❌ Workflow initialization failed: {e}")
            self.workflow = None
        
        # Assistant state
        self.current_state = AssistantState.SLEEPING
        self.session: Optional[ConversationSession] = None
        self.is_active = False
        
        print(f"🤖 {self.assistant_name} Voice Assistant initialized")
        print(f"🐝 Wake word: '{self.wake_word}'")

    def set_gui_callback(self, callback):
        """Set callback function for GUI message updates"""
        self.gui_callback = callback
    
    def _send_to_gui(self, sender: str, message: str):
        """Send message to GUI if callback is set"""
        if self.gui_callback:
            try:
                self.gui_callback(sender, message)
            except Exception as e:
                print(f"❌ GUI callback error: {e}")

    def _build_langgraph_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with proper async handling"""

        assistant_self = self
            
        async def router_node(state: VoiceAssistantState) -> VoiceAssistantState:
            """Route using local LLM with proper async handling"""
            prompt = f"""
            Route this request to the appropriate agent:
            User: {state['user_input']}
            
            Available agents:
            - CODER: Programming, code generation, debugging, technical questions
            - WEB: Web search, current information, news, weather, general knowledge
            - CHAT: General conversation, greetings, personal questions
            
            Respond with only the agent name: CODER, WEB, or CHAT
            """
            
            try:
                # FIXED: Use asyncio.to_thread for blocking ollama call
                response = await asyncio.to_thread(
                    ollama.generate, 
                    model="llama3.2:3b", 
                    prompt=prompt
                )
                agent_choice = response['response'].strip().upper()
                
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
        
        async def web_node(state: VoiceAssistantState) -> VoiceAssistantState:
            """Web agent node - proper async"""
            result = await assistant_self.web_agent.search_and_browse(state)
            return result
        
        async def coder_node(state: VoiceAssistantState) -> VoiceAssistantState:
            """Coder agent node - proper async"""
            result = await assistant_self.coder_agent.generate_code(state)
            return result
        
        async def chat_node(state: VoiceAssistantState) -> VoiceAssistantState:
            """Chat agent node using dedicated ChatAgent with Claude"""
            # Add assistant name to state for the chat agent
            state['assistant_name'] = assistant_self.assistant_name
            
            # Use the dedicated ChatAgent
            result = await assistant_self.chat_agent.chat(state)
            return result
        
        async def controller_node(state: VoiceAssistantState) -> VoiceAssistantState:
            """Controller node - proper async"""
            try:
                result = await assistant_self.controller.verify_output(state)
                return result
            except Exception as e:
                print(f"❌ Controller error: {e}")
                state['verification_result'] = 'approved'
                return state
        
        # Build the StateGraph with async nodes
        workflow = StateGraph(VoiceAssistantState)
        
        # Add async nodes
        workflow.add_node("router", router_node)
        workflow.add_node("web", web_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("chat", chat_node)
        workflow.add_node("controller", controller_node)
        
        # Rest of your workflow setup...
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
            lambda state: "end" if not state.get('verification_required', False) else "router",
            {
                "end": END,
                "router": "router"
            }
        )
        
        return workflow.compile()

    async def start(self):
        """Start the voice assistant"""
        print(f"🚀 Starting {self.assistant_name}...")
        print(f"🐝 Say '{self.wake_word}' to wake me up!")
        
        self.is_active = True
        self._set_state(AssistantState.SLEEPING)
        
        # Start wake word detection
        await self.speech_manager.start_wake_word_detection(
            callback=self._wake_word_detected
        )
        
        try:
            while self.is_active:
                await asyncio.sleep(1)
                
                if self.session and self._is_session_expired():
                    await self._end_session()
                    
        except KeyboardInterrupt:
            print(f"\n🛑 {self.assistant_name} shutting down...")
        finally:
            await self.stop()
    
    async def _wake_word_detected(self, wake_word: str, context: str):
        """Handle wake word detection with interrupt support"""
        
        if context == "interrupt_detection":
            # TTS was interrupted - start listening immediately
            print("🎤 TTS interrupted - starting conversation immediately")
            
            if not self.session:
                # Create new session if needed
                self.session = ConversationSession(
                    session_id=str(uuid.uuid4()),
                    start_time=time.time(),
                    last_interaction=time.time()
                )
            
            # Start conversation without intro message
            await self._start_conversation()
            
        elif self.current_state == AssistantState.SLEEPING:
            # Normal wake word detection
            print(f"🐝 Wake word detected! Mortey is waking up...")
            
            self.session = ConversationSession(
                session_id=str(uuid.uuid4()),
                start_time=time.time(),
                last_interaction=time.time()
            )
            
            await self.speech_manager.speak("Hello!")
            await asyncio.sleep(0.5)
            await self._start_conversation()

    
    async def _start_conversation(self):
        """Start conversation using LangGraph workflow"""
        self._set_state(AssistantState.LISTENING)
        
        user_input = await self.speech_manager.listen_with_vad(max_duration=30)
        
        if user_input and user_input != "timeout":
            await self._process_with_langgraph(user_input)
        else:
            await self.speech_manager.speak(f"I didn't hear anything. Say {self.wake_word} to wake me up again.")
            await self._end_session()
    
    async def _process_with_langgraph(self, user_input: str):
        """Process user input through your LangGraph workflow"""
        self._set_state(AssistantState.PROCESSING)
        self.session.conversation_count += 1
        self.session.last_interaction = time.time()
        
        print(f"🧠 Processing through LangGraph: {user_input}")
        
        initial_state = {
            'user_input': user_input,
            'agent_choice': '',
            'current_agent': '',
            'session_id': self.session.session_id,
            'conversation_count': self.session.conversation_count,
            'loop_count': 0,
            'max_loops': 3,
            'verification_required': False,
            'verification_result': '',
            'web_results': [],
            'output_content': '',
            'output_type': '',
            'thinking_state': {}
        }
        
        try:
            # FIXED: Use ainvoke for async workflow
            result = await self.workflow.ainvoke(initial_state)
            
            await self._respond_to_user(result)
            await self._continue_or_end()
            
        except Exception as e:
            print(f"❌ LangGraph processing error: {e}")
            await self.speech_manager.speak("I encountered an error processing your request.")
            await self._end_session()

    async def _process_speech(self, audio_text: str):
        """Process speech and send updates to GUI"""
        try:
            # Send user input to GUI
            self._send_to_gui("User", audio_text)
            
            # Process through your workflow
            response = await self._process_through_workflow(audio_text)
            
            # Send response to GUI
            self._send_to_gui("Mortey", response)
            
            # Speak the response
            await self.speech_manager.speak(response)
            
        except Exception as e:
            error_msg = f"Error processing speech: {str(e)}"
            self._send_to_gui("System", error_msg)


    async def _respond_to_user(self, state: Dict[str, Any]):
        """Respond to user with speech - wait until done"""
        self._set_state(AssistantState.RESPONDING)
        
        verification = state.get('verification_result', 'approved')
        
        if verification == 'approved':
            response = state['output_content']
            
            # Truncate very long responses for speech
            if len(response) > 500:
                response = response[:500] + "... I can provide more details if you'd like."
            
            # FIXED: Ensure TTS completes before continuing
            await self.speech_manager.speak(response)
            
            # ADDED: Small pause to ensure audio output is completely finished
            await asyncio.sleep(0.5)
        else:
            await self.speech_manager.speak("I need to be more careful with that response.")
            await asyncio.sleep(0.5)

    async def _continue_or_end(self):
        """Improved session management to prevent premature sleep"""
        # Ask if user wants to continue
        await self.speech_manager.speak("Is there anything else I can help you with?")
        await asyncio.sleep(1.0)  # Wait for TTS to finish
        
        # Listen for response with longer timeout
        response = await self.speech_manager.listen_with_vad(max_duration=15)
        
        if response and response != "timeout":
            # FIXED: Check if it's a new command vs just a continuation response
            response_lower = response.lower()
            
            # If user gives a new command/request, process it instead of ending
            command_indicators = [
                'search', 'find', 'look', 'tell me', 'what', 'how', 'where', 'when', 'why',
                'write', 'create', 'make', 'help', 'show', 'explain', 'restaurant', 'weather'
            ]
            
            continuation_words = ['yes', 'yeah', 'sure', 'okay', 'please']
            
            has_command = any(indicator in response_lower for indicator in command_indicators)
            has_continuation = any(word in response_lower for word in continuation_words)
            
            if has_command or has_continuation:
                # User gave a new command or wants to continue
                if has_command:
                    # Process the new command directly
                    await self._process_with_langgraph(response)
                else:
                    # Continue conversation
                    await self._start_conversation()
            else:
                # User said something like "no" or unclear - end session
                await self.speech_manager.speak("Alright! Say bumblebee anytime you need me.")
                await self._end_session()
        else:
            # No response or timeout - end session
            await self.speech_manager.speak("Alright! Say bumblebee anytime you need me.")
            await self._end_session()

    async def _end_session(self):
        """End conversation session"""
        if self.session:
            duration = time.time() - self.session.start_time
            print(f"📊 Session ended: {self.session.conversation_count} conversations in {duration:.1f}s")
            self.session = None
        
        self._set_state(AssistantState.SLEEPING)
        print(f"😴 {self.assistant_name} is sleeping. Say '{self.wake_word}' to wake me up!")
    
    def _is_session_expired(self) -> bool:
        """Check session expiration with longer timeout"""
        if not self.session:
            return False
        timeout = 600  # 10 minutes instead of 5
        return (time.time() - self.session.last_interaction) > timeout

    
    def _set_state(self, new_state: AssistantState):
        """Update assistant state"""
        if self.current_state != new_state:
            self.current_state = new_state
    
    async def stop(self):
        """Stop the assistant"""
        print(f"🛑 Stopping {self.assistant_name}...")
        self.is_active = False
        
        if self.session:
            await self._end_session()
        
        self.speech_manager.cleanup()
        print(f"✅ {self.assistant_name} stopped successfully")

# Factory function for your Mortey assistant
def create_mortey_assistant() -> VoiceAssistant:
    """Create a Mortey voice assistant instance"""
    return VoiceAssistant(
        assistant_name="Mortey",
        wake_word="bumblebee",
        voice_settings=VoiceSettings(rate=180, volume=0.9)
    )
