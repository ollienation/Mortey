import asyncio
import os
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
from services.speech.speech_manager import SpeechManager, VoiceSettings
from core.assistant_core import AssistantCore
from config.settings import config

class VoiceState(Enum):
    SLEEPING = "sleeping"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"

@dataclass
class VoiceSession:
    session_id: str
    start_time: float
    conversation_count: int = 0
    last_interaction: float = 0

class VoiceAssistant:
    """Voice assistant addon that layers on top of AssistantCore with singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self,
                 assistant_name: str = "Mortey",
                 wake_word: str = "bumblebee",
                 voice_settings: VoiceSettings = None):
        
        # Only initialize once (singleton pattern)
        if self._initialized:
            print(f"🔄 {assistant_name} Voice Assistant already initialized (reusing instance)")
            return
        
        self.assistant_name = assistant_name
        self.wake_word = wake_word
        self.voice_settings = voice_settings or VoiceSettings(rate=180, volume=0.9)
        
        # Initialize speech manager
        self.speech_manager = SpeechManager(self.voice_settings)
        
        # Use existing AssistantCore instead of duplicating logic
        self.core = AssistantCore()
        
        # Voice-specific state
        self.current_state = VoiceState.SLEEPING
        self.session: Optional[VoiceSession] = None
        self.is_active = False
        
        # GUI callback support
        self.gui_callback = None
        
        self._initialized = True
        print(f"🤖 {self.assistant_name} Voice Assistant initialized (singleton)")
        print(f"🐝 Wake word: '{self.wake_word}'")
        print(f"🔗 Using unified AssistantCore workflow")
    
    def set_gui_callback(self, callback):
        """Set callback function for GUI message updates"""
        self.gui_callback = callback
        # Pass callback to core for unified handling
        if hasattr(self.core, 'set_gui_callback'):
            self.core.set_gui_callback(callback)
    
    def _send_to_gui(self, sender: str, message: str):
        """Send message to GUI if callback is set"""
        if self.gui_callback:
            try:
                self.gui_callback(sender, message)
            except Exception as e:
                print(f"❌ GUI callback error: {e}")
    
    async def start(self):
        """Start the voice assistant (safe for multiple calls)"""
        if self.is_active:
            print(f"🎤 {self.assistant_name} is already active")
            return
            
        print(f"🚀 Starting {self.assistant_name}...")
        print(f"🐝 Say '{self.wake_word}' to wake me up!")
        self.is_active = True
        self._set_state(VoiceState.SLEEPING)
        
        try:
            # Start wake word detection
            await self.speech_manager.start_wake_word_detection(
                callback=self._wake_word_detected
            )
            
            # Main loop
            while self.is_active:
                await asyncio.sleep(1)
                if self.session and self._is_session_expired():
                    await self._end_session()
                    
        except KeyboardInterrupt:
            print(f"\n🛑 {self.assistant_name} shutting down...")
        except Exception as e:
            print(f"❌ Voice assistant error: {e}")
        finally:
            await self.stop()
    
    async def _wake_word_detected(self, wake_word: str, context: str):
        """Handle wake word detection"""
        try:
            if context == "interrupt_detection":
                print("🎤 TTS interrupted - starting conversation immediately")
                if not self.session:
                    self.session = VoiceSession(
                        session_id=str(uuid.uuid4()),
                        start_time=time.time(),
                        last_interaction=time.time()
                    )
                await self._start_conversation()
            elif self.current_state == VoiceState.SLEEPING:
                print(f"🐝 Wake word detected! {self.assistant_name} is waking up...")
                self.session = VoiceSession(
                    session_id=str(uuid.uuid4()),
                    start_time=time.time(),
                    last_interaction=time.time()
                )
                await self.speech_manager.speak("Hello!")
                await asyncio.sleep(0.5)
                await self._start_conversation()
        except Exception as e:
            print(f"❌ Wake word handling error: {e}")
    
    async def _start_conversation(self):
        """Start conversation using core workflow"""
        try:
            self._set_state(VoiceState.LISTENING)
            user_input = await self.speech_manager.listen_with_vad(max_duration=30)
            
            if user_input and user_input != "timeout":
                await self._process_with_core(user_input)
            else:
                await self.speech_manager.speak(f"I didn't hear anything. Say {self.wake_word} to wake me up again.")
                await self._end_session()
        except Exception as e:
            print(f"❌ Conversation start error: {e}")
            await self._end_session()
    
    async def _process_with_core(self, user_input: str):
        """Process user input through core workflow with voice extensions"""
        try:
            self._set_state(VoiceState.PROCESSING)
            self.session.conversation_count += 1
            self.session.last_interaction = time.time()
            
            print(f"🧠 Processing through core workflow: {user_input}")
            
            # Send user input to GUI (this ensures voice input shows up)
            if self.gui_callback:
                self.gui_callback("User", user_input)
            
            # Create voice-enhanced state
            voice_state = self._create_voice_state(user_input)
            
            # Process through core workflow
            result = await self.core.process_message_with_voice_state(voice_state)
            
            # Handle voice-specific output
            await self._handle_voice_output(result)
            
            # Continue conversation management
            await self._continue_or_end()
            
        except Exception as e:
            print(f"❌ Core workflow processing error: {e}")
            if self.gui_callback:
                self.gui_callback("System", f"Voice error: {str(e)}")
            await self.speech_manager.speak("I encountered an error processing your request.")
            await self._end_session()
    
    def _create_voice_state(self, user_input: str) -> Dict[str, Any]:
        """Create state with voice-specific extensions"""
        return {
            'user_input': user_input,
            'voice_mode': True,
            'requires_speech_output': True,
            'voice_session_id': self.session.session_id,
            'session_id': self.session.session_id,
            'message_count': self.session.conversation_count,
            'assistant_name': self.assistant_name,
            # Core workflow fields
            'agent_choice': '',
            'current_agent': '',
            'loop_count': 0,
            'max_loops': 3,
            'verification_required': False,
            'verification_result': '',
            'output_content': '',
            'output_type': '',
            'thinking_state': {},
            'temp_filename': '',
            'final_filename': '',
            'file_saved': False
        }
    
    async def _handle_voice_output(self, result: Dict[str, Any]):
        """Handle voice-specific output processing"""
        try:
            self._set_state(VoiceState.RESPONDING)
            
            output_content = result.get('output_content', 'I had trouble processing that.')
            verification = result.get('verification_result', 'approved')
            
            # Send to GUI (this was missing or not working)
            if self.gui_callback:
                self.gui_callback("Mortey", output_content)
            
            # Handle speech output
            if result.get('requires_speech_output', True):
                if verification == 'approved':
                    speech_content = output_content
                    if len(speech_content) > 500:
                        speech_content = speech_content[:500] + "... I can provide more details if you'd like."
                    
                    await self.speech_manager.speak(speech_content)
                    await asyncio.sleep(0.5)
                else:
                    await self.speech_manager.speak("I need to be more careful with that response.")
                    await asyncio.sleep(0.5)
        except Exception as e:
            print(f"❌ Voice output handling error: {e}")

    async def _continue_or_end(self):
        """Improved session management"""
        try:
            await self.speech_manager.speak("Is there anything else I can help you with?")
            await asyncio.sleep(1.0)
            
            response = await self.speech_manager.listen_with_vad(max_duration=15)
            
            if response and response != "timeout":
                response_lower = response.lower()
                
                # Check for new commands vs continuation
                command_indicators = [
                    'search', 'find', 'look', 'tell me', 'what', 'how', 'where', 'when', 'why',
                    'write', 'create', 'make', 'help', 'show', 'explain', 'list', 'read'
                ]
                continuation_words = ['yes', 'yeah', 'sure', 'okay', 'please']
                
                has_command = any(indicator in response_lower for indicator in command_indicators)
                has_continuation = any(word in response_lower for word in continuation_words)
                
                if has_command or has_continuation:
                    if has_command:
                        await self._process_with_core(response)
                    else:
                        await self._start_conversation()
                else:
                    await self.speech_manager.speak("Alright! Say bumblebee anytime you need me.")
                    await self._end_session()
            else:
                await self.speech_manager.speak("Alright! Say bumblebee anytime you need me.")
                await self._end_session()
        except Exception as e:
            print(f"❌ Session continuation error: {e}")
            await self._end_session()
    
    async def _end_session(self):
        """End conversation session"""
        try:
            if self.session:
                duration = time.time() - self.session.start_time
                print(f"📊 Session ended: {self.session.conversation_count} conversations in {duration:.1f}s")
                self.session = None
            
            self._set_state(VoiceState.SLEEPING)
            print(f"😴 {self.assistant_name} is sleeping. Say '{self.wake_word}' to wake me up!")
        except Exception as e:
            print(f"❌ Session end error: {e}")
    
    def _is_session_expired(self) -> bool:
        """Check session expiration"""
        if not self.session:
            return False
        timeout = 600  # 10 minutes
        return (time.time() - self.session.last_interaction) > timeout
    
    def _set_state(self, new_state: VoiceState):
        """Update voice assistant state"""
        if self.current_state != new_state:
            self.current_state = new_state
    
    async def stop(self):
        """Stop the assistant safely"""
        if not self.is_active:
            return
            
        print(f"🛑 Stopping {self.assistant_name}...")
        self.is_active = False
        
        try:
            if self.session:
                await self._end_session()
            self.speech_manager.cleanup()
        except Exception as e:
            print(f"❌ Stop error: {e}")
        
        print(f"✅ {self.assistant_name} stopped successfully")

# Factory function for Mortey assistant
def create_mortey_assistant() -> VoiceAssistant:
    """Create a Mortey voice assistant instance (singleton)"""
    return VoiceAssistant(
        assistant_name="Mortey",
        wake_word="bumblebee",
        voice_settings=VoiceSettings(rate=180, volume=0.9)
    )
