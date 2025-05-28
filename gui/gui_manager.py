import asyncio
import threading
from typing import Optional
from pathlib import Path
from gui.chat_gui import ChatGUI
from core.assistant_core import AssistantCore
from core.voice_assistant import VoiceAssistant
from config.settings import config
from core.voice_assistant import create_mortey_assistant

class GUIManager:
    """Manages GUI and integrates text + voice functionality"""
    
    def __init__(self):
        self.assistant_core = AssistantCore()
        self.voice_assistant: Optional = None
        
        # Initialize GUI with proper callback
        self.gui = ChatGUI(message_callback=self.handle_text_message)
        self.gui.set_voice_callback(self.handle_voice_toggle)
        
        # Set up assistant core GUI callback for unified message handling
        self.assistant_core.set_gui_callback(self.gui.add_message)
        
        # Welcome message with workspace info
        workspace_path = config.workspace_dir
        welcome_msg = f"Mortey Assistant ready! Workspace: {workspace_path}"
        self.gui.add_message("System", welcome_msg)
        
    def handle_text_message(self, message: str):
        """Handle text messages from GUI (sync wrapper for async)"""
        # Run async processing in thread to avoid blocking GUI
        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._handle_text_message_async(message))
            except Exception as e:
                # Use GUI thread-safe method to show error
                self.gui.root.after(0, lambda: self.gui.add_message("System", f"Error: {str(e)}"))
            finally:
                loop.close()
        
        # Run in background thread
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
    
    async def _handle_text_message_async(self, message: str):
        """Async text message handler"""
        try:
            # Show processing indicator (thread-safe)
            self.gui.root.after(0, lambda: self.gui.add_message("System", "Processing..."))
            
            # Process through core assistant
            response = await self.assistant_core.process_message(message)
            
        except Exception as e:
            # Display error (thread-safe)
            self.gui.root.after(0, lambda: self.gui.add_message("System", f"Error: {str(e)}"))
    
    def handle_voice_toggle(self, enable: bool):
        """Handle voice mode toggle (sync)"""
        if enable:
            self._start_voice_mode()
        else:
            self._stop_voice_mode()
    
    def _start_voice_mode(self):
        """Start voice mode"""
        try:
            if not self.voice_assistant:
                print("🎤 Creating voice assistant...")
                self.voice_assistant = create_mortey_assistant()
            
            # Start voice assistant in background thread
            def run_voice():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.voice_assistant.start())
                except Exception as e:
                    print(f"❌ Voice assistant error: {e}")
                    self.gui.root.after(0, lambda: self.gui.add_message("System", f"Voice error: {str(e)}"))
                finally:
                    loop.close()
            
            voice_thread = threading.Thread(target=run_voice, daemon=True)
            voice_thread.start()
            
            self.gui.add_message("System", "Voice mode enabled")
            
        except Exception as e:
            print(f"❌ Voice start error: {e}")
            self.gui.add_message("System", f"Voice error: {str(e)}")
    
    def _stop_voice_mode(self):
        """Stop voice mode safely with proper cleanup"""
        try:
            if self.voice_assistant:
                # Stop voice assistant gracefully
                def stop_voice():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Give it time to clean up properly
                        loop.run_until_complete(asyncio.wait_for(
                            self.voice_assistant.stop(), 
                            timeout=5.0
                        ))
                    except asyncio.TimeoutError:
                        print("⚠️ Voice assistant stop timeout - forcing cleanup")
                    except Exception as e:
                        print(f"❌ Voice stop error: {e}")
                    finally:
                        loop.close()
                
                stop_thread = threading.Thread(target=stop_voice, daemon=True)
                stop_thread.start()
                
                # Don't wait for thread to complete to avoid GUI blocking
                self.gui.add_message("System", "Voice mode disabled")
                
        except Exception as e:
            print(f"❌ Voice stop error: {e}")
            self.gui.add_message("System", f"Voice error: {str(e)}")

    def handle_voice_message(self, sender: str, message: str):
        """Handle messages from voice assistant (thread-safe)"""
        # Use thread-safe GUI update
        self.gui.root.after(0, lambda: self.gui.add_message(sender, message))

    def get_status(self) -> dict:
        """Get current system status for debugging"""
        return {
            'gui_initialized': self.gui is not None,
            'voice_assistant_created': self.voice_assistant is not None,
            'voice_active': getattr(self.voice_assistant, 'is_active', False) if self.voice_assistant else False,
            'assistant_core_ready': self.assistant_core is not None
        }
    
    def run(self):
        """Start the GUI"""
        try:
            print("🚀 Starting GUI...")
            self.gui.run()
        except Exception as e:
            print(f"❌ GUI run error: {e}")
            raise
