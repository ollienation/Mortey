import asyncio
from typing import Optional
from pathlib import Path

from gui.chat_gui import ChatGUI
from core.assistant_core import AssistantCore
from core.voice_assistant import VoiceAssistant
from config.settings import config

class GUIManager:
    """Manages GUI and integrates text + voice functionality"""
    
    def __init__(self):
        self.assistant_core = AssistantCore()
        self.voice_assistant: Optional[VoiceAssistant] = None
        self.gui = ChatGUI(message_callback=self.handle_text_message)
        self.gui.set_voice_callback(self.handle_voice_toggle)
        
        # Welcome message with workspace info
        workspace_path = config.workspace_dir
        welcome_msg = f"Mortey Assistant ready! Workspace: {workspace_path}"
        self.gui.add_message("System", welcome_msg)
    
    async def handle_text_message(self, message: str):
        """Handle text messages from GUI"""
        try:
            # Show processing indicator
            self.gui.add_message("System", "Processing...", "system")
            
            # Process through core assistant
            response = await self.assistant_core.process_message(message)
            
            # Display response
            self.gui.add_message("Mortey", response)
            
        except Exception as e:
            self.gui.add_message("System", f"Error: {str(e)}", "system")
    
    async def handle_voice_toggle(self, enable: bool):
        """Handle voice mode toggle"""
        try:
            if enable:
                if not self.voice_assistant:
                    self.voice_assistant = VoiceAssistant(
                        assistant_name="Mortey",
                        wake_word="bumblebee"
                    )
                    self.voice_assistant.set_gui_callback(self.handle_voice_message)
                
                await self.voice_assistant.start()
                self.gui.add_message("System", "Voice mode enabled", "system")
                
            else:
                if self.voice_assistant:
                    await self.voice_assistant.stop()
                    self.gui.add_message("System", "Voice mode disabled", "system")
                    
        except Exception as e:
            self.gui.add_message("System", f"Voice error: {str(e)}", "system")

    def handle_voice_message(self, sender: str, message: str):
        """Handle messages from voice assistant"""
        self.gui.add_message(sender, message)
        
        # Update GUI status based on voice assistant state
        if hasattr(self.voice_assistant, 'current_state'):
            self.gui.update_status(self.voice_assistant.current_state)

        self.start_status_monitoring()

    def start_status_monitoring(self):
        """Start monitoring agent status"""
        def update_status():
            from core.agent_monitor import agent_monitor
            status = agent_monitor.get_status_summary()
            
            # Update GUI status (you'll need to add this to your ChatGUI)
            if hasattr(self.gui, 'update_agent_status'):
                self.gui.update_agent_status(status)
            
            # Schedule next update
            self.gui.root.after(1000, update_status)  # Update every second
        
        update_status()
    
    def run(self):
        """Start the GUI"""
        self.gui.run()
