import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
import threading
from typing import Optional, Callable
from datetime import datetime
import queue
import os
from pathlib import Path

# Import our configuration
try:
    from config.settings import config
except ImportError:
    # Fallback for standalone testing
    config = None

class ChatGUI:
    """Enhanced chat GUI with voice toggle and configurable settings"""
    
    def __init__(self, message_callback: Optional[Callable] = None, gui_config: dict = None):
        self.message_callback = message_callback
        self.voice_callback: Optional[Callable] = None
        self.message_queue = queue.Queue()
        self.voice_enabled = False
        
        # Load GUI configuration
        self.gui_config = self._load_gui_config(gui_config)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.gui_config['window_title'])
        self.root.geometry(self.gui_config['window_size'])
        self.root.configure(bg=self.gui_config['bg_color'])
        
        # Set window icon if available
        self._set_window_icon()
        
        self.setup_styles()
        self.create_widgets()
        self.process_messages()
        
    def _load_gui_config(self, custom_config: dict = None) -> dict:
        """Load GUI configuration with fallbacks"""
        default_config = {
            'window_title': 'Mortey Assistant',
            'window_size': '900x700',
            'bg_color': '#2b2b2b',
            'chat_bg': '#1e1e1e',
            'input_bg': '#3c3c3c',
            'text_color': '#ffffff',
            'accent_color': '#0d7377',
            'voice_color': '#ff6b6b',
            'voice_active_color': '#4CAF50',
            'font_family': 'Arial',
            'font_size': 11,
            'title_font_size': 16,
            'wake_word': 'Bumblebee'
        }
        
        # Override with environment variables if available
        if config:
            env_overrides = {
                'wake_word': os.getenv('MORTEY_WAKE_WORD', default_config['wake_word']),
                'window_title': os.getenv('MORTEY_WINDOW_TITLE', default_config['window_title']),
                'font_family': os.getenv('MORTEY_FONT_FAMILY', default_config['font_family'])
            }
            default_config.update(env_overrides)
        
        # Override with custom config if provided
        if custom_config:
            default_config.update(custom_config)
            
        return default_config
    
    def _set_window_icon(self):
        """Set window icon if available"""
        if config:
            icon_path = config.project_root / "assets" / "mortey_icon.ico"
            if icon_path.exists():
                try:
                    self.root.iconbitmap(str(icon_path))
                except tk.TclError:
                    pass  # Icon format not supported, continue without icon
    
    def setup_styles(self):
        """Configure modern dark theme styles with configurable colors"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Use configuration colors
        style.configure('Chat.TFrame', background=self.gui_config['bg_color'])
        style.configure('Input.TFrame', background=self.gui_config['input_bg'])
        style.configure('Send.TButton', 
                       background=self.gui_config['accent_color'],
                       foreground=self.gui_config['text_color'],
                       borderwidth=0,
                       focuscolor='none')
        style.configure('Voice.TButton', 
                       background=self.gui_config['voice_color'],
                       foreground=self.gui_config['text_color'],
                       borderwidth=0,
                       focuscolor='none')
        style.configure('VoiceOn.TButton', 
                       background=self.gui_config['voice_active_color'],
                       foreground=self.gui_config['text_color'],
                       borderwidth=0,
                       focuscolor='none')
        
    def create_widgets(self):
        """Create and layout GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Chat.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header frame
        header_frame = tk.Frame(main_frame, bg=self.gui_config['bg_color'])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title with configuration
        title_text = f"🤖 {self.gui_config['window_title']}"
        title_label = tk.Label(header_frame, 
                              text=title_text, 
                              font=(self.gui_config['font_family'], 
                                   self.gui_config['title_font_size'], 'bold'),
                              bg=self.gui_config['bg_color'], 
                              fg=self.gui_config['text_color'])
        title_label.pack(side=tk.LEFT)
        
        # Voice toggle button
        self.voice_button = ttk.Button(
            header_frame,
            text="🎤 Enable Voice",
            style='Voice.TButton',
            command=self.toggle_voice
        )
        self.voice_button.pack(side=tk.RIGHT)
        
        # Status indicator with wake word
        wake_word = self.gui_config['wake_word']
        self.status_label = tk.Label(main_frame,
                                   text="💬 Text Mode - Voice Disabled",
                                   font=(self.gui_config['font_family'], 10),
                                   bg=self.gui_config['bg_color'],
                                   fg='#888888')
        self.status_label.pack(pady=(0, 10))
        
        # Chat display area
        chat_frame = tk.Frame(main_frame, bg=self.gui_config['bg_color'])
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=(self.gui_config['font_family'], self.gui_config['font_size']),
            bg=self.gui_config['chat_bg'],
            fg=self.gui_config['text_color'],
            insertbackground=self.gui_config['text_color'],
            selectbackground=self.gui_config['accent_color'],
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Input area
        input_frame = ttk.Frame(main_frame, style='Input.TFrame')
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Message input
        self.message_entry = tk.Entry(
            input_frame,
            font=(self.gui_config['font_family'], 12),
            bg=self.gui_config['input_bg'],
            fg=self.gui_config['text_color'],
            insertbackground=self.gui_config['text_color'],
            relief=tk.FLAT,
            bd=5
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.message_entry.bind('<Return>', self.send_message)
        
        # Send button
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            style='Send.TButton',
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Configure chat display tags with configurable colors
        self.chat_display.tag_configure("user", 
                                       foreground=self.gui_config['voice_active_color'], 
                                       font=(self.gui_config['font_family'], 
                                            self.gui_config['font_size'], 'bold'))
        self.chat_display.tag_configure("mortey", 
                                       foreground="#2196F3", 
                                       font=(self.gui_config['font_family'], 
                                            self.gui_config['font_size'], 'bold'))
        self.chat_display.tag_configure("system", 
                                       foreground="#FF9800", 
                                       font=(self.gui_config['font_family'], 
                                            self.gui_config['font_size'] - 1, 'italic'))
        self.chat_display.tag_configure("timestamp", 
                                       foreground="#666666", 
                                       font=(self.gui_config['font_family'], 
                                            self.gui_config['font_size'] - 2))
    
    def toggle_voice(self):
        """Toggle voice functionality"""
        self.voice_enabled = not self.voice_enabled
        wake_word = self.gui_config['wake_word']
        
        if self.voice_enabled:
            self.voice_button.config(text="🔇 Disable Voice", style='VoiceOn.TButton')
            status_text = f"🎤 Voice Mode - Say '{wake_word}' to activate"
            self.status_label.config(text=status_text, fg=self.gui_config['voice_active_color'])
            
            # Call voice callback (sync, not async)
            if self.voice_callback:
                self.voice_callback(True)
        else:
            self.voice_button.config(text="🎤 Enable Voice", style='Voice.TButton')
            self.status_label.config(text="💬 Text Mode - Voice Disabled", fg="#888888")
            
            # Call voice callback (sync, not async)
            if self.voice_callback:
                self.voice_callback(False)
    
    def set_voice_callback(self, callback: Callable):
        """Set callback for voice toggle"""
        self.voice_callback = callback
    
    def add_message(self, sender: str, message: str, message_type: str = "normal"):
        """Add a message to the chat display"""
        self.message_queue.put((sender, message, message_type))
    
    def process_messages(self):
        """Process messages from the queue"""
        try:
            while True:
                sender, message, message_type = self.message_queue.get_nowait()
                self._display_message(sender, message, message_type)
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_messages)
    
    def _display_message(self, sender: str, message: str, message_type: str):
        """Display a message in the chat area"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        if sender.lower() == "user":
            self.chat_display.insert(tk.END, "You: ", "user")
        elif sender.lower() == "mortey":
            self.chat_display.insert(tk.END, f"{self.gui_config['window_title']}: ", "mortey")
        else:
            self.chat_display.insert(tk.END, f"{sender}: ", "system")
        
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        """Send a message from the input field"""
        message = self.message_entry.get().strip()
        if message:
            self.add_message("User", message)
            self.message_entry.delete(0, tk.END)
            
            # Call message callback (sync, let gui_manager handle async)
            if self.message_callback:
                self.message_callback(message)
    
    def update_status(self, status: str, color: str = "#888888"):
        """Update the status indicator"""
        if not self.voice_enabled:
            return  # Don't update status if voice is disabled
        self.status_label.config(text=status, fg=color)
    
    def save_chat_log(self):
        """Save chat log to file"""
        if config:
            log_file = config.logs_dir / f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    chat_content = self.chat_display.get(1.0, tk.END)
                    f.write(chat_content)
                self.add_message("System", f"Chat log saved to {log_file.name}", "system")
            except Exception as e:
                self.add_message("System", f"Failed to save chat log: {str(e)}", "system")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

# For standalone testing
if __name__ == "__main__":
    def dummy_callback(message):
        import time
        time.sleep(1)  # Simulate processing
        return f"Echo: {message}"
    
    gui = ChatGUI(message_callback=dummy_callback)
    gui.run()
