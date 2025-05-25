import sys
sys.path.append('src')

from gui.chat_gui import ChatGUI
import asyncio

async def test_message_handler(message: str):
    """Test message handler"""
    print(f"Received message: {message}")
    # Simulate processing delay
    await asyncio.sleep(1)
    return f"Echo: {message}"

def main():
    # Create GUI with test handler
    gui = ChatGUI(message_callback=test_message_handler)
    
    # Add some test messages
    gui.add_message("System", "GUI initialized successfully!")
    gui.add_message("Mortey", "Hello! I'm ready to chat.")
    
    # Start GUI
    gui.run()

if __name__ == "__main__":
    main()