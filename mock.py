from fasthtml.common import *
import asyncio
import json
import uuid
from core.assistant_core import AssistantCore
from core.state import validate_and_filter_messages_v2

# ✅ CRITICAL FIX: Create a single global assistant instance
assistant = None
current_thread_id = None

async def get_assistant():
    """Get or create the assistant instance"""
    global assistant
    if assistant is None:
        assistant = AssistantCore()
        await assistant.initialize()
    return assistant

# Initialize FastHTML app
app, rt = fast_app()

@rt("/")
async def get():
    """Main chat interface"""
    global current_thread_id
    # ✅ CRITICAL FIX: Create a NEW thread ID each time to avoid corrupted history
    current_thread_id = str(uuid.uuid4())
    
    return Titled("Mortey Assistant",
        Div(
            H1("🤖 Mortey Assistant"),
            Div(
                P("New conversation started!", style="color: #666; font-style: italic; margin: 10px 0;"),
                Button("🔄 New Conversation", 
                       hx_get="/new-session", 
                       hx_target="#chat-messages",
                       hx_swap="innerHTML",
                       style="margin-bottom: 10px; padding: 5px 10px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;"),
                id="chat-messages", 
                cls="chat-container"
            ),
            Form(
                Input(
                    id="message-input", 
                    placeholder="Type your message...", 
                    name="message",
                    required=True,
                    autocomplete="off",
                    minlength="1"  # ✅ Prevent empty submissions
                ),
                Button("Send", type="submit"),
                Hidden(name="thread_id", value=current_thread_id),
                hx_post="/chat",
                hx_target="#chat-messages",
                hx_swap="beforeend",
                hx_on_submit="this.reset()"
            ),
            Style("""
                .chat-container { 
                    height: 400px; 
                    overflow-y: auto; 
                    border: 1px solid #ccc; 
                    padding: 10px; 
                    margin: 20px 0; 
                    background: #fafafa;
                }
                .user-message { 
                    background: #e3f2fd; 
                    padding: 10px; 
                    margin: 5px 0; 
                    border-radius: 10px; 
                    border-left: 4px solid #2196f3;
                }
                .assistant-message { 
                    background: #f5f5f5; 
                    padding: 10px; 
                    margin: 5px 0; 
                    border-radius: 10px; 
                    border-left: 4px solid #4caf50;
                }
                .error-message {
                    background: #ffebee;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 10px;
                    border-left: 4px solid #f44336;
                    color: #c62828;
                }
                form {
                    display: flex;
                    gap: 10px;
                    margin-top: 10px;
                }
                #message-input {
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                button {
                    padding: 10px 20px;
                    background: #2196f3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                button:hover {
                    background: #1976d2;
                }
            """)
        )
    )

@rt("/chat")
async def post(message: str, thread_id: str = None):
    """Handle chat messages with enhanced error handling"""
    
    # ✅ CRITICAL FIX: Strict message validation
    if not message or len(message.strip()) < 1:
        return Div("Please enter a valid message.", cls="error-message")
    
    # Clean and validate the message
    clean_message = message.strip()
    if len(clean_message) == 0:
        return Div("Message cannot be empty.", cls="error-message")
    
    # Display user message
    user_msg = Div(f"You: {clean_message}", cls="user-message")
    
    try:
        # Get the global assistant instance
        assistant_instance = await get_assistant()
        
        # ✅ CRITICAL FIX: Force a new thread if we detect issues
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        # ✅ ENHANCED: Add debug logging
        print(f"Processing message: '{clean_message}' with thread_id: {thread_id}")
        
        # Process the message
        response = await assistant_instance.process_message(
            message=clean_message,
            thread_id=thread_id
        )
        
        # Extract response content with enhanced error handling
        response_content = response.get('response', 'No response generated.')
        error_type = response.get('error')
        
        # ✅ CRITICAL FIX: Handle specific error types
        if error_type == 'empty_message_content':
            assistant_msg = Div(
                "I received an empty message. Please try again with a proper question.",
                cls="error-message"
            )
        elif error_type and 'empty content' in str(error_type):
            # This is the Anthropic empty content error
            assistant_msg = Div(
                "I'm having trouble with the conversation history. Let me start fresh.",
                cls="error-message"
            )
            # Force a new conversation
            global current_thread_id
            current_thread_id = str(uuid.uuid4())
        elif error_type:
            assistant_msg = Div(
                f"I encountered an issue: {response_content}",
                cls="error-message"
            )
        else:
            assistant_msg = Div(f"Assistant: {response_content}", cls="assistant-message")
        
        return user_msg, assistant_msg
        
    except Exception as e:
        error_str = str(e)
        print(f"Chat error: {error_str}")
        
        # ✅ CRITICAL FIX: Handle the specific Anthropic error
        if "empty content" in error_str.lower() or "messages." in error_str:
            error_msg = Div(
                "I'm having trouble with the conversation history. Starting a new conversation.",
                cls="error-message"
            )
            # Force a new conversation
            global current_thread_id
            current_thread_id = str(uuid.uuid4())
        else:
            error_msg = Div(
                f"Error: {error_str[:100]}{'...' if len(error_str) > 100 else ''}",
                cls="error-message"
            )
        
        return user_msg, error_msg

@rt("/new-session")
async def new_session():
    """Start a completely new conversation session"""
    global current_thread_id
    current_thread_id = str(uuid.uuid4())
    return P("New conversation started! Previous history cleared.", 
             style="color: #666; font-style: italic; margin: 10px 0;")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
