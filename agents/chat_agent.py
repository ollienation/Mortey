import os
import asyncio
from typing import Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class ChatAgent:
    """Chat agent using Claude for conversational responses"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service  # For consistency with other agents
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    async def chat(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat request using Claude"""
        
        user_input = state.get('user_input', '')
        assistant_name = state.get('assistant_name', 'Assistant')
        
        # Update thinking state
        state['thinking_state'] = {
            'active_agent': 'CHAT',
            'current_task': 'Generating conversational response',
            'progress': 0.5,
            'details': f'Processing: {user_input}'
        }
        
        prompt = f"""
        You are {assistant_name}, a helpful voice assistant. Respond naturally to this conversation:
        
        User: {user_input}
        
        Guidelines:
        - Keep responses under 50 words since this will be spoken aloud
        - Be conversational, friendly, and helpful
        - If the user asks for information you don't have, suggest they ask for a web search
        - Stay in character as a voice assistant
        - Be concise and direct
        """
        
        try:
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=150,  # Limit for concise responses
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            return {
                **state,
                'output_content': response_text,
                'output_type': 'chat',
                'thinking_state': {
                    'active_agent': 'CHAT',
                    'current_task': 'Chat response complete',
                    'progress': 1.0,
                    'details': 'Ready for verification'
                }
            }
            
        except Exception as e:
            print(f"❌ Claude chat error: {e}")
            
            # Fallback response
            fallback_response = "I'm having trouble processing that request right now. Please try again."
            
            return {
                **state,
                'output_content': fallback_response,
                'output_type': 'error',
                'thinking_state': {
                    'active_agent': 'CHAT',
                    'current_task': 'Error occurred',
                    'progress': 1.0,
                    'details': f'Error: {str(e)}'
                }
            }
    
    async def get_quick_response(self, user_input: str, assistant_name: str = "Assistant") -> str:
        """Get a quick chat response without full state management"""
        prompt = f"""
        You are {assistant_name}, a helpful voice assistant. Respond briefly to:
        
        User: {user_input}
        
        Keep it under 30 words and conversational.
        """
        
        try:
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"❌ Quick chat error: {e}")
            return "I'm having trouble right now. Please try again."
