# src/mortey/services/llm/llm_service.py
import ollama
import asyncio
from anthropic import AsyncAnthropic
from typing import Optional, Dict, Any

class LLMService:
    """Hybrid local/API LLM service for cost optimization"""
    
    def __init__(self, local_model: str, api_key: str):
        self.local_model = local_model
        self.anthropic = AsyncAnthropic(api_key=api_key)
        
        # Cost tracking
        self.api_calls_count = 0
        self.local_calls_count = 0
    
    async def route_request(self, user_input: str) -> str:
        """Use fast local model for routing decisions"""
        self.local_calls_count += 1
        
        prompt = f"""
        Route this request to the appropriate agent:
        User: {user_input}
        
        Agents: CODER, WEB, VISION, IMAGE
        Respond with only the agent name.
        """
        
        response = await asyncio.to_thread(
            ollama.generate,
            model=self.local_model,
            prompt=prompt,
            options={"num_predict": 10}  # Very short response
        )
        
        return response['response'].strip().upper()
    
    async def generate_complex(self, prompt: str, max_tokens: int = 2000) -> str:
        """Use API model for complex tasks like coding"""
        self.api_calls_count += 1
        
        message = await self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    async def verify_output(self, content: str, context: str) -> Dict[str, Any]:
        """Use local model for basic verification, API for complex safety checks"""
        
        # Simple checks with local model first
        simple_check = await asyncio.to_thread(
            ollama.generate,
            model=self.local_model,
            prompt=f"Is this safe and appropriate? Answer YES or NO: {content[:500]}",
            options={"num_predict": 5}
        )
        
        if "NO" in simple_check['response'].upper():
            # Use API for detailed safety analysis
            detailed_check = await self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{
                    "role": "user", 
                    "content": f"Analyze this content for safety issues:\n{content}\n\nContext: {context}"
                }]
            )
            
            return {
                "safe": False,
                "reason": detailed_check.content[0].text,
                "action": "block"
            }
        
        return {"safe": True, "action": "approve"}
