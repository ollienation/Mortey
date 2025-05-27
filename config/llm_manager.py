from typing import Optional, Dict, Any
from config.settings import config
import asyncio

class LLMManager:
    """Universal LLM client manager for all providers"""
    
    def __init__(self):
        self._clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize clients for available providers"""
        for provider_name, provider_config in config.llm_providers.items():
            if provider_name == "anthropic":
                from anthropic import AsyncAnthropic
                self._clients[provider_name] = AsyncAnthropic(api_key=provider_config.api_key)
            
            elif provider_name == "openai":
                from openai import AsyncOpenAI
                self._clients[provider_name] = AsyncOpenAI(api_key=provider_config.api_key)
            
            elif provider_name == "gemini":
                # Add Gemini client initialization
                pass
            
            elif provider_name == "local":
                # Add local model client (Ollama, etc.)
                pass
    
    async def generate(self, 
                      task: str, 
                      prompt: str, 
                      provider: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> str:
        """Generate response using specified or default provider"""
        
        # Use specified provider or get default for task
        if not provider:
            provider = config.get_default_provider_for_task(task)
        
        provider_config = config.get_provider_config(provider)
        if not provider_config:
            raise ValueError(f"Provider {provider} not configured")
        
        client = self._clients.get(provider)
        if not client:
            raise ValueError(f"Client for {provider} not initialized")
        
        # Use provided max_tokens or provider default
        max_tokens = max_tokens or provider_config.max_tokens
        
        # Route to appropriate provider
        if provider == "anthropic":
            return await self._generate_anthropic(client, provider_config, prompt, max_tokens)
        elif provider == "openai":
            return await self._generate_openai(client, provider_config, prompt, max_tokens)
        else:
            raise ValueError(f"Provider {provider} not implemented")
    
    async def _generate_anthropic(self, client, config, prompt, max_tokens):
        """Generate using Anthropic"""
        message = await client.messages.create(
            model=config.model,
            max_tokens=max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    async def _generate_openai(self, client, config, prompt, max_tokens):
        """Generate using OpenAI"""
        response = await client.chat.completions.create(
            model=config.model,
            max_tokens=max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Global LLM manager instance
llm_manager = LLMManager()
