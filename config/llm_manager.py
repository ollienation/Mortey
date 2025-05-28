from typing import Optional
from config.settings import config
import asyncio

class LLMManager:
    """Universal LLM client manager with YAML-based configuration"""
    
    def __init__(self):
        self._clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize clients for available providers"""
        for provider_name, provider_config in config.providers.items():
            try:
                if provider_name == "anthropic":
                    from anthropic import AsyncAnthropic
                    self._clients[provider_name] = AsyncAnthropic(api_key=provider_config.api_key)
                    print(f"✅ Initialized {provider_name} client")
                
                elif provider_name == "openai":
                    try:
                        from openai import AsyncOpenAI
                        self._clients[provider_name] = AsyncOpenAI(api_key=provider_config.api_key)
                        print(f"✅ Initialized {provider_name} client")
                    except ImportError:
                        print(f"⚠️ OpenAI package not installed - {provider_name} provider disabled")
                        continue
                    
            except ImportError as e:
                print(f"⚠️ Package not available for {provider_name}: {e}")
            except Exception as e:
                print(f"❌ Error initializing {provider_name} client: {e}")
    
    async def generate_for_node(self, 
                               node_name: str, 
                               prompt: str,
                               override_max_tokens: Optional[int] = None) -> str:
        """Generate response using node-specific configuration from YAML"""
        
        # Get node configuration
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured in llm_config.yaml")
        
        # Get provider and model configurations
        provider_config = config.get_provider_config(node_config.provider)
        if not provider_config:
            raise ValueError(f"Provider {node_config.provider} not configured")
        
        model_config = config.get_model_config(node_config.provider, node_config.model)
        if not model_config:
            raise ValueError(f"Model {node_config.model} not found for provider {node_config.provider}")
        
        client = self._clients.get(node_config.provider)
        if not client:
            raise ValueError(f"Client for {node_config.provider} not initialized")
        
        # Use override or node-specific max_tokens
        max_tokens = override_max_tokens or node_config.max_tokens
        
        print(f"🎯 {node_name} using {node_config.provider}/{model_config.model_id} (max_tokens: {max_tokens})")
        
        # Route to appropriate provider
        if node_config.provider == "anthropic":
            return await self._generate_anthropic(client, model_config, node_config, prompt, max_tokens)
        elif node_config.provider == "openai":
            return await self._generate_openai(client, model_config, node_config, prompt, max_tokens)
        else:
            raise ValueError(f"Provider {node_config.provider} not implemented")
    
    async def _generate_anthropic(self, client, model_config, node_config, prompt, max_tokens):
        """Generate using Anthropic with YAML configuration"""
        message = await client.messages.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            temperature=node_config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    async def _generate_openai(self, client, model_config, node_config, prompt, max_tokens):
        """Generate using OpenAI with YAML configuration"""
        response = await client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            temperature=node_config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Global LLM manager instance
llm_manager = LLMManager()
