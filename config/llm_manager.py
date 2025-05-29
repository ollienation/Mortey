from typing import Optional
import time
import os

from config.settings import config

import asyncio

# LangSmith imports - only need traceable decorator
from langsmith import traceable

class LLMManager:
    """Universal LLM client manager with YAML-based configuration and LangSmith tracing"""

    def __init__(self):
        self._clients = {}
        self._setup_langsmith()
        self._initialize_clients()

    def _setup_langsmith(self):
        """Setup LangSmith tracing if available and configured"""
        if (config.langsmith_tracing and config.langsmith_api_key):
            try:
                # Set environment variables for LangSmith
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
                os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
                os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
                
                self.langsmith_enabled = True
                print(f"✅ LLM Manager: LangSmith tracing enabled for project: {config.langsmith_project}")
                
            except Exception as e:
                print(f"⚠️ LLM Manager: Failed to initialize LangSmith: {e}")
                self.langsmith_enabled = False
        else:
            self.langsmith_enabled = False
            print("📊 LLM Manager: LangSmith tracing disabled")

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

    @traceable(name="llm_generation", run_type="llm")
    async def generate_for_node(self,
                               node_name: str,
                               prompt: str,
                               override_max_tokens: Optional[int] = None,
                               metadata: Optional[dict] = None) -> str:
        """Generate response using node-specific configuration with LangSmith tracing"""

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

        try:
            # Route to appropriate provider - tracing handled by @traceable decorator
            if node_config.provider == "anthropic":
                response = await self._generate_anthropic(client, model_config, node_config, prompt, max_tokens)
            elif node_config.provider == "openai":
                response = await self._generate_openai(client, model_config, node_config, prompt, max_tokens)
            else:
                raise ValueError(f"Provider {node_config.provider} not implemented")
            
            return response
            
        except Exception as e:
            print(f"❌ LLM generation error for {node_name}: {e}")
            raise

    async def _generate_anthropic(self, client, model_config, node_config, prompt, max_tokens):
        """Generate using Anthropic with enhanced error handling"""
        try:
            message = await client.messages.create(
                model=model_config.model_id,
                max_tokens=max_tokens,
                temperature=node_config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"❌ Anthropic generation error: {e}")
            raise

    async def _generate_openai(self, client, model_config, node_config, prompt, max_tokens):
        """Generate using OpenAI with enhanced error handling"""
        try:
            response = await client.chat.completions.create(
                model=model_config.model_id,
                max_tokens=max_tokens,
                temperature=node_config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI generation error: {e}")
            raise

    def get_usage_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "providers_initialized": list(self._clients.keys()),
            "langsmith_enabled": self.langsmith_enabled,
            "total_providers": len(config.providers)
        }

    async def health_check(self) -> dict:
        """Check health of all initialized providers"""
        health_status = {}
        
        for provider_name, client in self._clients.items():
            try:
                # Simple health check - attempt a minimal generation
                if provider_name == "anthropic":
                    test_response = await client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=5,
                        messages=[{"role": "user", "content": "Hi"}]
                    )
                    health_status[provider_name] = "healthy"
                elif provider_name == "openai":
                    test_response = await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        max_tokens=5,
                        messages=[{"role": "user", "content": "Hi"}]
                    )
                    health_status[provider_name] = "healthy"
            except Exception as e:
                health_status[provider_name] = f"unhealthy: {str(e)}"
        
        return health_status

# Global LLM manager instance
llm_manager = LLMManager()
