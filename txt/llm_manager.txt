# Modern LLM Client Manager
# June 2025 - Production Ready

import asyncio
import time
import os
import logging
from typing import Optional, Dict, Any, List
from asyncio import Semaphore
from functools import wraps
from langchain.chat_models import init_chat_model
from config.settings import config

# Setup logging
logger = logging.getLogger("llm_manager")

# LangSmith import - only need traceable decorator
try:
    from langsmith import traceable
except ImportError:
    # Fallback decorator if LangSmith isn't available
    def traceable(**kwargs):
        def decorator(func):
            return func
        return decorator

class LLMManager:
    """
    Universal LLM client manager with modern patterns.
    
    Key improvements for June 2025:
    - Uses string-based model references with init_chat_model
    - Implements proper concurrency controls with semaphores
    - Retry logic with exponential backoff
    - Built-in tracing with LangSmith
    - Token usage tracking and rate limiting
    """

    def __init__(self):
        self._models = {}  # Cache for initialized models
        self._setup_langsmith()
        self._initialize_concurrency_controls()

    def _setup_langsmith(self):
        """Setup LangSmith tracing if available and configured"""
        if config.langsmith_tracing and config.langsmith_api_key:
            try:
                # Set environment variables for LangSmith
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
                os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
                os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
                
                self.langsmith_enabled = True
                logger.info(f"✅ LLM Manager: LangSmith tracing enabled for project: {config.langsmith_project}")
            except Exception as e:
                logger.warning(f"⚠️ LLM Manager: Failed to initialize LangSmith: {e}")
                self.langsmith_enabled = False
        else:
            self.langsmith_enabled = False
            logger.info("📊 LLM Manager: LangSmith tracing disabled")

    def _initialize_concurrency_controls(self):
        """Initialize concurrency controls for rate limiting"""
        # Global semaphore for all API calls
        self.MAX_CONCURRENT_CALLS = 5
        self._global_semaphore = Semaphore(self.MAX_CONCURRENT_CALLS)
        
        # Provider-specific semaphores for rate limiting
        self.MAX_PROVIDER_CALLS = {
            "anthropic": 3,  # 3 concurrent calls to Anthropic
            "openai": 5,     # 5 concurrent calls to OpenAI
            "default": 2     # Default for other providers
        }
        
        self._provider_semaphores = {
            provider: Semaphore(self.MAX_PROVIDER_CALLS.get(provider, self.MAX_PROVIDER_CALLS["default"]))
            for provider in config.get_available_providers()
        }
        
        # Add default fallback semaphore
        self._provider_semaphores["default"] = Semaphore(self.MAX_PROVIDER_CALLS["default"])
        
        # Token usage tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_provider": {}
        }

    def _get_model_key(self, provider: str, model_name: str) -> str:
        """Generate a unique key for model caching"""
        return f"{provider}:{model_name}"

    def _get_model(self, node_name: str):
        """Get a language model for the specified node using modern patterns"""
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
        
        # Check if model already initialized
        model_key = self._get_model_key(node_config.provider, model_config.model_id)
        if model_key in self._models:
            return self._models[model_key]
            
        # Initialize model if not in cache
        try:
            # Set API key as environment variable
            os.environ[provider_config.api_key_env] = provider_config.api_key
            
            # Use modern string-based model initialization
            model_string = f"{node_config.provider}:{model_config.model_id}"
            
            # Initialize model with proper configuration
            model = init_chat_model(
                model_string,
                temperature=node_config.temperature,
                max_tokens=node_config.max_tokens
            )
            
            # Cache the model
            self._models[model_key] = model
            logger.info(f"✅ Initialized {model_key} model")
            
            return model
        except Exception as e:
            logger.error(f"❌ Error initializing model {model_key}: {e}")
            raise

    @traceable(name="llm_generation", run_type="llm")
    async def generate_for_node(self,
                               node_name: str,
                               prompt: str,
                               override_max_tokens: Optional[int] = None,
                               metadata: Optional[dict] = None) -> str:
        """
        Generate response with concurrency control and retry logic.
        
        Key improvements:
        - Semaphore-based concurrency control
        - Exponential backoff retries
        - Token usage tracking
        - Error handling and recovery
        """
        # Get node configuration
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured in llm_config.yaml")
            
        # Get provider and model configurations
        provider_name = node_config.provider
        provider_config = config.get_provider_config(provider_name)
        if not provider_config:
            raise ValueError(f"Provider {provider_name} not configured")
            
        model_config = config.get_model_config(provider_name, node_config.model)
        if not model_config:
            raise ValueError(f"Model {node_config.model} not found for provider {provider_name}")
        
        # Get semaphores for concurrency control
        global_semaphore = self._global_semaphore
        provider_semaphore = self._provider_semaphores.get(provider_name, self._provider_semaphores["default"])
        
        # Use override or node-specific max_tokens
        max_tokens = override_max_tokens or node_config.max_tokens
        
        logger.info(f"🎯 {node_name} using {provider_name}/{model_config.model_id} (max_tokens: {max_tokens})")
        
        # Implement retry logic with exponential backoff
        max_retries = config.retry_attempts
        retry_delay = 1.0  # Initial delay in seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Apply concurrency control with semaphores
                async with global_semaphore, provider_semaphore:
                    # Route to appropriate provider
                    model_string = f"{provider_name}:{model_config.model_id}"
                    
                    # Use modern string-based model for generation
                    # Prefer direct model creation for this single call to avoid caching
                    model = init_chat_model(
                        model_string,
                        temperature=node_config.temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Perform generation
                    messages = [{"role": "user", "content": prompt}]
                    response = await model.ainvoke(messages)
                    
                    # Track token usage if available
                    if hasattr(response, "usage") and response.usage:
                        usage = response.usage
                        self._update_token_usage(
                            provider_name,
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0)
                        )
                    
                    # Extract content
                    content = response.content if hasattr(response, "content") else str(response)
                    return content
                    
            except Exception as e:
                # Handle retries with exponential backoff
                if attempt < max_retries:
                    # Calculate delay with jitter for exponential backoff
                    jitter = 0.1 * retry_delay * (2 * (0.5 - 0.5 * (attempt / max_retries)))
                    current_delay = retry_delay + jitter
                    
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {current_delay:.2f}s")
                    
                    # Exponential backoff
                    await asyncio.sleep(current_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                else:
                    # Final attempt failed
                    logger.error(f"❌ All {max_retries+1} attempts failed: {e}")
                    raise
    
    def _update_token_usage(self, provider: str, prompt_tokens: int, completion_tokens: int):
        """Update token usage tracking"""
        # Update global counts
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += prompt_tokens + completion_tokens
        
        # Update provider-specific counts
        if provider not in self.token_usage["by_provider"]:
            self.token_usage["by_provider"][provider] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
        self.token_usage["by_provider"][provider]["prompt_tokens"] += prompt_tokens
        self.token_usage["by_provider"][provider]["completion_tokens"] += completion_tokens
        self.token_usage["by_provider"][provider]["total_tokens"] += prompt_tokens + completion_tokens

    def get_usage_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "providers_initialized": list(self._models.keys()),
            "langsmith_enabled": self.langsmith_enabled,
            "total_providers": len(config.providers),
            "token_usage": self.token_usage
        }

    async def health_check(self) -> dict:
        """Check health of all initialized providers with concurrency control"""
        health_status = {}
        
        # Check each provider with a simple health check
        for provider_name in config.get_available_providers():
            try:
                # Get semaphores for concurrency control
                provider_semaphore = self._provider_semaphores.get(provider_name, self._provider_semaphores["default"])
                
                async with self._global_semaphore, provider_semaphore:
                    # Get first available model for this provider
                    models = config.get_available_models(provider_name)
                    if not models:
                        health_status[provider_name] = "unhealthy: no models configured"
                        continue
                        
                    # Select first model for health check
                    model_name = models[0]
                    model_config = config.get_model_config(provider_name, model_name)
                    if not model_config:
                        health_status[provider_name] = "unhealthy: model configuration missing"
                        continue
                    
                    # Create a temporary model for health check
                    model_string = f"{provider_name}:{model_config.model_id}"
                    model = init_chat_model(
                        model_string,
                        temperature=0.0,
                        max_tokens=5
                    )
                    
                    # Simple health check
                    response = await model.ainvoke([{"role": "user", "content": "Hi"}])
                    
                    health_status[provider_name] = "healthy"
                    
            except Exception as e:
                health_status[provider_name] = f"unhealthy: {str(e)}"
        
        return health_status

# Global LLM manager instance
llm_manager = LLMManager()