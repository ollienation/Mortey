# config/llm_manager.py - ✅ FIXED MODEL CACHING WITH STANDARDIZED IMPORTS

# ✅ STANDARD LIBRARY IMPORTS
import asyncio
import logging
import os
import time
from functools import wraps
from typing import Optional, Dict, Any, List
from asyncio import Semaphore

# ✅ THIRD-PARTY IMPORTS
from langchain.chat_models import init_chat_model

# ✅ LOCAL IMPORTS (absolute paths)
from config.settings import config

# API circuit breaker
from core.circuit_breaker import global_circuit_breaker, with_circuit_breaker

# ✅ OPTIONAL IMPORTS WITH ERROR HANDLING
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(**kwargs):
        """Fallback decorator when LangSmith is not available"""
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger("llm_manager")

class LLMManager:
    """
    ✅ FIXED: Universal LLM client manager with proper model caching.
    
    Key improvements:
    - Proper model caching that actually works
    - Cache-aware model retrieval
    - Performance optimization through model reuse
    """

    def __init__(self):
        self._models = {}  # Cache for initialized models
        self._setup_langsmith()
        self._initialize_concurrency_controls()

    def _setup_langsmith(self):
        """Setup LangSmith tracing if available and configured"""
        if config.langsmith_tracing and config.langsmith_api_key:
            try:
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
        self.MAX_CONCURRENT_CALLS = 5
        self._global_semaphore = Semaphore(self.MAX_CONCURRENT_CALLS)
        
        self.MAX_PROVIDER_CALLS = {
            "anthropic": 3,
            "openai": 5,
            "default": 2
        }
        
        self._provider_semaphores = {
            provider: Semaphore(self.MAX_PROVIDER_CALLS.get(provider, self.MAX_PROVIDER_CALLS["default"]))
            for provider in config.get_available_providers()
        }
        
        self._provider_semaphores["default"] = Semaphore(self.MAX_PROVIDER_CALLS["default"])
        
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_provider": {}
        }

    def _get_cache_key(self, node_name: str, override_max_tokens: Optional[int] = None) -> str:
        """✅ FIXED: Generate cache key that includes all configuration parameters"""
        node_config = config.get_node_config(node_name)
        model_config = config.get_model_config(node_config.provider, node_config.model)
        
        # Include all parameters that affect model behavior
        effective_max_tokens = override_max_tokens or node_config.max_tokens
        cache_key = f"{node_config.provider}:{model_config.model_id}:{node_config.temperature}:{effective_max_tokens}"
        
        return cache_key

    def _get_model(self, node_name: str, override_max_tokens: Optional[int] = None):
        """✅ FIXED: Get cached model or create new one if not exists"""
        cache_key = self._get_cache_key(node_name, override_max_tokens)
        
        # Return cached model if exists
        if cache_key in self._models:
            logger.debug(f"✅ Using cached model: {cache_key}")
            return self._models[cache_key]
        
        # Get configurations
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured in llm_config.yaml")
            
        provider_config = config.get_provider_config(node_config.provider)
        if not provider_config:
            raise ValueError(f"Provider {node_config.provider} not configured")
            
        model_config = config.get_model_config(node_config.provider, node_config.model)
        if not model_config:
            raise ValueError(f"Model {node_config.model} not found for provider {node_config.provider}")
        
        try:
            # Set API key as environment variable
            os.environ[provider_config.api_key_env] = provider_config.api_key
            
            # Use effective max_tokens (override or config default)
            effective_max_tokens = override_max_tokens or node_config.max_tokens
            
            # Initialize model with proper configuration
            model_string = f"{node_config.provider}:{model_config.model_id}"
            model = init_chat_model(
                model_string,
                temperature=node_config.temperature,
                max_tokens=effective_max_tokens
            )
            
            # ✅ CRITICAL FIX: Actually cache the model
            self._models[cache_key] = model
            logger.info(f"✅ Cached new model: {cache_key}")
            
            return model
        except Exception as e:
            logger.error(f"❌ Error initializing model {cache_key}: {e}")
            raise

    @traceable(name="llm_generation", run_type="llm")
    async def generate_for_node(self,
                               node_name: str,
                               prompt: str,
                               override_max_tokens: Optional[int] = None,
                               metadata: Optional[dict] = None) -> str:
        """
        ✅ FIXED: Generate response using cached models with concurrency control.
        
        This now properly uses the cached models instead of creating new ones.
        """
        
        # Get node configuration for logging and semaphore selection
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured")
        
        # Get semaphores for concurrency control
        provider_name = node_config.provider
        global_semaphore = self._global_semaphore
        provider_semaphore = self._provider_semaphores.get(provider_name, self._provider_semaphores["default"])
        
        # Use override or node-specific max_tokens
        effective_max_tokens = override_max_tokens or node_config.max_tokens
        
        logger.info(f"🎯 {node_name} using cached model (max_tokens: {effective_max_tokens})")
        
        # Implement retry logic with exponential backoff
        max_retries = config.retry_attempts
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                # Apply concurrency control with semaphores
                async with global_semaphore, provider_semaphore:
                    # ✅ CRITICAL FIX: Use cached model instead of creating new one
                    model = self._get_model(node_name, override_max_tokens)
                    
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
                if attempt < max_retries:
                    jitter = 0.1 * retry_delay * (2 * (0.5 - 0.5 * (attempt / max_retries)))
                    current_delay = retry_delay + jitter
                    
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {current_delay:.2f}s")
                    await asyncio.sleep(current_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ All {max_retries+1} attempts failed: {e}")
                    raise

    def _update_token_usage(self, provider: str, prompt_tokens: int, completion_tokens: int):
        """Update token usage tracking"""
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += prompt_tokens + completion_tokens
        
        if provider not in self.token_usage["by_provider"]:
            self.token_usage["by_provider"][provider] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
        self.token_usage["by_provider"][provider]["prompt_tokens"] += prompt_tokens
        self.token_usage["by_provider"][provider]["completion_tokens"] += completion_tokens
        self.token_usage["by_provider"][provider]["total_tokens"] += prompt_tokens + completion_tokens

    def clear_cache(self):
        """Clear the model cache (useful for testing or memory management)"""
        self._models.clear()
        logger.info("🧹 Model cache cleared")

    def get_cache_info(self) -> dict:
        """Get information about cached models"""
        return {
            "cached_models": list(self._models.keys()),
            "cache_size": len(self._models),
            "memory_efficient": True
        }

    def get_usage_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "providers_initialized": list(self._models.keys()),
            "langsmith_enabled": self.langsmith_enabled,
            "total_providers": len(config.providers),
            "token_usage": self.token_usage,
            "cache_info": self.get_cache_info()
        }

    async def health_check(self) -> dict:
        """Check health of all initialized providers with concurrency control"""
        health_status = {}
        
        for provider_name in config.get_available_providers():
            try:
                provider_semaphore = self._provider_semaphores.get(provider_name, self._provider_semaphores["default"])
                
                async with self._global_semaphore, provider_semaphore:
                    models = config.get_available_models(provider_name)
                    if not models:
                        health_status[provider_name] = "unhealthy: no models configured"
                        continue
                        
                    model_name = models[0]
                    model_config = config.get_model_config(provider_name, model_name)
                    if not model_config:
                        health_status[provider_name] = "unhealthy: model configuration missing"
                        continue
                    
                    # Use cached model for health check
                    test_node_name = f"{provider_name}_health_check"
                    
                    # Create temporary node config for health check
                    from config.settings import NodeConfig
                    temp_config = NodeConfig(
                        provider=provider_name,
                        model=model_name,
                        max_tokens=5,
                        temperature=0.0,
                        description="Health check"
                    )
                    
                    # Temporarily add to config for health check
                    config.nodes[test_node_name] = temp_config
                    
                    try:
                        model = self._get_model(test_node_name)
                        response = await model.ainvoke([{"role": "user", "content": "Hi"}])
                        health_status[provider_name] = "healthy"
                    finally:
                        # Clean up temporary config
                        if test_node_name in config.nodes:
                            del config.nodes[test_node_name]
                    
            except Exception as e:
                health_status[provider_name] = f"unhealthy: {str(e)}"
        
        return health_status

# Global LLM manager instance
llm_manager = LLMManager()
