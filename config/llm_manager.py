# config/llm_manager.py - ✅ ENHANCED WITH PYTHON 3.13.4 COMPATIBILITY
import asyncio
import logging
import os
import time
from functools import wraps
from typing import Optional, Union, Any, Dict
from collections.abc import Mapping  # Python 3.13.4 preferred import
from asyncio import Semaphore, TaskGroup  # Python 3.13.4 TaskGroup
from dataclasses import dataclass, field
from enum import Enum
from langchain_openai import ChatOpenAI

from langchain.chat_models import init_chat_model

from config.settings import config
from core.circuit_breaker import global_circuit_breaker, with_circuit_breaker

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

class ModelState(Enum):
    """Model states for better tracking"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"

@dataclass
class ModelInfo:
    """Enhanced model information with Python 3.13.4 features"""
    model: Any
    cache_key: str
    state: ModelState = ModelState.READY
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_usage(self):
        """Update usage statistics"""
        self.last_used = time.time()
        self.usage_count += 1
    
    def record_error(self, error: str):
        """Record error for this model"""
        self.error_count += 1
        self.last_error = error
        self.state = ModelState.ERROR

@dataclass
class TokenUsage:
    """Token usage tracking with enhanced metrics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)  # Python 3.13.4 syntax
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)  # Python 3.13.4 syntax
    cost_estimate: float = 0.0
    
    def add_usage(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int, cost: float = 0.0):
        """Add usage statistics"""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.cost_estimate += cost
        
        # Track by provider
        if provider not in self.by_provider:
            self.by_provider[provider] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        self.by_provider[provider]["prompt_tokens"] += prompt_tokens
        self.by_provider[provider]["completion_tokens"] += completion_tokens
        self.by_provider[provider]["total_tokens"] += prompt_tokens + completion_tokens
        
        # Track by model
        model_key = f"{provider}:{model}"
        if model_key not in self.by_model:
            self.by_model[model_key] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        self.by_model[model_key]["prompt_tokens"] += prompt_tokens
        self.by_model[model_key]["completion_tokens"] += completion_tokens
        self.by_model[model_key]["total_tokens"] += prompt_tokens + completion_tokens

class LLMManager:
    """
    Universal LLM client manager with enhanced caching and Python 3.13.4 features.
    """

    def __init__(self):
        self._models: dict[str, ModelInfo] = {}  # Python 3.13.4 syntax
        self._initialization_locks: dict[str, asyncio.Lock] = {}  # Python 3.13.4 syntax
        self._setup_langsmith()
        self._initialize_concurrency_controls()
        self.token_usage = TokenUsage()
        self._health_check_interval = 300.0  # 5 minutes
        self._last_health_check = 0.0

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
        """Initialize concurrency controls for rate limiting with Python 3.13.4 enhancements"""
        self.MAX_CONCURRENT_CALLS = config.max_concurrent_requests
        self._global_semaphore = Semaphore(self.MAX_CONCURRENT_CALLS)
        
        # Enhanced provider-specific limits
        self.MAX_PROVIDER_CALLS = {
            "anthropic": 3,
            "openai": 5,
            "google": 3,
            "cohere": 2,
            "default": 2
        }
        
        self._provider_semaphores: dict[str, Semaphore] = {  # Python 3.13.4 syntax
            provider: Semaphore(self.MAX_PROVIDER_CALLS.get(provider, self.MAX_PROVIDER_CALLS["default"]))
            for provider in config.get_available_providers()
        }
        
        self._provider_semaphores["default"] = Semaphore(self.MAX_PROVIDER_CALLS["default"])

    def _get_cache_key(self, node_name: str, override_max_tokens: Optional[int] = None) -> str:
        """Generate comprehensive cache key that includes all configuration parameters"""
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured in llm_config.yaml")
            
        model_config = config.get_model_config(node_config.provider, node_config.model)
        if not model_config:
            raise ValueError(f"Model {node_config.model} not found for provider {node_config.provider}")
        
        # Include all parameters that affect model behavior
        effective_max_tokens = override_max_tokens or node_config.max_tokens
        cache_key = f"{node_config.provider}:{model_config.model_id}:{node_config.temperature}:{effective_max_tokens}"
        
        # Include custom system prompt if present
        if hasattr(node_config, 'custom_system_prompt') and node_config.custom_system_prompt:
            import hashlib
            prompt_hash = hashlib.md5(node_config.custom_system_prompt.encode()).hexdigest()[:8]
            cache_key += f":{prompt_hash}"
        
        return cache_key

    async def _get_model(self, node_name: str, override_max_tokens: Optional[int] = None) -> Any:
        """Get cached model or create new one if not exists with async safety"""
        cache_key = self._get_cache_key(node_name, override_max_tokens)
        
        # Return cached model if exists and ready
        if cache_key in self._models:
            model_info = self._models[cache_key]
            if model_info.state == ModelState.READY:
                model_info.update_usage()
                logger.debug(f"✅ Using cached model: {cache_key}")
                return model_info.model
            elif model_info.state == ModelState.ERROR:
                # Try to reinitialize error models after some time
                if time.time() - model_info.created_at > 300:  # 5 minutes
                    logger.info(f"🔄 Attempting to reinitialize error model: {cache_key}")
                    del self._models[cache_key]
                else:
                    raise Exception(f"Model {cache_key} is in error state: {model_info.last_error}")
        
        # Use lock to prevent concurrent initialization of the same model
        if cache_key not in self._initialization_locks:
            self._initialization_locks[cache_key] = asyncio.Lock()
        
        async with self._initialization_locks[cache_key]:
            # Double-check after acquiring lock
            if cache_key in self._models and self._models[cache_key].state == ModelState.READY:
                model_info = self._models[cache_key]
                model_info.update_usage()
                return model_info.model
            
            # Initialize new model
            try:
                # Mark as initializing
                if cache_key in self._models:
                    self._models[cache_key].state = ModelState.INITIALIZING
                else:
                    self._models[cache_key] = ModelInfo(
                        model=None,
                        cache_key=cache_key,
                        state=ModelState.INITIALIZING
                    )
                
                model = await self._create_model(node_name, override_max_tokens)
                
                # Update model info
                model_info = self._models[cache_key]
                model_info.model = model
                model_info.state = ModelState.READY
                model_info.update_usage()
                
                logger.info(f"✅ Cached new model: {cache_key}")
                return model
                
            except Exception as e:
                # Record error
                if cache_key in self._models:
                    self._models[cache_key].record_error(str(e))
                
                logger.error(f"❌ Error initializing model {cache_key}: {e}")
                raise

    async def _create_model(self, node_name: str, override_max_tokens: Optional[int] = None) -> Any:
        """Create a new model instance with enhanced error handling"""
        # Get configurations
        node_config = config.get_node_config(node_name)
        provider_config = config.get_provider_config(node_config.provider)  # Now returns ProviderConfig
        model_config = config.get_model_config(node_config.provider, node_config.model)
        
        if not provider_config:
            raise ValueError(f"Provider config not found for {node_config.provider}")
        
        if not provider_config.api_key:
            raise ValueError(f"API key not available for provider {node_config.provider}")
        
        try:
            # Set API key as environment variable
            os.environ[provider_config.api_key_env] = provider_config.api_key
            
            # Use effective max_tokens (override or config default)
            effective_max_tokens = override_max_tokens or node_config.max_tokens
            
            model_name = model_config.model_id
            temperature = node_config.temperature
            
            # Special handling for OpenAI models that only support temperature=1 
            if "o3" in model_name.lower() and temperature != 1.0:
                logger.warning(f"⚠️ Model {model_name} only supports temperature=1, adjusting from {temperature}")
                temperature = 1.0
            
            init_params = {
                "temperature": temperature,
                "max_tokens": effective_max_tokens
            }
            
            if hasattr(provider_config, 'base_url') and provider_config.base_url:
                init_params["base_url"] = provider_config.base_url
            
            model_string = f"{node_config.provider}:{model_name}"
            
            model = init_chat_model(model_string, **init_params)
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error creating model {node_name}: {e}")
            raise

    @traceable(name="llm_generation", run_type="llm")
    async def generate_for_node(
        self,
        node_name: str,
        prompt: Union[str, list, Any],
        override_max_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None  # Python 3.13.4 syntax
    ) -> str:
        """
        Generate response with flexible input handling
        """
        if isinstance(prompt, str):
            # Already a string
            normalized_prompt = prompt
        elif isinstance(prompt, list):
            # List of messages - extract content from last human message
            last_human_content = None
            for msg in reversed(prompt):
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if getattr(msg, 'type', '') == 'human':
                        last_human_content = msg.content
                        break
            normalized_prompt = last_human_content or str(prompt[-1]) if prompt else "Hello"
        elif hasattr(prompt, 'content'):
            # Single message object
            normalized_prompt = prompt.content
        else:
            # Fallback to string conversion
            normalized_prompt = str(prompt)

        # Continue with logic using normalized_prompt
        return await self._generate_with_normalized_input(node_name, normalized_prompt, override_max_tokens, metadata)

    async def _generate_with_normalized_input(
        self,
        node_name: str,
        prompt: str,  # Now guaranteed to be string
        override_max_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """Internal method with original logic"""
        # Get node configuration for logging and semaphore selection
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured")
        
        # Rest of your existing logic...
        max_retries = config.retry_attempts
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                async with self._global_semaphore, self._provider_semaphores.get(node_config.provider, self._provider_semaphores["default"]):
                    result = await global_circuit_breaker.call_with_circuit_breaker(
                        f"llm_{node_config.provider}",
                        self._generate_with_model,
                        node_name,
                        prompt,  # Using normalized string prompt
                        override_max_tokens,
                        metadata
                    )
                    return result
            except Exception as e:
                if attempt < max_retries:
                    jitter = 0.1 * retry_delay * (2 * (0.5 - 0.5 * (attempt / max_retries)))
                    current_delay = retry_delay + jitter
                    logger.warning(f"Attempt {attempt+1}/{max_retries+1} failed: {e}. Retrying in {current_delay:.2f}s")
                    await asyncio.sleep(current_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ All {max_retries+1} attempts failed: {e}")
                    raise

    async def _generate_with_model(
        self,
        node_name: str,
        prompt: str,
        override_max_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None  # Python 3.13.4 syntax
    ) -> str:
        """Internal method to generate with model"""
        model = await self._get_model(node_name, override_max_tokens)
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Add custom system prompt if configured
        node_config = config.get_node_config(node_name)
        if hasattr(node_config, 'custom_system_prompt') and node_config.custom_system_prompt:
            messages.insert(0, {"role": "system", "content": node_config.custom_system_prompt})
        
        # Perform generation
        start_time = time.time()
        response = await model.ainvoke(messages)
        generation_time = time.time() - start_time
        
        # Track token usage if available
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            model_config = config.get_model_config(node_config.provider, node_config.model)
            cost = self._calculate_cost(usage, model_config)
            
            self.token_usage.add_usage(
                node_config.provider,
                node_config.model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                cost
            )
        
        # Log performance metrics
        if metadata:
            metadata.update({
                "generation_time": generation_time,
                "provider": node_config.provider,
                "model": node_config.model
            })
        
        # Extract content
        content = response.content if hasattr(response, "content") else str(response)
        return content

    def _calculate_cost(self, usage: dict[str, Any], model_config) -> float:  # Python 3.13.4 syntax
        """Calculate estimated cost based on token usage"""
        try:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Use cost per 1k tokens from model config
            cost_per_1k = getattr(model_config, 'cost_per_1k_tokens', 0.0)
            
            total_cost = ((prompt_tokens + completion_tokens) / 1000) * cost_per_1k
            return total_cost
        except Exception as e:
            logger.warning(f"Error calculating cost: {e}")
            return 0.0

    async def generate_batch_for_nodes(
        self, 
        requests: list[tuple[str, str, Optional[int]]]
    ) -> list[str]:
        """Batch processing with TaskGroup - FIXED"""
        final_results = [""] * len(requests)  # Pre-initialize
        
        try:
            tasks_and_indices = []
            
            async with TaskGroup() as tg:
                for i, (node_name, prompt, max_tokens) in enumerate(requests):
                    try:
                        task = tg.create_task(
                            self._safe_generate_for_node(node_name, prompt, max_tokens)
                        )
                        tasks_and_indices.append((i, task))
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to create generation task {i}: {e}")
                        final_results[i] = f"Error: {e}"
            
            # Collect results
            for i, task in tasks_and_indices:
                try:
                    final_results[i] = task.result()
                except Exception as e:
                    logger.warning(f"⚠️ Generation task {i} failed: {e}")
                    final_results[i] = f"Error: {e}"
            
        except* Exception as eg:  # 🔥 FIX: Handle exception group without return
            logger.error(f"❌ Batch generation TaskGroup failed with {len(eg.exceptions)} exceptions")
            # Fill any remaining empty results with errors
            for i, result in enumerate(final_results):
                if not result:
                    final_results[i] = "Error: TaskGroup exception"
        
        # 🔥 FIX: Return statement outside except* block
        return final_results

    # 🔥 NEW: Safe wrapper for generation
    async def _safe_generate_for_node(self, node_name: str, prompt: str, max_tokens: Optional[int]) -> str:
        """Safe wrapper that never raises exceptions"""
        try:
            return await self.generate_for_node(node_name, prompt, max_tokens)
        except Exception as e:
            logger.debug(f"Generation failed for {node_name}: {e}")
            return f"Error: {e}"

    async def _sequential_fallback(self, requests: list[tuple[str, str, Optional[int]]]) -> list[str]:
        """Sequential fallback when TaskGroup fails"""
        results = []
        for node_name, prompt, max_tokens in requests:
            try:
                result = await self.generate_for_node(node_name, prompt, max_tokens)
                results.append(result)
            except Exception as e:
                logger.error(f"Sequential request failed for {node_name}: {e}")
                results.append(f"Error: {e}")
        return results


    async def get_openai_model(self, node_name: str) -> ChatOpenAI:
        """Get ChatOpenAI instance for function calling"""
        node_config = config.get_node_config(node_name)
        if node_config.provider == 'openai':
            return ChatOpenAI(
                model=node_config.model,
                temperature=node_config.temperature,
                max_tokens=node_config.max_tokens,
            )
        else:
            raise ValueError(f"Node {node_name} is not an OpenAI provider")

    def clear_cache(self):
        """Clear the model cache (useful for testing or memory management)"""
        self._models.clear()
        self._initialization_locks.clear()
        logger.info("🧹 Model cache cleared")

    def cleanup_stale_models(self, max_age_hours: float = 24.0):
        """Remove stale models from cache"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        stale_keys = []
        for cache_key, model_info in self._models.items():
            if current_time - model_info.last_used > max_age_seconds:
                stale_keys.append(cache_key)
        
        for key in stale_keys:
            del self._models[key]
            if key in self._initialization_locks:
                del self._initialization_locks[key]
        
        if stale_keys:
            logger.info(f"🧹 Cleaned up {len(stale_keys)} stale models")

    def get_cache_info(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get comprehensive information about cached models"""
        cache_stats = {
            "total_models": len(self._models),
            "models_by_state": {},
            "models_by_provider": {},
            "memory_efficient": True,
            "models": []
        }
        
        for cache_key, model_info in self._models.items():
            # Count by state
            state = model_info.state.value
            cache_stats["models_by_state"][state] = cache_stats["models_by_state"].get(state, 0) + 1
            
            # Count by provider
            provider = cache_key.split(":")[0]
            cache_stats["models_by_provider"][provider] = cache_stats["models_by_provider"].get(provider, 0) + 1
            
            # Model details
            cache_stats["models"].append({
                "cache_key": cache_key,
                "state": state,
                "usage_count": model_info.usage_count,
                "error_count": model_info.error_count,
                "last_used": model_info.last_used,
                "age_hours": (time.time() - model_info.created_at) / 3600
            })
        
        return cache_stats

    def get_usage_stats(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get comprehensive usage statistics"""
        return {
            "providers_initialized": list(set(key.split(":")[0] for key in self._models.keys())),
            "langsmith_enabled": self.langsmith_enabled,
            "total_providers": len(config.providers),
            "token_usage": {
                "total_tokens": self.token_usage.total_tokens,
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "estimated_cost": self.token_usage.cost_estimate,
                "by_provider": dict(self.token_usage.by_provider),
                "by_model": dict(self.token_usage.by_model)
            },
            "cache_info": self.get_cache_info(),
            "concurrency_limits": {
                "global_max": self.MAX_CONCURRENT_CALLS,
                "provider_limits": dict(self.MAX_PROVIDER_CALLS)
            }
        }

    # config/llm_manager.py - FIX HEALTH CHECK
    async def _health_check_provider(self, provider_name: str) -> str:
        """Health check for a specific provider"""
        try:
            provider_semaphore = self._provider_semaphores.get(provider_name, self._provider_semaphores["default"])
            
            async with self._global_semaphore, provider_semaphore:
                models = config.get_available_models(provider_name)
                if not models:
                    return "unhealthy: no models configured"
                
                model_name = models[0]
                model_config = config.get_model_config(provider_name, model_name)
                if not model_config:
                    return "unhealthy: model configuration missing"
                
                # 🔥 FIX: Use existing nodes instead of creating temporary ones
                existing_nodes = [name for name, node in config.nodes.items() if node.provider == provider_name]
                if not existing_nodes:
                    return "unhealthy: no nodes configured for provider"
                
                test_node = existing_nodes[0]
                
                try:
                    # Test with minimal tokens and safe temperature
                    await asyncio.wait_for(
                        self.generate_for_node(test_node, "Hi", override_max_tokens=5),
                        timeout=10.0
                    )
                    return "healthy"
                    
                except Exception as e:
                    return f"unhealthy: {str(e)}"
                    
        except asyncio.TimeoutError:
            return "unhealthy: timeout"
        except Exception as e:
            return f"unhealthy: {str(e)}"


    async def _health_check_provider(self, provider_name: str) -> str:
        """Health check for a specific provider"""
        try:
            provider_semaphore = self._provider_semaphores.get(provider_name, self._provider_semaphores["default"])
            
            async with self._global_semaphore, provider_semaphore:
                models = config.get_available_models(provider_name)
                if not models:
                    return "unhealthy: no models configured"
                
                model_name = models[0]
                model_config = config.get_model_config(provider_name, model_name)
                if not model_config:
                    return "unhealthy: model configuration missing"
                
                # Create temporary node config for health check
                from config.settings import NodeConfig
                temp_node_name = f"{provider_name}_health_check"
                temp_config = NodeConfig(
                    provider=provider_name,
                    model=model_name,
                    max_tokens=5,
                    temperature=0.0,
                    description="Health check"
                )
                
                # Temporarily add to config for health check
                config.nodes[temp_node_name] = temp_config
                
                try:
                    # Perform actual health check with timeout
                    await asyncio.wait_for(
                        self.generate_for_node(temp_node_name, "Hi", override_max_tokens=5),
                        timeout=10.0
                    )
                    return "healthy"
                finally:
                    # Clean up temporary config
                    if temp_node_name in config.nodes:
                        del config.nodes[temp_node_name]
                
        except asyncio.TimeoutError:
            return "unhealthy: timeout"
        except Exception as e:
            return f"unhealthy: {str(e)}"

    async def warm_up_models(self, node_names: Optional[list[str]] = None):  # Python 3.13.4 syntax
        """Warm up models by pre-initializing them"""
        if node_names is None:
            node_names = list(config.get_available_nodes())
        
        logger.info(f"🔥 Warming up {len(node_names)} models...")
        
        try:
            async with TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._warm_up_model(node_name))
                    for node_name in node_names
                ]
            
            successful = sum(1 for task in tasks if task.result())
            logger.info(f"✅ Warmed up {successful}/{len(node_names)} models")
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")

    async def _warm_up_model(self, node_name: str) -> bool:
        """Warm up a specific model"""
        try:
            await self._get_model(node_name)
            logger.debug(f"✅ Warmed up model: {node_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to warm up model {node_name}: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """
        Comprehensive health check for LLM Manager - MISSING METHOD ADDED
        """
        try:
            logger.debug("🔍 Starting LLM Manager health check...")
            
            # Get available providers from config
            available_providers = config.get_available_providers()
            
            # Health check all providers using TaskGroup
            provider_health = {}
            
            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = []
                    for provider in available_providers:
                        task = tg.create_task(self._safe_health_check_provider(provider))
                        tasks.append((provider, task))
                
                # Collect results
                for provider, task in tasks:
                    try:
                        provider_health[provider] = task.result()
                    except Exception as e:
                        logger.warning(f"⚠️ Health check failed for provider {provider}: {e}")
                        provider_health[provider] = f"unhealthy: {e}"
                        
            except* Exception as eg:
                logger.error(f"❌ Provider health check TaskGroup failed: {len(eg.exceptions)} exceptions")
                # Fill remaining providers with error status
                for provider in available_providers:
                    if provider not in provider_health:
                        provider_health[provider] = "unhealthy: TaskGroup exception"
            
            # Calculate overall health
            healthy_providers = sum(1 for status in provider_health.values() if status == "healthy")
            total_providers = len(provider_health)
            health_score = (healthy_providers / total_providers) if total_providers > 0 else 0
            
            # Determine overall health status
            if health_score >= 0.8:
                overall_status = "healthy"
            elif health_score >= 0.5:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            # Gather additional metrics
            cache_info = self.get_cache_info()
            usage_stats = self.get_usage_stats()
            performance_metrics = self.get_performance_metrics()
            
            health_report = {
                "healthy": overall_status == "healthy",
                "overall_status": overall_status,
                "health_score": health_score,
                "provider_health": provider_health,
                "providers": {
                    "total": total_providers,
                    "healthy": healthy_providers,
                    "unhealthy": total_providers - healthy_providers
                },
                "cache": {
                    "total_models": cache_info["total_models"],
                    "models_by_state": cache_info["models_by_state"],
                    "cache_hit_rate": performance_metrics.get("cache_hit_rate", 0)
                },
                "usage": {
                    "total_tokens": usage_stats["token_usage"]["total_tokens"],
                    "estimated_cost": usage_stats["token_usage"]["estimated_cost"],
                    "providers_initialized": len(usage_stats["providers_initialized"])
                },
                "performance": {
                    "error_rate": performance_metrics.get("error_rate", 0),
                    "models_in_error_state": performance_metrics.get("models_in_error_state", 0)
                },
                "timestamp": time.time()
            }
            
            logger.debug(f"✅ LLM Manager health check completed - Status: {overall_status}")
            return health_report
            
        except Exception as e:
            logger.error(f"❌ LLM Manager health check failed: {e}")
            return {
                "healthy": False,
                "overall_status": "error",
                "health_score": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _safe_health_check_provider(self, provider_name: str) -> str:
        """Safe wrapper for provider health check that never raises exceptions"""
        try:
            return await self._health_check_provider(provider_name)
        except Exception as e:
            logger.debug(f"Provider health check failed for {provider_name}: {e}")
            return f"unhealthy: {e}"

    def get_performance_metrics(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get detailed performance metrics"""
        cache_info = self.get_cache_info()
        
        # Calculate cache hit rate
        total_usage = sum(model["usage_count"] for model in cache_info["models"])
        cache_hits = len([m for m in cache_info["models"] if m["usage_count"] > 1])
        cache_hit_rate = (cache_hits / len(cache_info["models"])) if cache_info["models"] else 0
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_model_usage": total_usage,
            "average_usage_per_model": total_usage / len(cache_info["models"]) if cache_info["models"] else 0,
            "error_rate": sum(m["error_count"] for m in cache_info["models"]) / total_usage if total_usage > 0 else 0,
            "models_in_error_state": len([m for m in cache_info["models"] if m["state"] == "error"]),
            "token_efficiency": {
                "tokens_per_dollar": self.token_usage.total_tokens / max(self.token_usage.cost_estimate, 0.01),
                "average_tokens_per_request": self.token_usage.total_tokens / total_usage if total_usage > 0 else 0
            }
        }

# Global LLM manager instance
llm_manager = LLMManager()
