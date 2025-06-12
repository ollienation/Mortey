# config/llm_manager.py - ✅ ENHANCED WITH LANGCHAIN NATIVE CACHING
import asyncio
import logging
import os
import time
from typing import Optional, Union, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain.chat_models import init_chat_model
from langchain.globals import set_verbose, set_debug

from config.settings import config
from core.circuit_breaker import global_circuit_breaker

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

@dataclass
class TokenUsage:
    """Simplified token usage tracking"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
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
    Simplified LLM manager using LangChain's native caching and pre-initialized models.
    """
    
    # Singleton pattern to prevent duplicate initialization
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
            
        # Enable LangChain's global caching - handles all caching automatically
        set_llm_cache(InMemoryCache())
        logger.info("✅ LangChain global caching enabled")
        
        # Pre-initialized models - no more lazy loading
        self.models: dict[str, Any] = {}
        
        # Simplified tracking
        self.token_usage = TokenUsage()
        self._health_check_cache = {}
        self._last_health_check = 0.0
        self._health_check_cooldown = 30.0  # 30 seconds
        
        self._setup_langsmith()
        self._initialize_concurrency_controls()
        
        # Mark as initialized
        self._initialized = True
        logger.info("✅ LLM Manager initialized with LangChain caching")

    def _setup_langsmith(self):
        """Setup LangSmith tracing if available and configured"""
        if config.langsmith_tracing and config.langsmith_api_key:
            try:
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
                os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
                os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
                
                self.langsmith_enabled = True
                logger.info(f"✅ LangSmith tracing enabled for project: {config.langsmith_project}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LangSmith: {e}")
                self.langsmith_enabled = False
        else:
            self.langsmith_enabled = False
            logger.info("📊 LangSmith tracing disabled")

    def _initialize_concurrency_controls(self):
        """Simplified concurrency controls"""
        from asyncio import Semaphore
        
        self.MAX_CONCURRENT_CALLS = config.max_concurrent_requests
        self._global_semaphore = Semaphore(self.MAX_CONCURRENT_CALLS)
        
        # Simplified provider limits
        self.MAX_PROVIDER_CALLS = {
            "anthropic": 5,
            "openai": 10, 
            "google": 5,
            "cohere": 3,
            "default": 3
        }
        
        self._provider_semaphores: dict[str, Semaphore] = {
            provider: Semaphore(self.MAX_PROVIDER_CALLS.get(provider, self.MAX_PROVIDER_CALLS["default"]))
            for provider in config.get_available_providers()
        }
        
        self._provider_semaphores["default"] = Semaphore(self.MAX_PROVIDER_CALLS["default"])

    async def initialize_models(self):
        """Pre-initialize all models during startup"""
        logger.info("🔥 Pre-initializing all models...")
        
        initialization_tasks = []
        all_node_names = []
        
        # Add regular nodes
        regular_nodes = config.get_available_nodes()
        all_node_names.extend(regular_nodes)
        
        # FIX: Check custom_nodes using proper config structure
        try:
            if hasattr(config, 'llm_config') and hasattr(config.llm_config, 'custom_nodes'):
                custom_node_names = list(config.llm_config.custom_nodes.keys())
                all_node_names.extend(custom_node_names)
                logger.debug(f"Found custom nodes via llm_config: {custom_node_names}")
            elif hasattr(config, 'custom_nodes'):
                custom_node_names = list(config.custom_nodes.keys())
                all_node_names.extend(custom_node_names)
                logger.debug(f"Found custom nodes via config: {custom_node_names}")
            else:
                logger.debug("No custom_nodes found in config")
        except Exception as e:
            logger.warning(f"Error accessing custom_nodes: {e}")
        
        # Remove duplicates
        all_node_names = list(set(all_node_names))
        logger.info(f"Total nodes to initialize: {all_node_names}")
        
        for node_name in all_node_names:
            task = asyncio.create_task(self._initialize_model(node_name))
            initialization_tasks.append((node_name, task))
        
        # Wait for all models to initialize
        for node_name, task in initialization_tasks:
            try:
                model = await task
                self.models[node_name] = model
                logger.debug(f"✅ Model pre-initialized: {node_name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize model {node_name}: {e}")
        
        logger.info(f"✅ Pre-initialized {len(self.models)} models")

    async def _initialize_model(self, node_name: str) -> Any:
        """Initialize a single model"""
        node_config = None
        
        # Try regular nodes first
        try:
            node_config = config.get_node_config(node_name)
            logger.debug(f"Found regular node config for {node_name}")
        except:
            pass
        
        # FIX: If not found, try custom nodes with proper structure
        if node_config is None:
            try:
                if hasattr(config, 'llm_config') and hasattr(config.llm_config, 'custom_nodes'):
                    if node_name in config.llm_config.custom_nodes:
                        node_config = config.llm_config.custom_nodes[node_name]
                        logger.debug(f"Found custom node config for {node_name} via llm_config")
                elif hasattr(config, 'custom_nodes') and node_name in config.custom_nodes:
                    node_config = config.custom_nodes[node_name]
                    logger.debug(f"Found custom node config for {node_name} via config")
            except Exception as e:
                logger.warning(f"Error accessing custom node {node_name}: {e}")
        
        if node_config is None:
            raise ValueError(f"Node configuration not found for {node_name}")
        
        provider_config = config.get_provider_config(node_config.provider)
        
        # For custom nodes, get model directly from config
        if hasattr(node_config, 'model'):
            model_name = node_config.model
            model_config = config.get_model_config(node_config.provider, model_name)
        else:
            model_config = config.get_model_config(node_config.provider, node_config.model)
            model_name = model_config.model_id
        
        if not provider_config or not provider_config.api_key:
            raise ValueError(f"Provider config or API key not available for {node_config.provider}")
        
        # Set API key as environment variable
        os.environ[provider_config.api_key_env] = provider_config.api_key
        
        # Prepare model parameters
        temperature = getattr(node_config, 'temperature', 0.7)
        max_tokens = getattr(node_config, 'max_tokens', 1000)
        
        # Special handling for models with restricted temperature
        if "o3" in model_name.lower() and temperature != 1.0:
            logger.warning(f"⚠️ Model {model_name} only supports temperature=1, adjusting from {temperature}")
            temperature = 1.0
        
        init_params = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if hasattr(provider_config, 'base_url') and provider_config.base_url:
            init_params["base_url"] = provider_config.base_url
        
        model_string = f"{node_config.provider}:{model_name}"
        
        # Create model - LangChain will handle caching automatically
        model = await asyncio.to_thread(init_chat_model, model_string, **init_params)
        return model

    async def get_model(self, node_name: str) -> Any:
        """Get pre-initialized model - simple and fast"""
        if node_name not in self.models:
            # Fallback: initialize on demand if not pre-initialized
            logger.warning(f"⚠️ Model {node_name} not pre-initialized, creating on demand")
            self.models[node_name] = await self._initialize_model(node_name)
        
        return self.models[node_name]

    @traceable(name="llm_generation", run_type="llm")
    async def generate_for_node(
        self,
        node_name: str,
        prompt: Union[str, list, Any],
        override_max_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """Generate response - simplified with LangChain caching"""
        
        # Normalize prompt to string
        if isinstance(prompt, str):
            normalized_prompt = prompt
        elif isinstance(prompt, list):
            # Extract content from last human message
            last_human_content = None
            for msg in reversed(prompt):
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if getattr(msg, 'type', '') == 'human':
                        last_human_content = msg.content
                        break
            normalized_prompt = last_human_content or str(prompt[-1]) if prompt else "Hello"
        elif hasattr(prompt, 'content'):
            normalized_prompt = prompt.content
        else:
            normalized_prompt = str(prompt)
        
        # Get node configuration
        node_config = config.get_node_config(node_name)
        if not node_config:
            raise ValueError(f"Node {node_name} not configured")
        
        # Retry logic with exponential backoff
        max_retries = config.retry_attempts
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                async with self._global_semaphore, self._provider_semaphores.get(node_config.provider, self._provider_semaphores["default"]):
                    result = await global_circuit_breaker.call_with_circuit_breaker(
                        f"llm_{node_config.provider}",
                        self._generate_with_model,
                        node_name,
                        normalized_prompt,
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
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """Internal method to generate with pre-initialized model"""
        model = await self.get_model(node_name)
        node_config = config.get_node_config(node_name)
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Add custom system prompt if configured
        if hasattr(node_config, 'custom_system_prompt') and node_config.custom_system_prompt:
            messages.insert(0, {"role": "system", "content": node_config.custom_system_prompt})
        
        # Generate response - LangChain handles caching automatically
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

    def _calculate_cost(self, usage: dict[str, Any], model_config) -> float:
        """Calculate estimated cost based on token usage"""
        try:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            cost_per_1k = getattr(model_config, 'cost_per_1k_tokens', 0.0)
            total_cost = ((prompt_tokens + completion_tokens) / 1000) * cost_per_1k
            return total_cost
        except Exception as e:
            logger.warning(f"Error calculating cost: {e}")
            return 0.0

    async def generate_batch_for_nodes(self, requests: list[tuple[str, str, Optional[int]]]) -> list[str]:
        """Simplified batch processing"""
        tasks = [
            self._safe_generate_for_node(node_name, prompt, max_tokens)
            for node_name, prompt, max_tokens in requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error strings
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(f"Error: {result}")
            else:
                final_results.append(result)
        
        return final_results

    async def _safe_generate_for_node(self, node_name: str, prompt: str, max_tokens: Optional[int]) -> str:
        """Safe wrapper that never raises exceptions"""
        try:
            return await self.generate_for_node(node_name, prompt, max_tokens)
        except Exception as e:
            logger.debug(f"Generation failed for {node_name}: {e}")
            return f"Error: {e}"

    async def health_check(self) -> dict[str, Any]:
        """Rate-limited health check with debouncing"""
        current_time = time.time()
        
        # Check if we can skip health check due to cooldown
        if current_time - self._last_health_check < self._health_check_cooldown:
            return self._health_check_cache.get('result', {
                "healthy": False,
                "overall_status": "unknown",
                "error": "Health check in cooldown",
                "timestamp": current_time
            })
        
        try:
            logger.debug("🔍 Starting LLM Manager health check...")
            
            # Get available providers
            available_providers = config.get_available_providers()
            provider_health = {}
            
            # Health check providers using existing models
            for provider in available_providers:
                try:
                    provider_health[provider] = await self._health_check_provider(provider)
                except Exception as e:
                    logger.warning(f"⚠️ Health check failed for provider {provider}: {e}")
                    provider_health[provider] = f"unhealthy: {e}"
            
            # Calculate overall health
            healthy_providers = sum(1 for status in provider_health.values() if status == "healthy")
            total_providers = len(provider_health)
            health_score = (healthy_providers / total_providers) if total_providers > 0 else 0
            
            # Determine overall status
            if health_score >= 0.8:
                overall_status = "healthy"
            elif health_score >= 0.5:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
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
                "models": {
                    "total_initialized": len(self.models),
                    "cache_enabled": True  # LangChain caching always enabled
                },
                "usage": {
                    "total_tokens": self.token_usage.total_tokens,
                    "estimated_cost": self.token_usage.cost_estimate
                },
                "timestamp": current_time
            }
            
            # Cache result and update last check time
            self._health_check_cache['result'] = health_report
            self._last_health_check = current_time
            
            logger.debug(f"✅ LLM Manager health check completed - Status: {overall_status}")
            return health_report
            
        except Exception as e:
            logger.error(f"❌ LLM Manager health check failed: {e}")
            error_result = {
                "healthy": False,
                "overall_status": "error",
                "health_score": 0.0,
                "error": str(e),
                "timestamp": current_time
            }
            self._health_check_cache['result'] = error_result
            return error_result

    async def _health_check_provider(self, provider_name: str) -> str:
        try:
            # Simple fix: check BOTH regular nodes AND custom nodes
            existing_nodes = [name for name, node in config.nodes.items() if node.provider == provider_name]
            
            # Add custom nodes to the list
            if hasattr(config, 'custom_nodes'):
                existing_nodes.extend([name for name, node in config.custom_nodes.items() if node.provider == provider_name])
            
            if not existing_nodes:
                return "unhealthy: no nodes configured for provider"
            
            # Test with the first available node
            test_node = existing_nodes[0]
            await asyncio.wait_for(
                self.generate_for_node(test_node, "Hi", override_max_tokens=5),
                timeout=5.0
            )
            return "healthy"
            
        except asyncio.TimeoutError:
            return "unhealthy: timeout"
        except Exception as e:
            return f"unhealthy: {str(e)}"

    def get_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive usage statistics"""
        return {
            "models_initialized": len(self.models),
            "langsmith_enabled": self.langsmith_enabled,
            "langchain_caching": True,  # Always enabled
            "token_usage": {
                "total_tokens": self.token_usage.total_tokens,
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "estimated_cost": self.token_usage.cost_estimate,
                "by_provider": dict(self.token_usage.by_provider),
                "by_model": dict(self.token_usage.by_model)
            },
            "concurrency_limits": {
                "global_max": self.MAX_CONCURRENT_CALLS,
                "provider_limits": dict(self.MAX_PROVIDER_CALLS)
            }
        }

    def clear_cache(self):
        """Clear models and reinitialize - useful for testing"""
        self.models.clear()
        self._health_check_cache.clear()
        # Note: LangChain's global cache is managed automatically
        logger.info("🧹 Model cache cleared")

# Global LLM manager instance
llm_manager = LLMManager()
