# config/settings.py - ✅ ENHANCED WITH PYTHON 3.13.4 COMPATIBILITY
import os
import pathlib
import logging
import yaml
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Protocol, runtime_checkable, Any
from collections.abc import Mapping
from dotenv import load_dotenv

if TYPE_CHECKING:
    from config.providers import ProviderRegistry

logger = logging.getLogger("settings")

@dataclass
class ModelConfig:
    """Configuration for a specific model with enhanced validation"""
    model_id: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float = 0.0
    
    def __post_init__(self):
        """Validate model configuration parameters"""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.cost_per_1k_tokens < 0:
            raise ValueError(f"cost_per_1k_tokens cannot be negative, got {self.cost_per_1k_tokens}")

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider with enhanced features"""
    api_key_env: str
    models: dict[str, ModelConfig] = field(default_factory=dict)  # Python 3.13.4 syntax
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For custom endpoints
    rate_limit_per_minute: int = 60  # Default rate limit
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Validate provider configuration"""
        if not self.api_key_env:
            raise ValueError("api_key_env cannot be empty")
        if self.rate_limit_per_minute <= 0:
            raise ValueError(f"rate_limit_per_minute must be positive, got {self.rate_limit_per_minute}")

@dataclass
class NodeConfig:
    """Configuration for a specific node/agent with enhanced validation"""
    provider: str
    model: str
    max_tokens: int
    temperature: float
    description: str = ""
    custom_system_prompt: Optional[str] = None
    tools_enabled: bool = True
    
    def __post_init__(self):
        """Validate node configuration"""
        if not self.provider:
            raise ValueError("provider cannot be empty")
        if not self.model:
            raise ValueError("model cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

@dataclass
class DatabaseConfig:
    """Database configuration with connection validation"""
    type: str = "sqlite"  # sqlite, postgresql
    url: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    
    def __post_init__(self):
        """Validate database configuration"""
        valid_types = {"sqlite", "postgresql"}
        if self.type not in valid_types:
            raise ValueError(f"Database type must be one of {valid_types}, got {self.type}")

@dataclass
class HealthCheckConfig:
    """Configuration for system health monitoring"""
    enabled: bool = True
    base_interval_seconds: int = 180  # 3 minutes
    degraded_interval_seconds: int = 90  # 1.5 minutes  
    critical_interval_seconds: int = 60  # 1 minute
    
    # Token limits for health checks
    max_health_check_tokens: int = 1
    health_check_timeout_seconds: int = 10
    
    # Caching
    healthy_agent_cache_seconds: int = 300  # 5 minutes
    llm_health_cache_seconds: int = 300  # 5 minutes
    
    # Recovery settings
    failures_before_recovery: int = 3
    recovery_cooldown_seconds: int = 600  # 10 minutes

# Add to main config
health_config: HealthCheckConfig = HealthCheckConfig()

@runtime_checkable  
class ProviderRegistryProtocol(Protocol):
    """Interface for provider registry"""
    def get_provider(self, provider_name: str) -> Optional[object]: ...
    def get_available_providers(self) -> list[str]: ...

@dataclass
class MorteyConfig:
    """
    Centralized configuration for Mortey with Python 3.13.4 enhancements
    """
    
    # Project paths
    project_root: pathlib.Path
    workspace_dir: pathlib.Path
    config_dir: pathlib.Path
    logs_dir: pathlib.Path
    
    # LLM configurations from YAML
    providers: dict[str, ProviderConfig] = field(default_factory=dict)  # Python 3.13.4 syntax
    nodes: dict[str, NodeConfig] = field(default_factory=dict)  # Python 3.13.4 syntax
    
    # Global LLM settings
    default_provider: str = "openai"
    fallback_provider: str = "anthropic"
    retry_attempts: int = 3
    timeout_seconds: int = 180
    enable_caching: bool = True
    log_requests: bool = True
    
    # Database configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # API keys for external services
    tavily_api_key: Optional[str] = None
    picovoice_access_key: Optional[str] = None
    audio_device_index: Optional[int] = None

    # Langsmith configuration
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "mortey-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: float = 180.0
    circuit_breaker_enabled: bool = True
    
    # Security settings
    allowed_file_extensions: set[str] = field(default_factory=lambda: { 
        ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml", 
        ".csv", ".xml", ".log", ".cfg", ".ini", ".toml"
    })
    max_file_size_mb: int = 50

    provider_registry: Optional[ProviderRegistryProtocol] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """Initialize with validation"""
        from config.providers import get_provider_registry
        
        registry = get_provider_registry()
        
        # Runtime validation
        required_methods = ['get_provider', 'get_available_providers']
        if not all(hasattr(registry, method) for method in required_methods):
            missing = [m for m in required_methods if not hasattr(registry, m)]
            raise TypeError(f"Provider registry missing required methods: {missing}")
        
        self.llm_config = {
            'global': {
                'default_provider': self.default_provider,
                'fallback_provider': self.fallback_provider,
                'retry_attempts': self.retry_attempts,
                'timeout_seconds': self.timeout_seconds,
                'enable_caching': self.enable_caching,
                'max_concurrent_requests': self.max_concurrent_requests,
                'request_timeout': self.request_timeout,
                'circuit_breaker_enabled': self.circuit_breaker_enabled
            },
            'providers': {name: self._provider_to_dict(provider) for name, provider in self.providers.items()},
            'nodes': {name: self._node_to_dict(node) for name, node in self.nodes.items()}
        }

        self.provider_registry = registry
        logger.debug("✅ Provider registry initialized and validated")
    

    # Quick fix NEEDS CLEANING
    def _provider_to_dict(self, provider: ProviderConfig) -> dict[str, Any]:
        """Convert ProviderConfig to dict for backward compatibility"""
        return {
            'api_key_env': provider.api_key_env,
            'api_key': provider.api_key,  # Add this
            'base_url': provider.base_url,
            'rate_limit_per_minute': provider.rate_limit_per_minute,
            'retry_attempts': provider.retry_attempts,
            'models': {name: {
                'model_id': model.model_id,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'cost_per_1k_tokens': model.cost_per_1k_tokens
            } for name, model in provider.models.items()}
        }
    
    def _node_to_dict(self, node: NodeConfig) -> dict[str, Any]:
        """Convert NodeConfig to dict for backward compatibility"""
        return {
            'provider': node.provider,
            'model': node.model,
            'max_tokens': node.max_tokens,
            'temperature': node.temperature,
            'description': node.description,
            'tools_enabled': node.tools_enabled
        }

    @classmethod
    def from_environment(cls) -> 'MorteyConfig':
        """
        Create config from environment variables and YAML config with enhanced error handling
        """
        try:
            # Detect project root
            current_file = pathlib.Path(__file__).resolve()
            project_root = current_file.parent.parent
            
            # Load .env file for API keys
            env_file = project_root / '.env'
            if env_file.exists():
                load_dotenv(env_file, override=True)
                logger.info(f"✅ Loaded .env from: {env_file}")
            else:
                logger.warning(f"⚠️ No .env file found at: {env_file}")
            
            # Set up directories with enhanced error handling
            workspace_dir = cls._get_workspace_dir(project_root)
            config_dir = project_root / "config"
            logs_dir = project_root / "logs"
            
            # Create directories with proper error handling
            for directory in [workspace_dir, config_dir, logs_dir]:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"✅ Directory ensured: {directory}")
                except PermissionError as e:
                    logger.error(f"❌ Permission denied creating directory {directory}: {e}")
                    raise
                except Exception as e:
                    logger.error(f"❌ Error creating directory {directory}: {e}")
                    raise
            
            # Load LLM configuration from YAML with enhanced error handling
            llm_config_file = config_dir / "llm_config.yaml"
            providers, nodes, global_settings = cls._load_llm_config(llm_config_file)
            
            # Create database configuration
            database_config = cls._create_database_config()
            
            # Create configuration instance
            config_instance = cls(
                project_root=project_root,
                workspace_dir=workspace_dir,
                config_dir=config_dir,
                logs_dir=logs_dir,
                providers=providers,
                nodes=nodes,
                database=database_config,
                default_provider=global_settings.get("default_provider", "anthropic"),
                fallback_provider=global_settings.get("fallback_provider", "openai"),
                retry_attempts=global_settings.get("retry_attempts", 3),
                timeout_seconds=global_settings.get("timeout_seconds", 30),
                enable_caching=global_settings.get("enable_caching", True),
                log_requests=global_settings.get("log_requests", True),
                max_concurrent_requests=global_settings.get("max_concurrent_requests", 10),
                request_timeout=global_settings.get("request_timeout", 30.0),
                circuit_breaker_enabled=global_settings.get("circuit_breaker_enabled", True),
                tavily_api_key=os.getenv("TAVILY_API_KEY"),
                picovoice_access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
                audio_device_index=cls._get_audio_device_index(),
                langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
                langsmith_project=os.getenv("LANGSMITH_PROJECT", "mortey-assistant"),
                langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                langsmith_tracing=os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
            )
            
            # Validate the configuration
            config_instance._validate_configuration()
            
            logger.info("✅ Configuration loaded successfully")
            return config_instance
            
        except Exception as e:
            logger.error(f"❌ Failed to create configuration: {e}")
            raise
    
    @classmethod
    def _load_llm_config(cls, config_file: pathlib.Path) -> tuple[dict[str, ProviderConfig], dict[str, NodeConfig], dict[str, Any]]:
        """Load LLM configuration with modular provider support"""
        
        if not config_file.exists():
            logger.warning(f"❌ LLM config file not found: {config_file}")
            return {}, {}, {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not isinstance(config_data, dict):
                logger.error("❌ LLM config file must contain a YAML dictionary")
                return {}, {}, {}
            
            logger.info(f"✅ Loaded LLM config from: {config_file}")
            
            # 🔥 NEW: Check if using modular provider system
            if 'provider_config_path' in config_data:
                logger.info("🔧 Using modular provider configuration")
                return cls._load_modular_config(config_file.parent, config_data)
            else:
                logger.info("🔧 Using traditional provider configuration")
                return cls._load_traditional_config(config_data)
                
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error in {config_file}: {e}")
            return {}, {}, {}
        except Exception as e:
            logger.error(f"❌ Error loading LLM config: {e}")
            return {}, {}, {}

    @classmethod  
    def _load_modular_config(cls, config_dir: pathlib.Path, main_config: dict) -> tuple[dict[str, ProviderConfig], dict[str, NodeConfig], dict[str, Any]]:
        """Load configuration from modular provider system"""
        
        try:
            # Load provider registry
            registry_path = config_dir / main_config['provider_config_path']
            
            if not registry_path.exists():
                logger.error(f"❌ Provider registry file not found: {registry_path}")
                return {}, {}, {}
            
            with open(registry_path, 'r') as f:
                registry_data = yaml.safe_load(f)
            
            providers = {}
            
            # Load each provider configuration
            active_providers = registry_data.get('active_providers', [])
            logger.info(f"🔧 Loading {len(active_providers)} active providers: {active_providers}")
            
            for provider_name in active_providers:
                provider_file = config_dir / 'providers' / f'{provider_name}.yaml'
                
                if not provider_file.exists():
                    logger.warning(f"⚠️ Provider file not found: {provider_file}")
                    continue
                
                try:
                    with open(provider_file, 'r') as f:
                        provider_data = yaml.safe_load(f)
                    
                    # Check if API key is available
                    api_key_env = provider_data['connection']['api_key_env']
                    api_key = os.getenv(api_key_env)
                    
                    if not api_key:
                        logger.warning(f"⚠️ Provider {provider_name} skipped (no API key for {api_key_env})")
                        continue
                    
                    # Convert modular format to ProviderConfig
                    models = {}
                    for model_name, model_data in provider_data['models'].items():
                        models[model_name] = ModelConfig(
                            model_id=model_name,  # 🔥 FIX: Use model_name, not nested model_id
                            max_tokens=model_data['max_tokens'],
                            temperature=0.7,  # Default
                            cost_per_1k_tokens=model_data.get('cost_per_1k_tokens', 0.0)
                        )
                    
                    providers[provider_name] = ProviderConfig(
                        api_key_env=api_key_env,
                        models=models,
                        api_key=api_key,
                        base_url=provider_data['connection']['base_url'],
                        rate_limit_per_minute=provider_data['rate_limits']['requests_per_minute'],
                        retry_attempts=provider_data['connection']['max_retries']
                    )
                    
                    logger.info(f"✅ Provider {provider_name} configured with {len(models)} models")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading provider {provider_name}: {e}")
                    continue
            
            # Process nodes with template expansion        
            nodes = {}
            for node_name, node_data in main_config.get('nodes', {}).items():
                try:
                    provider_name = node_data['provider']
                    template_name = node_data.get('template', 'chat_default')
                    
                    if provider_name not in providers:
                        logger.warning(f"⚠️ Node {node_name} references unknown provider {provider_name}")
                        continue
                    
                    # Load provider file to get template
                    provider_file = config_dir / 'providers' / f'{provider_name}.yaml'
                    with open(provider_file, 'r') as f:
                        provider_config = yaml.safe_load(f)
                    
                    template = provider_config.get('node_templates', {}).get(template_name, {})
                    
                    # Merge template with node overrides
                    merged_config = {**template, **node_data}
                    
                    # 🔥 FIX: Use actual model names from provider
                    model_name = merged_config['model']
                    if model_name not in providers[provider_name].models:
                        # Try to find a valid model
                        available_models = list(providers[provider_name].models.keys())
                        if available_models:
                            model_name = available_models[0]
                            logger.warning(f"⚠️ Node {node_name} model not found, using {model_name}")
                        else:
                            logger.warning(f"⚠️ No models available for provider {provider_name}")
                            continue
                    
                    nodes[node_name] = NodeConfig(
                        provider=provider_name,
                        model=model_name,
                        max_tokens=merged_config.get('max_tokens', 1500),
                        temperature=merged_config.get('temperature', 0.7),
                        description=merged_config.get('description', ''),
                        tools_enabled=merged_config.get('enable_functions', False)
                    )
                    
                    logger.debug(f"✅ Node {node_name} configured with model {model_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing node {node_name}: {e}")
                    continue
            
            # Global settings from registry
            global_settings = registry_data.get('global', {})
            
            logger.info(f"✅ Modular config loaded: {len(providers)} providers, {len(nodes)} nodes")
            return providers, nodes, global_settings
            
        except Exception as e:
            logger.error(f"❌ Error loading modular config: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}, {}

    @classmethod
    def _load_traditional_config(cls, config_data: dict) -> tuple[dict[str, ProviderConfig], dict[str, NodeConfig], dict[str, Any]]:
        """Load configuration from traditional format (your existing logic)"""
        
        # Parse providers with enhanced validation
        providers: dict[str, ProviderConfig] = {}
        for provider_name, provider_data in config_data.get('providers', {}).items():
            try:
                api_key = os.getenv(provider_data['api_key_env'])
                
                # Parse models for this provider
                models: dict[str, ModelConfig] = {}
                for model_name, model_data in provider_data.get('models', {}).items():
                    try:
                        models[model_name] = ModelConfig(
                            model_id=model_data['model_id'],
                            max_tokens=model_data['max_tokens'],
                            temperature=model_data['temperature'],
                            cost_per_1k_tokens=model_data.get('cost_per_1k_tokens', 0.0)
                        )
                    except Exception as e:
                        logger.error(f"❌ Error parsing model {model_name}: {e}")
                        continue
                
                if api_key:  # Only add provider if API key is available
                    providers[provider_name] = ProviderConfig(
                        api_key_env=provider_data['api_key_env'],
                        models=models,
                        api_key=api_key,
                        base_url=provider_data.get('base_url'),
                        rate_limit_per_minute=provider_data.get('rate_limit_per_minute', 60),
                        retry_attempts=provider_data.get('retry_attempts', 3)
                    )
                    logger.info(f"✅ Provider {provider_name} configured with {len(models)} models")
                else:
                    logger.warning(f"⚠️ Provider {provider_name} skipped (no API key for {provider_data['api_key_env']})")
                    
            except Exception as e:
                logger.error(f"❌ Error parsing provider {provider_name}: {e}")
                continue
        
        # Parse nodes with enhanced validation
        nodes: dict[str, NodeConfig] = {}
        for node_name, node_data in config_data.get('nodes', {}).items():
            try:
                # Validate that the provider exists
                provider_name = node_data['provider']
                if provider_name not in providers:
                    logger.warning(f"⚠️ Node {node_name} references unknown provider {provider_name}")
                    continue
                
                # Validate that the model exists for the provider
                model_name = node_data['model']
                if model_name not in providers[provider_name].models:
                    logger.warning(f"⚠️ Node {node_name} references unknown model {model_name} for provider {provider_name}")
                    continue
                
                nodes[node_name] = NodeConfig(
                    provider=provider_name,
                    model=model_name,
                    max_tokens=node_data['max_tokens'],
                    temperature=node_data['temperature'],
                    description=node_data.get('description', ''),
                    custom_system_prompt=node_data.get('custom_system_prompt'),
                    tools_enabled=node_data.get('tools_enabled', True)
                )
                logger.debug(f"✅ Node {node_name} configured")
                
            except Exception as e:
                logger.error(f"❌ Error parsing node {node_name}: {e}")
                continue
        
        global_settings = config_data.get('global', {})
        
        return providers, nodes, global_settings

    
    @classmethod
    def _create_database_config(cls) -> DatabaseConfig:
        """Create database configuration from environment variables"""
        database_url = os.getenv("DATABASE_URL")
        postgres_url = os.getenv("POSTGRES_URL")
        
        if postgres_url:
            return DatabaseConfig(
                type="postgresql",
                url=postgres_url,
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                pool_timeout=float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
            )
        elif database_url:
            db_type = "postgresql" if database_url.startswith("postgresql://") else "sqlite"
            return DatabaseConfig(type=db_type, url=database_url)
        else:
            return DatabaseConfig(type="sqlite")
    
    def _validate_configuration(self) -> None:
        """Validate the entire configuration for consistency and completeness"""
        validation_errors: list[str] = []  # Python 3.13.4 syntax
        
        # Validate that we have at least one provider
        if not self.providers:
            validation_errors.append("No LLM providers configured")
        
        # Validate that default provider exists
        if self.default_provider not in self.providers:
            if self.providers:
                # Auto-fix: use first available provider
                first_provider = next(iter(self.providers.keys()))
                logger.warning(f"⚠️ Default provider '{self.default_provider}' not found. Using '{first_provider}'")
                self.default_provider = first_provider
            else:
                validation_errors.append(f"Default provider '{self.default_provider}' not configured")
        
        # Validate that fallback provider exists
        if self.fallback_provider not in self.providers:
            if len(self.providers) > 1:
                # Auto-fix: use a different provider than default
                fallback_candidates = [p for p in self.providers.keys() if p != self.default_provider]
                if fallback_candidates:
                    self.fallback_provider = fallback_candidates[0]
                    logger.warning(f"⚠️ Fallback provider '{self.fallback_provider}' not found. Using '{self.fallback_provider}'")
            else:
                logger.warning(f"⚠️ Fallback provider '{self.fallback_provider}' not configured")
        
        # Validate nodes reference valid providers and models
        for node_name, node_config in self.nodes.items():
            if node_config.provider not in self.providers:
                validation_errors.append(f"Node '{node_name}' references unknown provider '{node_config.provider}'")
                continue
            
            provider = self.providers[node_config.provider]
            if node_config.model not in provider.models:
                validation_errors.append(f"Node '{node_name}' references unknown model '{node_config.model}' for provider '{node_config.provider}'")
        
        # Validate workspace directory is writable
        try:
            test_file = self.workspace_dir / ".config_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            validation_errors.append(f"Workspace directory not writable: {e}")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validation passed")
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.providers.get(provider_name)

    def get_provider_config_from_registry(self, provider_name: str):
        """Get configuration for a specific provider from registry (if needed)"""
        return self.provider_registry.get_provider(provider_name)
    
    def get_node_config(self, node_name: str) -> Optional[NodeConfig]:
        """Get configuration for a specific node"""
        return self.nodes.get(node_name)
    
    def get_model_config(self, provider_name: str, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        provider = self.providers.get(provider_name)
        if provider:
            return provider.models.get(model_name)
        return None
    
    def get_available_providers(self) -> list[str]:
        """Get list of available providers"""
        return self.provider_registry.get_available_providers()

    def get_node_config_with_provider(self, node_name: str) -> dict[str, Any]:
        """Get node configuration merged with provider settings"""
        node_config = self.llm_config['nodes'].get(node_name, {})
        provider_name = node_config.get('provider')
        
        if provider_name:
            provider_config = self.provider_registry.get_provider(provider_name)
            if provider_config:
                # Merge node config with provider template
                template_name = node_config.get('template', 'chat_default')
                template_config = provider_config.node_templates.get(template_name, {})
                
                # Merge: template -> node overrides
                merged_config = {**template_config, **node_config}
                merged_config['provider_info'] = provider_config
                return merged_config
        
        return node_config
    
    def get_available_models(self, provider_name: str) -> list[str]:  # Python 3.13.4 syntax
        """Get list of available models for a provider"""
        provider = self.providers.get(provider_name)
        if provider:
            return list(provider.models.keys())
        return []
    
    def get_available_nodes(self) -> list[str]:  # Python 3.13.4 syntax
        """Get list of available nodes"""
        return list(self.nodes.keys())
    
    @staticmethod
    def _get_workspace_dir(project_root: pathlib.Path) -> pathlib.Path:
        """Get workspace directory with fallback options and enhanced logic"""
        # Priority 1: Environment variable
        if workspace_env := os.getenv("MORTEY_WORKSPACE_DIR"):
            workspace_path = pathlib.Path(workspace_env).resolve()
            if workspace_path.exists() or workspace_path.parent.exists():
                return workspace_path
            else:
                logger.warning(f"⚠️ MORTEY_WORKSPACE_DIR points to invalid path: {workspace_path}")
        
        # Priority 2: Project workspace directory
        workspace = project_root / "workspace"
        if workspace.exists() or workspace.parent.exists():
            return workspace
        
        # Priority 3: Home directory workspace
        home_workspace = pathlib.Path.home() / "Mortey" / "workspace"
        if home_workspace.exists() or home_workspace.parent.exists():
            return home_workspace
        
        # Fallback: Use project workspace (will be created)
        return workspace
    
    @staticmethod
    def _get_audio_device_index() -> Optional[int]:
        """Get audio device index from environment with validation"""
        if device_str := os.getenv("MORTEY_AUDIO_DEVICE"):
            try:
                device_index = int(device_str)
                if device_index >= 0:
                    return device_index
                else:
                    logger.warning(f"⚠️ Invalid audio device index: {device_index} (must be >= 0)")
            except ValueError:
                logger.warning(f"⚠️ Invalid audio device index format: {device_str}")
        return None
    
    def validate_workspace(self) -> None:
        """Ensure workspace directory is writable with detailed error reporting"""
        try:
            # Test write permission
            test_file = self.workspace_dir / "permission_test"
            test_file.touch()
            test_file.unlink()
            logger.debug(f"✅ Workspace directory is writable: {self.workspace_dir}")
        except PermissionError as e:
            error_msg = f"Workspace directory not writable: {self.workspace_dir}"
            logger.critical(f"❌ {error_msg}: {e}")
            raise PermissionError(error_msg) from e
        except Exception as e:
            error_msg = f"Workspace validation failed: {self.workspace_dir}"
            logger.critical(f"❌ {error_msg}: {e}")
            raise RuntimeError(error_msg) from e
    
    def get_configuration_summary(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get a summary of the current configuration for debugging"""
        return {
            "project_root": str(self.project_root),
            "workspace_dir": str(self.workspace_dir),
            "providers_count": len(self.providers),
            "providers": list(self.providers.keys()),
            "nodes_count": len(self.nodes),
            "nodes": list(self.nodes.keys()),
            "default_provider": self.default_provider,
            "fallback_provider": self.fallback_provider,
            "database_type": self.database.type,
            "langsmith_enabled": bool(self.langsmith_api_key),
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "max_concurrent_requests": self.max_concurrent_requests
        }
    
    def update_provider_config(self, provider_name: str, config: ProviderConfig) -> None:
        """Update configuration for a specific provider"""
        self.providers[provider_name] = config
        logger.info(f"✅ Updated configuration for provider: {provider_name}")
    
    def update_node_config(self, node_name: str, config: NodeConfig) -> None:
        """Update configuration for a specific node"""
        # Validate that the provider and model exist
        if config.provider not in self.providers:
            raise ValueError(f"Provider '{config.provider}' not configured")
        
        provider = self.providers[config.provider]
        if config.model not in provider.models:
            raise ValueError(f"Model '{config.model}' not found for provider '{config.provider}'")
        
        self.nodes[node_name] = config
        logger.info(f"✅ Updated configuration for node: {node_name}")
    
    def reload_llm_config(self) -> None:
        """Reload LLM configuration from YAML file"""
        llm_config_file = self.config_dir / "llm_config.yaml"
        providers, nodes, global_settings = self._load_llm_config(llm_config_file)
        
        self.providers = providers
        self.nodes = nodes
        
        # Update global settings if provided
        if global_settings:
            for key, value in global_settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Re-validate configuration
        self._validate_configuration()
        logger.info("✅ LLM configuration reloaded successfully")

# Global config instance
config = MorteyConfig.from_environment()
