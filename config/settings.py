import os
import pathlib
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from dotenv import load_dotenv

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_id: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float = 0.0

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider"""
    api_key_env: str
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    api_key: Optional[str] = None

@dataclass
class NodeConfig:
    """Configuration for a specific node/agent"""
    provider: str
    model: str
    max_tokens: int
    temperature: float
    description: str = ""

@dataclass
class MorteyConfig:
    """Centralized configuration for Mortey"""
    
    # Project paths
    project_root: pathlib.Path
    workspace_dir: pathlib.Path
    config_dir: pathlib.Path
    logs_dir: pathlib.Path
    
    # LLM configurations from YAML
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    nodes: Dict[str, NodeConfig] = field(default_factory=dict)
    
    # Global LLM settings
    default_provider: str = "anthropic"
    fallback_provider: str = "openai"
    retry_attempts: int = 3
    timeout_seconds: int = 30
    enable_caching: bool = True
    log_requests: bool = True
    
    # Other API keys (still from env)
    tavily_api_key: Optional[str] = None
    picovoice_access_key: Optional[str] = None
    audio_device_index: Optional[int] = None
    
    @classmethod
    def from_environment(cls) -> 'MorteyConfig':
        """Create config from environment variables and YAML config"""
        
        # Detect project root
        current_file = pathlib.Path(__file__).resolve()
        project_root = current_file.parent.parent
        
        # Load .env file for API keys
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file, override=True)
            print(f"✅ Loaded .env from: {env_file}")
        
        # Set up directories
        workspace_dir = cls._get_workspace_dir(project_root)
        config_dir = project_root / "config"
        logs_dir = project_root / "logs"
        
        for directory in [workspace_dir, config_dir, logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load LLM configuration from YAML
        llm_config_file = config_dir / "llm_config.yaml"
        providers, nodes, global_settings = cls._load_llm_config(llm_config_file)
        
        return cls(
            project_root=project_root,
            workspace_dir=workspace_dir,
            config_dir=config_dir,
            logs_dir=logs_dir,
            providers=providers,
            nodes=nodes,
            default_provider=global_settings.get("default_provider", "anthropic"),
            fallback_provider=global_settings.get("fallback_provider", "openai"),
            retry_attempts=global_settings.get("retry_attempts", 3),
            timeout_seconds=global_settings.get("timeout_seconds", 30),
            enable_caching=global_settings.get("enable_caching", True),
            log_requests=global_settings.get("log_requests", True),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            picovoice_access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            audio_device_index=cls._get_audio_device_index()
        )
    
    @classmethod
    def _load_llm_config(cls, config_file: pathlib.Path) -> tuple:
        """Load LLM configuration from YAML file"""
        
        if not config_file.exists():
            print(f"❌ LLM config file not found: {config_file}")
            return {}, {}, {}
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            print(f"✅ Loaded LLM config from: {config_file}")
            
            # Parse providers
            providers = {}
            for provider_name, provider_data in config_data.get('providers', {}).items():
                api_key = os.getenv(provider_data['api_key_env'])
                
                models = {}
                for model_name, model_data in provider_data.get('models', {}).items():
                    models[model_name] = ModelConfig(
                        model_id=model_data['model_id'],
                        max_tokens=model_data['max_tokens'],
                        temperature=model_data['temperature'],
                        cost_per_1k_tokens=model_data.get('cost_per_1k_tokens', 0.0)
                    )
                
                if api_key:  # Only add provider if API key is available
                    providers[provider_name] = ProviderConfig(
                        api_key_env=provider_data['api_key_env'],
                        models=models,
                        api_key=api_key
                    )
                    print(f"✅ Provider {provider_name} configured with {len(models)} models")
                else:
                    print(f"⚠️ Provider {provider_name} skipped (no API key)")
            
            # Parse nodes
            nodes = {}
            for node_name, node_data in config_data.get('nodes', {}).items():
                nodes[node_name] = NodeConfig(
                    provider=node_data['provider'],
                    model=node_data['model'],
                    max_tokens=node_data['max_tokens'],
                    temperature=node_data['temperature'],
                    description=node_data.get('description', '')
                )
            
            global_settings = config_data.get('global', {})
            
            return providers, nodes, global_settings
            
        except Exception as e:
            print(f"❌ Error loading LLM config: {e}")
            return {}, {}, {}
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.providers.get(provider_name)
    
    def get_node_config(self, node_name: str) -> Optional[NodeConfig]:
        """Get configuration for a specific node"""
        return self.nodes.get(node_name)
    
    def get_model_config(self, provider_name: str, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        provider = self.providers.get(provider_name)
        if provider:
            return provider.models.get(model_name)
        return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_available_models(self, provider_name: str) -> List[str]:
        """Get list of available models for a provider"""
        provider = self.providers.get(provider_name)
        if provider:
            return list(provider.models.keys())
        return []
    
    @staticmethod
    def _get_workspace_dir(project_root: pathlib.Path) -> pathlib.Path:
        """Get workspace directory with fallback options"""
        if workspace_env := os.getenv("MORTEY_WORKSPACE_DIR"):
            return pathlib.Path(workspace_env).resolve()
        
        workspace = project_root / "workspace"
        if not workspace.exists():
            home_workspace = pathlib.Path.home() / "Mortey" / "workspace"
            if home_workspace.exists():
                return home_workspace
        
        return workspace
    
    @staticmethod
    def _get_audio_device_index() -> Optional[int]:
        """Get audio device index from environment"""
        if device_str := os.getenv("MORTEY_AUDIO_DEVICE"):
            try:
                return int(device_str)
            except ValueError:
                return None
        return None

# Global config instance
config = MorteyConfig.from_environment()
