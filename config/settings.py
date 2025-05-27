import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from dotenv import load_dotenv

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider"""
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None  # For custom endpoints
    max_tokens: int = 4000
    temperature: float = 0.7

@dataclass
class MorteyConfig:
    """Centralized configuration for Mortey with multi-provider support"""
    
    # Project paths
    project_root: pathlib.Path
    workspace_dir: pathlib.Path
    config_dir: pathlib.Path
    logs_dir: pathlib.Path
    
    # LLM Provider configurations
    llm_providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    
    # Default providers for different tasks
    default_chat_provider: str = "anthropic"
    default_routing_provider: str = "anthropic"
    default_coding_provider: str = "anthropic"
    default_web_provider: str = "anthropic"
    
    # Other API keys
    tavily_api_key: Optional[str] = None
    picovoice_access_key: Optional[str] = None
    
    # Audio settings
    audio_device_index: Optional[int] = None
    
    @classmethod
    def from_environment(cls) -> 'MorteyConfig':
        """Create config from environment variables and auto-detection"""
        
        # Detect project root
        current_file = pathlib.Path(__file__).resolve()
        project_root = current_file.parent.parent
        
        # Load .env file
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file, override=True)
        
        # Set up directories
        workspace_dir = cls._get_workspace_dir(project_root)
        config_dir = project_root / "config"
        logs_dir = project_root / "logs"
        
        for directory in [workspace_dir, config_dir, logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load all provider configurations
        llm_providers = cls._load_provider_configs()
        
        return cls(
            project_root=project_root,
            workspace_dir=workspace_dir,
            config_dir=config_dir,
            logs_dir=logs_dir,
            llm_providers=llm_providers,
            default_chat_provider=os.getenv("MORTEY_DEFAULT_CHAT_PROVIDER", "anthropic"),
            default_routing_provider=os.getenv("MORTEY_DEFAULT_ROUTING_PROVIDER", "anthropic"),
            default_coding_provider=os.getenv("MORTEY_DEFAULT_CODING_PROVIDER", "anthropic"),
            default_web_provider=os.getenv("MORTEY_DEFAULT_WEB_PROVIDER", "anthropic"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            picovoice_access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            audio_device_index=cls._get_audio_device_index()
        )
    
    @classmethod
    def _load_provider_configs(cls) -> Dict[str, ProviderConfig]:
        """Load all available LLM provider configurations"""
        providers = {}
        
        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            providers["anthropic"] = ProviderConfig(
                api_key=anthropic_key,
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
            )
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            providers["openai"] = ProviderConfig(
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
            )
        
        # Google Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            providers["gemini"] = ProviderConfig(
                api_key=gemini_key,
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
            )
        
        # Local models (Ollama, etc.)
        local_model = os.getenv("LOCAL_MODEL")
        if local_model:
            providers["local"] = ProviderConfig(
                api_key=None,
                model=local_model,
                base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434"),
                max_tokens=int(os.getenv("LOCAL_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("LOCAL_TEMPERATURE", "0.7"))
            )
        
        # Cohere
        cohere_key = os.getenv("COHERE_API_KEY")
        if cohere_key:
            providers["cohere"] = ProviderConfig(
                api_key=cohere_key,
                model=os.getenv("COHERE_MODEL", "command-r-plus"),
                max_tokens=int(os.getenv("COHERE_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("COHERE_TEMPERATURE", "0.7"))
            )
        
        return providers
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.llm_providers.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.llm_providers.keys())
    
    def get_default_provider_for_task(self, task: str) -> str:
        """Get default provider for a specific task"""
        task_mapping = {
            "chat": self.default_chat_provider,
            "routing": self.default_routing_provider,
            "coding": self.default_coding_provider,
            "web": self.default_web_provider
        }
        return task_mapping.get(task, self.default_chat_provider)
    
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
