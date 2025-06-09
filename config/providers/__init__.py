# PROVIDER CONFIGURATION LOADER
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    name: str
    display_name: str
    type: str
    connection: Dict[str, Any]
    capabilities: Dict[str, bool]
    rate_limits: Dict[str, int]
    models: Dict[str, Any]
    node_templates: Dict[str, Any]

class ProviderRegistry:
    """Modular provider configuration registry"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.registry_config = self._load_registry()
        self.providers: Dict[str, ProviderConfig] = {}
        self._load_all_providers()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the main provider registry"""
        registry_path = self.config_dir / "registry.yaml"
        with open(registry_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_all_providers(self):
        """Load all active provider configurations"""
        active_providers = self.registry_config.get('active_providers', [])
        
        for provider_name in active_providers:
            try:
                provider_config = self._load_provider_config(provider_name)
                self.providers[provider_name] = provider_config
            except Exception as e:
                print(f"⚠️ Failed to load provider {provider_name}: {e}")
    
    def _load_provider_config(self, provider_name: str) -> ProviderConfig:
        """Load configuration for a specific provider"""
        provider_file = self.config_dir / f"{provider_name}.yaml"
        
        with open(provider_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return ProviderConfig(
            name=provider_name,
            display_name=config_data['provider_info']['display_name'],
            type=config_data['provider_info']['type'],
            connection=config_data['connection'],
            capabilities=config_data['capabilities'],
            rate_limits=config_data['rate_limits'],
            models=config_data['models'],
            node_templates=config_data['node_templates']
        )
    
    def get_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.providers.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def get_default_provider(self) -> str:
        """Get the default provider name"""
        return self.registry_config.get('default_provider', 'openai')
    
    def get_fallback_provider(self) -> str:
        """Get the fallback provider name"""
        return self.registry_config.get('fallback_provider', 'anthropic')
    
    def get_node_config(self, node_name: str, provider_override: str = None) -> Dict[str, Any]:
        """Get complete configuration for a node"""
        # This would integrate with your existing node configuration logic
        pass

# Global registry instance
provider_registry: Optional[ProviderRegistry] = None

def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance"""
    global provider_registry
    if provider_registry is None:
        config_dir = Path(__file__).parent
        provider_registry = ProviderRegistry(config_dir)
    return provider_registry
