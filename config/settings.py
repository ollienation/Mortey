import os
import pathlib
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class MorteyConfig:
    """Centralized configuration for Mortey"""
    
    # Project paths
    project_root: pathlib.Path
    workspace_dir: pathlib.Path
    config_dir: pathlib.Path
    logs_dir: pathlib.Path
    
    # API keys
    anthropic_api_key: Optional[str]
    tavily_api_key: Optional[str]
    picovoice_access_key: Optional[str]
    
    # Audio settings
    audio_device_index: Optional[int]
        
    @classmethod
    def from_environment(cls) -> 'MorteyConfig':
        """Create config from environment variables and auto-detection"""
        
        # Detect project root (where this file is relative to)
        current_file = pathlib.Path(__file__).resolve()
        project_root = current_file.parent.parent

        # Load .env file
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file, override=True)
            print(f"✅ Loaded .env from: {env_file}")
        else:
            print(f"❌ .env file not found at: {env_file}")
            print(f"🔍 Looking in: {project_root}")
        
        # Set up directories relative to project root
        workspace_dir = cls._get_workspace_dir(project_root)
        config_dir = project_root / "config"
        logs_dir = project_root / "logs"
        
        # Create directories if they don't exist
        for directory in [workspace_dir, config_dir, logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        return cls(
            project_root=project_root,
            workspace_dir=workspace_dir,
            config_dir=config_dir,
            logs_dir=logs_dir,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            picovoice_access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            audio_device_index=cls._get_audio_device_index()
        )
    
    @staticmethod
    def _get_workspace_dir(project_root: pathlib.Path) -> pathlib.Path:
        """Get workspace directory with fallback options"""
        
        # Option 1: Environment variable
        if workspace_env := os.getenv("MORTEY_WORKSPACE_DIR"):
            return pathlib.Path(workspace_env).resolve()
        
        # Option 2: Project-relative workspace
        workspace = project_root / "workspace"
        
        # Option 3: User home fallback (for backwards compatibility)
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
