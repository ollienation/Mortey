#!/usr/bin/env python3
"""
Mortey GUI Launcher
"""
import sys
import os
import signal
import asyncio
from pathlib import Path

# Much simpler path setup now
def setup_project_path():
    """Setup Python path - now much simpler"""
    script_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(script_dir))

# Setup path before imports
setup_project_path()

try:
    from gui.gui_manager import GUIManager
    from config.settings import config
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Shutting down Mortey GUI...")
    os._exit(0)

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import anthropic
    except ImportError:
        missing_deps.append("anthropic")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_configuration():
    """Check if essential configuration is present"""
    issues = []
    
    if not config.anthropic_api_key:
        issues.append("ANTHROPIC_API_KEY not found in environment")
    
    if not config.workspace_dir.exists():
        try:
            config.workspace_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created workspace directory: {config.workspace_dir}")
        except Exception as e:
            issues.append(f"Cannot create workspace directory: {e}")
    
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("💡 Copy .env.template to .env and configure your settings")
        return False
    
    return True

def print_startup_info():
    """Print startup information"""
    print("🤖 Mortey Assistant - GUI Mode")
    print("=" * 50)
    print(f"📁 Project root: {config.project_root}")
    print(f"🗂️  Workspace: {config.workspace_dir}")
    print(f"📝 Logs: {config.logs_dir}")
    print(f"🎤 Wake word: {os.getenv('MORTEY_WAKE_WORD', 'Bumblebee')}")
    print("=" * 50)
    print("💡 Use the GUI to chat with Mortey")
    print("🎤 Enable voice mode for hands-free interaction")
    print("💾 Chat logs are automatically saved")
    print("🛑 Press Ctrl+C to exit")
    print()

async def async_main():
    """Async main function for future compatibility"""
    try:
        manager = GUIManager()
        
        # Add welcome message with system info
        welcome_msg = f"Welcome! Running from {config.project_root.name}"
        manager.gui.add_message("System", welcome_msg, "system")
        
        # Start the GUI (this blocks until window is closed)
        manager.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error starting Mortey GUI: {e}")
        print("💡 Check your configuration and try again")
        sys.exit(1)

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    if not check_configuration():
        print("⚠️  Starting with limited functionality...")
        print()
    
    # Print startup information
    print_startup_info()
    
    # Start the application
    try:
        # For now, run synchronously since tkinter is not async
        # In the future, we might want to integrate with async event loops
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
