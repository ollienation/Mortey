#!/usr/bin/env python3
import asyncio
import sys
import signal
import os
from pathlib import Path

# Add project root to path dynamically
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root / "src"))

from core.voice_assistant import create_mortey_assistant
from config.settings import config

def signal_handler(signum, frame):
    print(f"\n🛑 Shutting down...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

async def main():
    print("🤖 Welcome to Mortey - Your Voice Assistant!")
    print(f"📁 Workspace: {config.workspace_dir}")
    print(f"🔧 Project root: {config.project_root}")
    print("🐝 Say 'Bumblebee' to wake Mortey")
    print("💡 Press Ctrl+C to exit")
    print("=" * 50)
    
    # Verify configuration
    if not config.anthropic_api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("💡 Copy .env.template to .env and add your API keys")
        return
    
    mortey = create_mortey_assistant()
    await mortey.start()

if __name__ == "__main__":
    asyncio.run(main())