#!/usr/bin/env python3
import asyncio
from core.checkpointer import create_checkpointer
from config.settings import config

async def init_db():
    checkpointer = create_checkpointer()
    if isinstance(checkpointer, SqliteSaver):
        print(f"SQLite database initialized at {config.workspace_dir}/assistant_memory.db")
    elif isinstance(checkpointer, PostgresSaver):
        print("PostgreSQL database initialized successfully")
        
if __name__ == "__main__":
    asyncio.run(init_db())
