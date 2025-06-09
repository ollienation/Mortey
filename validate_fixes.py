# test_checkpointer_fix.py
import asyncio
import os
from core.checkpointer import CheckpointerFactory

# Test script to verify the new pattern works
async def test_all_checkpointers():
    try:
        factory = CheckpointerFactory()
        
        # Test development (SQLite)
        dev_checkpointer = await factory._create_development_checkpointer()
        print(f"✅ Dev checkpointer: {type(dev_checkpointer).__name__}")
        
        # Test production (if PostgreSQL available)
        if os.getenv("POSTGRES_URL"):
            prod_checkpointer = await factory._create_production_checkpointer()
            print(f"✅ Prod checkpointer: {type(prod_checkpointer).__name__}")
        
        print("🎉 All checkpointer patterns working!")
        
    except Exception as e:
        print(f"❌ Pattern test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_checkpointers())
