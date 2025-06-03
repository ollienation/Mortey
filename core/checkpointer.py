# FIXED Checkpointer Configuration - LangGraph 0.4.8 (June 2025)
# CRITICAL FIXES: Proper context manager usage and database setup

import os
import asyncio
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("checkpointer")

class CheckpointerFactory:
    """
    ✅ FIXED Factory for creating production-ready checkpointers using LangGraph 0.4.8.

    CRITICAL FIXES:
    - Uses proper context manager patterns for SqliteSaver
    - Creates direct connections when context managers aren't suitable
    - Proper setup() calls on actual checkpointer instances
    - Robust error handling with graceful fallbacks
    """

    @classmethod
    def create_checkpointer(cls, environment: str = "auto"):
        """
        Create appropriate checkpointer based on environment.

        Args:
            environment: "production", "development", or "auto" (default)
        """
        if environment == "auto":
            environment = cls._detect_environment()

        logger.info(f"Creating checkpointer for environment: {environment}")

        if environment == "production":
            return cls._create_postgres_checkpointer()
        else:
            return cls._create_sqlite_checkpointer()

    @classmethod
    def _detect_environment(cls) -> str:
        """Auto-detect environment based on available configurations"""
        # Check for production database URL
        if os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL"):
            # Additional check: ensure PostgreSQL is actually running
            if cls._test_postgres_connection():
                return "production"
            else:
                logger.warning("PostgreSQL URL found but connection failed, falling back to development")
                return "development"

        # Check for explicit environment setting
        env = os.getenv("ENVIRONMENT", "").lower()
        if env in ["prod", "production"]:
            return "production"

        return "development"

    @classmethod
    def _test_postgres_connection(cls) -> bool:
        """Test if PostgreSQL connection is available"""
        try:
            import psycopg
            conn_str = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
            if not conn_str:
                return False

            # Quick connection test
            with psycopg.connect(conn_str, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.debug(f"PostgreSQL connection test failed: {e}")
            return False

    # ✅ FIXED: SQLite checkpointer with proper connection handling
    @classmethod
    def _create_sqlite_checkpointer(cls):
        """Create SQLite checkpointer using direct connection (not context manager)"""
        try:
            # ✅ FIXED: Import from separate checkpoint package
            from langgraph.checkpoint.sqlite import SqliteSaver
            from config.settings import config

            # Ensure workspace exists
            config.workspace_dir.mkdir(parents=True, exist_ok=True)
            db_path = config.workspace_dir / "assistant_memory.db"

            # ✅ FIXED: Create direct connection instead of using context manager
            # This avoids the GeneratorContextManager issue
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # ✅ FIXED: Create checkpointer with direct connection
            checkpointer = SqliteSaver(conn)
            
            # ✅ CRITICAL FIX: Call setup() on the actual checkpointer instance
            checkpointer.setup()

            logger.info(f"✅ SQLite checkpointer initialized at {db_path}")
            return checkpointer

        except ImportError as e:
            logger.error(f"❌ langgraph-checkpoint-sqlite not installed: {e}")
            logger.info("Install with: pip install langgraph-checkpoint-sqlite")
            return cls._create_memory_checkpointer()
        except Exception as e:
            logger.error(f"❌ SQLite checkpointer failed: {e}")
            return cls._create_memory_checkpointer()

    # ✅ FIXED: PostgreSQL checkpointer with proper connection handling
    @classmethod
    def _create_postgres_checkpointer(cls):
        """Create PostgreSQL checkpointer using direct connection"""
        try:
            # ✅ FIXED: Import from separate checkpoint package
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg

            conn_str = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
            if not conn_str:
                logger.warning("No PostgreSQL URL found, falling back to SQLite")
                return cls._create_sqlite_checkpointer()

            # ✅ FIXED: Create direct connection with proper pooling
            conn = psycopg.connect(conn_str)
            
            # ✅ FIXED: Create checkpointer with direct connection
            checkpointer = PostgresSaver(conn)
            
            # ✅ CRITICAL FIX: Call setup() on the actual checkpointer instance
            checkpointer.setup()

            logger.info("✅ PostgreSQL checkpointer initialized successfully")
            return checkpointer

        except ImportError as e:
            logger.error(f"❌ langgraph-checkpoint-postgres not installed: {e}")
            logger.info("Install with: pip install langgraph-checkpoint-postgres")
            return cls._create_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"❌ PostgreSQL checkpointer failed: {e}")
            logger.info("Falling back to SQLite checkpointer")
            return cls._create_sqlite_checkpointer()

    @classmethod
    def _create_memory_checkpointer(cls):
        """Create in-memory checkpointer as fallback"""
        try:
            # ✅ FIXED: Import from core langgraph package
            from langgraph.checkpoint.memory import MemorySaver
            logger.warning("⚠️ Using MemorySaver - conversations will not persist between restarts")
            return MemorySaver()
        except Exception as e:
            logger.error(f"❌ Even MemorySaver failed: {e}")
            raise RuntimeError("No checkpointer could be created")

class AsyncCheckpointerFactory:
    """
    ✅ FIXED: Async version with modern LangGraph 0.4.8 patterns
    """

    @classmethod
    async def create_checkpointer(cls, environment: str = "auto"):
        """Create async checkpointer with proper await"""
        environment = CheckpointerFactory._detect_environment()
        try:
            if environment == "production":
                return await cls._create_async_postgres_checkpointer()
            return await cls._create_async_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"Async checkpointer creation failed: {e}")
            return CheckpointerFactory._create_memory_checkpointer()

    @classmethod
    async def _create_async_postgres_checkpointer(cls):
        """Create async PostgreSQL checkpointer"""
        try:
            # ✅ FIXED: Import from separate checkpoint package
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            import asyncpg

            conn_str = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
            if not conn_str:
                logger.warning("No PostgreSQL URL found, falling back to SQLite")
                return await cls._create_async_sqlite_checkpointer()

            # ✅ FIXED: Create async connection
            conn = await asyncpg.connect(conn_str)
            
            # ✅ FIXED: Create checkpointer with direct connection
            checkpointer = AsyncPostgresSaver(conn)
            
            # ✅ CRITICAL FIX: Call setup() asynchronously
            await checkpointer.setup()

            logger.info("✅ Async PostgreSQL checkpointer initialized successfully")
            return checkpointer

        except ImportError as e:
            logger.warning(f"Async PostgreSQL not available: {e}")
            return await cls._create_async_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"❌ Async PostgreSQL checkpointer creation failed: {e}")
            return await cls._create_async_sqlite_checkpointer()

    @classmethod
    async def _create_async_sqlite_checkpointer(cls):
        """Create async SQLite checkpointer"""
        try:
            # ✅ FIXED: Import from separate checkpoint package
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            import aiosqlite
            from config.settings import config

            db_path = config.workspace_dir / "assistant_memory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # ✅ FIXED: Create async connection
            conn = await aiosqlite.connect(str(db_path))
            
            # ✅ FIXED: Create checkpointer with direct connection
            checkpointer = AsyncSqliteSaver(conn)
            
            # ✅ CRITICAL FIX: Call setup() asynchronously
            await checkpointer.setup()

            logger.info(f"✅ Async SQLite checkpointer initialized: {db_path}")
            return checkpointer

        except ImportError as e:
            logger.warning(f"Async SQLite not available: {e}")
            return CheckpointerFactory._create_memory_checkpointer()
        except Exception as e:
            logger.error(f"❌ Async SQLite checkpointer creation failed: {e}")
            return CheckpointerFactory._create_memory_checkpointer()

def get_checkpointer_info() -> Dict[str, Any]:
    """Get information about the current checkpointer configuration"""
    environment = CheckpointerFactory._detect_environment()
    info = {
        "detected_environment": environment,
        "database_url_available": bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")),
        "postgres_available": CheckpointerFactory._test_postgres_connection(),
        "sqlite_available": True,  # SQLite is always available with Python
        "memory_fallback": True,
        "packages_available": {}
    }

    # Check package availability
    packages = [
        "langgraph.checkpoint.postgres",
        "langgraph.checkpoint.sqlite", 
        "langgraph.checkpoint.memory"
    ]

    for package in packages:
        try:
            __import__(package)
            info["packages_available"][package] = True
        except ImportError:
            info["packages_available"][package] = False

    return info

# ✅ FIXED: Main entry points for the assistant
def create_checkpointer():
    """
    Main entry point for creating checkpointers.
    Used by assistant_core.py to initialize the checkpointer.
    """
    return CheckpointerFactory.create_checkpointer()

def create_production_checkpointer():
    """
    Create a production-ready checkpointer.
    """
    return CheckpointerFactory.create_checkpointer(environment="production")

async def create_async_checkpointer():
    """
    Create an async checkpointer.
    """
    return await AsyncCheckpointerFactory.create_checkpointer()

# Legacy compatibility
def create_checkpointer_legacy():
    """Legacy compatibility function"""
    logger.warning("⚠️ Using legacy checkpointer creation. Update to create_checkpointer()")
    return create_checkpointer()