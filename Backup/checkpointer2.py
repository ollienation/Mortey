# Fixed Checkpointer Configuration - For LangGraph 0.4.8
# June 2025 - Production Ready

import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("checkpointer")

class CheckpointerFactory:
    """
    Factory for creating production-ready checkpointers using dedicated libraries.
    
    ✅ FIXED for LangGraph 0.4.8 with correct import paths and packages.
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
            
        if environment == "production":
            return cls._create_postgres_checkpointer()
        else:
            return cls._create_sqlite_checkpointer()

    @classmethod
    def _detect_environment(cls) -> str:
        """Auto-detect environment based on available configurations"""
        # Check for production database URL
        if os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL"):
            return "production"
            
        # Check for explicit environment setting
        env = os.getenv("ENVIRONMENT", "").lower()
        if env in ["prod", "production"]:
            return "production"
            
        return "development"

    # Updated SQLite checkpointer with proper path handling
    @classmethod
    def _create_sqlite_checkpointer(cls):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from config.settings import config
            
            # Ensure workspace exists
            config.workspace_dir.mkdir(parents=True, exist_ok=True)
            db_path = config.workspace_dir / "assistant_memory.db"
            
            # Use connection pool for production reliability
            checkpointer = SqliteSaver.from_conn_string(
                f"sqlite:///{db_path}?check_same_thread=false",
                pool_size=20,
                timeout=30
            )
            checkpointer.setup()
            logger.info(f"✅ SQLite checkpointer initialized at {db_path}")
            return checkpointer
        except Exception as e:
            logger.error(f"SQLite checkpointer failed: {e}")
            return cls._create_memory_checkpointer()

    # Modern PostgreSQL checkpointer with connection pool
    @classmethod
    def _create_postgres_checkpointer(cls):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg_pool
            
            conn_str = os.getenv("POSTGRES_URL")
            pool = psycopg_pool.ConnectionPool(conn_str, min_size=5, max_size=20)
            
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            logger.info("✅ PostgreSQL checkpointer with connection pool initialized")
            return checkpointer
        except Exception as e:
            logger.error(f"PostgreSQL checkpointer failed: {e}")
            return cls._create_sqlite_checkpointer()

    @classmethod
    def _create_memory_checkpointer(cls):
        """Create in-memory checkpointer as fallback"""
        try:
            # ✅ FIXED: Correct import for memory checkpointer
            from langgraph.checkpoint.memory import MemorySaver
            
            logger.warning("⚠️ Using MemorySaver - conversations will not persist between restarts")
            return MemorySaver()
        except Exception as e:
            logger.error(f"❌ Even MemorySaver failed: {e}")
            raise RuntimeError("No checkpointer could be created")

class AsyncCheckpointerFactory:
    """Async version of the checkpointer factory for async applications"""

    @classmethod
    async def create_checkpointer(cls, environment: str = "auto") -> Any:
        """Create async checkpointer with proper await"""
        environment = cls._detect_environment()
        try:
            if environment == "production":
                return await cls._create_async_postgres_checkpointer()
            return await cls._create_async_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"Checkpointer creation failed: {e}")
            return cls._create_memory_checkpointer()

    @classmethod
    def _detect_environment(cls) -> str:
        """Auto-detect environment based on available configurations"""
        return CheckpointerFactory._detect_environment()

    @classmethod
    async def _create_async_postgres_checkpointer(cls):
        """
        Create async PostgreSQL checkpointer using modern pattern.
        
        ✅ FIXED: Uses proper package langgraph-checkpoint-postgres
        """
        try:
            # ✅ FIXED: Use the separate checkpoint package
            import langgraph.checkpoint.postgres
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            
            db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
            if not db_url:
                logger.warning("No PostgreSQL URL found, falling back to SQLite")
                return await cls._create_async_sqlite_checkpointer()
                
            # Create async checkpointer with simplified initialization
            checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
            
            # Setup database tables
            try:
                await checkpointer.setup()
                logger.info("✅ Async PostgreSQL checkpointer initialized successfully")
                return checkpointer
            except Exception as setup_error:
                logger.error(f"❌ Async PostgreSQL setup failed: {setup_error}")
                return await cls._create_async_sqlite_checkpointer()
                
        except ImportError as e:
            logger.warning(f"Async PostgreSQL not available: {e}")
            return await cls._create_async_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"❌ Async PostgreSQL checkpointer creation failed: {e}")
            return await cls._create_async_sqlite_checkpointer()

    @classmethod
    async def _create_async_sqlite_checkpointer(cls):
        """
        Create async SQLite checkpointer using modern pattern.
        
        ✅ FIXED: Uses proper package langgraph-checkpoint-sqlite
        """
        try:
            # ✅ FIXED: Use the separate checkpoint package
            import langgraph.checkpoint.sqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            
            from config.settings import config
            db_path = config.workspace_dir / "assistant_memory.db"
            
            # Ensure directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create async checkpointer with simplified initialization
            checkpointer = AsyncSqliteSaver.from_conn_string(f"sqlite:///{db_path}")
            
            # Setup database tables
            try:
                await checkpointer.setup()
                logger.info(f"✅ Async SQLite checkpointer initialized: {db_path}")
                return checkpointer
            except Exception as setup_error:
                logger.error(f"❌ Async SQLite setup failed: {setup_error}")
                return cls._create_memory_checkpointer()
                
        except ImportError as e:
            logger.warning(f"Async SQLite not available: {e}")
            return cls._create_memory_checkpointer()
        except Exception as e:
            logger.error(f"❌ Async SQLite checkpointer creation failed: {e}")
            return cls._create_memory_checkpointer()

    @classmethod
    def _create_memory_checkpointer(cls):
        """Create in-memory checkpointer as fallback"""
        return CheckpointerFactory._create_memory_checkpointer()

def get_checkpointer_info() -> Dict[str, Any]:
    """Get information about the current checkpointer configuration"""
    environment = CheckpointerFactory._detect_environment()
    
    info = {
        "detected_environment": environment,
        "database_url_available": bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")),
        "postgres_available": False,
        "sqlite_available": False,
        "memory_fallback": True
    }
    
    # Check package availability
    try:
        import langgraph.checkpoint.postgres
        info["postgres_available"] = True
    except ImportError:
        pass
    
    try:
        import langgraph.checkpoint.sqlite
        info["sqlite_available"] = True
    except ImportError:
        pass
    
    return info

def create_production_checkpointer():
    """
    Create a production-ready checkpointer using CheckpointerFactory.
    This function is required by assistant_core.py for initialization.
    """
    return CheckpointerFactory.create_checkpointer(environment="production")

def create_checkpointer():
    """
    Create a checkpointer using CheckpointerFactory.
    This is the main entry point for creating checkpointers.
    Used by assistant_core.py to initialize the checkpointer.
    
    Returns:
        A checkpointer instance from CheckpointerFactory
    """
    return CheckpointerFactory.create_checkpointer()

# ✅ FIXED: Migration helper for old code
def create_checkpointer_legacy():
    """Legacy compatibility function"""
    logger.warning("⚠️ Using legacy checkpointer creation. Update to create_production_checkpointer()")
    return create_production_checkpointer()