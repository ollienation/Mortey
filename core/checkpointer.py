import os
import asyncio
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("checkpointer")

class CheckpointerFactory:
    """
    Factory for creating modern checkpointers using LangGraph 0.4.8.
    """
    @classmethod
    async def _create_async_postgres_checkpointer(cls):
        """
        Creates the checkpointer using
        `psycopg_pool` library that LangGraph's saver expects.
        """
        try:
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            conn_str = os.getenv("POSTGRES_URL")
            if not conn_str:
                logger.warning("POSTGRES_URL not set, falling back to SQLite.")
                return await cls._create_async_sqlite_checkpointer()

            pool = None
            try:
                pool = AsyncConnectionPool(conninfo=conn_str)
                checkpointer = AsyncPostgresSaver(conn=pool)
                await checkpointer.setup()
                logger.info("✅ Async PostgreSQL checkpointer (psycopg_pool) instance created.")
                return checkpointer 
                
            except Exception as setup_error:
                logger.error(f"PostgreSQL checkpointer creation failed: {setup_error}")
                if pool:
                    await pool.close()
                return await cls._create_async_sqlite_checkpointer()
                
        except ImportError:
            logger.error("❌ Required packages for PostgreSQL are missing. Please install 'psycopg[binary]' and 'psycopg-pool'.")
            return await cls._create_async_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while creating async PostgreSQL checkpointer: {e}")
            return await cls._create_async_sqlite_checkpointer()
    
    @classmethod
    def create_checkpointer_sync(cls, environment: str = "auto"):
        """Create appropriate sync checkpointer based on environment."""
        if environment == "auto":
            environment = cls._detect_environment()

        logger.info(f"Attempting to create SYNC checkpointer for determined environment: {environment}")
        
        if environment == "production":
            pg_checkpointer = cls._create_postgres_checkpointer_sync()
            if pg_checkpointer and 'postgres' in type(pg_checkpointer).__name__.lower():
                return pg_checkpointer
            logger.warning("SYNC PostgreSQL checkpointer failed. Using SQLite fallback.")
            return cls._create_sqlite_checkpointer_sync()
        else:
            logger.info("Using SYNC SQLite checkpointer for DEVELOPMENT environment.")
            return cls._create_sqlite_checkpointer_sync()

    @classmethod
    def create_checkpointer_async(cls, environment: str = "auto"):
        """Create appropriate async checkpointer based on environment."""
        if environment == "auto":
            environment = cls._detect_environment()

        logger.info(f"Attempting to create ASYNC checkpointer for determined environment: {environment}")
        
        if environment == "production":
            pg_checkpointer = cls._create_postgres_checkpointer_async()
            if pg_checkpointer and 'postgres' in type(pg_checkpointer).__name__.lower():
                return pg_checkpointer
            logger.warning("ASYNC PostgreSQL checkpointer failed. Using SQLite fallback.")
            return cls._create_sqlite_checkpointer_async()
        else:
            logger.info("Using ASYNC SQLite checkpointer for DEVELOPMENT environment.")
            return cls._create_sqlite_checkpointer_async()

    @classmethod
    def _detect_environment(cls) -> str:
        """Auto-detect environment based on available configurations."""
        postgres_url = os.getenv("POSTGRES_URL")
        env_var_setting = os.getenv("ENVIRONMENT", "").lower()
        
        if postgres_url:
            logger.info(f"POSTGRES_URL is set: {postgres_url[:30]}...")
            if cls._test_postgres_connection_sync():
                logger.info("PostgreSQL connection successful via POSTGRES_URL. Detected environment: production")
                return "production"
            else:
                logger.warning("POSTGRES_URL is set, but PostgreSQL connection test failed.")
                if env_var_setting == "production":
                    logger.error("CRITICAL: ENVIRONMENT is 'production' and POSTGRES_URL is set, but connection failed.")
                    return "production"
                else:
                    logger.info("Falling back to development mode due to failed PostgreSQL test.")
                    return "development"
        
        if env_var_setting == "production":
            logger.warning("ENVIRONMENT is 'production', but POSTGRES_URL is not set.")
            return "production"
            
        logger.info("No explicit production indicators. Detected environment: development")
        return "development"
        
    @classmethod
    async def _create_async_sqlite_checkpointer(cls):
        """Create async SQLite checkpointer."""
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            import aiosqlite
            from config.settings import config 
            
            config.workspace_dir.mkdir(parents=True, exist_ok=True)
            sqlite_path_str = os.getenv("DATABASE_URL")
            if sqlite_path_str and sqlite_path_str.startswith("sqlite:///"):
                db_path = Path(sqlite_path_str.replace("sqlite:///", ""))
            else:
                db_path = config.workspace_dir / "assistant.db" 
            
            logger.info(f"Initializing ASYNC SQLite checkpointer at: {db_path}")
            conn = await aiosqlite.connect(str(db_path))
            checkpointer = AsyncSqliteSaver(conn=conn)
            logger.info(f"✅ Async SQLite checkpointer initialized: {db_path}")
            return checkpointer
        except ImportError:
            logger.error("Async SQLite dependencies not available.")
            return MemorySaver()
        except Exception as e:
            logger.error(f"❌ Async SQLite checkpointer creation failed: {e}")
            return MemorySaver()

    @classmethod
    def _create_sqlite_checkpointer_sync(cls):
        """Create synchronous SQLite checkpointer."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from config.settings import config
            
            config.workspace_dir.mkdir(parents=True, exist_ok=True)
            sqlite_path_str = os.getenv("DATABASE_URL")
            if sqlite_path_str and sqlite_path_str.startswith("sqlite:///"):
                db_path = Path(sqlite_path_str.replace("sqlite:///", ""))
            else:
                db_path = config.workspace_dir / "assistant.db"
            
            logger.info(f"Initializing SYNC SQLite checkpointer at: {db_path}")
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn=conn)
            logger.info(f"✅ Sync SQLite checkpointer initialized at {db_path}")
            return checkpointer
        except ImportError:
            logger.error("❌ SQLite checkpointer not available")
            return MemorySaver()
        except Exception as e:
            logger.error(f"❌ Sync SQLite checkpointer failed: {e}")
            return MemorySaver()
    
    @classmethod
    def _create_postgres_checkpointer_sync(cls):
        """Create synchronous PostgreSQL checkpointer using psycopg2."""
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg2
            
            conn_str = os.getenv("POSTGRES_URL")
            if not conn_str:
                logger.error("POSTGRES_URL not set. Cannot create SYNC PostgreSQL checkpointer.")
                return cls._create_sqlite_checkpointer_sync()

            if not cls._test_postgres_connection_sync():
                 logger.error("PostgreSQL (sync) connection test failed before creating saver.")
                 return cls._create_sqlite_checkpointer_sync()
                
            logger.info("Attempting to connect to SYNC PostgreSQL for saver...")
            conn = psycopg2.connect(conn_str)
            checkpointer = PostgresSaver(conn=conn)
            logger.info("✅ Sync PostgreSQL checkpointer initialized successfully")
            return checkpointer
        except ImportError:
            logger.error("❌ psycopg2-binary or PostgreSQL checkpointer not installed.")
            return cls._create_sqlite_checkpointer_sync()
        except Exception as e:
            logger.error(f"❌ Sync PostgreSQL checkpointer failed: {e}")
            return cls._create_sqlite_checkpointer_sync()

    @classmethod
    def _test_postgres_connection_sync(cls) -> bool:
        """Test PostgreSQL connection using psycopg2 for sync operations"""
        try:
            import psycopg2
            
            conn_str = os.getenv("POSTGRES_URL")
            if not conn_str:
                logger.warning("POSTGRES_URL not set for connection test")
                return False
            
            conn = psycopg2.connect(conn_str)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] == 1:
                logger.info("PostgreSQL (sync) connection test successful.")
                return True
            else:
                logger.error("PostgreSQL (sync) connection test failed: unexpected result")
                return False
                
        except ImportError:
            logger.error("psycopg2 not available for PostgreSQL connection test")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL (sync) connection test failed: {e}")
            return False

def get_checkpointer_info() -> Dict[str, Any]:
    """Get information about the current checkpointer configuration"""
    environment = CheckpointerFactory._detect_environment()
    
    info = {
        "detected_environment": environment,
        "DATABASE_URL_env": os.getenv("DATABASE_URL", "Not Set"),
        "POSTGRES_URL_env": os.getenv("POSTGRES_URL", "Not Set")[:30] + "..." if os.getenv("POSTGRES_URL") else "Not Set",
        "ENVIRONMENT_env": os.getenv("ENVIRONMENT", "Not Set"),
        "postgres_connection_sync_test_passed": CheckpointerFactory._test_postgres_connection_sync(),
        "sqlite_available": True, 
        "memory_fallback_possible": True,
        "packages_available": {}
    }
    
    packages = [
        "psycopg2", "langgraph.checkpoint.postgres", "langgraph.checkpoint.postgres.aio",
        "aiosqlite", "langgraph.checkpoint.sqlite", "langgraph.checkpoint.sqlite.aio",
        "asyncpg", "langgraph.checkpoint.memory"
    ]
    
    for package_path in packages:
        try:
            base_module = package_path.split('.')[0]
            __import__(base_module)
            info["packages_available"][package_path] = True
        except ImportError:
            info["packages_available"][package_path] = False
            
    return info

async def create_checkpointer(
    environment: str = "auto",
    use_async: bool = True
) -> Any:
    """Main entry point for creating checkpointers"""
    if use_async:   
        return await CheckpointerFactory.create_checkpointer_async(environment)
    else:
        return CheckpointerFactory.create_checkpointer_sync(environment)

async def create_async_checkpointer(environment: str = "auto") -> Any:
    return await CheckpointerFactory.create_checkpointer_async(environment)

def create_sync_checkpointer(environment: str = "auto") -> Any:
    return CheckpointerFactory.create_checkpointer_sync(environment)
