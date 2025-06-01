# Production Checkpointer Configuration
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
    
    Key improvements for June 2025:
    - Uses dedicated checkpointer libraries (langgraph-checkpoint-postgres, langgraph-checkpoint-sqlite)
    - Simplified initialization patterns with proper error handling
    - Connection pooling and automatic setup
    - Supports both sync and async interfaces
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
    
    @classmethod
    def _create_postgres_checkpointer(cls):
        """Create PostgreSQL checkpointer for production using modern pattern"""
        try:
            # Import the dedicated PostgreSQL checkpointer library
            from langgraph_checkpoint_postgres import PostgresSaver
            
            # Get database URL from environment
            db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
            if not db_url:
                logger.warning("No PostgreSQL URL found, falling back to SQLite")
                return cls._create_sqlite_checkpointer()
                
            # Create checkpointer with simplified initialization
            checkpointer = PostgresSaver.from_conn_string(db_url)
            
            # Setup database tables
            try:
                checkpointer.setup()
                logger.info("✅ PostgreSQL checkpointer initialized successfully")
                return checkpointer
            except Exception as setup_error:
                logger.error(f"❌ PostgreSQL setup failed: {setup_error}")
                logger.info("Falling back to SQLite checkpointer")
                return cls._create_sqlite_checkpointer()
                
        except ImportError:
            logger.warning("langgraph-checkpoint-postgres not installed, using SQLite")
            return cls._create_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"❌ PostgreSQL checkpointer creation failed: {e}")
            logger.info("Falling back to SQLite checkpointer")
            return cls._create_sqlite_checkpointer()
    
    @classmethod
    def _create_sqlite_checkpointer(cls):
        """Create SQLite checkpointer for development using modern pattern"""
        try:
            # Import the dedicated SQLite checkpointer library
            from langgraph_checkpoint_sqlite import SqliteSaver
            
            # Use workspace directory for SQLite database
            from config.settings import config
            db_path = config.workspace_dir / "assistant_memory.db"
            
            # Ensure directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create checkpointer with simplified initialization
            checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
            
            # Setup database tables
            try:
                checkpointer.setup()
                logger.info(f"✅ SQLite checkpointer initialized: {db_path}")
                return checkpointer
            except Exception as setup_error:
                logger.error(f"❌ SQLite setup failed: {setup_error}")
                # Fall back to in-memory as last resort
                return cls._create_memory_checkpointer()
                
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite not installed, using in-memory")
            return cls._create_memory_checkpointer()
        except Exception as e:
            logger.error(f"❌ SQLite checkpointer creation failed: {e}")
            logger.info("Falling back to in-memory checkpointer")
            return cls._create_memory_checkpointer()
    
    @classmethod
    def _create_memory_checkpointer(cls):
        """Create in-memory checkpointer as fallback"""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            logger.warning("⚠️ Using MemorySaver - conversations will not persist between restarts")
            return MemorySaver()
        except Exception as e:
            logger.error(f"❌ Even MemorySaver failed: {e}")
            raise RuntimeError("No checkpointer could be created")

class AsyncCheckpointerFactory:
    """Async version of the checkpointer factory for async applications"""
    
    @classmethod
    async def create_checkpointer(cls, environment: str = "auto"):
        """Create async checkpointer"""
        if environment == "auto":
            environment = cls._detect_environment()
            
        if environment == "production":
            return await cls._create_async_postgres_checkpointer()
        else:
            return await cls._create_async_sqlite_checkpointer()
    
    @classmethod
    def _detect_environment(cls) -> str:
        """Auto-detect environment based on available configurations"""
        return CheckpointerFactory._detect_environment()
    
    @classmethod
    async def _create_async_postgres_checkpointer(cls):
        """Create async PostgreSQL checkpointer using modern pattern"""
        try:
            # Import the dedicated async PostgreSQL checkpointer
            from langgraph_checkpoint_postgres.aio import AsyncPostgresSaver
            
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
                
        except ImportError:
            logger.warning("Async PostgreSQL not available, using SQLite")
            return await cls._create_async_sqlite_checkpointer()
        except Exception as e:
            logger.error(f"❌ Async PostgreSQL checkpointer creation failed: {e}")
            return await cls._create_async_sqlite_checkpointer()
    
    @classmethod
    async def _create_async_sqlite_checkpointer(cls):
        """Create async SQLite checkpointer using modern pattern"""
        try:
            # Import the dedicated async SQLite checkpointer
            from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
            
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
                
        except ImportError:
            logger.warning("Async SQLite not available, using in-memory")
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
        import langgraph_checkpoint_postgres
        info["postgres_available"] = True
    except ImportError:
        pass
        
    try:
        import langgraph_checkpoint_sqlite
        info["sqlite_available"] = True
    except ImportError:
        pass
    
    return info

# Convenience function for most common use case
def create_production_checkpointer():
    """Create the best available checkpointer for the current environment"""
    return CheckpointerFactory.create_checkpointer()

async def create_async_production_checkpointer():
    """Create the best available async checkpointer for the current environment"""
    return await AsyncCheckpointerFactory.create_checkpointer()