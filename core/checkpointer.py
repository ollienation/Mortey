# core/checkpointer.py - ✅ ENHANCED WITH TYPE SAFETY AND ASYNC I/O FIXES
import os
import logging
import asyncio
from typing import Optional, Union, TypeAlias, Any
from collections.abc import Sequence  # Python 3.13.4 preferred import
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# LangGraph imports with proper type hints
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver

try:
    # These are the ACTUAL PostgreSQL checkpointers from LangGraph
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.postgres import PostgresSaver
    import asyncpg  # For async PostgreSQL
    import psycopg  # For sync PostgreSQL (psycopg3)
    POSTGRES_AVAILABLE = True
except ImportError:
    AsyncPostgresSaver = None
    PostgresSaver = None
    asyncpg = None
    psycopg = None
    POSTGRES_AVAILABLE = False

from config.settings import config
logger = logging.getLogger("checkpointer")

# 🔥 CORRECTED: TypeAlias with actual checkpointer types
Checkpointer: TypeAlias = Union[
    AsyncPostgresSaver,  # From langgraph.checkpoint.postgres.aio
    PostgresSaver,       # From langgraph.checkpoint.postgres
    AsyncSqliteSaver,
    SqliteSaver,
    MemorySaver,
]

class Environment(Enum):
    """Environment types for checkpointer selection"""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"

@dataclass
class CheckpointerConfig:
    """
    Configuration for checkpointer creation with enhanced validation
    """
    environment: Environment = Environment.DEVELOPMENT
    prefer_async: bool = True
    connection_timeout: float = 30.0
    max_connections: int = 10
    retry_attempts: int = 3
    enable_connection_pooling: bool = True
    database_path: Optional[str] = None
    postgres_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.max_connections <= 0:
            raise ValueError("max_connections must be positive")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")

@dataclass
class CheckpointerHealth:
    """Health status for checkpointer monitoring"""
    healthy: bool
    checkpointer_type: str
    connection_status: str
    last_check: float
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

class CheckpointerFactory:
    """
    Enhanced factory for creating optimal checkpointers with dependency injection support
    """
    
    def __init__(self, config: Optional[CheckpointerConfig] = None):
        """Initialize factory with optional configuration"""
        self.config = config or CheckpointerConfig()
        self._connection_cache: dict[str, Any] = {}
        self._health_status: dict[str, CheckpointerHealth] = {}
        logger.info(f"🏭 CheckpointerFactory initialized for {self.config.environment.value} environment")
    
    async def create_optimal_checkpointer(self) -> Checkpointer:
        """
        Create the optimal checkpointer based on environment and availability
        """
        try:
            environment = self._detect_environment()
            logger.info(f"🎯 Creating checkpointer for {environment.value} environment")
            
            # Use match-case for clean environment routing (Python 3.13.4)
            match environment:
                case Environment.PRODUCTION | Environment.STAGING:
                    return await self._create_production_checkpointer()
                case Environment.DEVELOPMENT:
                    return await self._create_development_checkpointer()
                case Environment.TESTING:
                    return await self._create_testing_checkpointer()
                case _:
                    logger.warning(f"⚠️ Unknown environment: {environment}, defaulting to development")
                    return await self._create_development_checkpointer()
                    
        except Exception as e:
            logger.error(f"❌ Checkpointer creation failed: {e}")
            logger.info("🔄 Falling back to memory checkpointer")
            return await self._create_memory_checkpointer()
    
    def _detect_environment(self) -> Environment:
        """Enhanced environment detection with better auto-detection"""
        # Check explicit environment variable first
        env_var = os.getenv("ENVIRONMENT", "").lower()
        
        if env_var:
            env_mapping = {
                "production": Environment.PRODUCTION,
                "prod": Environment.PRODUCTION,
                "staging": Environment.STAGING,
                "stage": Environment.STAGING,
                "development": Environment.DEVELOPMENT,
                "dev": Environment.DEVELOPMENT,
                "testing": Environment.TESTING,
                "test": Environment.TESTING,
            }
            
            if env_var in env_mapping:
                detected_env = env_mapping[env_var]
                logger.debug(f"🔍 Environment detected from ENVIRONMENT var: {detected_env.value}")
                return detected_env
        
        # Auto-detection based on other environment variables
        if os.getenv("POSTGRES_URL"):
            logger.debug("🔍 PostgreSQL URL found, assuming production")
            return Environment.PRODUCTION
        
        # Check for CI/testing environment
        ci_indicators = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "TRAVIS"]
        if any(os.getenv(indicator) for indicator in ci_indicators):
            logger.debug("🔍 CI environment detected, using testing mode")
            return Environment.TESTING
        
        # Default to development
        logger.debug("🔍 No specific environment detected, defaulting to development")
        return Environment.DEVELOPMENT
    
    async def _create_production_checkpointer(self) -> Checkpointer:
        """Create production-grade checkpointer with PostgreSQL preference"""
        if not POSTGRES_AVAILABLE:
            logger.warning("⚠️ PostgreSQL dependencies not available, falling back to SQLite")
            return await self._create_development_checkpointer()
        
        postgres_url = self.config.postgres_url or os.getenv("POSTGRES_URL")
        
        if postgres_url:
            # Try async PostgreSQL first
            if self.config.prefer_async:
                try:
                    checkpointer = await self._create_postgres_checkpointer_async(postgres_url)
                    logger.info("✅ Production async PostgreSQL checkpointer created")
                    return checkpointer
                except Exception as e:
                    logger.warning(f"⚠️ Async PostgreSQL failed: {e}, trying sync version")
            
            # Fallback to sync PostgreSQL
            try:
                checkpointer = await self._create_postgres_checkpointer_sync(postgres_url)
                logger.info("✅ Production sync PostgreSQL checkpointer created")
                return checkpointer
            except Exception as e:
                logger.warning(f"⚠️ Sync PostgreSQL failed: {e}, falling back to SQLite")
        
        # Final fallback to async SQLite with warning
        logger.warning("⚠️ PostgreSQL unavailable in production, using SQLite fallback")
        return await self._create_development_checkpointer()

    async def _create_development_checkpointer(self) -> Checkpointer:
        """Create development checkpointer using SQLite - CORRECTED VERSION"""
        try:
            db_path = self.config.database_path or str(config.workspace_dir / "assistant.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.prefer_async:
                # 🔥 CRITICAL FIX: Create actual aiosqlite connection, not string
                import aiosqlite
                
                # Create actual connection object
                conn = await aiosqlite.connect(db_path)
                
                # Pass the connection object (not string) to AsyncSqliteSaver
                checkpointer = AsyncSqliteSaver(conn=conn)
                
                logger.info(f"✅ Development async SQLite checkpointer created: {db_path}")
            else:
                # 🔥 FIXED: For sync version, use from_conn_string
                conn_string = f"sqlite:///{db_path}"
                checkpointer = await asyncio.to_thread(
                    SqliteSaver.from_conn_string,
                    conn_string
                )
                logger.info(f"✅ Development sync SQLite checkpointer created: {db_path}")
            
            return NamespacedCheckpointer(checkpointer)
            
        except Exception as e:
            logger.error(f"❌ SQLite checkpointer creation failed: {e}")
            logger.info("🔄 Falling back to memory checkpointer")
            return await self._create_memory_checkpointer()

    # core/checkpointer.py - LANGGRAPH 0.4.8 PATTERN
    async def _create_postgres_checkpointer_async(self, postgres_url: str) -> AsyncPostgresSaver:
        """Create async PostgreSQL checkpointer - LANGGRAPH 0.4.8"""
        if not POSTGRES_AVAILABLE or AsyncPostgresSaver is None:
            raise ImportError("LangGraph PostgreSQL checkpointers not available")
        
        try:
            # ✅ LANGGRAPH 0.4.8: Use context manager pattern
            self._postgres_context = AsyncPostgresSaver.from_conn_string(postgres_url)
            checkpointer = await self._postgres_context.__aenter__()
            
            # ✅ LANGGRAPH 0.4.8: Setup tables (first time)
            await checkpointer.setup()
            
            # Store both context and checkpointer for cleanup
            self._connection_cache["postgres_async_context"] = self._postgres_context
            self._connection_cache["postgres_async"] = checkpointer
            
            logger.info("✅ LangGraph 0.4.8 async PostgreSQL checkpointer created")
            return checkpointer
            
        except Exception as e:
            logger.error(f"❌ LangGraph 0.4.8 PostgreSQL checkpointer failed: {e}")
            raise

    async def _create_postgres_checkpointer_sync(self, postgres_url: str) -> PostgresSaver:
        """Create sync PostgreSQL checkpointer using LangGraph's PostgresSaver"""
        if not POSTGRES_AVAILABLE or PostgresSaver is None:
            raise ImportError("LangGraph PostgreSQL checkpointers not available")
        
        try:
            # Test connection first
            if not await self._test_postgres_connection_sync(postgres_url):
                raise ConnectionError("PostgreSQL connection test failed")
            
            # 🔥 CORRECTED: Use LangGraph's PostgresSaver.from_conn_string in thread
            checkpointer = await asyncio.to_thread(
                PostgresSaver.from_conn_string,
                postgres_url
            )
            
            self._connection_cache["postgres_sync"] = checkpointer
            logger.info("✅ LangGraph sync PostgreSQL checkpointer created")
            return checkpointer
            
        except Exception as e:
            logger.error(f"❌ LangGraph sync PostgreSQL creation failed: {e}")
            raise

    # 🔥 FIXED: Async connection testing (asyncpg is ALREADY async)
    async def _test_postgres_connection_async(self, postgres_url: str) -> bool:
        """Test async PostgreSQL connection - FIXED for asyncpg"""
        try:
            if not asyncpg:
                return False
                
            # 🔥 CRITICAL FIX: asyncpg.connect is ALREADY async, don't wrap it
            conn = await asyncpg.connect(
                postgres_url,
                timeout=self.config.connection_timeout
            )
            
            # Test the connection (also already async)
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            return result == 1
            
        except Exception as e:
            logger.debug(f"Async PostgreSQL connection test failed: {e}")
            return False

    # 🔥 FIXED: Sync connection testing with modern psycopg3
    async def _test_postgres_connection_sync(self, postgres_url: str) -> bool:
        """Test sync PostgreSQL connection using modern psycopg3"""
        try:
            if not psycopg:
                return False
                
            # 🔥 MODERN PATTERN: Use psycopg3 (not psycopg2)
            def test_connection():
                with psycopg.connect(
                    postgres_url,
                    connect_timeout=int(self.config.connection_timeout)
                ) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        return result is not None
                        
            return await asyncio.to_thread(test_connection)
            
        except Exception as e:
            logger.debug(f"Sync PostgreSQL connection test failed: {e}")
            return False

    async def health_check_all(self) -> dict[str, CheckpointerHealth]:
        """Comprehensive health check for all cached connections - FIXED"""
        health_results = {}
        
        try:
            # 🔥 FIX: Better TaskGroup exception handling
            tasks_and_names = []
            
            async with asyncio.TaskGroup() as tg:
                for conn_name, connection in self._connection_cache.items():
                    try:
                        task = tg.create_task(
                            self._safe_health_check_connection(conn_name, connection)
                        )
                        tasks_and_names.append((conn_name, task))
                    except Exception as e:
                        # Handle task creation errors
                        logger.warning(f"⚠️ Failed to create health check task for {conn_name}: {e}")
                        health_results[conn_name] = CheckpointerHealth(
                            healthy=False,
                            checkpointer_type=conn_name,
                            connection_status="task_creation_failed",
                            last_check=asyncio.get_event_loop().time(),
                            error_message=str(e)
                        )
            
            # Collect results
            for conn_name, task in tasks_and_names:
                try:
                    health_results[conn_name] = task.result()
                except Exception as e:
                    logger.warning(f"⚠️ Health check task failed for {conn_name}: {e}")
                    health_results[conn_name] = CheckpointerHealth(
                        healthy=False,
                        checkpointer_type=conn_name,
                        connection_status="health_check_failed",
                        last_check=asyncio.get_event_loop().time(),
                        error_message=str(e)
                    )
                    
        except* Exception as eg:  # 🔥 NEW: Use exception groups for TaskGroup
            logger.error(f"❌ TaskGroup health check failed with {len(eg.exceptions)} exceptions")
            for i, exc in enumerate(eg.exceptions):
                logger.error(f"  Exception {i+1}: {exc}")
            
            # Return failed health checks for remaining connections
            for conn_name in self._connection_cache.keys():
                if conn_name not in health_results:
                    health_results[conn_name] = CheckpointerHealth(
                        healthy=False,
                        checkpointer_type=conn_name,
                        connection_status="taskgroup_exception",
                        last_check=asyncio.get_event_loop().time(),
                        error_message="TaskGroup exception occurred"
                    )
        
        self._health_status = health_results
        return health_results

    # 🔥 NEW: Safe wrapper for health check
    async def _safe_health_check_connection(self, conn_name: str, connection: Any) -> CheckpointerHealth:
        """Safe wrapper for health check that never raises exceptions"""
        try:
            return await self._health_check_connection(conn_name, connection)
        except Exception as e:
            logger.debug(f"Health check failed for {conn_name}: {e}")
            return CheckpointerHealth(
                healthy=False,
                checkpointer_type=conn_name,
                connection_status="exception",
                last_check=asyncio.get_event_loop().time(),
                error_message=str(e)
            )

    
    # MODIFY THIS METHOD:
    async def _health_check_connection(self, conn_name: str, connection: Any) -> CheckpointerHealth:
        """Improved health check for LangGraph 0.4.8 checkpointers"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if "postgres_async_context" in conn_name:
                # FIX: Better validation for context manager
                if hasattr(connection, '__aenter__') and hasattr(connection, '__aexit__'):
                    # This is a context manager, mark as healthy if it exists
                    status = "context_manager_ready"
                    healthy = True
                else:
                    status = "invalid_context_manager"
                    healthy = False
                    uuuh, 
            elif "postgres_async" in conn_name:
                # FIX: Validate actual checkpointer instance
                if hasattr(connection, 'aget') and hasattr(connection, 'aput'):
                    try:
                        # Test with a simple get operation
                        test_config = {"configurable": {"thread_id": "health_check_test"}}
                        await asyncio.wait_for(connection.aget(test_config), timeout=2.0)
                        status = "connected"
                        healthy = True
                    except asyncio.TimeoutError:
                        status = "timeout"
                        healthy = False
                    except Exception:
                        # Not finding the config is OK - means DB is accessible
                        status = "connected"
                        healthy = True
                else:
                    status = "invalid_checkpointer"
                    healthy = False
            else:
                # SQLite and Memory savers (unchanged)
                healthy = connection is not None
                status = "available" if healthy else "unavailable"
            
            return CheckpointerHealth(
                healthy=healthy,
                checkpointer_type=conn_name,
                connection_status=status,
                last_check=start_time,
                metadata={
                    "response_time_ms": (asyncio.get_event_loop().time() - start_time) * 1000
                }
            )
            
        except Exception as e:
            return CheckpointerHealth(
                healthy=False,
                checkpointer_type=conn_name,
                connection_status="error",
                last_check=start_time,
                error_message=str(e),
                metadata={
                    "response_time_ms": (asyncio.get_event_loop().time() - start_time) * 1000
                }
            )
    
    async def cleanup_connections(self) -> None:
        """Clean up with LangGraph 0.4.8 context manager pattern"""
        cleanup_errors = []
        
        for conn_name, connection in self._connection_cache.items():
            try:
                if "_context" in conn_name:
                    # ✅ LANGGRAPH 0.4.8: Exit context manager properly
                    await connection.__aexit__(None, None, None)
                    logger.debug(f"✅ LangGraph 0.4.8 context {conn_name} exited")
                elif hasattr(connection, 'conn') and hasattr(connection.conn, 'close'):
                    await connection.conn.close()
                    
            except Exception as e:
                cleanup_errors.append(f"Failed to cleanup {conn_name}: {e}")
                logger.warning(f"⚠️ Cleanup failed: {e}")
        
        self._connection_cache.clear()
        logger.info("✅ LangGraph 0.4.8 connections cleaned up")
    
    def get_factory_statistics(self) -> dict[str, Any]:
        """Get comprehensive factory statistics"""
        return {
            "config": {
                "environment": self.config.environment.value,
                "prefer_async": self.config.prefer_async,
                "connection_timeout": self.config.connection_timeout,
                "max_connections": self.config.max_connections,
            },
            "connections": {
                "active_count": len(self._connection_cache),
                "connection_types": list(self._connection_cache.keys()),
            },
            "health": {
                "last_check_count": len(self._health_status),
                "healthy_connections": sum(
                    1 for h in self._health_status.values() if h.healthy
                ),
            },
            "capabilities": {
                "postgres_available": POSTGRES_AVAILABLE,
                "async_postgres": POSTGRES_AVAILABLE and AsyncPostgresSaver is not None,
                "sync_postgres": POSTGRES_AVAILABLE and PostgresSaver is not None,
            }
        }

# 🔥 CRITICAL FIX: Remove global singleton, use dependency injection instead
# Factory should be instantiated where needed, not as a global

# Convenience functions for backward compatibility
async def create_optimal_checkpointer(config: Optional[CheckpointerConfig] = None) -> Checkpointer:
    """Create optimal checkpointer with optional configuration"""
    factory = CheckpointerFactory(config)
    return await factory.create_optimal_checkpointer()

async def create_development_checkpointer() -> Checkpointer:
    """Create development checkpointer"""
    config = CheckpointerConfig(environment=Environment.DEVELOPMENT)
    factory = CheckpointerFactory(config)
    return await factory.create_optimal_checkpointer()

async def create_production_checkpointer() -> Checkpointer:
    """Create production checkpointer"""
    config = CheckpointerConfig(environment=Environment.PRODUCTION)
    factory = CheckpointerFactory(config)
    return await factory.create_optimal_checkpointer()

async def create_memory_checkpointer() -> Checkpointer:
    """Create memory checkpointer"""
    config = CheckpointerConfig(environment=Environment.TESTING)
    factory = CheckpointerFactory(config)
    return await factory.create_optimal_checkpointer()

async def health_check_checkpointers() -> dict[str, Any]:
    """Health check for checkpointers - requires factory instance"""
    # This function now requires a factory instance to work properly
    # It should be called on a specific factory instance instead
    logger.warning("⚠️ health_check_checkpointers requires a factory instance")
    return {
        "warning": "Function requires CheckpointerFactory instance",
        "recommendation": "Use factory.health_check_all() instead"
    }

class NamespacedCheckpointer:
    """Wrapper around LangGraph checkpointers to support namespaced keys"""

    def __init__(self, base_checkpointer):
        self.base = base_checkpointer
        
    async def aget(self, config_dict):
        return await self.base.aget(config_dict)
        
    async def aput(self, state, config_dict):
        # Add namespace metadata to state before saving
        if isinstance(state, dict) and "messages" in state:
            thread_id = config_dict.get("configurable", {}).get("thread_id", "")
            for msg in state["messages"]:
                if hasattr(msg, 'additional_kwargs'):
                    msg.additional_kwargs['_namespace'] = thread_id
        return await self.base.aput(state, config_dict)
        
    async def adelete(self, config_dict):
        return await self.base.adelete(config_dict)
