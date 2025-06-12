# core/circuit_breaker.py - ✅ ENHANCED WITH PYTHON 3.13.4 COMPATIBILITY
"""
This module provides sophisticated circuit breaker patterns specifically designed
for protecting external API calls in AI assistant applications.
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Awaitable, Union, Any
from collections.abc import Mapping  # Python 3.13.4 preferred import
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from functools import wraps
from asyncio import TaskGroup  # Python 3.13.4 TaskGroup

logger = logging.getLogger("circuit_breaker")

class CircuitState(Enum):
    """Circuit breaker states with enhanced descriptions"""
    CLOSED = "closed"          # Normal operation, requests flow through
    OPEN = "open"              # Circuit is open, requests are rejected
    HALF_OPEN = "half_open"    # Testing if service has recovered
    FORCED_OPEN = "forced_open"  # Manually opened for maintenance

@dataclass
class ServiceConfig:
    """Configuration for a specific service's circuit breaker with Python 3.13.4 enhancements"""
    failure_threshold: int = 5           # Failures needed to open circuit
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    call_timeout: float = 30.0           # Timeout for individual calls
    success_threshold: int = 3           # Successes needed to close from half-open
    rolling_window_size: int = 100       # Size of rolling window for statistics
    min_throughput: int = 10             # Minimum calls before circuit can trip
    error_rate_threshold: float = 0.5    # Error rate (0.0-1.0) to trip circuit
    slow_call_threshold: float = 10.0    # Seconds to consider a call "slow"
    adaptive_threshold: bool = True      # Enable adaptive failure thresholds
    health_check_interval: float = 300.0 # Health check interval in seconds
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if not 0.0 <= self.error_rate_threshold <= 1.0:
            raise ValueError("error_rate_threshold must be between 0.0 and 1.0")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")

class ServiceMetrics:
    """Metrics tracking for a service with enhanced analytics"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.call_history = deque(maxlen=config.rolling_window_size)
        self.total_calls = 0
        self.total_failures = 0
        self.total_timeouts = 0
        self.total_slow_calls = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.error_types: dict[str, int] = defaultdict(int)  # Python 3.13.4 syntax
        self.performance_history = deque(maxlen=1000)  # Track performance trends
        
    def record_call(self, success: bool, duration: float, error: Optional[Exception] = None):
        """Record the result of a service call with enhanced tracking"""
        call_time = time.time()
        
        call_record = {
            "timestamp": call_time,
            "success": success,
            "duration": duration,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error)[:100] if error else None
        }
        
        self.call_history.append(call_record)
        self.total_calls += 1
        
        # Track performance trends
        self.performance_history.append({
            "timestamp": call_time,
            "duration": duration,
            "success": success
        })
        
        if success:
            self.last_success_time = call_time
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.last_failure_time = call_time
            self.total_failures += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Track specific failure types
            if error:
                error_type = type(error).__name__
                self.error_types[error_type] += 1
                
                # Enhanced error classification using match-case (Python 3.13.4)
                match error_type:
                    case "TimeoutError" | "asyncio.TimeoutError":
                        self.total_timeouts += 1
                    case "ConnectionError" | "OSError":
                        # Network-related errors
                        pass
                    case "HTTPError":
                        # HTTP-specific errors
                        pass
        
        # Track slow calls
        if duration > self.config.slow_call_threshold:
            self.total_slow_calls += 1
    
    def get_error_rate(self) -> float:
        """Calculate current error rate from rolling window"""
        if not self.call_history:
            return 0.0
        
        recent_calls = list(self.call_history)
        if len(recent_calls) < self.config.min_throughput:
            return 0.0
        
        failures = sum(1 for call in recent_calls if not call["success"])
        return failures / len(recent_calls)
    
    def get_average_response_time(self) -> float:
        """Calculate average response time from rolling window"""
        if not self.call_history:
            return 0.0
        
        durations = [call["duration"] for call in self.call_history]
        return sum(durations) / len(durations)
    
    def get_throughput(self) -> float:
        """Calculate calls per second over the last minute"""
        if not self.call_history:
            return 0.0
        
        current_time = time.time()
        recent_calls = [
            call for call in self.call_history 
            if current_time - call["timestamp"] <= 60
        ]
        
        return len(recent_calls) / 60.0
    
    def get_percentile_response_time(self, percentile: float) -> float:
        """Calculate percentile response time (Python 3.13.4 enhanced)"""
        if not self.call_history:
            return 0.0
        
        durations = sorted([call["duration"] for call in self.call_history])
        if not durations:
            return 0.0
        
        index = int(len(durations) * percentile / 100)
        return durations[min(index, len(durations) - 1)]
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        if self.total_calls == 0:
            return 1.0
        
        # Combine multiple factors for health score
        success_rate = 1.0 - (self.total_failures / self.total_calls)
        
        # Factor in response time performance
        avg_response = self.get_average_response_time()
        response_score = max(0.0, 1.0 - (avg_response / (self.config.slow_call_threshold * 2)))
        
        # Factor in recent performance
        recent_error_rate = self.get_error_rate()
        recent_score = 1.0 - recent_error_rate
        
        # Weighted combination
        health_score = (success_rate * 0.4 + response_score * 0.3 + recent_score * 0.3)
        return max(0.0, min(1.0, health_score))
    
    def get_error_distribution(self) -> dict[str, int]:  # Python 3.13.4 syntax
        """Get distribution of error types"""
        return dict(self.error_types)

@dataclass 
class CircuitBreakerState:
    """State of an individual circuit breaker with enhanced tracking"""
    service_name: str
    state: CircuitState = CircuitState.CLOSED
    config: ServiceConfig = field(default_factory=ServiceConfig)
    metrics: ServiceMetrics = None
    last_state_change: float = field(default_factory=time.time)
    half_open_successes: int = 0
    forced_reason: Optional[str] = None  # Reason for forced state
    last_health_check: float = 0.0
    adaptive_threshold: int = 0  # Dynamic failure threshold
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ServiceMetrics(self.config)
        self.adaptive_threshold = self.config.failure_threshold
    
    def update_adaptive_threshold(self):
        """Update adaptive failure threshold based on recent performance"""
        if not self.config.adaptive_threshold:
            return
        
        health_score = self.metrics.get_health_score()
        base_threshold = self.config.failure_threshold
        
        # Adjust threshold based on health score using match-case (Python 3.13.4)
        match health_score:
            case score if score >= 0.9:
                # Very healthy - increase tolerance
                self.adaptive_threshold = int(base_threshold * 1.5)
            case score if score >= 0.7:
                # Healthy - slight increase
                self.adaptive_threshold = int(base_threshold * 1.2)
            case score if score >= 0.5:
                # Average - use base threshold
                self.adaptive_threshold = base_threshold
            case score if score >= 0.3:
                # Poor health - decrease tolerance
                self.adaptive_threshold = max(1, int(base_threshold * 0.8))
            case _:
                # Very poor health - very low tolerance
                self.adaptive_threshold = max(1, int(base_threshold * 0.5))

class AdvancedCircuitBreaker:
    """
    Circuit breaker with adaptive behavior and Python 3.13.4 enhancements
    """
    
    def __init__(self):
        self.circuits: dict[str, CircuitBreakerState] = {}  # Python 3.13.4 syntax
        self.default_configs = self._initialize_default_configs()
        self._health_check_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = True
        
    def _initialize_default_configs(self) -> dict[str, ServiceConfig]:  # Python 3.13.4 syntax
        """Initialize default configurations for known services"""
        return {
            "anthropic": ServiceConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                call_timeout=240.0,
                error_rate_threshold=0.4,
                min_throughput=5,
                adaptive_threshold=True,
                health_check_interval=300.0
            ),
            "openai": ServiceConfig(
                failure_threshold=3,
                recovery_timeout=30.0, 
                call_timeout=240.0,
                error_rate_threshold=0.4,
                min_throughput=5,
                adaptive_threshold=True,
                health_check_interval=300.0
            ),
            "google": ServiceConfig(
                failure_threshold=4,
                recovery_timeout=45.0,
                call_timeout=240.0,
                error_rate_threshold=0.5,
                min_throughput=3,
                adaptive_threshold=True
            ),
            "tavily": ServiceConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                call_timeout=60.0,
                error_rate_threshold=0.6,
                min_throughput=3,
                adaptive_threshold=False  # Search API is more volatile
            ),
            "file_system": ServiceConfig(
                failure_threshold=10,
                recovery_timeout=5.0,
                call_timeout=10.0,
                error_rate_threshold=0.7,
                min_throughput=5,
                adaptive_threshold=True
            ),
            "database": ServiceConfig(
                failure_threshold=8,
                recovery_timeout=15.0,
                call_timeout=60.0,
                error_rate_threshold=0.6,
                min_throughput=5,
                adaptive_threshold=True
            ),
            "default": ServiceConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                call_timeout=180.0,
                error_rate_threshold=0.5,
                min_throughput=5,
                adaptive_threshold=True
            )
        }
    
    def get_circuit(self, service_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuits:
            config = self.default_configs.get(service_name, self.default_configs["default"])
            self.circuits[service_name] = CircuitBreakerState(
                service_name=service_name,
                config=config
            )
        return self.circuits[service_name]
    
    def should_allow_request(self, service_name: str) -> bool:
        """Check if a request should be allowed through the circuit"""
        circuit = self.get_circuit(service_name)
        current_time = time.time()
        
        # Update adaptive threshold
        circuit.update_adaptive_threshold()
        
        # Enhanced state handling using match-case (Python 3.13.4)
        match circuit.state:
            case CircuitState.CLOSED:
                # Check if we should trip the circuit
                if self._should_trip_circuit(circuit):
                    self._transition_to_open(circuit)
                    return False
                return True
            
            case CircuitState.OPEN:
                # Check if we should try half-open
                if current_time - circuit.last_state_change >= circuit.config.recovery_timeout:
                    self._transition_to_half_open(circuit)
                    return True
                return False
            
            case CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True
            
            case CircuitState.FORCED_OPEN:
                # Manually opened - don't allow requests
                return False
            
            case _:
                # Unknown state - default to closed
                logger.warning(f"Unknown circuit state for {service_name}: {circuit.state}")
                circuit.state = CircuitState.CLOSED
                return True
    
    def _should_trip_circuit(self, circuit: CircuitBreakerState) -> bool:
        """Determine if circuit should be tripped to open state"""
        # Check consecutive failures with adaptive threshold
        if circuit.metrics.consecutive_failures >= circuit.adaptive_threshold:
            logger.warning(
                f"Tripping circuit {circuit.service_name}: "
                f"{circuit.metrics.consecutive_failures} consecutive failures "
                f"(adaptive threshold: {circuit.adaptive_threshold})"
            )
            return True
        
        # Check error rate over rolling window
        error_rate = circuit.metrics.get_error_rate()
        if (error_rate >= circuit.config.error_rate_threshold and 
            len(circuit.metrics.call_history) >= circuit.config.min_throughput):
            logger.warning(
                f"Tripping circuit {circuit.service_name}: "
                f"error rate {error_rate:.2%} exceeds threshold {circuit.config.error_rate_threshold:.2%}"
            )
            return True
        
        # Check for excessive slow calls
        slow_call_rate = (circuit.metrics.total_slow_calls / 
                         max(circuit.metrics.total_calls, 1))
        if slow_call_rate > 0.8 and circuit.metrics.total_calls >= circuit.config.min_throughput:
            logger.warning(
                f"Tripping circuit {circuit.service_name}: "
                f"slow call rate {slow_call_rate:.2%} too high"
            )
            return True
        
        return False
    
    def _transition_to_open(self, circuit: CircuitBreakerState):
        """Transition circuit to open state"""
        circuit.state = CircuitState.OPEN
        circuit.last_state_change = time.time()
        logger.error(f"🔴 Circuit OPEN: {circuit.service_name}")
    
    def _transition_to_half_open(self, circuit: CircuitBreakerState):
        """Transition circuit to half-open state"""
        circuit.state = CircuitState.HALF_OPEN
        circuit.last_state_change = time.time()
        circuit.half_open_successes = 0
        logger.warning(f"🟡 Circuit HALF-OPEN: {circuit.service_name} (testing recovery)")
    
    def _transition_to_closed(self, circuit: CircuitBreakerState):
        """Transition circuit to closed state"""
        circuit.state = CircuitState.CLOSED
        circuit.last_state_change = time.time()
        circuit.half_open_successes = 0
        logger.info(f"🟢 Circuit CLOSED: {circuit.service_name} (service recovered)")
    
    async def call_with_circuit_breaker(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> any:  # Python 3.13.4 syntax
        """Execute a function call with circuit breaker protection"""
        circuit = self.get_circuit(service_name)
        
        # Check if request should be allowed
        if not self.should_allow_request(service_name):
            raise CircuitBreakerOpenException(
                f"Circuit breaker open for service: {service_name} "
                f"(state: {circuit.state.value})"
            )
        
        start_time = time.time()
        try:
            # Apply timeout with enhanced error handling
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=circuit.config.call_timeout
            )
            
            # Record successful call
            duration = time.time() - start_time
            circuit.metrics.record_call(True, duration)
            
            # Handle half-open state transitions
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.half_open_successes += 1
                if circuit.half_open_successes >= circuit.config.success_threshold:
                    self._transition_to_closed(circuit)
            
            return result
            
        except Exception as e:
            # Record failed call
            duration = time.time() - start_time
            circuit.metrics.record_call(False, duration, e)
            
            # If in half-open state, failed call means back to open
            if circuit.state == CircuitState.HALF_OPEN:
                self._transition_to_open(circuit)
            
            # Re-raise the original exception
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> any:  # Python 3.13.4 syntax
        """Execute function handling both sync and async callables"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            elif asyncio.iscoroutine(func):
                return await func
            else:
                # Run sync function in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
        except Exception as e:
            logger.debug(f"Function execution failed: {type(e).__name__}: {e}")
            raise
    
    def get_circuit_status(self, service_name: str) -> dict[str, any]:  # Python 3.13.4 syntax
        """Get detailed status of a circuit breaker"""
        if service_name not in self.circuits:
            return {"status": "not_initialized"}
        
        circuit = self.circuits[service_name]
        metrics = circuit.metrics
        
        return {
            "service": service_name,
            "state": circuit.state.value,
            "last_state_change": circuit.last_state_change,
            "time_since_change": time.time() - circuit.last_state_change,
            "forced_reason": circuit.forced_reason,
            "adaptive_threshold": circuit.adaptive_threshold,
            "health_score": metrics.get_health_score(),
            "metrics": {
                "total_calls": metrics.total_calls,
                "total_failures": metrics.total_failures,
                "consecutive_failures": metrics.consecutive_failures,
                "consecutive_successes": metrics.consecutive_successes,
                "error_rate": metrics.get_error_rate(),
                "average_response_time": metrics.get_average_response_time(),
                "p95_response_time": metrics.get_percentile_response_time(95),
                "p99_response_time": metrics.get_percentile_response_time(99),
                "throughput": metrics.get_throughput(),
                "slow_call_rate": (metrics.total_slow_calls / max(metrics.total_calls, 1)),
                "last_success": metrics.last_success_time,
                "last_failure": metrics.last_failure_time,
                "error_distribution": metrics.get_error_distribution()
            },
            "config": {
                "failure_threshold": circuit.config.failure_threshold,
                "recovery_timeout": circuit.config.recovery_timeout,
                "error_rate_threshold": circuit.config.error_rate_threshold,
                "adaptive_threshold_enabled": circuit.config.adaptive_threshold
            }
        }
    
    def get_all_circuit_status(self) -> dict[str, dict[str, any]]:  # Python 3.13.4 syntax
        """Get status of all circuit breakers"""
        return {
            service_name: self.get_circuit_status(service_name)
            for service_name in self.circuits.keys()
        }
    
    async def health_check_all_services(self) -> dict[str, bool]:  # Python 3.13.4 syntax
        """Perform health check for all services using TaskGroup (Python 3.13.4)"""
        try:
            async with TaskGroup() as tg:
                tasks = {
                    service_name: tg.create_task(self.health_check(service_name))
                    for service_name in self.circuits.keys()
                }
            
            return {
                service_name: task.result()
                for service_name, task in tasks.items()
            }
        except Exception as e:
            logger.error(f"Health check batch failed: {e}")
            # Fallback to sequential checks
            results = {}
            for service_name in self.circuits.keys():
                try:
                    results[service_name] = await self.health_check(service_name)
                except Exception as service_error:
                    logger.error(f"Health check failed for {service_name}: {service_error}")
                    results[service_name] = False
            return results
    
    async def health_check(self, service_name: str) -> bool:
        """Perform health check for a specific service"""
        try:
            circuit = self.get_circuit(service_name)
            current_time = time.time()
            
            # Rate limit health checks
            if (current_time - circuit.last_health_check < 
                circuit.config.health_check_interval):
                return circuit.state == CircuitState.CLOSED
            
            circuit.last_health_check = current_time
            
            # Enhanced health check using match-case (Python 3.13.4)
            match service_name.lower():
                case name if "anthropic" in name:
                    return await self._health_check_anthropic()
                case name if "openai" in name:
                    return await self._health_check_openai()
                case name if "google" in name:
                    return await self._health_check_google()
                case name if "tavily" in name:
                    return await self._health_check_tavily()
                case name if "file" in name or "filesystem" in name:
                    return await self._health_check_file_system()
                case name if "database" in name or "db" in name:
                    return await self._health_check_database()
                case _:
                    return await self._health_check_generic(service_name)
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return False
    
    async def _health_check_anthropic(self) -> bool:
        """Enhanced health check for Anthropic API"""
        try:
            from config.llm_manager import llm_manager
            # Use a very small request to minimize cost and latency
            await asyncio.wait_for(
                llm_manager.generate_for_node("chat", "Hi", override_max_tokens=1),
                timeout=10.0
            )
            return True
        except Exception as e:
            logger.debug(f"Anthropic health check failed: {e}")
            return False
    
    async def _health_check_openai(self) -> bool:
        """Enhanced health check for OpenAI API"""
        try:
            from config.llm_manager import llm_manager
            from config.settings import config
            
            # FIX: Use actual node names, not provider-prefixed names
            openai_nodes = [name for name, node in config.nodes.items() if node.provider == "openai"]
            if not openai_nodes:
                logger.debug("No OpenAI nodes configured")
                return False
            
            # Use first available OpenAI node
            test_node = openai_nodes[0]
            logger.debug(f"Testing OpenAI with node: {test_node}")
            
            # Try a minimal request
            await asyncio.wait_for(
                llm_manager.generate_for_node(test_node, "Hi", override_max_tokens=1),
                timeout=10.0
            )
            return True
        except Exception as e:
            logger.debug(f"OpenAI health check failed: {e}")
            return False
    
    async def _health_check_google(self) -> bool:
        """Enhanced health check for Google API"""
        try:
            from config.llm_manager import llm_manager
            from config.settings import config
            if "google" not in config.get_available_providers():
                return False
            
            await asyncio.wait_for(
                llm_manager.generate_for_node("google_chat", "Hi", override_max_tokens=1),
                timeout=10.0
            )
            return True
        except Exception as e:
            logger.debug(f"Google health check failed: {e}")
            return False
    
    async def _health_check_tavily(self) -> bool:
        """Enhanced health check for Tavily API"""
        import os
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return False
            
            # Could implement actual API test here
            # For now, just check if API key is configured
            return bool(api_key.strip())
        except Exception as e:
            logger.debug(f"Tavily health check failed: {e}")
            return False
    
    async def _health_check_file_system(self) -> bool:
        """Enhanced health check for file system operations"""
        try:
            from config.settings import config
            # Test basic file operations
            test_file = config.workspace_dir / ".circuit_breaker_health_check"
            
            # Write test
            test_file.write_text("health_check")
            
            # Read test
            content = test_file.read_text()
            
            # Cleanup
            test_file.unlink()
            
            return content == "health_check"
        except Exception as e:
            logger.debug(f"File system health check failed: {e}")
            return False
    
    async def _health_check_database(self) -> bool:
        """Enhanced health check for database operations"""
        try:
            # Test database connectivity
            import sqlite3
            from config.settings import config
            
            db_path = config.workspace_dir / "assistant.db"
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] == 1
        except Exception as e:
            logger.debug(f"Database health check failed: {e}")
            return False
    
    async def _health_check_generic(self, service_name: str) -> bool:
        """Generic health check for unknown services"""
        try:
            # Basic connectivity test or service-specific logic
            await asyncio.sleep(0.1)  # Simulate check
            return True
        except Exception as e:
            logger.debug(f"Generic health check failed for {service_name}: {e}")
            return False
    
    def force_open_circuit(self, service_name: str, reason: str = "Manual intervention"):
        """Manually force a circuit to open state"""
        circuit = self.get_circuit(service_name)
        circuit.state = CircuitState.FORCED_OPEN
        circuit.forced_reason = reason
        circuit.last_state_change = time.time()
        logger.warning(f"🔴 Circuit FORCED OPEN: {service_name} - {reason}")
    
    def force_close_circuit(self, service_name: str):
        """Manually force a circuit to closed state"""
        circuit = self.get_circuit(service_name)
        circuit.state = CircuitState.CLOSED
        circuit.forced_reason = None
        circuit.last_state_change = time.time()
        # Reset metrics for fresh start
        circuit.metrics = ServiceMetrics(circuit.config)
        logger.info(f"🟢 Circuit FORCED CLOSED: {service_name}")
    
    def reset_circuit(self, service_name: str):
        """Manually reset a circuit breaker to closed state"""
        if service_name in self.circuits:
            circuit = self.circuits[service_name]
            self._transition_to_closed(circuit)
            # Reset metrics
            circuit.metrics = ServiceMetrics(circuit.config)
            circuit.forced_reason = None
            logger.info(f"✅ Circuit breaker reset for {service_name}")
    
    def get_statistics_summary(self) -> dict[str, any]:  # Python 3.13.4 syntax
        """Get comprehensive statistics across all circuits"""
        summary = {
            "total_circuits": len(self.circuits),
            "circuits_by_state": defaultdict(int),
            "total_calls": 0,
            "total_failures": 0,
            "overall_health_score": 0.0,
            "services": []
        }
        
        health_scores = []
        
        for service_name, circuit in self.circuits.items():
            summary["circuits_by_state"][circuit.state.value] += 1
            summary["total_calls"] += circuit.metrics.total_calls
            summary["total_failures"] += circuit.metrics.total_failures
            
            health_score = circuit.metrics.get_health_score()
            health_scores.append(health_score)
            
            summary["services"].append({
                "name": service_name,
                "state": circuit.state.value,
                "calls": circuit.metrics.total_calls,
                "error_rate": circuit.metrics.get_error_rate(),
                "avg_response_time": circuit.metrics.get_average_response_time(),
                "health_score": health_score,
                "adaptive_threshold": circuit.adaptive_threshold
            })
        
        # Calculate overall health score
        if health_scores:
            summary["overall_health_score"] = sum(health_scores) / len(health_scores)
        
        return summary
    
    async def start_monitoring(self):
        """Start background monitoring of circuit breakers"""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._monitoring_enabled = True
        self._health_check_task = asyncio.create_task(self._monitoring_loop())
        logger.info("✅ Circuit breaker monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_enabled = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Circuit breaker monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_enabled:
            try:
                # Perform health checks for all circuits
                await self.health_check_all_services()
                
                # Update adaptive thresholds
                for circuit in self.circuits.values():
                    circuit.update_adaptive_threshold()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60.0)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30.0)  # Shorter sleep on error

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    
    def __init__(self, message: str, service_name: Optional[str] = None):
        super().__init__(message)
        self.service_name = service_name

# Decorator for easy circuit breaker application
def with_circuit_breaker(service_name: str, circuit_manager: Optional[AdvancedCircuitBreaker] = None):
    """Decorator to apply circuit breaker to a function"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = circuit_manager or global_circuit_breaker
            return await manager.call_with_circuit_breaker(service_name, func, *args, **kwargs)
        return wrapper
    return decorator

# Global instance
global_circuit_breaker = AdvancedCircuitBreaker()

# Convenience functions
async def call_with_breaker(service_name: str, func: Callable, *args, **kwargs) -> any:  # Python 3.13.4 syntax
    """Convenience function for calling with circuit breaker"""
    return await global_circuit_breaker.call_with_circuit_breaker(service_name, func, *args, **kwargs)

def get_circuit_status(service_name: str) -> dict[str, any]:  # Python 3.13.4 syntax
    """Get circuit breaker status for a service"""
    return global_circuit_breaker.get_circuit_status(service_name)

def get_all_circuits_status() -> dict[str, dict[str, any]]:  # Python 3.13.4 syntax
    """Get status of all circuit breakers"""
    return global_circuit_breaker.get_all_circuit_status()

async def start_circuit_monitoring():
    """Start global circuit breaker monitoring"""
    await global_circuit_breaker.start_monitoring()

async def stop_circuit_monitoring():
    """Stop global circuit breaker monitoring"""
    await global_circuit_breaker.stop_monitoring()

def force_open_circuit(service_name: str, reason: str = "Manual intervention"):
    """Force a circuit to open state"""
    global_circuit_breaker.force_open_circuit(service_name, reason)

def force_close_circuit(service_name: str):
    """Force a circuit to closed state"""
    global_circuit_breaker.force_close_circuit(service_name)

def reset_circuit(service_name: str):
    """Reset a circuit breaker"""
    global_circuit_breaker.reset_circuit(service_name)
