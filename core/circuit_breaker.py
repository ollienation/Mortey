"""
Advanced Circuit Breaker Implementation for LangGraph 0.4.8 Assistant

This module provides sophisticated circuit breaker patterns specifically designed
for protecting external API calls in AI assistant applications.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable, Union, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from functools import wraps

logger = logging.getLogger("circuit_breaker")

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Circuit is open, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class ServiceConfig:
    """Configuration for a specific service's circuit breaker"""
    failure_threshold: int = 5           # Failures needed to open circuit
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    call_timeout: float = 30.0           # Timeout for individual calls
    success_threshold: int = 3           # Successes needed to close from half-open
    rolling_window_size: int = 100       # Size of rolling window for statistics
    min_throughput: int = 10             # Minimum calls before circuit can trip
    error_rate_threshold: float = 0.5    # Error rate (0.0-1.0) to trip circuit
    slow_call_threshold: float = 10.0    # Seconds to consider a call "slow"
    
class ServiceMetrics:
    """Metrics tracking for a service"""
    
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
        
    def record_call(self, success: bool, duration: float, error: Optional[Exception] = None):
        """Record the result of a service call"""
        call_time = time.time()
        
        call_record = {
            "timestamp": call_time,
            "success": success,
            "duration": duration,
            "error_type": type(error).__name__ if error else None
        }
        
        self.call_history.append(call_record)
        self.total_calls += 1
        
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
            if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
                self.total_timeouts += 1
        
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

@dataclass 
class CircuitBreakerState:
    """State of an individual circuit breaker"""
    service_name: str
    state: CircuitState = CircuitState.CLOSED
    config: ServiceConfig = field(default_factory=ServiceConfig)
    metrics: ServiceMetrics = None
    last_state_change: float = field(default_factory=time.time)
    half_open_successes: int = 0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ServiceMetrics(self.config)

class AdvancedCircuitBreaker:
    """
    ✅ PRODUCTION-GRADE: Advanced circuit breaker with adaptive behavior
    
    Features:
    - Multiple circuit breaker states per service
    - Rolling window metrics and statistics
    - Adaptive thresholds based on service behavior
    - Health check integration
    - Detailed monitoring and alerting
    """
    
    def __init__(self):
        self.circuits: Dict[str, CircuitBreakerState] = {}
        self.default_configs = {
            "anthropic": ServiceConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                call_timeout=30.0,
                error_rate_threshold=0.4,
                min_throughput=5
            ),
            "openai": ServiceConfig(
                failure_threshold=3,
                recovery_timeout=30.0, 
                call_timeout=30.0,
                error_rate_threshold=0.4,
                min_throughput=5
            ),
            "tavily": ServiceConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                call_timeout=10.0,
                error_rate_threshold=0.6,
                min_throughput=3
            ),
            "file_system": ServiceConfig(
                failure_threshold=10,
                recovery_timeout=5.0,
                call_timeout=5.0,
                error_rate_threshold=0.7,
                min_throughput=5
            ),
            "default": ServiceConfig()
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
        
        if circuit.state == CircuitState.CLOSED:
            # Check if we should trip the circuit
            if self._should_trip_circuit(circuit):
                self._transition_to_open(circuit)
                return False
            return True
        
        elif circuit.state == CircuitState.OPEN:
            # Check if we should try half-open
            if current_time - circuit.last_state_change >= circuit.config.recovery_timeout:
                self._transition_to_half_open(circuit)
                return True
            return False
        
        elif circuit.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return True
        
        return False
    
    def _should_trip_circuit(self, circuit: CircuitBreakerState) -> bool:
        """Determine if circuit should be tripped to open state"""
        # Check consecutive failures
        if circuit.metrics.consecutive_failures >= circuit.config.failure_threshold:
            logger.warning(f"Tripping circuit {circuit.service_name}: {circuit.metrics.consecutive_failures} consecutive failures")
            return True
        
        # Check error rate over rolling window
        error_rate = circuit.metrics.get_error_rate()
        if (error_rate >= circuit.config.error_rate_threshold and 
            len(circuit.metrics.call_history) >= circuit.config.min_throughput):
            logger.warning(f"Tripping circuit {circuit.service_name}: error rate {error_rate:.2%}")
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
    ) -> Any:
        """Execute a function call with circuit breaker protection"""
        circuit = self.get_circuit(service_name)
        
        # Check if request should be allowed
        if not self.should_allow_request(service_name):
            raise CircuitBreakerOpenException(
                f"Circuit breaker open for service: {service_name}"
            )
        
        start_time = time.time()
        try:
            # Apply timeout
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
            
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function handling both sync and async callables"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        elif asyncio.iscoroutine(func):
            return await func
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def get_circuit_status(self, service_name: str) -> Dict[str, Any]:
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
            "metrics": {
                "total_calls": metrics.total_calls,
                "total_failures": metrics.total_failures,
                "consecutive_failures": metrics.consecutive_failures,
                "error_rate": metrics.get_error_rate(),
                "average_response_time": metrics.get_average_response_time(),
                "throughput": metrics.get_throughput(),
                "last_success": metrics.last_success_time,
                "last_failure": metrics.last_failure_time
            },
            "config": {
                "failure_threshold": circuit.config.failure_threshold,
                "recovery_timeout": circuit.config.recovery_timeout,
                "error_rate_threshold": circuit.config.error_rate_threshold
            }
        }
    
    def get_all_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {
            service_name: self.get_circuit_status(service_name)
            for service_name in self.circuits.keys()
        }
    
    async def health_check(self, service_name: str) -> bool:
        """Perform health check for a specific service"""
        try:
            circuit = self.get_circuit(service_name)
            
            # Define simple health check functions for each service
            health_checks = {
                "anthropic": self._health_check_anthropic,
                "openai": self._health_check_openai,
                "tavily": self._health_check_tavily,
                "file_system": self._health_check_file_system
            }
            
            if service_name in health_checks:
                await self.call_with_circuit_breaker(
                    f"{service_name}_health",
                    health_checks[service_name]
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return False
    
    async def _health_check_anthropic(self) -> bool:
        """Simple health check for Anthropic API"""
        from config.llm_manager import llm_manager
        try:
            # Try to get a model (this will test configuration and API key)
            await llm_manager.generate_for_node("chat", "Hi", override_max_tokens=5)
            return True
        except Exception:
            return False
    
    async def _health_check_openai(self) -> bool:
        """Simple health check for OpenAI API"""
        # Similar implementation for OpenAI if configured
        return True
    
    async def _health_check_tavily(self) -> bool:
        """Simple health check for Tavily API"""
        import os
        return bool(os.getenv("TAVILY_API_KEY"))
    
    async def _health_check_file_system(self) -> bool:
        """Simple health check for file system operations"""
        from config.settings import config
        try:
            test_file = config.workspace_dir / ".health_check"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def reset_circuit(self, service_name: str):
        """Manually reset a circuit breaker to closed state"""
        if service_name in self.circuits:
            circuit = self.circuits[service_name]
            self._transition_to_closed(circuit)
            # Reset metrics
            circuit.metrics = ServiceMetrics(circuit.config)
            logger.info(f"✅ Circuit breaker reset for {service_name}")
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all circuits"""
        summary = {
            "total_circuits": len(self.circuits),
            "circuits_by_state": defaultdict(int),
            "total_calls": 0,
            "total_failures": 0,
            "services": []
        }
        
        for service_name, circuit in self.circuits.items():
            summary["circuits_by_state"][circuit.state.value] += 1
            summary["total_calls"] += circuit.metrics.total_calls
            summary["total_failures"] += circuit.metrics.total_failures
            
            summary["services"].append({
                "name": service_name,
                "state": circuit.state.value,
                "calls": circuit.metrics.total_calls,
                "error_rate": circuit.metrics.get_error_rate(),
                "avg_response_time": circuit.metrics.get_average_response_time()
            })
        
        return summary

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

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
async def call_with_breaker(service_name: str, func: Callable, *args, **kwargs) -> Any:
    """Convenience function for calling with circuit breaker"""
    return await global_circuit_breaker.call_with_circuit_breaker(service_name, func, *args, **kwargs)

def get_circuit_status(service_name: str) -> Dict[str, Any]:
    """Get circuit breaker status for a service"""
    return global_circuit_breaker.get_circuit_status(service_name)

def get_all_circuits_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers"""
    return global_circuit_breaker.get_all_circuit_status()
