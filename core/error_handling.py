# core/error_handling.py - ✅ ENHANCED ERROR CLASSIFICATION WITH SOPHISTICATED PATTERNS
from typing import Dict, Any, Optional, Union, Callable, Awaitable, Type, Set, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import inspect
import time
import json
from collections import defaultdict, deque
from functools import wraps

logger = logging.getLogger("error_handling")

class ErrorType(Enum):
    """Comprehensive error types for intelligent classification"""
    EMPTY_CONTENT = "empty_content"
    RATE_LIMIT = "rate_limit"
    CONNECTION_ERROR = "connection_error"
    SUPERVISOR_ERROR = "supervisor_error"
    VALIDATION_ERROR = "validation_error"
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_ERROR = "timeout_error"
    CONTEXT_MANAGER_ERROR = "context_manager_error"
    CHECKPOINTER_ERROR = "checkpointer_error"
    STATE_ERROR = "state_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_OVERLOADED = "model_overloaded"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"

@dataclass
class ErrorPattern:
    """Pattern for intelligent error classification"""
    exception_types: Set[Type[Exception]]
    string_patterns: Set[str]
    error_type: ErrorType
    retryable: bool
    priority: int = 1  # Higher priority patterns are checked first

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for external API calls"""
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    call_timeout: float = 30.0
    
    def should_allow_call(self) -> bool:
        """Check if calls should be allowed based on circuit breaker state"""
        current_time = time.time()
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """Record successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

@dataclass
class AssistantError:
    """Enhanced error response structure with context awareness"""
    error_type: ErrorType
    message: str
    details: Optional[str] = None
    fallback_response: Optional[str] = None
    retryable: bool = False
    user_facing: bool = True
    context: str = ""
    retry_after: Optional[float] = None
    recovery_suggestions: List[str] = field(default_factory=list)

class ErrorClassifier:
    """✅ ENHANCED: Sophisticated error classification using exception types and patterns"""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.dynamic_patterns = defaultdict(list)  # Learning patterns
        self.error_history = deque(maxlen=1000)  # Track error patterns
        
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize comprehensive error patterns"""
        patterns = [
            # LLM Provider Errors
            ErrorPattern(
                exception_types={ConnectionError, TimeoutError, OSError},
                string_patterns={"connection", "timeout", "network", "unreachable"},
                error_type=ErrorType.CONNECTION_ERROR,
                retryable=True,
                priority=3
            ),
            
            # Rate Limiting
            ErrorPattern(
                exception_types={Exception},  # Often wrapped in generic exceptions
                string_patterns={"rate limit", "429", "quota", "too many requests", "throttling"},
                error_type=ErrorType.RATE_LIMIT,
                retryable=True,
                priority=5
            ),
            
            # Authentication Issues
            ErrorPattern(
                exception_types={PermissionError, ValueError},
                string_patterns={"unauthorized", "401", "forbidden", "403", "api key", "authentication", "invalid_api_key"},
                error_type=ErrorType.AUTHENTICATION_ERROR,
                retryable=False,
                priority=4
            ),
            
            # LangGraph Specific Errors
            ErrorPattern(
                exception_types={AttributeError, TypeError},
                string_patterns={"_asyncgeneratorcontextmanager", "get_next_version", "langgraph"},
                error_type=ErrorType.CONTEXT_MANAGER_ERROR,
                retryable=False,
                priority=5
            ),
            
            # Checkpointer Errors
            ErrorPattern(
                exception_types={ConnectionError, OSError, ValueError},
                string_patterns={"checkpointer", "checkpoint", "database", "conn", "sqlite", "postgres"},
                error_type=ErrorType.CHECKPOINTER_ERROR,
                retryable=True,
                priority=4
            ),
            
            # State Management Errors
            ErrorPattern(
                exception_types={KeyError, TypeError, AttributeError},
                string_patterns={"state", "messages", "dict", "typeddict", "session_id"},
                error_type=ErrorType.STATE_ERROR,
                retryable=True,
                priority=3
            ),
            
            # Model Overloaded
            ErrorPattern(
                exception_types={Exception},
                string_patterns={"overloaded", "busy", "502", "503", "service unavailable", "capacity"},
                error_type=ErrorType.MODEL_OVERLOADED,
                retryable=True,
                priority=4
            ),
            
            # Configuration Errors
            ErrorPattern(
                exception_types={ImportError, ModuleNotFoundError, KeyError},
                string_patterns={"not found", "missing", "config", "import", "module"},
                error_type=ErrorType.CONFIGURATION_ERROR,
                retryable=False,
                priority=3
            ),
            
            # Tool Errors
            ErrorPattern(
                exception_types={FileNotFoundError, PermissionError, ValueError},
                string_patterns={"tool", "function", "file", "permission", "directory"},
                error_type=ErrorType.TOOL_ERROR,
                retryable=True,
                priority=2
            )
        ]
        
        # Sort by priority (higher first)
        return sorted(patterns, key=lambda p: p.priority, reverse=True)
    
    def classify_error(self, error: Exception, context: str = "") -> ErrorType:
        """
        ✅ ENHANCED: Classify errors using multiple strategies
        """
        error_str = str(error).lower()
        error_type = type(error)
        
        # Strategy 1: Exception type matching (most reliable)
        for pattern in self.error_patterns:
            if error_type in pattern.exception_types:
                # Additional string pattern check for precision
                if any(pattern_str in error_str for pattern_str in pattern.string_patterns):
                    self._record_pattern_match(pattern.error_type, error, context)
                    return pattern.error_type
        
        # Strategy 2: String pattern matching
        for pattern in self.error_patterns:
            if any(pattern_str in error_str for pattern_str in pattern.string_patterns):
                self._record_pattern_match(pattern.error_type, error, context)
                return pattern.error_type
        
        # Strategy 3: Context-based classification
        context_error_type = self._classify_by_context(error, context)
        if context_error_type:
            return context_error_type
        
        # Strategy 4: Dynamic pattern matching (learned patterns)
        dynamic_type = self._check_dynamic_patterns(error, context)
        if dynamic_type:
            return dynamic_type
        
        # Fallback
        self._record_pattern_match(ErrorType.SYSTEM_ERROR, error, context)
        return ErrorType.SYSTEM_ERROR
    
    def _classify_by_context(self, error: Exception, context: str) -> Optional[ErrorType]:
        """Classify errors based on context"""
        context_mappings = {
            "llm_generation": ErrorType.LLM_ERROR,
            "supervisor": ErrorType.SUPERVISOR_ERROR,
            "checkpointer": ErrorType.CHECKPOINTER_ERROR,
            "tool_execution": ErrorType.TOOL_ERROR,
            "state_management": ErrorType.STATE_ERROR,
            "message_processing": ErrorType.VALIDATION_ERROR
        }
        
        for context_pattern, error_type in context_mappings.items():
            if context_pattern in context.lower():
                return error_type
        
        return None
    
    def _check_dynamic_patterns(self, error: Exception, context: str) -> Optional[ErrorType]:
        """Check dynamically learned patterns"""
        error_signature = f"{type(error).__name__}:{str(error)[:100]}"
        
        for error_type, signatures in self.dynamic_patterns.items():
            if any(sig in error_signature for sig in signatures):
                return ErrorType(error_type)
        
        return None
    
    def _record_pattern_match(self, error_type: ErrorType, error: Exception, context: str):
        """Record error pattern for learning"""
        error_record = {
            "error_type": error_type.value,
            "exception_type": type(error).__name__,
            "error_message": str(error)[:200],
            "context": context,
            "timestamp": time.time()
        }
        
        self.error_history.append(error_record)
        
        # Learn dynamic patterns
        error_signature = f"{type(error).__name__}:{str(error)[:50]}"
        if error_signature not in self.dynamic_patterns[error_type.value]:
            self.dynamic_patterns[error_type.value].append(error_signature)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error classification statistics"""
        error_counts = defaultdict(int)
        recent_errors = []
        
        current_time = time.time()
        for record in self.error_history:
            error_counts[record["error_type"]] += 1
            if current_time - record["timestamp"] < 3600:  # Last hour
                recent_errors.append(record)
        
        return {
            "total_errors": len(self.error_history),
            "error_distribution": dict(error_counts),
            "recent_errors_count": len(recent_errors),
            "dynamic_patterns_count": sum(len(patterns) for patterns in self.dynamic_patterns.values())
        }

class CircuitBreakerManager:
    """✅ NEW: Circuit breaker manager for external API calls"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreakerState()
        return self.circuit_breakers[service_name]
    
    async def call_with_circuit_breaker(
        self, 
        service_name: str, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        if not circuit_breaker.should_allow_call():
            raise Exception(f"Circuit breaker open for {service_name}")
        
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                ErrorHandler.with_error_handling(func, *args, **kwargs),
                timeout=circuit_breaker.call_timeout
            )
            circuit_breaker.record_success()
            return result
        except Exception as e:
            circuit_breaker.record_failure()
            raise

class ErrorHandler:
    """
    ✅ ENHANCED: Sophisticated error handling with intelligent classification and circuit breakers
    """
    
    _classifier = ErrorClassifier()
    _circuit_breaker_manager = CircuitBreakerManager()
    
    @staticmethod
    def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """
        ✅ ENHANCED: Handle errors with intelligent classification and context awareness
        """
        # Classify the error using sophisticated patterns
        error_type = ErrorHandler._classifier.classify_error(error, context)
        
        # Generate intelligent error response
        error_info = ErrorHandler._generate_error_info(error, error_type, context)
        
        # Log with appropriate level
        log_level = logging.ERROR if not error_info.retryable else logging.WARNING
        logger.log(log_level, f"Error in {context}: {error_type.value} - {error}")
        
        # Build response
        response = {
            "response": error_info.fallback_response,
            "error": error_info.error_type.value,
            "error_message": error_info.message,
            "details": error_info.details,
            "retryable": error_info.retryable,
            "fallback_used": True,
            "context": context,
            "user_facing": error_info.user_facing
        }
        
        # Add retry information if applicable
        if error_info.retry_after:
            response["retry_after"] = error_info.retry_after
        
        if error_info.recovery_suggestions:
            response["recovery_suggestions"] = error_info.recovery_suggestions
        
        return response
    
    @staticmethod
    def _generate_error_info(error: Exception, error_type: ErrorType, context: str) -> AssistantError:
        """Generate comprehensive error information"""
        error_configs = {
            ErrorType.RATE_LIMIT: AssistantError(
                error_type=error_type,
                message="Rate limit exceeded",
                fallback_response="I'm experiencing high demand. Please try again in a moment.",
                retryable=True,
                retry_after=60.0,
                recovery_suggestions=["Wait a few minutes before retrying", "Try a simpler request"]
            ),
            ErrorType.CONNECTION_ERROR: AssistantError(
                error_type=error_type,
                message="Connection error",
                fallback_response="I'm having trouble connecting to my services. Please try again.",
                retryable=True,
                retry_after=10.0,
                recovery_suggestions=["Check your internet connection", "Try again in a few seconds"]
            ),
            ErrorType.AUTHENTICATION_ERROR: AssistantError(
                error_type=error_type,
                message="Authentication error",
                fallback_response="I'm experiencing an authentication issue. Please contact support.",
                retryable=False,
                user_facing=False,
                recovery_suggestions=["Check API key configuration", "Verify credentials"]
            ),
            ErrorType.MODEL_OVERLOADED: AssistantError(
                error_type=error_type,
                message="Model overloaded",
                fallback_response="The AI service is currently busy. Please try again shortly.",
                retryable=True,
                retry_after=30.0,
                recovery_suggestions=["Wait a moment and retry", "Try during off-peak hours"]
            ),
            ErrorType.CONTEXT_MANAGER_ERROR: AssistantError(
                error_type=error_type,
                message="Async context manager error",
                fallback_response="I'm experiencing a technical issue. Let me try a different approach.",
                retryable=False,
                recovery_suggestions=["Restart the assistant", "Check system configuration"]
            )
        }
        
        # Get configured error info or create default
        error_info = error_configs.get(error_type, AssistantError(
            error_type=error_type,
            message="System error",
            fallback_response=ErrorHandler.generate_fallback_response(error_type, context),
            retryable=ErrorHandler.is_retryable_error(error)
        ))
        
        # Set details from actual error
        error_info.details = str(error)
        error_info.context = context
        
        return error_info
    
    @staticmethod
    async def with_error_handling(
        func: Union[Callable, Awaitable], 
        *args, 
        context: str = "", 
        **kwargs
    ) -> Any:
        """
        ✅ ENHANCED: Execute function with comprehensive error handling and retry logic
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                # Handle different types of callables
                if inspect.iscoroutine(func):
                    result = await func
                elif inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                elif callable(func):
                    result = func(*args, **kwargs)
                    if inspect.iscoroutine(result):
                        result = await result
                else:
                    result = func
                
                return result
                
            except Exception as e:
                error_type = ErrorHandler._classifier.classify_error(e, context)
                
                # Don't retry non-retryable errors
                if not ErrorHandler.is_retryable_error(e) or attempt == max_retries:
                    return ErrorHandler.handle_error(e, context)
                
                # Calculate retry delay with exponential backoff and jitter
                retry_delay = base_delay * (2 ** attempt)
                jitter = retry_delay * 0.1 * (0.5 - 0.5 * (attempt / max_retries))
                total_delay = retry_delay + jitter
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed in {context}: {e}. "
                    f"Retrying in {total_delay:.2f}s"
                )
                await asyncio.sleep(total_delay)
        
        # This shouldn't be reached, but just in case
        return ErrorHandler.handle_error(Exception("Max retries exceeded"), context)
    
    @staticmethod
    def generate_fallback_response(error_type: ErrorType, context: str = "") -> str:
        """
        ✅ ENHANCED: Generate contextual fallback responses
        """
        context_responses = {
            "message_processing": "I'm ready to help with your next question.",
            "supervisor_processing": "Let me try routing your request differently.",
            "agent_execution": "I'll attempt to handle your request using a different approach.",
            "checkpointer_operation": "Your message was processed, but conversation history may not be saved.",
            "tool_execution": "I encountered an issue with a tool. Let me try another method.",
            "llm_generation": "I had trouble generating a response. Please try rephrasing your request."
        }
        
        # Get context-specific response
        for context_pattern, response in context_responses.items():
            if context_pattern in context.lower():
                return response
        
        # Fallback to error type defaults
        error_responses = {
            ErrorType.CONTEXT_MANAGER_ERROR: "I'm experiencing a technical issue. Please try again.",
            ErrorType.CHECKPOINTER_ERROR: "I'm having trouble with conversation persistence. Please continue.",
            ErrorType.STATE_ERROR: "I encountered an issue with message handling. Please try rephrasing.",
            ErrorType.SUPERVISOR_ERROR: "I'm experiencing a routing issue. Let me try a different approach.",
            ErrorType.TOOL_ERROR: "I encountered an issue while using a tool. Let me try another method.",
            ErrorType.CONNECTION_ERROR: "I'm having trouble connecting to services. Please try again.",
            ErrorType.RATE_LIMIT: "I'm experiencing high demand. Please try again in a moment.",
            ErrorType.TIMEOUT_ERROR: "The request took too long. Please try again.",
            ErrorType.EMPTY_CONTENT: "I detected an issue with the response. Please try rephrasing.",
            ErrorType.AUTHENTICATION_ERROR: "I'm experiencing authentication issues. Please contact support.",
            ErrorType.MODEL_OVERLOADED: "The AI service is currently busy. Please try again shortly.",
            ErrorType.SYSTEM_ERROR: "I encountered an unexpected issue. Please try again."
        }
        
        return error_responses.get(error_type, "I'm ready to help with your next question.")
    
    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """
        ✅ ENHANCED: Determine if an error is retryable using intelligent classification
        """
        error_type = ErrorHandler._classifier.classify_error(error)
        
        # Non-retryable error types
        non_retryable_types = {
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.CONFIGURATION_ERROR,
            ErrorType.CONTEXT_MANAGER_ERROR,
            ErrorType.DEPENDENCY_ERROR
        }
        
        return error_type not in non_retryable_types
    
    @staticmethod
    async def with_circuit_breaker(
        service_name: str,
        func: Callable,
        *args,
        context: str = "",
        **kwargs
    ) -> Any:
        """
        ✅ NEW: Execute function with circuit breaker protection
        """
        try:
            return await ErrorHandler._circuit_breaker_manager.call_with_circuit_breaker(
                service_name, func, *args, **kwargs
            )
        except Exception as e:
            return ErrorHandler.handle_error(e, f"{context}_circuit_breaker")
    
    @staticmethod
    def get_error_statistics() -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        classifier_stats = ErrorHandler._classifier.get_error_statistics()
        circuit_breaker_stats = {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time
            }
            for name, cb in ErrorHandler._circuit_breaker_manager.circuit_breakers.items()
        }
        
        return {
            "classification": classifier_stats,
            "circuit_breakers": circuit_breaker_stats,
            "patterns_learned": len(ErrorHandler._classifier.dynamic_patterns)
        }
    
    @staticmethod
    async def safe_execute_with_fallback(
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        context: str = "",
        *args,
        **kwargs
    ) -> Any:
        """
        ✅ ENHANCED: Execute with intelligent fallback and circuit breaker protection
        """
        try:
            return await ErrorHandler.with_error_handling(
                primary_func, *args, context=f"{context}_primary", **kwargs
            )
        except Exception as primary_error:
            logger.warning(f"Primary function failed in {context}: {primary_error}")
            
            if fallback_func:
                try:
                    return await ErrorHandler.with_error_handling(
                        fallback_func, *args, context=f"{context}_fallback", **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback function also failed in {context}: {fallback_error}")
                    return ErrorHandler.handle_error(fallback_error, f"{context}_fallback_failed")
            else:
                return ErrorHandler.handle_error(primary_error, context)

# Export convenience functions
handle_error = ErrorHandler.handle_error
with_error_handling = ErrorHandler.with_error_handling
with_circuit_breaker = ErrorHandler.with_circuit_breaker
