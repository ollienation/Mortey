## **File 1: `checkpointer.py` Analysis**

### **What Should Be Tested:**

**Classes and Methods Available:**

- `CheckpointerFactory` class with static methods:
    - `_create_async_postgres_checkpointer()`
    - `create_checkpointer_sync(environment)`
    - `_detect_environment()`
    - `_create_async_sqlite_checkpointer()`
    - `_create_sqlite_checkpointer_sync()`
    - `_create_postgres_checkpointer_sync()`
    - `_test_postgres_connection_sync()`

**Main Entry Points:**

- `create_checkpointer(environment, use_async)` - Main function to test
- `get_checkpointer_info()` - Returns configuration info
- `create_async_checkpointer(environment)` - Async wrapper
- `create_sync_checkpointer(environment)` - Sync wrapper

**Test Cases Needed:**

1. **Environment Detection Test**: Test `_detect_environment()` with different env vars
2. **Checkpointer Info Test**: Test `get_checkpointer_info()` returns correct structure
3. **Async Checkpointer Creation**: Test `create_checkpointer(use_async=True)`
4. **Sync Checkpointer Creation**: Test `create_checkpointer(use_async=False)`
5. **PostgreSQL Connection Test**: Test `_test_postgres_connection_sync()`
6. **SQLite Fallback Test**: Ensure SQLite fallback works when PostgreSQL fails
7. **Memory Fallback Test**: Ensure MemorySaver fallback works when everything fails

**Key Testing Points:**

- Test with `POSTGRES_URL` set and unset
- Test with `ENVIRONMENT` variable set to "production" vs "development"
- Verify proper fallback chain: PostgreSQL → SQLite → Memory
- Test error handling for missing dependencies
- Verify returned checkpointer types are correct

## **File 2: `llm_config.yaml` Analysis**

### **What Should Be Tested:**

**YAML Configuration Structure:**

- Global settings section with defaults
- Provider configurations (anthropic, openai)
- Individual model configurations within providers
- Node configurations for different agents

**Key Configuration Components:**

1. **Global Settings**:
    - `default_provider: "anthropic"`
    - `fallback_provider: "openai"`
    - `retry_attempts: 3`
    - `timeout_seconds: 30`
    - `enable_caching: true`
    - `log_requests: true`
2. **Provider Configurations**:
    - **Anthropic**: claude-3-haiku, claude-3-5-haiku, claude-4-sonnet
    - **OpenAI**: gpt-4o, gpt-4o-mini
    - Each with `api_key_env`, `model_id`, `max_tokens`, `temperature`, `cost_per_1m_tokens`
3. **Node Configurations**:
    - `router`: Fast routing with claude-3-haiku
    - `chat`: General conversation with claude-3-haiku
    - `coder`: Programming tasks with claude-4-sonnet
    - `web`: Web search with claude-3-5-haiku
    - `controller`: Security verification with claude-4-sonnet

**Test Cases Needed:**

1. **YAML Loading Test**: Verify the YAML file parses correctly
2. **Provider Validation**: Test that all required provider fields exist
3. **Model Configuration Test**: Verify model settings are complete
4. **Node Configuration Test**: Ensure all nodes have valid provider/model references
5. **Environment Variable Test**: Check that API key environment variables are referenced correctly
6. **Cost Configuration Test**: Verify cost tracking settings are present
7. **Temperature/Token Limits**: Validate reasonable ranges for these parameters

**Key Testing Points:**

- Validate YAML structure matches expected schema
- Test that provider API key environment variables are correctly specified
- Verify node configurations reference existing providers and models
- Check that temperature values are between 0.0 and 1.0
- Ensure max_tokens values are reasonable
- Test fallback provider configuration exists

## **File 3: `settings.py` Analysis**

### **What Should Be Tested:**

**Main Configuration Class:**

- `MorteyConfig` dataclass with comprehensive configuration management

**Class Methods and Properties:**

- `MorteyConfig.from_environment()` - Main factory method that loads configuration
- `_load_llm_config(config_file)` - YAML configuration loader
- `get_provider_config(provider_name)` - Provider configuration retrieval
- `get_node_config(node_name)` - Node configuration retrieval
- `get_model_config(provider_name, model_name)` - Model configuration retrieval
- `get_available_providers()` - List available providers
- `get_available_models(provider_name)` - List models for a provider
- `_get_workspace_dir(project_root)` - Workspace directory resolution
- `_get_audio_device_index()` - Audio device configuration
- `validate_workspace()` - Workspace write permissions check

**Configuration Components to Test:**

1. **Path Management**:
    - `project_root`, `workspace_dir`, `config_dir`, `logs_dir` path resolution
    - Directory creation and permissions
2. **LLM Configuration Loading**:
    - YAML file parsing from `llm_config.yaml`
    - Provider configurations (anthropic, openai) with API keys
    - Model configurations with proper data types
    - Node configurations with valid references
3. **Environment Variable Loading**:
    - `.env` file loading with `load_dotenv()`
    - API key extraction (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, etc.)
    - LangSmith configuration
    - Audio device configuration
4. **Global Configuration**:
    - Default settings from YAML (`default_provider`, `retry_attempts`, etc.)
    - Configuration validation and type checking

**Test Cases Needed:**

1. **Environment Detection Test**: Test with/without `.env` file
2. **YAML Loading Test**: Validate YAML parsing and data structure creation
3. **Provider Configuration Test**: Test provider availability based on API keys
4. **Model/Node Lookup Test**: Test configuration retrieval methods
5. **Path Resolution Test**: Test workspace directory fallback logic
6. **Global Config Instance Test**: Test the global `config` object initialization
7. **Validation Test**: Test `validate_workspace()` method
8. **Missing File Handling Test**: Test behavior when `llm_config.yaml` is missing

**Key Testing Focus:**

- Configuration loading without errors
- Proper dataclass instantiation with all fields
- API key presence/absence handling
- YAML structure validation
- Method return types match expected formats

## **File 4: `state.py` Analysis**

### **What Should Be Tested:**

**Main Classes:**

- `AssistantState` TypedDict with proper field definitions
- `StateValidator` class with comprehensive validation methods
- `StateValidationError` custom exception

**Key Functions and Methods:**

1. **State Creation Functions**:
    - `create_optimized_state(session_id, user_id, initial_context, validate)` - Main state factory
    - `create_assistant_state()` - Alias for backward compatibility
2. **State Validation Functions**:
    - `StateValidator.validate_state(state, strict)` - Returns (bool, List[str]) tuple
    - `StateValidator._validate_messages(messages, strict)` - Message-specific validation
    - `StateValidator.sanitize_state(state)` - State cleaning and normalization
    - `StateValidator._sanitize_messages(messages)` - Message list sanitization
3. **Message Handling Functions**:
    - `validate_and_filter_messages_v3(messages)` - Enhanced message validation
    - `safe_state_access(state, key, default)` - Safe dictionary access with type checking
4. **Utility Functions**:
    - `migrate_legacy_state(old_state)` - Legacy state format migration
    - `get_state_summary(state)` - State debugging and monitoring info
    - `optimize_state_for_processing(state, max_messages)` - Performance optimization
    - `smart_trim_messages_v2(state, max_messages)` - Message trimming alias

**Test Cases Needed:**

1. **TypedDict State Creation**: Test `create_optimized_state()` with various parameters
2. **State Validation**: Test both strict and non-strict validation modes
3. **Message Validation**: Test with HumanMessage, AIMessage, ToolMessage, SystemMessage
4. **State Sanitization**: Test with malformed state objects and edge cases
5. **Safe Access Testing**: Test `safe_state_access()` with missing keys and wrong types
6. **Legacy Migration**: Test migration from old state formats
7. **Performance Optimization**: Test message trimming and state optimization
8. **Error Handling**: Test validation error reporting and error recovery

**Key Testing Focus:**

- Validate TypedDict structure matches expected format
- Test state validation with various message combinations
- Verify safe access patterns work with malformed states
- Test message filtering with tool calls and empty content
- Validate error handling for invalid state structures

## **File 5: `error_handling.py` Analysis**

### **What Should Be Tested:**

**Main Classes and Components:**

- `ErrorHandler` class with static methods for error handling
- `ErrorClassifier` class with sophisticated error pattern matching
- `CircuitBreakerManager` class for external API protection
- `ErrorType` enum with comprehensive error categories
- Various dataclasses: `ErrorPattern`, `CircuitBreakerState`, `AssistantError`

**Key Methods to Test:**

1. **Error Classification Methods**:
    - `ErrorHandler.handle_error(error, context)` - Main error handling entry point
    - `ErrorClassifier.classify_error(error, context)` - Intelligent error classification
    - `ErrorClassifier._validate_messages()`, `_classify_by_context()`, `_check_dynamic_patterns()`
2. **Async Error Handling Methods**:
    - `ErrorHandler.with_error_handling(func, *args, context, **kwargs)` - Async function wrapper
    - `ErrorHandler.safe_execute_with_fallback()` - Execute with fallback functions
3. **Circuit Breaker Methods**:
    - `CircuitBreakerManager.call_with_circuit_breaker(service_name, func, *args, **kwargs)`
    - `CircuitBreakerState.should_allow_call()`, `record_success()`, `record_failure()`
4. **Utility Methods**:
    - `ErrorHandler.generate_fallback_response(error_type, context)` - Context-aware responses
    - `ErrorHandler.is_retryable_error(error)` - Determine retry eligibility
    - `ErrorHandler.get_error_statistics()` - Error analytics and monitoring

**Test Cases Needed:**

1. **Error Classification Test**: Test various exception types (ConnectionError, ValueError, TimeoutError, etc.)
2. **Context-Based Classification**: Test error classification with different contexts ("llm_generation", "supervisor", etc.)
3. **Async Error Handling**: Test `with_error_handling()` with both sync and async functions
4. **Circuit Breaker Functionality**: Test circuit breaker state transitions (closed → open → half_open)
5. **Fallback Response Generation**: Test context-aware fallback message generation
6. **Error Statistics**: Test error tracking and analytics collection
7. **Retry Logic**: Test retry determination for different error types
8. **Dynamic Pattern Learning**: Test error pattern learning and classification improvement

**Key Testing Focus:**

- Test error classification accuracy with real exception objects
- Validate circuit breaker state management and timeouts
- Test async error handling with actual coroutines and functions
- Verify fallback response appropriateness for different contexts
- Test error statistics collection and pattern learning

## **File 6: `agents.py` Analysis**

### **What Should Be Tested:**

**Main Classes and Methods:**

- `AgentFactory` class with agent creation methods
- Individual agent creation methods: `create_chat_agent()`, `create_coder_agent()`, `create_web_agent()`
- Tool provider methods: `_get_chat_tools()`, `_get_coder_tools()`, `_get_web_tools()`
- `get_all_tools()` method for supervisor integration

**Key Methods to Test:**

1. **Agent Creation Methods**:
    - `create_chat_agent()` - Creates ReAct agent with chat tools
    - `create_coder_agent()` - Creates ReAct agent with file management tools
    - `create_web_agent()` - Creates ReAct agent with Tavily search tools
2. **Tool Management Methods**:
    - `_get_chat_tools()` - Returns list of chat-specific tools (workspace browsing)
    - `_get_coder_tools()` - Returns FileSystemTools.get_tools() for file operations
    - `_get_web_tools()` - Returns Tavily search tools (if API key available)
    - `get_all_tools()` - Combines all tools for supervisor, with deduplication
3. **Utility Methods**:
    - `_get_model(node_name)` - Gets LLM model from llm_manager for specified node

**Test Cases Needed:**

1. **Agent Factory Initialization**: Test AgentFactory() creates instance with FileSystemTools
2. **Individual Agent Creation**: Test each agent type creates valid ReAct agent
3. **Tool Collection**: Test `_get_chat_tools()`, `_get_coder_tools()`, `_get_web_tools()` return tool lists
4. **Tool Deduplication**: Test `get_all_tools()` properly combines and deduplicates tools
5. **Model Integration**: Test `_get_model()` correctly interfaces with llm_manager
6. **Tavily Integration**: Test web tools handle missing/present TAVILY_API_KEY gracefully
7. **Error Handling**: Test agent creation with invalid node configurations

**Key Testing Focus:**

- Verify all three agent types are created successfully
- Test tool collection and integration with FileSystemTools
- Validate Tavily API key handling (present/missing scenarios)
- Test model retrieval and error handling
- Verify tool deduplication logic works correctly

## **File 7: `assistant_core.py` Analysis**

### **What Should Be Tested:**

**Main Classes:**

- `AssistantCore` class - The main orchestrator for the entire assistant system
- `AssistantSession` dataclass - Enhanced session management with persistence
- `SessionPersistenceManager` class - Database-backed session storage

**Key Methods to Test:**

1. **Initialization Methods**:
    - `AssistantCore.initialize()` - Complete system initialization
    - `_initialize_session_storage()` - Database table setup for sessions
    - `_initialize_supervisor()` - Supervisor setup with agents and tools
    - `_attempt_fallback_initialization()` - Graceful degradation on init failure
2. **Session Management Methods**:
    - `_get_or_create_session(thread_id, user_id)` - Session creation/retrieval with persistence
    - `_cleanup_expired_sessions()` - Automatic cleanup of old sessions
    - `get_session_info()` - Current session information
    - `restore_session(session_id)` - Session recovery from storage
    - `get_user_session_history(user_id, limit)` - User's previous sessions
3. **Message Processing Methods**:
    - `process_message(message, thread_id, user_id, max_tokens)` - Main entry point with `@traceable`
    - `_sanitize_user_input(message)` - Input validation and cleaning
    - `_create_validated_state(session, message)` - State creation with validation
    - `_extract_response_comprehensively(result, session)` - Multi-strategy response extraction
4. **Utility Methods**:
    - `get_system_status()` - System health and configuration info
    - `graceful_shutdown()` - Clean shutdown with session persistence.txt
    - `_extract_topic_from_response(response)` - Simple topic classification
    - `_generate_contextual_fallback(agent_used, session)` - Context-aware fallback responses

**Test Cases Needed:**

1. **Complete System Initialization**: Test full initialization chain including agents, supervisor, checkpointer
2. **Session Persistence**: Test session creation, storage, retrieval, and cleanup across restarts
3. **Message Processing Pipeline**: Test end-to-end message processing with state management
4. **Error Recovery**: Test fallback initialization and error handling throughout
5. **Database Integration**: Test session storage with both database and file fallbacks
6. **Concurrent Processing**: Test semaphore-controlled concurrent message processing
7. **System Status**: Test comprehensive status reporting and monitoring
8. **Graceful Shutdown**: Test proper cleanup and session persistence on shutdown

**Key Testing Focus:**

- Session persistence across application restarts
- Database integration with fallback mechanisms
- Comprehensive response extraction with multiple strategies
- Error handling and graceful degradation
- Performance with concurrent sessions and message processing

## **File 8: `simplified_supervisor.py`**

### **What Should Be Tested:**

**Main Classes:**

- `SimplifiedSupervisor` class - The core supervisor with intelligent routing
- `SupervisorConfig` dataclass - Configuration management for supervisor behavior
- `SupervisorError` custom exception class

**Key Methods Available for Testing:**

1. **Initialization Methods**:
    - `__init__(config)` - Constructor with optional configuration
    - `initialize(agents, all_tools, checkpointer)` - Main setup method
    - `_validate_initialization_inputs(agents, tools)` - Input validation
2. **Core Processing Methods**:
    - `process(state, config_dict)` - Main entry point for request processing
    - `_create_agent_node(agent, agent_name)` - Agent wrapper creation
    - `_route_to_agent(state)` - Intelligent routing logic with keyword matching
3. **Configuration Methods**:
    - `update_routing_keywords(agent_name, keywords)` - Dynamic keyword updates
    - `get_configuration()` - Get current config
    - `set_configuration(new_config)` - Update configuration
4. **Monitoring Methods**:
    - `get_routing_statistics()` - Routing stats and metrics
    - Internal routing statistics tracking via `_routing_stats`

**Configuration Components to Test:**

- `SupervisorConfig` with routing keywords for "coder" and "web" agents
- Default agent fallback logic
- Routing logs enable/disable functionality
- Max replays prevention

**Test Cases Needed:**

1. **Supervisor Initialization**: Test initialization with various agent/tool combinations
2. **Agent Routing Logic**: Test keyword-based routing to different agents (coder, web, chat)
3. **Configuration Management**: Test dynamic configuration updates and keyword modification
4. **Error Handling**: Test supervisor error handling and fallback mechanisms
5. **State Processing**: Test end-to-end state processing through the supervisor graph
6. **Statistics Tracking**: Test routing statistics collection and reporting
7. **LangGraph Integration**: Test proper StateGraph compilation and execution
8. **Tool Integration**: Test ToolNode creation and tool call routing

**Key Testing Focus:**

- Validate intelligent routing based on message content analysis
- Test configuration flexibility and dynamic updates
- Verify proper LangGraph 0.4.8 patterns and START node usage
- Test error recovery and fallback agent routing
- Validate statistics collection and monitoring capabilities

## **File 9: `file_tools.py`**

### **What Should Be Tested:**

**Main Class:**

- `FileSystemTools` class - Enhanced file system tools for comprehensive file management

**Key Methods Available for Testing:**

1. **Initialization Methods**:
    - `__init__(workspace_dir)` - Constructor with workspace directory setup
    - Uses `FileManagementToolkit` from LangChain with selected base tools
2. **Main Tool Collection**:
    - `get_tools()` - Returns combined list of base tools + custom enhanced tools
3. **Base LangChain Tools** (from FileManagementToolkit):
    - `read_file`, `write_file`, `list_directory`, `copy_file`, `move_file`
4. **Custom Enhanced Tools** (defined with `@tool` decorator):
    - `create_project(project_name, project_type)` - Creates project structures (python, web, data)
    - `search_in_files(query, file_extension)` - Search text across workspace files
    - `backup_file(filename)` - Create timestamped backups
    - `file_info(filename)` - Detailed file information with MD5, mime types, stats
    - `organize_workspace()` - Organize files by type into subdirectories
    - `convert_file_format(filename, output_format)` - Convert between JSON, YAML, CSV, TXT, MD
    - `summarize_file(filename, max_length)` - Generate file summaries with statistics

**Test Cases Needed:**

1. **Tool Initialization**: Test `FileSystemTools()` creates with proper workspace and toolkit
2. **Base Tool Integration**: Test that LangChain base tools are properly included
3. **Project Creation**: Test creating different project types (python, web, data, generic)
4. **File Search**: Test searching across files with and without extension filters
5. **File Operations**: Test backup, info, organization, conversion, summarization
6. **Error Handling**: Test with missing files, invalid formats, permission errors
7. **Tool Collection**: Test `get_tools()` returns complete list of base + enhanced tools
8. **Workspace Management**: Test workspace directory creation and file organization

**Key Testing Focus:**

- Verify all 12 tools (5 base + 7 enhanced) are returned by `get_tools()`
- Test project scaffolding for different project types
- Validate file format conversions work correctly
- Test file search functionality across different file types
- Verify workspace organization maintains file integrity
- Test error handling for missing dependencies (yaml, csv modules)

## **File 10: `llm_manager.py`**

### **What Should Be Tested:**

**Main Class:**

- `LLMManager` class - Universal LLM client manager with model caching and circuit breaker integration

**Key Methods to Test:**

1. **Initialization Methods**:
    - `__init__()` - Sets up model cache, LangSmith, and concurrency controls
    - `_setup_langsmith()` - LangSmith tracing configuration
    - `_initialize_concurrency_controls()` - Semaphore setup for rate limiting
2. **Model Management Methods**:
    - `_get_model(node_name, override_max_tokens)` - **Fixed model caching** - the core method
    - `_get_cache_key(node_name, override_max_tokens)` - Cache key generation with all parameters
    - `clear_cache()` - Cache clearing for testing/memory management
    - `get_cache_info()` - Cache inspection and debugging
3. **Core Generation Method**:
    - `generate_for_node(node_name, prompt, override_max_tokens, metadata)` - **Main entry point with @traceable**
    - Uses cached models, concurrency control, retry logic, and token tracking
4. **Monitoring and Health Methods**:
    - `health_check()` - Provider health checking with temporary node configs
    - `get_usage_stats()` - Comprehensive usage statistics
    - `_update_token_usage(provider, prompt_tokens, completion_tokens)` - Token usage tracking
5. **Concurrency Control Features**:
    - Global semaphore (5 concurrent calls)
    - Provider-specific semaphores (anthropic: 3, openai: 5, default: 2)
    - Retry logic with exponential backoff and jitter

**Integration Points to Test**:

- **Circuit Breaker Integration**: `from core.circuit_breaker import global_circuit_breaker, with_circuit_breaker`
- **Configuration Integration**: Uses `config.get_node_config()`, `get_provider_config()`, `get_model_config()`
- **LangSmith Integration**: Optional tracing with fallback decorator
- **Global Instance**: `llm_manager = LLMManager()` singleton pattern

**Test Cases Needed:**

1. **Model Caching Test**: Verify models are actually cached and reused (the critical fix)
2. **Cache Key Generation**: Test cache keys include all parameters (provider, model, temperature, max_tokens)
3. **Configuration Integration**: Test node/provider/model config retrieval and validation
4. **Concurrency Control**: Test semaphore limits and provider-specific controls
5. **Retry Logic**: Test exponential backoff with jitter for failed calls
6. **Token Usage Tracking**: Test token counting and provider-specific statistics
7. **Health Check**: Test provider health checking with temporary node creation
8. **Error Handling**: Test configuration errors, API failures, and fallbacks
9. **LangSmith Integration**: Test with and without LangSmith available
10. **Performance**: Test that cached models provide significant performance improvement

**Key Testing Focus:**

- **Verify the caching fix works**: Models should be cached and reused, not created fresh each time
- **Test concurrency limits**: Ensure semaphores properly limit concurrent API calls
- **Validate configuration integration**: Ensure proper YAML config reading and error handling
- **Test circuit breaker integration**: Verify integration with the circuit breaker system
- **Performance validation**: Measure response time improvement from model caching

## **File 11: `circuit_breaker.py`**

### **What Should Be Tested:**

**Main Classes:**

- `AdvancedCircuitBreaker` class - Production-grade circuit breaker with adaptive behavior
- `ServiceMetrics` class - Comprehensive metrics tracking for services
- `CircuitBreakerState` dataclass - Individual circuit state management
- `ServiceConfig` dataclass - Configuration for service-specific circuit breakers
- `CircuitState` enum - Circuit states (CLOSED, OPEN, HALF_OPEN)
- `CircuitBreakerOpenException` - Custom exception for open circuits

**Key Methods to Test:**

1. **Circuit Management Methods**:
    - `get_circuit(service_name)` - Get or create circuit breaker for a service
    - `should_allow_request(service_name)` - Request permission checking
    - `call_with_circuit_breaker(service_name, func, *args, **kwargs)` - Main protected call method
2. **State Transition Methods**:
    - `_should_trip_circuit(circuit)` - Circuit tripping logic
    - `_transition_to_open(circuit)`, `_transition_to_half_open(circuit)`, `_transition_to_closed(circuit)`
3. **Metrics and Monitoring**:
    - `ServiceMetrics.record_call(success, duration, error)` - Call result recording
    - `ServiceMetrics.get_error_rate()`, `get_average_response_time()`, `get_throughput()`
    - `get_circuit_status(service_name)` - Detailed circuit status
    - `get_all_circuit_status()` - Status of all circuits
    - `get_statistics_summary()` - Comprehensive statistics
4. **Health Check Methods**:
    - `health_check(service_name)` - Service-specific health checks
    - `_health_check_anthropic()`, `_health_check_openai()`, `_health_check_tavily()`, `_health_check_file_system()`
5. **Utility Methods**:
    - `reset_circuit(service_name)` - Manual circuit reset
    - `_execute_function(func, *args, **kwargs)` - Sync/async function execution

**Global Functions to Test**:

- `call_with_breaker(service_name, func, *args, **kwargs)` - Convenience function
- `get_circuit_status(service_name)` - Global status getter
- `get_all_circuits_status()` - Global status for all circuits
- `@with_circuit_breaker` decorator functionality

**Test Cases Needed:**

1. **Circuit State Transitions**: Test CLOSED → OPEN → HALF_OPEN → CLOSED flow
2. **Service Configuration**: Test different configs for anthropic, openai, tavily, file_system
3. **Metrics Tracking**: Test call recording, error rates, response times, throughput
4. **Failure Threshold Testing**: Test consecutive failures and error rate thresholds
5. **Recovery Testing**: Test half-open recovery and success threshold logic
6. **Health Check Integration**: Test health checks for all service types
7. **Decorator Functionality**: Test `@with_circuit_breaker` decorator
8. **Exception Handling**: Test `CircuitBreakerOpenException` and error scenarios
9. **Statistics and Monitoring**: Test comprehensive statistics collection
10. **Manual Circuit Control**: Test manual circuit reset functionality