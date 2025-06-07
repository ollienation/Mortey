Mortey/
├── core/
│   ├── __init__.py
│   ├── assistant_core.py          # AssistantCore, AssistantSession, SessionPersistenceManager
│   ├── checkpointer.py            # CheckpointerFactory, create_checkpointer, environment detection
│   ├── simplified_supervisor.py   # SimplifiedSupervisor, SupervisorConfig, routing logic
│   ├── state.py                   # AssistantState TypedDict, StateValidator, state utilities
│   ├── error_handling.py          # ErrorHandler, ErrorClassifier, circuit breaker components
│   └── circuit_breaker.py         # AdvancedCircuitBreaker, ServiceMetrics, (not implemented)
│
├── config/
│   ├── __init__.py
│   ├── settings.py                # MorteyConfig, environment loading, configuration management
│   ├── llm_manager.py             # LLMManager with caching, concurrency, circuit breaker integration
│   └── llm_config.yaml            # YAML configuration for providers, models, nodes
│
├── agents/
│   ├── __init__.py
│   └── agents.py                  # AgentFactory, agent creation methods, tool management
│
├── tools/
│   ├── __init__.py
│   └── file_tools.py              # FileSystemTools, enhanced file operations, project creation
│
├── logs/                          # Will be created by our test suite
│   └── test_results_YYYYMMDD_HHMMSS.log
│
├── .env                           # Environment variables (API keys, etc.)
├── requirements.txt               # Dependencies
├── debug.py                       # Our new comprehensive testing suite (under development)
└── README.md                      # Project documentation
