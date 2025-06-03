# Modern LangGraph Assistant

## June 2025 - Production Ready

This project implements a modern LangGraph assistant using best practices as of June 2025. The codebase demonstrates proper use of LangGraph features including supervisor patterns, concurrency control, memory management, and human-in-the-loop interactions.

## Key Improvements

### Modern Supervisor Pattern
- Uses `langgraph-supervisor` package for hierarchical agent systems
- Proper handoff mechanisms between specialized agents
- Human-in-the-loop integration with interrupt patterns

### Production-Ready Architecture
- Dedicated checkpointer libraries for production databases
- Semaphore-based concurrency control to prevent rate limiting
- Memory management with automatic context trimming
- Comprehensive error handling and recovery

### Modern LLM Integration
- String-based model references with `init_chat_model`
- Token usage tracking and optimization
- Improved tracing with LangSmith
- Proper human-in-the-loop flows

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Modern LangGraph dependencies (new for June 2025)
pip install langgraph-supervisor langgraph-checkpoint-sqlite 

# Optional: For production deployments
pip install langgraph-checkpoint-postgres
```

## Requirements

```
# requirements.txt

# Core dependencies
langgraph>=0.3.0
langchain>=0.4.0
langchain-anthropic>=0.1.0
langchain-openai>=0.1.0

# Supervisor support
langgraph-supervisor>=0.2.0

# Database persistence
langgraph-checkpoint-sqlite>=0.1.0
langgraph-checkpoint-postgres>=0.1.0  # Optional for production

# Tracing and monitoring
langsmith>=0.1.0

# Web search capabilities
tavily-python>=0.1.0

# Environment management
python-dotenv>=0.21.0

# Utilities
pyyaml>=6.0
```

## Environment Setup

Create a `.env` file with the following variables:

```
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Search API Keys
TAVILY_API_KEY=your_tavily_key

# Optional: LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=mortey-assistant
LANGSMITH_TRACING=true

# Optional: Database for production
DATABASE_URL=postgres://user:password@localhost:5432/mortey
```

## Architecture

The assistant uses a hierarchical architecture with a supervisor managing specialized agents:

1. **Supervisor**: Routes user requests to the appropriate specialized agent
2. **Chat Agent**: Handles general conversation and file browsing
3. **Coder Agent**: Specialized for code generation and programming tasks
4. **Web Agent**: Performs web searches and research

## Key Components

- **State Management**: Modern state with proper reducers for concurrent updates
- **Controller**: Security verification with interrupt-based human approval
- **Checkpointer**: Production-ready persistence with database support
- **LLM Manager**: Efficient model handling with concurrency controls

## Memory Management

The assistant implements automatic memory management to prevent context overflows:

- **Trimming**: Removes oldest messages when exceeding context limits
- **Summarization**: Can replace old messages with summaries (configurable)
- **Token Tracking**: Monitors token usage across conversations

## Human-in-the-Loop

The system integrates human feedback at key points:

- **Security Approvals**: Reviews potentially dangerous operations
- **Content Moderation**: Flags content for human review when needed
- **Clarifications**: Can request additional information from users

## Usage Example

```python
from Core.assistant_core import assistant

async def example():
    # Process a message
    response = await assistant.process_message("Hello, can you help me with Python?")
    print(f"Assistant: {response}")
    
    # Get conversation history
    history = await assistant.get_conversation_history()
    
    # Get system status
    status = assistant.get_system_status()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(example())
```

## Testing

Run the test script to verify functionality:

```bash
python test_assistant.py
```

## Extending

To add new capabilities:

1. Define new tools in the appropriate agent file
2. Add new agent types to the state management system
3. Update the supervisor prompt to route to the new agent
4. Configure appropriate model in `llm_config.yaml`

## Deployment Considerations

For production deployment:

- Use PostgreSQL persistence with `langgraph-checkpoint-postgres`
- Configure appropriate concurrency limits based on API rate limits
- Enable LangSmith tracing for monitoring and debugging
- Set up proper error alerting and monitoring

## License

MIT License