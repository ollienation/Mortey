# Migration Guide: Legacy to Modern LangGraph
## June 2025 - Production Ready Patterns

This guide outlines the key changes needed to modernize your LangGraph assistant from legacy patterns to production-ready code as of June 2025.

## Critical Changes Summary

### 1. Supervisor Pattern Updates

**OLD (Deprecated):**
```python
from langgraph_supervisor import create_supervisor  # Wrong import
```

**NEW (Modern):**
```python
from langgraph_supervisor import create_supervisor  # Correct - separate package
pip install langgraph-supervisor
```

### 2. Model Initialization Modernization

**OLD (Manual Model Creation):**
```python
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(api_key=provider_config.api_key, ...)
```

**NEW (String-Based References):**
```python
from langchain.chat_models import init_chat_model
model = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest",
    temperature=0.7,
    max_tokens=2000
)
```

### 3. State Management with Proper Reducers

**OLD (No Reducers):**
```python
class AssistantState(MessagesState):
    messages: list[BaseMessage]  # No reducer
    web_results: List[Dict[str, Any]] = []  # No concurrent safety
```

**NEW (With Reducers):**
```python
from typing import Annotated
import operator
from langgraph.graph.message import add_messages

class AssistantState(MessagesState):
    messages: Annotated[List[BaseMessage], add_messages]  # Concurrent-safe
    web_results: Annotated[List[Dict[str, Any]], operator.add]  # Proper reducer
```

### 4. Security Controller Modernization

**OLD (Force Approval Bypass):**
```python
# Dangerous pattern that bypassed security
if force_approval:
    return self._create_approval_response(state)
```

**NEW (Interrupt Pattern):**
```python
from langgraph.types import interrupt

human_response = interrupt(
    f"🔍 Human review required: {reason}",
    approval_payload
)
```

### 5. Checkpointer Simplification

**OLD (Complex Factory):**
```python
from langgraph.checkpoint.postgres import PostgresSaver
# Complex manual initialization
```

**NEW (Dedicated Libraries):**
```python
from langgraph_checkpoint_postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(db_url)
checkpointer.setup()  # One-time setup
```

### 6. Concurrency Control Implementation

**OLD (No Concurrency Management):**
```python
# No rate limiting or concurrent execution control
```

**NEW (Semaphore-Based Control):**
```python
import asyncio
from asyncio import Semaphore

MAX_CONCURRENCY = 3
semaphore = Semaphore(MAX_CONCURRENCY)

async def rate_limited_node(state):
    async with semaphore:
        # Your rate-limited operations here
        return await process_state(state)
```

### 7. Memory Management Integration

**OLD (No Memory Management):**
```python
# Messages accumulate without limit
```

**NEW (Automatic Trimming):**
```python
from langchain_core.messages import trim_messages

def manage_memory(state):
    if len(state["messages"]) > max_messages:
        trimmed = trim_messages(state["messages"], max_tokens=4000)
        return {"messages": trimmed}
    return {}
```

## Migration Steps

### Step 1: Update Dependencies

Replace your `requirements.txt`:

```bash
# Remove old dependencies
pip uninstall langgraph langchain

# Install modern dependencies
pip install -r requirements.txt  # Use the new requirements.txt
```

### Step 2: Update State Management

Replace your `state.py` with the new version that includes:
- Proper reducers for concurrent operations
- Memory management configuration
- Enhanced type annotations

### Step 3: Modernize Security Controller

Replace `controller.py` with the new version featuring:
- `interrupt()` function instead of force-approval
- Semaphore-based concurrency control
- Improved error handling

### Step 4: Update Agent Factory

Replace `agents.py` with modern patterns:
- String-based model initialization
- Async tool definitions with concurrency control
- Proper interrupt patterns for human-in-the-loop

### Step 5: Modernize Core Assistant

Replace `assistant_core.py` with:
- `langgraph-supervisor` import
- Proper session management with semaphores
- Built-in memory management
- Modern error recovery

### Step 6: Update Checkpointer

Replace `checkpointer.py` with:
- Dedicated library imports
- Simplified initialization patterns
- Better error handling

### Step 7: Update LLM Manager

Replace `llm_manager.py` with:
- `init_chat_model` usage
- Semaphore-based concurrency control
- Token usage tracking
- Retry logic with exponential backoff

## Configuration Updates

### Update Environment Variables

Add these new environment variables:

```
# Modern dependencies might need specific versions
ENVIRONMENT=development  # or production

# Enhanced tracing
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your-project-name
```

### Update YAML Configuration

Your `llm_config.yaml` can remain the same, but you may want to add concurrency settings:

```yaml
global:
  default_provider: "anthropic"
  fallback_provider: "openai"
  retry_attempts: 3
  timeout_seconds: 30
  max_concurrent_calls: 5  # NEW: Concurrency control
  enable_memory_management: true  # NEW: Memory management
```

## Breaking Changes

### Import Changes
- `from langgraph.prebuilt import create_supervisor` → `from langgraph_supervisor import create_supervisor`
- Manual model creation → `init_chat_model` with string references

### API Changes
- Force approval pattern removed - must use interrupt patterns
- State updates now require proper reducers
- Tool definitions should be async with semaphore control

### Behavior Changes
- Memory is automatically managed (trimmed) when it exceeds limits
- Concurrency is controlled by semaphores to prevent rate limiting
- Human-in-the-loop flows now use interrupt patterns

## Testing Your Migration

Run the new test script to verify migration:

```bash
python test_assistant.py
```

Expected outputs:
- ✅ Modern checkpointer initialization
- ✅ Agent creation with string-based models
- ✅ Supervisor pattern with proper handoffs
- ✅ Concurrency control verification
- ✅ Memory management validation

## Rollback Plan

If you need to rollback:

1. Keep backup copies of your original files
2. Restore original `requirements.txt`
3. Reinstall old dependencies
4. Restore original Python files

## Performance Benefits

After migration, you should see:

- **Reduced rate limiting errors** due to semaphore controls
- **Better memory efficiency** with automatic trimming
- **Improved error recovery** with retry logic
- **Enhanced security** with proper human-in-the-loop patterns
- **Better observability** with modern tracing

## Support and Troubleshooting

Common issues and solutions:

### Import Errors
```bash
ModuleNotFoundError: No module named 'langgraph_supervisor'
```
**Solution:** `pip install langgraph-supervisor`

### Model Initialization Errors
```python
ValueError: Model string not recognized
```
**Solution:** Ensure API keys are set and use correct format: `"provider:model-name"`

### Concurrency Issues
```python
Too many concurrent requests
```
**Solution:** Adjust semaphore limits in the configuration

### Memory Issues
```python
Context length exceeded
```
**Solution:** Ensure memory management is enabled and configured properly

For additional support, check the LangGraph documentation or create an issue in the project repository.