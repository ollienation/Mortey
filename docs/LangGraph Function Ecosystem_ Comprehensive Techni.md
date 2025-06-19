<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# LangGraph Function Ecosystem: Comprehensive Technical Reference Guide

## Executive Summary

LangGraph represents a paradigm shift in building stateful, multi-actor LLM applications through its graph-based orchestration framework[^1][^2]. The ecosystem encompasses over 100+ core functions organized into ten distinct categories, from basic graph construction to advanced deployment operations[^1]. Unlike traditional directed acyclic graph (DAG) frameworks, LangGraph enables cyclical workflows essential for agent runtimes, providing low-level infrastructure for persistent execution, human-in-the-loop interactions, and comprehensive memory management[^2][^3]. The framework's architectural patterns center around three fundamental abstractions: StateGraph for workflow definition, checkpointers for persistence, and the Pregel-inspired computation model for distributed execution[^1][^4]. Critical implementation considerations include proper state schema design using TypedDict or Pydantic models, strategic checkpointer selection for production environments, and understanding the distinction between the Graph API and the newer Functional API paradigms[^5][^6]. This comprehensive function reference enables enterprise development teams to implement production-ready agent systems with confidence, leveraging LangGraph's mature ecosystem for building resilient, stateful workflows[^2].

## Function Reference by Category

### 1. Graph Construction Functions

#### StateGraph Class and Initialization

The `StateGraph` class serves as the primary abstraction for building workflows in LangGraph[^4][^6].

**Function Signature:**

```python
class StateGraph(Generic[StateType]):
    def __init__(self, state_schema: Type[StateType])
```

**Core Parameters:**

- `state_schema`: TypedDict, Pydantic BaseModel, or dataclass defining state structure[^6]

**Usage Examples:**

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    messages: list[str]
    count: int

workflow = StateGraph(State)
```


#### Node Addition Functions

**add_node Method:**

```python
def add_node(self, key: str, action: Callable[[StateType], StateType]) -> None
```

The `add_node` function registers callable functions as graph nodes[^4][^7]. Each node receives the current state as input and returns state updates[^8].

**Usage Pattern:**

```python
def increment_node(state: State):
    return {"count": state["count"] + 1}

workflow.add_node("increment", increment_node)
```

**add_sequence Method:**

```python
def add_sequence(self, nodes: Sequence[str]) -> None
```

Creates sequential execution chains between multiple nodes[^8].

#### Edge Creation Methods

**add_edge Function:**

```python
def add_edge(self, start_key: str, end_key: str) -> None
```

Establishes fixed transitions between nodes[^4][^9]. The constraint exists that when using list inputs, nodes must be added before edges[^9].

**add_conditional_edges Function:**

```python
def add_conditional_edges(
    self, 
    start_key: str, 
    condition: Callable[[StateType], str],
    condition_map: Dict[str, str]
) -> None
```

Enables dynamic routing based on state conditions[^10][^11]. The condition function determines the next node based on current state[^12].

**Implementation Example:**

```python
def should_continue(state: State) -> str:
    if state["count"] < 5:
        return "continue"
    return "end"

workflow.add_conditional_edges(
    "increment",
    should_continue,
    {"continue": "increment", "end": END}
)
```


#### Graph Compilation and Validation

**compile Method:**

```python
def compile(
    self,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False
) -> CompiledGraph
```

The compilation process transforms the graph definition into an executable runtime[^13][^14]. Checkpointers enable persistence, while interrupt parameters support human-in-the-loop workflows[^15].

### 2. State Management Functions

#### State Schema Definition Patterns

LangGraph supports multiple state definition approaches: TypedDict, Pydantic BaseModel, and dataclasses[^6][^8].

**TypedDict Pattern:**

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int
```


#### Reducer Functions

**add_messages Reducer:**
The `add_messages` function serves as a specialized reducer for message lists, enabling automatic message appending rather than replacement[^6][^16].

**Function Signature:**

```python
def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]
```

**Custom Reducer Implementation:**

```python
from typing_extensions import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    numbers: Annotated[list[int], operator.add]
```

Custom reducers control how state updates are merged, with each state key supporting independent reducer logic[^6].

#### State Update Mechanisms

Node functions return partial state updates that are merged using configured reducers[^6]. The merge strategy depends on the reducer function associated with each state key.

**Update Pattern:**

```python
def node_function(state: State) -> dict:
    return {"messages": [new_message], "count": state["count"] + 1}
```


### 3. Control Flow Functions

#### Conditional Routing Logic

Conditional edges enable dynamic workflow routing based on state evaluation[^10][^12]. Router functions examine current state and return string identifiers for next nodes.

**Advanced Routing Example:**

```python
def complex_router(state: AgentState) -> Literal["process", "retry", "error", "end"]:
    if state.status == "SUCCESS":
        return "end"
    elif state.status == "ERROR":
        return "retry" if state.error_count < 3 else "error"
    return "process"
```


#### Parallel Execution Methods

**Send API for Dynamic Distribution:**
The Send API enables parallel execution by distributing different states to multiple node instances[^17][^18].

**Function Signature:**

```python
from langgraph.types import Send

def parallel_node(state: State) -> list[Send]:
    tasks = identify_tasks(state)
    return [Send("worker_node", task) for task in tasks]
```

This pattern supports map-reduce workflows and unknown object count scenarios[^17].

#### Command Objects

**Command Class:**

```python
from langgraph.types import Command

def node_with_command(state: State) -> Command:
    return Command(
        update={"new_data": "value"},
        goto="target_node"
    )
```

Commands combine state updates with control flow decisions in a single operation[^12][^19].

### 4. Persistence and Memory Functions

#### Checkpointer Implementations

**MemorySaver (InMemorySaver):**

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

The InMemorySaver provides in-memory persistence suitable for development and experimentation[^13][^20].

**Thread Management:**

```python
config = {"configurable": {"thread_id": "conversation_1"}}
result = graph.invoke(input_data, config=config)
```

Thread IDs organize conversations and enable state isolation between different user sessions[^13][^20].

#### Long-term Memory Storage

**InMemoryStore Implementation:**

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("user_1", "memories")
store.put(namespace, "preference_key", {"food": "pizza"})
memories = store.search(namespace)
```

The store provides cross-thread memory capabilities for user preferences and learned information[^21][^22].

#### State Serialization Methods

**get_state Function:**

```python
def get_state(config: RunnableConfig) -> StateSnapshot
```

Retrieves current state snapshots with metadata including next nodes and execution history[^13].

**get_state_history Function:**

```python
def get_state_history(config: RunnableConfig) -> Iterator[StateSnapshot]
```

Returns chronological state history for debugging and replay scenarios[^13].

### 5. Human-in-the-Loop Functions

#### interrupt() Primitive

**Function Signature:**

```python
from langgraph.types import interrupt

def human_approval_node(state: State):
    user_input = interrupt({"data_to_review": state["content"]})
    return {"approved_content": user_input}
```

The interrupt function pauses execution and surfaces data to humans for review[^15].

#### Resume Functionality

**Command(resume=...) Pattern:**

```python
from langgraph.types import Command

# After interrupt occurs:
graph.invoke(Command(resume="approved"), config=config)
```

Resume operations inject human feedback and continue execution from the interruption point[^15][^19].

#### State Inspection APIs

Interrupted graphs return Interrupt objects containing payload data and resumption metadata[^15]. The `resumable` flag indicates whether execution can continue.

### 6. Agent-Specific Functions

#### Prebuilt Agent Constructors

**create_react_agent Function:**

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    return f"Sunny in {city}"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)
```

The `create_react_agent` function provides a complete ReAct agent implementation with tool calling capabilities[^23][^24].

**Function Parameters:**

- `model`: Language model instance or string identifier[^24]
- `tools`: List of tools or ToolNode instance[^24]
- `prompt`: System prompt for agent behavior[^23]
- `checkpointer`: Optional persistence layer[^24]
- `interrupt_before/after`: Human-in-the-loop configuration[^24]


#### Tool Integration Methods

**ToolNode Class:**

```python
from langgraph.prebuilt import ToolNode

tools = [calculator_tool, search_tool]
tool_node = ToolNode(tools)
```

ToolNode manages tool execution by matching LLM tool calls with registered functions[^25][^26].

**Tool Binding Pattern:**

```python
model_with_tools = model.bind_tools(tools)
```

Tool binding makes tool schemas available to the language model for structured calling[^25].

### 7. Error Handling and Debugging Functions

#### GraphRecursionError Management

**Error Class:**

```python
class GraphRecursionError(RecursionError):
    """Raised when graph exhausts maximum steps"""
```

GraphRecursionError prevents infinite loops by enforcing configurable recursion limits[^27][^28].

**Recursion Limit Configuration:**

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(recursion_limit=50)
result = graph.invoke(input_data, config)
```

The recursion_limit parameter controls maximum execution steps before termination[^28][^29].

#### Error Isolation Mechanisms

**Try-Catch Patterns:**

```python
def safe_execution(app, inputs, config):
    try:
        return app.invoke(inputs, config)
    except GraphRecursionError as e:
        return handle_recursion_error(e)
```

Error isolation prevents cascading failures in multi-agent systems[^30].

#### Debugging Utilities

**Visualization Methods:**

```python
graph.get_graph().draw_mermaid_png()
```

Built-in visualization generates Mermaid diagrams for graph structure analysis[^31][^32].

### 8. Streaming and Execution Functions

#### Stream Methods

**stream() Function:**

```python
def stream(
    self,
    input: Any,
    config: Optional[RunnableConfig] = None,
    stream_mode: str = "values"
) -> Iterator[Any]
```

Stream modes include "values", "updates", and "messages" for different output granularities[^33].

**astream() Function:**

```python
async def astream(
    self,
    input: Any,
    config: Optional[RunnableConfig] = None,
    stream_mode: str = "values"
) -> AsyncIterator[Any]
```

Asynchronous streaming enables non-blocking execution patterns[^33].

#### Invoke Patterns

**invoke() and ainvoke():**

```python
# Synchronous execution
result = graph.invoke({"messages": [user_message]}, config)

# Asynchronous execution  
result = await graph.ainvoke({"messages": [user_message]}, config)
```

Invoke methods provide complete execution results after workflow completion[^34].

### 9. Visualization and Tooling Functions

#### LangGraph Studio Integration

LangGraph Studio provides visual debugging capabilities with breakpoint setting and node-level execution control[^32]. The studio integrates with the `langgraph dev` command for local development.

**Development Server:**

```bash
langgraph dev
```

This command launches the LangGraph Platform locally with Studio interface access[^35].

#### LangSmith Tracing Integration

**Tracing Configuration:**

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your_key"
```

LangSmith integration provides observability for graph execution with automatic trace collection[^36].

### 10. Deployment and Production Functions

#### LangGraph Platform Deployment

**Local Development:**

```bash
langgraph dev
```

**Container Deployment:**
LangGraph supports multiple deployment options including Cloud SaaS, Self-Hosted Data Plane, and Standalone Container modes[^35].

#### Server API Configuration

The LangGraph Server provides REST API endpoints for remote graph execution with SDK support for Python and JavaScript[^1].

**SDK Usage:**

```python
from langgraph_sdk import LangGraphClient

client = LangGraphClient(url="http://localhost:8123")
```


## Implementation Patterns Guide

### Common Workflow Sequences

#### Basic Agent Workflow Pattern

The fundamental agent pattern combines state management, conditional routing, and tool execution:

```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")
```

This pattern implements the classic ReAct (Reasoning and Acting) loop where agents reason about actions and execute tools iteratively[^37][^23].

#### Human-in-the-Loop Approval Workflow

Human intervention workflows require careful interrupt placement and resume handling:

```python
def approval_node(state: State):
    decision = interrupt({"content": state["draft"]})
    return {"approved": decision == "approve"}

workflow.add_node("generate", generate_content)
workflow.add_node("approve", approval_node)
workflow.add_conditional_edges("approve", route_approval, {
    "approved": "finalize",
    "rejected": "generate"
})
```

This pattern enables quality control and human oversight in automated workflows[^15].

#### Multi-Agent Collaboration Pattern

Collaborative agents share state through message passing:

```python
workflow = StateGraph(CollaborativeState)
workflow.add_node("researcher", research_agent)
workflow.add_node("writer", writing_agent)
workflow.add_node("reviewer", review_agent)

# Parallel execution from coordinator
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
```

This enables specialized agent coordination for complex task decomposition[^38][^39].

### Best Practices for Function Combination

#### State Schema Design

Effective state schemas balance completeness with performance:

```python
class OptimalState(TypedDict):
    # Core workflow data
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Execution metadata
    iteration_count: int
    current_stage: str
    
    # Performance optimization
    cached_results: dict[str, Any]
```

Use type annotations and reducers to prevent state conflicts and ensure predictable updates[^6].

#### Checkpointer Selection Strategy

Development environments benefit from InMemorySaver for rapid iteration, while production systems require persistent storage:

```python
# Development
checkpointer = InMemorySaver()

# Production
checkpointer = PostgresSaver.from_conn_string(conn_string)
```

Thread management becomes critical for multi-user applications requiring state isolation[^13].

#### Error Recovery Patterns

Implement graceful degradation through error handling nodes:

```python
def error_handler(state: State):
    if state.get("error_count", 0) > 3:
        return {"status": "failed", "needs_human_intervention": True}
    return {"status": "retry", "error_count": state.get("error_count", 0) + 1}
```

This pattern prevents system failures from cascading through multi-agent workflows[^40].

### Anti-Patterns to Avoid

#### Excessive State Complexity

Avoid overloading state with unnecessary data that degrades performance:

```python
# Anti-pattern: Storing large objects in state
class BadState(TypedDict):
    full_document_content: str  # Potentially massive
    all_search_results: list[dict]  # Unbounded growth

# Better: Reference external storage
class GoodState(TypedDict):
    document_id: str
    relevant_excerpts: list[str]
```


#### Improper Reducer Usage

Mismatched reducers cause unexpected state behavior:

```python
# Anti-pattern: Wrong reducer for data type
class BadState(TypedDict):
    message_count: Annotated[int, add_messages]  # add_messages expects lists

# Correct: Appropriate reducer
class GoodState(TypedDict):
    message_count: Annotated[int, operator.add]
```


#### Synchronous Tool Execution in Parallel Contexts

Avoid blocking operations in parallel execution paths:

```python
# Anti-pattern: Blocking calls in parallel nodes
def slow_node(state: State):
    time.sleep(10)  # Blocks entire execution
    return {"result": "done"}

# Better: Async operations or Send API
async def fast_node(state: State):
    result = await async_operation()
    return {"result": result}
```


## Integration Matrix

### LangChain Compatibility Mapping

LangGraph maintains seamless integration with the LangChain ecosystem while providing enhanced workflow capabilities[^2]. Key integration points include:

#### Language Model Integration

LangGraph accepts any LangChain-compatible language model:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Direct model usage
agent = create_react_agent(model=ChatOpenAI(model="gpt-4"), tools=tools)

# String-based model specification
agent = create_react_agent(model="anthropic:claude-3-sonnet", tools=tools)
```

The model parameter supports both LangChain model instances and string identifiers for simplified configuration[^23][^41].

#### Tool Ecosystem Compatibility

LangGraph ToolNode accepts standard LangChain tools without modification:

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def custom_calculator(expression: str) -> str:
    """Evaluate mathematical expressions"""
    return str(eval(expression))

# All tools work seamlessly
tools = [TavilySearchResults(), custom_calculator]
tool_node = ToolNode(tools)
```

This compatibility enables leveraging the extensive LangChain tool library[^25][^26].

### Third-Party Service Integration Points

#### Vector Store Integration

LangGraph workflows integrate with LangChain vector stores for retrieval operations:

```python
from langchain_core.vectorstores import InMemoryVectorStore

def retrieval_node(state: State):
    vectorstore = InMemoryVectorStore(embeddings)
    docs = vectorstore.similarity_search(state["query"])
    return {"retrieved_docs": docs}
```


#### External API Integration

HTTP services integrate through standard tool patterns:

```python
@tool
def api_call(endpoint: str, payload: dict) -> dict:
    """Make HTTP requests to external services"""
    response = requests.post(endpoint, json=payload)
    return response.json()
```

This approach maintains type safety and error handling consistency[^26].

#### Database Persistence

Production deployments integrate with PostgreSQL and other databases through custom checkpointers:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)
```

Database integration enables audit trails and compliance requirements[^13].

### Message Format Standardization

LangGraph uses LangChain's message abstraction for consistent communication:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Standard message handling across all components
state = {
    "messages": [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="User query"),
        AIMessage(content="Assistant response")
    ]
}
```

This standardization ensures compatibility across the entire ecosystem[^6].

## Troubleshooting Reference

### Function-Specific Error Handling Strategies

#### GraphRecursionError Resolution

When agents exceed recursion limits, implement these strategies[^27][^28]:

**Immediate Fix:**

```python
config = {"recursion_limit": 100}
result = graph.invoke(input_data, config)
```

**Long-term Solution:**

```python
def termination_checker(state: State) -> str:
    if state.get("iterations", 0) > 10:
        return "force_end"
    return "continue"
```

Add explicit termination conditions to prevent infinite loops[^29].

#### State Update Conflicts

When state updates fail, verify reducer compatibility:

```python
# Debug state updates
def debug_node(state: State):
    print(f"Current state: {state}")
    update = {"field": "value"}
    print(f"Applying update: {update}")
    return update
```

Use logging to trace state modifications and identify conflicting updates[^6].

#### Tool Execution Failures

Handle tool failures gracefully through error isolation:

```python
def safe_tool_node(state: State):
    try:
        return tool_function(state)
    except Exception as e:
        return {
            "error": str(e),
            "status": "tool_failed",
            "retry_count": state.get("retry_count", 0) + 1
        }
```

This pattern prevents tool failures from terminating entire workflows[^40].

### Debug Utilities and Applications

#### State Inspection Methods

Use built-in inspection utilities for workflow debugging:

```python
# Get current state
current_state = graph.get_state(config)
print(f"State values: {current_state.values}")
print(f"Next nodes: {current_state.next}")

# Get execution history
history = list(graph.get_state_history(config))
for checkpoint in history:
    print(f"Step {checkpoint.metadata['step']}: {checkpoint.values}")
```

State inspection reveals execution flow and identifies bottlenecks[^13].

#### Visualization-Based Debugging

Generate visual representations for complex workflows:

```python
from IPython.display import Image, display

# Render graph structure
display(Image(graph.get_graph().draw_mermaid_png()))

# Export for external tools
graph.get_graph().draw_mermaid()
```

Visual debugging helps identify structural issues and optimization opportunities[^31].

### Performance Optimization Function Usage

#### Parallel Execution Optimization

Leverage parallel execution for independent operations:

```python
# Sequential (slow)
workflow.add_edge(START, "search")
workflow.add_edge("search", "analyze")

# Parallel (fast)  
workflow.add_edge(START, "search")
workflow.add_edge(START, "analyze")
workflow.add_edge("search", "combine")
workflow.add_edge("analyze", "combine")
```

Parallel patterns reduce total execution time significantly[^18].

#### Memory Usage Optimization

Control memory consumption through strategic state management:

```python
class EfficientState(TypedDict):
    # Keep only essential data in state
    current_step: str
    essential_data: dict
    
    # Store large objects externally
    large_data_refs: list[str]
```

External storage references prevent state bloat while maintaining functionality[^21].

#### Streaming Performance Tuning

Optimize streaming behavior for responsive applications:

```python
# Configure streaming granularity
for chunk in graph.stream(input_data, stream_mode="updates"):
    # Process incremental updates
    process_partial_result(chunk)
```

Stream mode selection balances responsiveness with processing overhead[^33].

<div style="text-align: center">⁂</div>

[^1]: https://langchain-ai.github.io/langgraph/reference/

[^2]: https://langchain-ai.github.io/langgraph/

[^3]: https://blog.langchain.dev/langgraph/

[^4]: https://langchain-ai.github.io/langgraph/concepts/low_level/

[^5]: https://langchain-ai.github.io/langgraph/concepts/functional_api/

[^6]: https://langchain-ai.github.io/langgraph/how-tos/state-reducers/

[^7]: https://docs.smith.langchain.com/evaluation/how_to_guides/langgraph

[^8]: https://langchain-ai.github.io/langgraph/how-tos/graph-api/

[^9]: https://github.com/langchain-ai/langgraph/issues/1462

[^10]: https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn

[^11]: https://www.youtube.com/watch?v=EKxoCVbXZwY

[^12]: https://www.linkedin.com/pulse/exploring-control-flow-langgraph-conditional-edges-vs-pranjal-dwivedi-emrjf

[^13]: https://langchain-ai.github.io/langgraph/concepts/persistence/

[^14]: https://www.getzep.com/ai-agents/langgraph-tutorial

[^15]: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/

[^16]: https://github.com/langchain-ai/langgraph/issues/1568

[^17]: https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd

[^18]: https://focused.io/lab/parallel-execution-with-langchain-and-langgraph

[^19]: https://github.com/langchain-ai/langgraph/discussions/3061

[^20]: https://langchain-ai.github.io/langgraph/agents/memory/

[^21]: https://langchain-ai.github.io/langgraph/concepts/memory/

[^22]: https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/

[^23]: https://langchain-ai.github.io/langgraph/agents/agents/

[^24]: https://langchain-ai.github.io/langgraph/reference/agents/

[^25]: https://www.reddit.com/r/LangChain/comments/1e9ap85/langgraph_what_is_the_advantage_of_use_toolnodes/

[^26]: https://www.youtube.com/watch?v=MoHtLAhoMp4

[^27]: https://langchain-ai.github.io/langgraph/reference/errors/

[^28]: https://stackoverflow.com/questions/78337975/setting-recursion-limit-in-langgraphs-stategraph-with-pregel-engine

[^29]: https://www.reddit.com/r/LangChain/comments/1jw3umq/infinite_loop_graphrecursionerror_with/

[^30]: https://github.com/langchain-ai/langchain/discussions/20379

[^31]: https://github.com/pakagronglb/langgraph-visualisation-tool

[^32]: https://www.youtube.com/watch?v=5vEC0Y4sV8g

[^33]: https://www.reddit.com/r/LangChain/comments/1fusj4e/streaming_with_langgraph/

[^34]: https://python.langchain.com/docs/how_to/migrate_agent/

[^35]: https://langchain-ai.github.io/langgraph/tutorials/deployment/

[^36]: https://docs.smith.langchain.com

[^37]: https://ai.google.dev/gemini-api/docs/langgraph-example

[^38]: https://dev.to/sreeni5018/building-multi-agent-systems-with-langgraph-swarm-a-new-approach-to-agent-collaboration-15kj

[^39]: https://blog.langchain.dev/langgraph-multi-agent-workflows/

[^40]: https://aiproduct.engineer/tutorials/langgraph-tutorial-error-handling-patterns-unit-23-exercise-6

[^41]: https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html

[^42]: https://blog.langchain.dev/introducing-the-langgraph-functional-api/

[^43]: https://github.com/langchain-ai/langgraph

[^44]: https://www.youtube.com/watch?v=kDmlKFGeN9k

[^45]: https://python.langchain.com/docs/how_to/graph_constructing/

[^46]: https://www.youtube.com/watch?v=iwPeT_I_GEc

[^47]: https://www.linkedin.com/pulse/fast-explain-langgraph-lưu-võ-7u2qc

[^48]: https://pypi.org/project/langgraph-reducer/

[^49]: https://www.youtube.com/watch?v=p_xDjyaJxD8

[^50]: https://github.com/langchain-ai/langgraph/discussions/4390

[^51]: https://github.com/langchain-ai/langgraph/discussions/4417

[^52]: https://www.youtube.com/watch?v=NXhyWJozM8A

[^53]: https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/

[^54]: https://datasciencedojo.com/blog/langgraph-tutorial/

[^55]: https://www.youtube.com/watch?v=AN06TYJCFgk

[^56]: https://github.com/langchain-ai/langgraph/discussions/3359

[^57]: https://python.langchain.com/docs/versions/migrating_memory/long_term_memory_agent/

[^58]: https://github.com/langchain-ai/langgraph/discussions/4352

[^59]: https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html

[^60]: https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.StateGraph.html

[^61]: https://generativeai.pub/what-are-state-sategraph-and-workflow-in-langgraph-afc3f4392c6f

[^62]: https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/

[^63]: https://github.com/xbeat/Machine-Learning/blob/main/Basics of LangChain's LangGraph.md

[^64]: https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4

[^65]: https://pypi.org/project/langgraph-checkpoint/

[^66]: https://github.com/langchain-ai/langgraph/discussions/1340

[^67]: https://blog.langchain.dev/langgraph-0-3-release-prebuilt-agents/

[^68]: https://www.youtube.com/watch?v=hMHyPtwruVs

[^69]: https://github.com/langchain-ai/langgraph/discussions/674

[^70]: https://blog.langchain.dev/langgraph-platform-announce/

[^71]: https://langchain-ai.github.io/langgraph/concepts/

[^72]: https://github.com/langchain-ai/langgraph/issues/4140

[^73]: https://langchain-ai.github.io/langgraph/how-tos/persistence/

[^74]: https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryStore.html

