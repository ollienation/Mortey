<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# LangGraph Detailed Function Reference: Comprehensive AI Instruction Set

## Core Graph Construction Functions

### StateGraph Class and Initialization

The `StateGraph` class serves as the primary building block for creating LangGraph applications [^1][^2]. Initialize a StateGraph by passing a state schema, which can be a `TypedDict`, Pydantic `BaseModel`, or dataclass [^1][^2]. The state schema defines the structure and data types that nodes will operate on throughout the graph execution [^1].

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[str]
    counter: int

builder = StateGraph(State)
```


### Node Management Functions

#### add_node()

Add computational units to your graph using `builder.add_node(name, function)` [^2][^3]. Nodes are Python functions where the first argument is always the state, and optionally a second argument for configuration [^1]. Functions can be synchronous or asynchronous [^1].

```python
def my_node(state: State, config: RunnableConfig):
    return {"counter": state["counter"] + 1}

builder.add_node("my_node", my_node)
```


#### add_sequence()

For sequential node execution, use `builder.add_sequence([node1, node2, node3])` to create a linear chain of operations [^2]. This provides a shorthand for adding multiple nodes with sequential edges [^2].

### Edge Management Functions

#### add_edge()

Create direct connections between nodes using `builder.add_edge(source, target)` [^2][^3]. Use the special `START` node to define entry points and `END` node for termination [^1][^2].

```python
from langgraph.graph import START, END
builder.add_edge(START, "first_node")
builder.add_edge("first_node", "second_node")
builder.add_edge("second_node", END)
```


#### add_conditional_edges()

Implement dynamic routing with `builder.add_conditional_edges(source, routing_function, path_map)` [^1][^4]. The routing function receives the current state and returns the name of the next node or a list of nodes for parallel execution [^1][^4].

```python
def routing_function(state: State):
    if state["counter"] > 5:
        return "high_value_node"
    return "low_value_node"

builder.add_conditional_edges("decision_node", routing_function)
```


## State Management and Reducers

### Default and Custom Reducers

State updates are controlled by reducer functions that determine how node outputs modify the existing state [^1][^5]. By default, updates override existing values, but custom reducers enable sophisticated state accumulation [^1][^5].

#### Annotated Types with Reducers

Use `Annotated[type, reducer_function]` to specify how state updates should be processed [^1][^2]. Common patterns include `operator.add` for list concatenation and custom functions for complex merging logic [^1][^2].

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    results: Annotated[list[dict], custom_merge_function]
```


#### add_messages Reducer

The specialized `add_messages` reducer handles message lists with ID tracking, deserialization, and update capabilities [^1][^2]. This function automatically converts various message formats and manages message updates by ID [^1][^2].

### MessagesState and MessagesAnnotation

LangGraph provides prebuilt state schemas for common chat applications [^1][^5]. `MessagesState` includes a `messages` key with the `add_messages` reducer pre-configured [^1]. Extend this class to add additional state fields while maintaining message handling capabilities [^1].

## Graph Compilation and Execution

### compile()

Transform your graph builder into an executable object using `graph = builder.compile()` [^1][^3]. The compilation process validates graph structure, checks for orphaned nodes, and prepares the graph for execution [^1]. Optional parameters include checkpointers for persistence and interrupts for human-in-the-loop workflows [^1].

### Execution Methods

#### invoke()

Execute the graph synchronously with `graph.invoke(input_state, config)` [^3][^6]. The input must contain at least one state key, and the method returns the complete final state [^2][^3].

#### stream()

Stream intermediate results during execution using `graph.stream(input_state, config, stream_mode)` [^6][^7]. Multiple stream modes are available: `"values"` returns complete state after each step, `"updates"` returns only state changes, and `"custom"` enables custom streaming data [^6][^7].

#### astream()

Asynchronous streaming execution with `graph.astream()` provides the same functionality as `stream()` but returns an async iterator [^6][^8].

#### batch()

Process multiple inputs simultaneously using `graph.batch([input1, input2, input3])` for efficient parallel execution [^6].

## Advanced Control Flow Features

### Send API for Dynamic Parallel Execution

The `Send` API enables map-reduce patterns where the number of parallel branches is determined at runtime [^9][^10]. Return `Send` objects from conditional edges to distribute different states to multiple node instances [^9][^10].

```python
from langgraph.types import Send

def dynamic_routing(state: State):
    return [Send("process_item", {"item": item}) for item in state["items"]]

builder.add_conditional_edges("dispatcher", dynamic_routing)
```


### Command API for Combined State Updates and Routing

The `Command` object allows nodes to both update state and control routing in a single return statement [^1][^2]. This eliminates the need for separate conditional edges when state updates and routing decisions are coupled [^1].

```python
from langgraph.types import Command

def my_node(state: State) -> Command[Literal["next_node"]]:
    return Command(
        update={"processed": True},
        goto="next_node"
    )
```


#### Command.PARENT for Subgraph Navigation

Navigate from subgraph nodes to parent graph nodes using `Command(goto="parent_node", graph=Command.PARENT)` [^1]. This enables hierarchical workflow management and multi-agent handoffs [^1].

## Persistence and Memory Management

### Checkpointing Functions

#### get_state()

Retrieve the current state snapshot for a thread using `graph.get_state(config)` [^11]. The config must include a `thread_id` to identify the conversation or workflow instance [^11].

#### get_state_history()

Access the complete execution history with `graph.get_state_history(config)` [^11]. This returns a chronologically ordered list of `StateSnapshot` objects representing each step in the graph execution [^11].

#### update_state()

Manually modify graph state using `graph.update_state(config, values, as_node)` [^11][^12]. This function enables human-in-the-loop modifications and state corrections during execution [^11].

### Checkpointer Implementations

#### InMemorySaver

Basic in-memory persistence using `InMemorySaver()` for development and testing [^11][^13]. This checkpointer stores state in memory and is lost when the application terminates [^11].

#### Database Checkpointers

Production-ready persistence with PostgreSQL, Redis, and MongoDB checkpointers available through the LangGraph ecosystem [^13]. These provide durable state storage and enable resumption across application restarts [^13].

## Human-in-the-Loop Functions

### interrupt()

Pause graph execution and collect human input using `interrupt("message_to_human")` [^14][^15]. This function marks the graph as interrupted and saves the current state for later resumption [^14][^15].

### Resume with Command

Resume interrupted execution using `graph.invoke(Command(resume="user_input"), config)` [^15]. The human input is passed to the interrupted node, which can then process the response and continue execution [^15].

## Parallel Execution and Branching

### Defer Parameter

Control execution timing with `builder.add_node("node_name", function, defer=True)` [^2]. Deferred nodes wait until all other pending tasks in the current superstep complete before executing [^2].

### Fan-out and Fan-in Patterns

Create parallel execution branches by adding multiple edges from a single source node [^2][^16]. Nodes with multiple incoming edges automatically wait for all predecessors to complete before executing [^2].

## Error Handling and Retry Policies

### RetryPolicy Configuration

Add resilience to nodes with `RetryPolicy(max_attempts=5, retry_on=SpecificException)` [^2]. Configure which exceptions trigger retries and set maximum retry attempts [^2][^17].

### Recursion Limit Management

Control maximum execution steps using `graph.invoke(input, {"recursion_limit": 100})` [^17][^18]. The default limit is 25 steps, and exceeding this limit raises `GraphRecursionError` [^17][^18].

## Subgraph Integration

### Adding Compiled Subgraphs

Integrate pre-compiled subgraphs as nodes using `builder.add_node("subgraph_name", compiled_subgraph)` [^19][^20]. This approach works when parent and subgraph share state key structures [^19][^20].

### State Schema Mapping

Handle different state schemas between parent graphs and subgraphs through input/output filtering [^20]. Define transformation functions to convert between different state representations [^20].

## Visualization and Debugging

### draw_mermaid_png()

Generate visual representations of your graph structure using `graph.get_graph().draw_mermaid_png()` [^21][^22]. Customize appearance with parameters for curve styles, node colors, and layout options [^21][^22].

### draw_mermaid()

Output Mermaid syntax strings with `graph.get_graph().draw_mermaid()` for integration with external visualization tools [^21][^22].

## Configuration and Runtime Parameters

### config_schema Parameter

Define runtime configuration schemas when creating StateGraph instances [^1][^2]. This enables dynamic parameter injection without polluting the graph state [^1][^2].

### Configurable Parameter Access

Access runtime configuration within nodes using `config["configurable"]["parameter_name"]` [^1][^2]. This pattern enables model switching, prompt customization, and environment-specific behavior [^1][^2].

## Functional API Components

### @entrypoint and @task Decorators

The Functional API provides `@entrypoint` and `@task` decorators for traditional programming paradigms [^23][^24]. These decorators enable LangGraph features like persistence and streaming without explicit graph construction [^23][^24].

### MemorySaver Integration

Combine checkpointing with the Functional API using `MemorySaver` to enable conversation memory and state persistence [^23][^25].

This comprehensive function reference provides the detailed technical foundation needed to implement sophisticated LangGraph workflows, from basic sequential processing to complex multi-agent systems with dynamic routing and human oversight capabilities.

<div style="text-align: center">⁂</div>

[^1]: https://langchain-ai.github.io/langgraph/concepts/low_level/

[^2]: https://langchain-ai.github.io/langgraph/how-tos/graph-api/

[^3]: https://pypi.org/project/langgraph/0.0.64/

[^4]: https://www.baihezi.com/mirrors/langgraph/reference/graphs/index.html

[^5]: https://langchain-ai.github.io/langgraphjs/concepts/low_level/

[^6]: https://langchain-ai.github.io/langgraph/how-tos/streaming/

[^7]: https://dev.to/jamesli/two-basic-streaming-response-techniques-of-langgraph-ioo

[^8]: https://langgraph-doc.ailand123.cn/reference/graphs/

[^9]: https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd

[^10]: https://langchain-ai.github.io/langgraphjs/how-tos/map-reduce/

[^11]: https://langchain-ai.github.io/langgraph/concepts/persistence/

[^12]: https://stackoverflow.com/questions/79456387/why-updated-state-in-langgraph-doesnt-reach-the-next-node

[^13]: https://blog.langchain.dev/langgraph-v0-2/

[^14]: https://langchain-ai.github.io/langgraphjs/concepts/human_in_the_loop/

[^15]: https://changelog.langchain.com/announcements/interrupt-simplifying-human-in-the-loop-agents

[^16]: https://focused.io/lab/parallel-execution-with-langchain-and-langgraph

[^17]: https://stackoverflow.com/questions/78337975/setting-recursion-limit-in-langgraphs-stategraph-with-pregel-engine

[^18]: https://langchain-ai.github.io/langgraph/troubleshooting/errors/GRAPH_RECURSION_LIMIT/

[^19]: https://dev.to/sreeni5018/langgraph-subgraphs-a-guide-to-modular-ai-agents-development-31ob

[^20]: https://langchain-ai.github.io/langgraph/how-tos/subgraph/

[^21]: https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.graph_mermaid.draw_mermaid.html

[^22]: https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.graph.Graph.html

[^23]: https://langchain-ai.github.io/langgraph/concepts/functional_api/

[^24]: https://blog.langchain.dev/introducing-the-langgraph-functional-api/

[^25]: https://www.youtube.com/watch?v=NXhyWJozM8A

[^26]: https://blog.langchain.dev/langgraph/

[^27]: https://github.com/Hadi2525/langgraph-builder

[^28]: https://ai.google.dev/gemini-api/docs/langgraph-example

[^29]: https://github.com/langchain-ai/langgraph-builder

[^30]: https://www.js-craft.io/blog/langgraph-js-conditional-edges-graphs/

[^31]: https://www.youtube.com/watch?v=qRxsCunfhws

[^32]: https://github.com/langchain-ai/langgraph/issues/2169

[^33]: https://langfuse.com/docs/integrations/langchain/example-python-langgraph

[^34]: https://www.youtube.com/watch?v=UrVno_5wB08

[^35]: https://github.com/langchain-ai/langgraph/discussions/583

[^36]: https://www.reddit.com/r/LangChain/comments/1hy6zq8/help_with_langgraph_state_not_updating_when_tool/

[^37]: https://www.youtube.com/watch?v=HgRJ5LUC4XY

[^38]: https://github.com/bytedance/deer-flow/issues/55

[^39]: https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html

[^40]: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/index.md

[^41]: https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.StateGraph.html

[^42]: https://www.langchain.com/langgraph

[^43]: https://realpython.com/langgraph-python/

[^44]: https://langchain-ai.github.io/langgraph/concepts/streaming/

[^45]: https://js.langchain.com/docs/concepts/streaming/

[^46]: https://github.com/langchain-ai/langgraph/discussions/2212

[^47]: https://github.com/langchain-ai/langgraph/discussions/1726

[^48]: https://www.reddit.com/r/LangChain/comments/1h226yc/discussion_why_does_the_recursion_limit_exist_in/

