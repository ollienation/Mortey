import asyncio
import ollama
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class BasicState(TypedDict):
    user_input: str
    agent_choice: str
    response: str

def router_node(state: BasicState):
    """Test router using your local LLM"""
    prompt = f"""
    Route this request to the appropriate agent:
    User: {state['user_input']}
    
    Available agents: CODER, WEB, VISION, IMAGE
    Respond with only the agent name.
    """
    
    response = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )
    
    agent = response['response'].strip().upper()
    
    return {
        **state,
        "agent_choice": agent,
        "response": f"Routing to {agent} agent"
    }

# Build simple test graph
workflow = StateGraph(BasicState)
workflow.add_node("router", router_node)
workflow.add_edge(START, "router")
workflow.add_edge("router", END)

graph = workflow.compile()

# Test it
if __name__ == "__main__":
    test_input = {
        "user_input": "Can you write a Python function to sort a list?",
        "agent_choice": "",
        "response": ""
    }
    
    result = graph.invoke(test_input)
    print(f"User: {result['user_input']}")
    print(f"Router decision: {result['agent_choice']}")
    print(f"Response: {result['response']}")
