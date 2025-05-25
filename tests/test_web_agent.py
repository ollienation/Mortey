import asyncio
import os
import uuid
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import ollama
import sys
sys.path.append('src')

# Load environment variables
load_dotenv()

class TestState(TypedDict):
    user_input: str
    agent_choice: str
    web_results: list
    output_content: str
    output_type: str
    session_id: str
    thinking_state: dict

def router_node(state: TestState):
    """Route using local LLM"""
    prompt = f"""
    Route this request to the appropriate agent:
    User: {state['user_input']}
    
    Available agents:
    - CODER: Programming, code generation, debugging
    - WEB: Web search, browsing, finding information online, news, weather, restaurants
    - VISION: Screen capture, image analysis
    - IMAGE: Image generation, photo editing
    
    For requests about searching, finding information, news, weather, or looking up anything online, use WEB.
    
    Respond with only the agent name: CODER, WEB, VISION, or IMAGE
    """
    
    response = ollama.generate(model="llama3.2:3b", prompt=prompt)
    agent = response['response'].strip().upper()
    
    return {**state, "agent_choice": agent}


def web_node(state: TestState):
    """Web browsing using our Web Agent"""
    if state['agent_choice'] != 'WEB':
        return {**state, "output_content": "Not a web request", "output_type": "text"}
    
    from agents.web_agent import WebAgent
    
    web_agent = WebAgent(None)  # llm_service not needed for this test
    
    # Run web search/browsing
    result = asyncio.run(web_agent.search_and_browse(state))
    
    return result

# Build workflow
workflow = StateGraph(TestState)
workflow.add_node("router", router_node)
workflow.add_node("web", web_node)

workflow.add_edge(START, "router")
workflow.add_edge("router", "web")
workflow.add_edge("web", END)

graph = workflow.compile()

# Test cases
test_cases = [
    "Search for the latest news about AI",
    "Find information about Python programming tutorials",
    "Look up the weather in New York",
    "Search for the best restaurants in Paris"
]

if __name__ == "__main__":
    for test_input in test_cases:
        print(f"\n=== Testing: {test_input} ===")
        
        test_state = {
            "user_input": test_input,
            "agent_choice": "",
            "web_results": [],
            "output_content": "",
            "output_type": "",
            "session_id": str(uuid.uuid4()),
            "thinking_state": {}
        }
        
        try:
            result = graph.invoke(test_state)
            print(f"Routed to: {result['agent_choice']}")
            print(f"Results found: {len(result.get('web_results', []))}")
            print(f"Summary: {result['output_content'][:200]}...")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 60)
