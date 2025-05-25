import asyncio
import os
import uuid
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import ollama
from anthropic import Anthropic

# Load environment variables
load_dotenv()

class TestState(TypedDict):
    user_input: str
    agent_choice: str
    output_content: str
    output_type: str

def router_node(state: TestState):
    """Route using local LLM"""
    prompt = f"""
    Route this request to the appropriate agent:
    User: {state['user_input']}
    
    Available agents: CODER, WEB, VISION, IMAGE
    Respond with only the agent name.
    """
    
    response = ollama.generate(model="llama3.2:3b", prompt=prompt)
    agent = response['response'].strip().upper()
    
    return {**state, "agent_choice": agent}

def coder_node(state: TestState):
    """Generate code using Anthropic"""
    if state['agent_choice'] != 'CODER':
        return {**state, "output_content": "Not a coding request", "output_type": "text"}
    
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    prompt = f"""
    Write clean Python code for: {state['user_input']}
    
    Provide working code with comments. Keep it concise.
    """
    
    try:
        message = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            **state,
            "output_content": message.content[0].text,
            "output_type": "code"
        }
    except Exception as e:
        return {
            **state,
            "output_content": f"Error: {str(e)}",
            "output_type": "error"
        }

# Build workflow
workflow = StateGraph(TestState)
workflow.add_node("router", router_node)
workflow.add_node("coder", coder_node)

workflow.add_edge(START, "router")
workflow.add_edge("router", "coder")
workflow.add_edge("coder", END)

graph = workflow.compile()

# Test
if __name__ == "__main__":
    test_input = {
        "user_input": "Write a function to reverse a string",
        "agent_choice": "",
        "output_content": "",
        "output_type": ""
    }
    
    result = graph.invoke(test_input)
    print("User: " + result['user_input'])
    print("Routed to: " + result['agent_choice'])
    print("Output type: " + result['output_type'])
    print("Generated code:")
    print(result['output_content'])
