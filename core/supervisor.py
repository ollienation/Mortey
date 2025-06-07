import logging
from typing import Dict, Any, Literal, Optional, List, Union
from dataclasses import dataclass

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from core.state import AssistantState
from core.error_handling import ErrorHandler

logger = logging.getLogger("supervisor")

@dataclass
class SupervisorConfig:
    
    """
    Configuration class for supervisor behavior.

    This allows for easy customization without modifying core logic.
    """
    max_replays: int = 5  # Prevent infinite agent-tool loops
    default_agent: str = "chat"  # Default fallback agent
    enable_routing_logs: bool = True  # Log routing decisions
    routing_keywords: Optional[Dict[str, List[str]]] = None  # Custom keyword mapping
    
    def __post_init__(self):
        """Initialize default routing keywords if not provided"""
        if self.routing_keywords is None:
            self.routing_keywords = {
                "coder": [
                    "code", "python", "function", "class", "def", "import", 
                    "script", "programming", "debug", "refactor", "algorithm",
                    "variable", "loop", "condition", "syntax", "error",
                    "compile", "execute", "method", "object", "inheritance"
                ],
                "web": [
                    "search", "web", "news", "internet", "google", "find", 
                    "lookup", "research", "browse", "website", "url",
                    "online", "information", "current", "latest", "recent"
                ]
            }

class SupervisorError(Exception):
    """Custom exception for supervisor-specific errors"""
    pass

class Supervisor:
    """
    A supervisor with intelligent routing
    """
    
    def __init__(self, config: Optional[SupervisorConfig] = None):
        self.config = config or SupervisorConfig()
        self.supervisor_graph = None
        self.agents: Dict[str, Any] = {}
        self.tools: List[Any] = []
        self._routing_stats = {
            "total_routes": 0,
            "routes_by_agent": {},
            "routing_errors": 0
        }
        
    async def initialize(
        self, 
        agents: Dict[str, Any], 
        all_tools: List[Any], 
        checkpointer=None
    ) -> None:
        """
        Initialize supervisor with agents and tools.
        
        Args:
            agents: Dictionary of agent instances from AgentFactory
            all_tools: List of all tools from AgentFactory.get_all_tools()
            checkpointer: Optional checkpointer for state persistence
        """
        try:
            self.agents = agents
            self.tools = all_tools
            
            # Validate inputs
            self._validate_initialization_inputs(agents, all_tools)
            
            # Create the workflow graph
            workflow = StateGraph(AssistantState)

            # Use the clean tool approach from AgentFactory
            tool_node = ToolNode(all_tools)
            workflow.add_node("call_tool", tool_node)
            
            # Add agent nodes with proper error handling
            for agent_name, agent in agents.items():
                workflow.add_node(agent_name, self._create_agent_node(agent, agent_name))

            # Set entry point to START node
            workflow.set_entry_point(START)

            # Configurable routing with fallback
            workflow.add_conditional_edges(
                START,
                self._route_to_agent,
                {agent_name: agent_name for agent_name in agents.keys()}
            )

            # Tool continuation logic
            def should_continue(state: AssistantState) -> Literal["call_tool", "__end__"]:
                """Determine if the agent should call tools or end"""
                messages = state.get("messages", [])
                last_message = messages[-1] if messages else None
                
                if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
                    if last_message.tool_calls:
                        return "call_tool"
                return "__end__"

            # Wire up agent nodes to continuation logic
            for agent_name in agents.keys():
                workflow.add_conditional_edges(agent_name, should_continue)
            
            # Wire up tool node to route back to calling agent
            workflow.add_conditional_edges(
                "call_tool",
                lambda state: state.get("current_agent", self.config.default_agent),
                {agent_name: agent_name for agent_name in agents.keys()}
            )

            # Compile with provided checkpointer
            self.supervisor_graph = workflow.compile(checkpointer=checkpointer)
            
            # Initialize routing statistics
            for agent_name in agents.keys():
                self._routing_stats["routes_by_agent"][agent_name] = 0
            
            if self.config.enable_routing_logs:
                logger.info(
                    f"✅ Modular supervisor initialized with {len(agents)} agents "
                    f"and {len(all_tools)} tools"
                )
            
        except Exception as e:
            error_msg = f"Supervisor initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            await ErrorHandler.handle_error(SupervisorError(error_msg), "supervisor_initialization")
            raise
    
    def _validate_initialization_inputs(self, agents: Dict[str, Any], tools: List[Any]) -> None:
        """Validate initialization inputs for better error messages"""
        if not agents:
            raise SupervisorError("No agents provided for initialization")
        
        if not tools:
            logger.warning("⚠️ No tools provided - agents will have limited capabilities")
        
        # Validate default agent exists
        if self.config.default_agent not in agents:
            available_agents = list(agents.keys())
            if available_agents:
                self.config.default_agent = available_agents[0]
                logger.warning(
                    f"⚠️ Default agent '{self.config.default_agent}' not found. "
                    f"Using '{self.config.default_agent}' instead."
                )
            else:
                raise SupervisorError("No valid agents available")
    
    def _create_agent_node(self, agent: Any, agent_name: str):
        """
        Creates a graph-compatible node wrapper for agents.
        """
        async def agent_node(state: AssistantState) -> Dict[str, Any]:
            try:
                # Set current agent for tool routing
                state["current_agent"] = agent_name
                
                # Track routing statistics
                self._routing_stats["routes_by_agent"][agent_name] += 1
                self._routing_stats["total_routes"] += 1
                
                # Execute the agent
                result = await agent.ainvoke(state)
                
                # Ensure result is properly formatted
                if not isinstance(result, dict):
                    logger.warning(f"Agent {agent_name} returned non-dict result, wrapping")
                    result = {"messages": [AIMessage(content=str(result))]}
                
                return result
                
            except Exception as e:
                self._routing_stats["routing_errors"] += 1
                error_msg = f"Agent {agent_name} execution failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                
                # Return error state instead of crashing
                return {
                    "messages": [AIMessage(content="I encountered an issue. Please try rephrasing your request.")],
                    "current_agent": agent_name
                }
        
        return agent_node

    def _route_to_agent(self, state: AssistantState) -> str:
        """
        Enhanced intelligent routing with configurable keywords.
        """
        try:
            messages = state.get("messages", [])
            
            # Find the most recent human message
            last_human_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    last_human_message = message
                    break
            
            if not last_human_message:
                if self.config.enable_routing_logs:
                    logger.info(f"🎯 No human message found, routing to default: {self.config.default_agent}")
                return self.config.default_agent
            
            # Extract and analyze content
            content = str(getattr(last_human_message, 'content', '')).lower().strip()
            
            if not content:
                if self.config.enable_routing_logs:
                    logger.info(f"🎯 Empty content, routing to default: {self.config.default_agent}")
                return self.config.default_agent
            
            # Use configurable keyword matching
            for agent_name, keywords in self.config.routing_keywords.items():
                if agent_name in self.agents:  # Only route to available agents
                    if any(keyword in content for keyword in keywords):
                        if self.config.enable_routing_logs:
                            matched_keywords = [kw for kw in keywords if kw in content]
                            logger.info(
                                f"🎯 Routing to {agent_name} agent "
                                f"(matched: {', '.join(matched_keywords[:3])})"
                            )
                        return agent_name
            
            # Default routing
            if self.config.enable_routing_logs:
                logger.info(f"🎯 No specific keywords matched, routing to default: {self.config.default_agent}")
            return self.config.default_agent
            
        except Exception as e:
            self._routing_stats["routing_errors"] += 1
            logger.error(f"❌ Routing error: {e}, using default agent: {self.config.default_agent}")
            return self.config.default_agent
    
    async def process(self, state: AssistantState, config_dict: Dict[str, Any]) -> AssistantState:
        """
        Process a request through the supervisor graph.
        """
        try:
            if not self.supervisor_graph:
                raise SupervisorError("Supervisor not initialized. Call initialize() first.")
            
            # Validate state
            if not isinstance(state, dict):
                raise SupervisorError(f"Invalid state type: {type(state)}. Expected dict.")
            
            # Process through the graph
            result = await self.supervisor_graph.ainvoke(state, config_dict)
            
            # Validate result
            if not isinstance(result, dict):
                logger.warning("Graph returned non-dict result, wrapping")
                result = {
                    "messages": [AIMessage(content=str(result))],
                    "session_id": state.get("session_id", ""),
                    "user_id": state.get("user_id", ""),
                    "current_agent": state.get("current_agent", self.config.default_agent)
                }
            
            return result
            
        except Exception as e:
            self._routing_stats["routing_errors"] += 1
            logger.error(f"❌ Supervisor processing error: {e}")
            
            # Return error state with fallback response
            error_response = await ErrorHandler.handle_error(e, "supervisor_processing")
            
            # Maintain state structure
            error_state = dict(state)  # Copy original state
            error_state["messages"] = state.get("messages", []) + [
                AIMessage(content=error_response.get("response", "I encountered an issue processing your request."))
            ]
            
            return error_state
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring and debugging"""
        return {
            "config": {
                "default_agent": self.config.default_agent,
                "max_replays": self.config.max_replays,
                "routing_logs_enabled": self.config.enable_routing_logs
            },
            "stats": dict(self._routing_stats),
            "agents_available": list(self.agents.keys()),
            "tools_count": len(self.tools),
            "graph_initialized": self.supervisor_graph is not None
        }
    
    def update_routing_keywords(self, agent_name: str, keywords: List[str]) -> None:
        """
        Update routing keywords for a specific agent.
        
        This allows dynamic configuration of routing behavior.
        """
        if agent_name not in self.agents:
            logger.warning(f"⚠️ Agent '{agent_name}' not found, cannot update keywords")
            return
        
        self.config.routing_keywords[agent_name] = keywords
        logger.info(f"✅ Updated routing keywords for {agent_name}: {len(keywords)} keywords")
    
    def get_configuration(self) -> SupervisorConfig:
        """Get current supervisor configuration"""
        return self.config
    
    def set_configuration(self, new_config: SupervisorConfig) -> None:
        """
        Update supervisor configuration.
        
        Note: This requires reinitialization to take full effect.
        """
        self.config = new_config
        logger.info("✅ Supervisor configuration updated. Reinitialize for full effect.")
