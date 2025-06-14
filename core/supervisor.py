import logging
from typing import Any, Literal, Optional, List, Union, Self
from dataclasses import dataclass
from collections.abc import Sequence  # Python 3.13.4 preferred import

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from core.state import AssistantState
from core.error_handling import ErrorHandler
from core.checkpointer import Checkpointer

logger = logging.getLogger("supervisor")

@dataclass
class SupervisorConfig:
    """Enhanced configuration class for context-aware supervisor behavior"""
    max_replays: int = 5
    default_agent: str = "chat"
    enable_routing_logs: bool = True
    enable_human_in_loop: bool = False
    
    # Context-aware routing settings
    enable_context_aware_routing: bool = True
    conversation_history_limit: int = 6  # Messages to analyze for context
    continuity_bonus_weight: float = 1.5  # Weight for staying with current agent
    keyword_match_weight: float = 2.0     # Weight for keyword matching
    
    routing_keywords: Optional[dict[str, list[str]]] = None

    def __post_init__(self):
        """Initialize default routing keywords if not provided"""
        if self.routing_keywords is None:
            self.routing_keywords = {
                "coder": [
                    "code", "python", "function", "class", "def", "import",
                    "script", "programming", "debug", "refactor", "algorithm",
                    "variable", "loop", "condition", "syntax", "error",
                    "compile", "execute", "method", "object", "inheritance",
                    "save", "write", "json", "csv", "txt",
                    "generate", "build", "make", "output", "export", "yaml",
                    "xml", "html", "css", "js", "javascript", "sql",
                    "config", "configuration", "data", "structure"
                ],
                "web": [
                    "search", "web", "news", "internet", "google", "find",
                    "lookup", "research", "browse", "website", "url",
                    "online", "information", "current", "latest", "recent"
                ],
                "file_manager": [
                    "file", "document", "upload", "download", "analyze",
                    "backup", "organize", "folder", "directory", "archive"
                ],
                "project_management": [
                    "project", "schedule", "activity", "resource", "primavera", "p6",
                    "critical path", "milestone", "baseline", "progress", "completion",
                    "allocation", "utilization", "wbs", "work breakdown", "gantt",
                    "task", "assignment", "calendar", "budget", "cost", "duration",
                    "start date", "finish date", "float", "slack", "dependency",
                    "relationship", "predecessor", "successor", "constraint",
                    "portfolio", "program", "subproject", "phase", "deliverable",
                    "risk", "issue", "change", "approval", "status report",
                    "dashboard", "kpi", "performance", "variance", "forecast"
                ]
            }

class SupervisorError(Exception):
    """Custom exception for supervisor-specific errors"""
    pass

class Supervisor:
    """
    A supervisor with intelligent routing and enhanced LangGraph 0.4.8 patterns
    """
    
    def __init__(self, config: Optional[SupervisorConfig] = None):
        self.config = config or SupervisorConfig()
        self.supervisor_graph = None
        self.agents: dict[str, Any] = {}  # Python 3.13.4 syntax
        self.tools: list[Any] = []  # Python 3.13.4 syntax
        self._routing_stats = {
            "total_routes": 0,
            "routes_by_agent": {},
            "routing_errors": 0
        }
        
    async def initialize(
        self, 
        agents: dict[str, Any],  # Python 3.13.4 syntax
        all_tools: list[Any],  # Python 3.13.4 syntax
        checkpointer: Optional[Checkpointer] = None
    ) -> None:
        """
        Initialize supervisor with agents and tools using modern LangGraph patterns.
        
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
            
            # Create the workflow graph with modern LangGraph 0.4.8 patterns
            workflow = StateGraph(AssistantState)

            # Use enhanced ToolNode with proper error handling
            tool_node = ToolNode(all_tools)
            workflow.add_node("call_tool", tool_node)
            
            # Add agent nodes with circuit breaker protection
            for agent_name, agent in agents.items():
                workflow.add_node(
                    agent_name, 
                    self._create_resilient_agent_node(agent, agent_name)
                )

            # Modern conditional routing with explicit fallback
            workflow.add_conditional_edges(
                START,
                self._enhanced_route_to_agent,
                {**{agent_name: agent_name for agent_name in agents.keys()}, 
                 "__end__": END}  # Explicit end condition
            )

            # Enhanced tool continuation with smart routing
            workflow.add_conditional_edges(
                "call_tool",
                self._smart_continuation_logic,
                {**{agent_name: agent_name for agent_name in agents.keys()},
                 "__end__": END}
            )

            # Wire up agent nodes to tool continuation
            for agent_name in agents.keys():
                workflow.add_conditional_edges(
                    agent_name, 
                    self._should_continue_to_tools,
                    {
                        "call_tool": "call_tool",
                        "__end__": END
                    }
                )
            
            # Compile with enhanced options for LangGraph 0.4.8
            compile_options = {
                "checkpointer": checkpointer,
                "debug": self.config.enable_routing_logs,
            }
            
            # Add human-in-the-loop if enabled
            if self.config.enable_human_in_loop:
                compile_options["interrupt_before"] = ["call_tool"]
            
            self.supervisor_graph = workflow.compile(**compile_options)
            
            # Initialize routing statistics
            for agent_name in agents.keys():
                self._routing_stats["routes_by_agent"][agent_name] = 0
            
            if self.config.enable_routing_logs:
                logger.info(
                    f"✅ Enhanced supervisor initialized with {len(agents)} agents "
                    f"and {len(all_tools)} tools (LangGraph 0.4.8)"
                )
            
        except Exception as e:
            error_msg = f"Supervisor initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            await ErrorHandler.handle_error(SupervisorError(error_msg), "supervisor_initialization")
            raise
    
    def _validate_initialization_inputs(self, agents: dict[str, Any], tools: list[Any]) -> None:
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
    
    def _create_resilient_agent_node(self, agent: Any, agent_name: str):
        """
        Wrap an agent in a LangGraph-compatible async node that
        – tracks routing statistics
        – is protected by the global circuit breaker
        – guarantees the returned state always contains `current_agent`
        """
        async def agent_node(state: AssistantState) -> dict[str, Any]:
            try:
                # Persist the agent we are about to call
                state["current_agent"] = agent_name

                # --- metrics -------------------------------------------------
                self._routing_stats["routes_by_agent"][agent_name] += 1
                self._routing_stats["total_routes"] += 1
                # -------------------------------------------------------------

                # Execute the agent behind the circuit breaker
                from core.circuit_breaker import global_circuit_breaker
                result: dict[str, Any] | Any = await global_circuit_breaker.call_with_circuit_breaker(
                    f"agent_{agent_name}",
                    agent.ainvoke,
                    state,
                )

                # Normalise return type
                if not isinstance(result, dict):
                    logger.warning("Agent %s returned non-dict, wrapping in {'messages': …}", agent_name)
                    result = {"messages": [AIMessage(content=str(result))]}

                # 🔑 PROPOSED CHANGE — keep the routing context in the new state
                result.setdefault("current_agent", agent_name)
                result.setdefault("session_id", state.get("session_id", ""))
                result.setdefault("user_id", state.get("user_id", ""))

                return result

            except Exception as e:               # noqa: BLE001
                self._routing_stats["routing_errors"] += 1
                logger.error("❌ Agent %s execution failed: %s", agent_name, e)

                # Gracefully degrade instead of exploding the graph
                return {
                    "messages": [
                        AIMessage(
                            content="I encountered an issue. Please try re-phrasing your request."
                        )
                    ],
                    "current_agent": agent_name,
                    "session_id": state.get("session_id", ""),
                    "user_id": state.get("user_id", ""),
                }

        return agent_node

    def _analyze_conversation_context(self, messages: list, current_agent: str) -> dict:
        """Analyze conversation context for routing decisions"""
        
        context = {
            "agent_usage_history": [],
            "topics_discussed": set(),
            "conversation_flow": "new",
            "last_ai_response_type": None,
            "user_satisfaction_indicators": [],
            "complexity_level": "simple"
        }
        # Filter messages by session if available
        session_filtered_messages = []
        current_session = None
        
        # Extract session ID from the most recent message
        for msg in reversed(messages):
            if hasattr(msg, 'additional_kwargs') and 'session_id' in msg.additional_kwargs:
                current_session = msg.additional_kwargs['session_id']
                break
        
        # Only analyze messages from current session
        if current_session:
            for msg in messages:
                if not hasattr(msg, 'additional_kwargs') or \
                msg.additional_kwargs.get('session_id') == current_session:
                    session_filtered_messages.append(msg)
            messages = session_filtered_messages  # Replace with filtered messages
        
        # Track agent usage pattern
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Infer which agent generated this response
                inferred_agent = self._infer_agent_from_message(msg)
                if inferred_agent:
                    context["agent_usage_history"].append(inferred_agent)
        
        # Analyze topics from all messages
        all_content = " ".join([
            getattr(msg, 'content', '') for msg in messages 
            if hasattr(msg, 'content')
        ]).lower()
        
        # Extract topics using your existing keywords
        for agent_name, keywords in self.config.routing_keywords.items():
            matched_keywords = [kw for kw in keywords if kw in all_content]
            if matched_keywords:
                context["topics_discussed"].update(matched_keywords)
        
        # Determine conversation flow
        if len(context["agent_usage_history"]) == 0:
            context["conversation_flow"] = "new"
        elif len(set(context["agent_usage_history"])) == 1:
            context["conversation_flow"] = "continuing"
        else:
            context["conversation_flow"] = "switching"
        
        # Analyze last AI response
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = getattr(msg, 'content', '')
                
                # Check for completion indicators
                completion_indicators = [
                    "here's", "completed", "done", "finished", "created", 
                    "generated", "found", "search results", "implemented"
                ]
                if any(indicator in content.lower() for indicator in completion_indicators):
                    context["last_ai_response_type"] = "completion"
                
                # Check for question/clarification
                elif "?" in content or any(q in content.lower() for q in ["what", "how", "which", "would you like"]):
                    context["last_ai_response_type"] = "question"
                
                # Check for error/limitation
                elif any(err in content.lower() for err in ["error", "can't", "unable", "sorry", "failed"]):
                    context["last_ai_response_type"] = "error"
                
                break
        
        return context

    def _infer_agent_from_message(self, message: AIMessage) -> Optional[str]:
        """Infer which agent generated a message based on patterns"""
        content = getattr(message, 'content', '').lower()
        
        # Look for agent-specific patterns in responses
        if any(pattern in content for pattern in ['code', 'function', 'python', 'script']):
            return 'coder'
        elif any(pattern in content for pattern in ['search', 'found', 'website', 'information']):
            return 'web'  # ✅ FIXED: Use your actual agent name
        elif any(pattern in content for pattern in ['file', 'document', 'upload', 'analyze']):
            return 'file_manager'   # ✅ FIXED: Use your actual agent name
        
        # Check tool calls for more definitive agent identification
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', '').lower()
                if 'code' in tool_name or 'python' in tool_name:
                    return 'coder'
                elif 'search' in tool_name or 'web' in tool_name or 'tavily' in tool_name:
                    return 'web'
                elif 'file' in tool_name:
                    return 'file_manager'
        
        return None

    def _make_contextual_routing_decision(
        self, 
        content: str, 
        context_analysis: dict, 
        current_agent: str,
        recent_messages: list
    ) -> dict:
        """Make routing decision with BALANCED content vs context weighting"""
        
        # 1. Basic keyword scoring
        keyword_scores = {}
        for agent_name, keywords in self.config.routing_keywords.items():
            if agent_name in self.agents:
                score = sum(1 for keyword in keywords if keyword in content)
                if score > 0:
                    keyword_scores[agent_name] = score
        
        # 2. Context-based scoring adjustments
        context_scores = {}
        
        # ✅ FIXED: Reduce continuity bonus and add topic-switch detection
        if context_analysis["conversation_flow"] == "continuing":
            if context_analysis["last_ai_response_type"] == "question":
                # If AI asked a question, user's answer should go to same agent
                context_scores[current_agent] = context_scores.get(current_agent, 0) + 2  # Reduced from 3
            elif context_analysis["last_ai_response_type"] == "completion":
                # If AI completed a task, slight bonus to continue with same agent
                context_scores[current_agent] = context_scores.get(current_agent, 0) + 0.5  # Reduced from 1
        
        # ✅ NEW: Detect clear topic switches and reduce context bonus
        clear_switch_keywords = {
            'web': ['search', 'find', 'look up', 'latest', 'current', 'recent', 'news', 'research'],
            'file_manager': ['upload', 'file', 'document', 'analyze file'],
            'coder': ['code', 'function', 'write', 'debug', 'implement']
        }
        
        # Check if this is a clear topic switch
        is_clear_switch = False
        for agent_name, switch_keywords in clear_switch_keywords.items():
            if agent_name != current_agent:
                if any(keyword in content for keyword in switch_keywords):
                    if sum(1 for keyword in switch_keywords if keyword in content) >= 2:
                        is_clear_switch = True
                        # Boost the target agent and reduce current agent context bonus
                        context_scores[agent_name] = context_scores.get(agent_name, 0) + 1
                        context_scores[current_agent] = context_scores.get(current_agent, 0) - 1
                        break
        
        # Topic consistency bonus (reduced when clear switch detected)
        consistency_multiplier = 0.3 if is_clear_switch else 1.0
        for agent_name in self.agents.keys():
            agent_keywords = set(self.config.routing_keywords.get(agent_name, []))
            discussed_topics = context_analysis["topics_discussed"]
            
            overlap = len(agent_keywords.intersection(discussed_topics))
            if overlap > 0:
                bonus = overlap * consistency_multiplier
                context_scores[agent_name] = context_scores.get(agent_name, 0) + bonus
        
        # 3. ✅ FIXED: Rebalanced scoring weights
        final_scores = {}
        all_agents = set(keyword_scores.keys()) | set(context_scores.keys()) | {current_agent}
        
        for agent in all_agents:
            if agent in self.agents:
                keyword_contribution = keyword_scores.get(agent, 0) * 2.0  # Increased keyword weight
                context_contribution = context_scores.get(agent, 0) * 1.2  # Reduced context weight
                
                final_scores[agent] = keyword_contribution + context_contribution
        
        # 4. Make decision with improved logic
        if final_scores:
            best_agent = max(final_scores.keys(), key=lambda x: final_scores[x])
            confidence = final_scores[best_agent] / (sum(final_scores.values()) + 0.001)
            
            # Determine reason with switch detection
            keyword_contribution = keyword_scores.get(best_agent, 0) * 3.0
            context_contribution = context_scores.get(best_agent, 0) * 1.0
            
            if is_clear_switch:
                reason = "topic_switch_detected"
            elif keyword_contribution > context_contribution:
                reason = "keyword_match"
            elif context_contribution > 0:
                reason = "context_continuity"
            else:
                reason = "default"
                
        else:
            best_agent = current_agent or self.config.default_agent
            confidence = 0.5
            reason = "fallback"
        
        return {
            "agent": best_agent,
            "confidence": confidence,
            "reason": reason,
            "keyword_scores": keyword_scores,
            "context_scores": context_scores,
            "topic_switch_detected": is_clear_switch
        }

    def _enhanced_route_to_agent(self, state: AssistantState) -> str:
        """Context-aware routing with detailed debugging"""
        try:
            messages = state.get("messages", [])
            current_agent = state.get("current_agent", self.config.default_agent)
            
            # 🔍 DEBUG: Log current state
            logger.info(f"🎯 ROUTING DEBUG - Current agent: {current_agent}")
            logger.info(f"🎯 ROUTING DEBUG - Message count: {len(messages)}")
            logger.info(f"🎯 ROUTING DEBUG - Context-aware enabled: {self.config.enable_context_aware_routing}")
            
            if not messages:
                logger.info(f"🎯 ROUTING DEBUG - No messages, using default: {self.config.default_agent}")
                return self.config.default_agent
            
            # Find last human message
            last_human_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    last_human_message = message
                    break
            
            if last_human_message:
                content = str(getattr(last_human_message, 'content', '')).lower().strip()
                logger.info(f"🎯 ROUTING DEBUG - Last human message: {content[:100]}...")
            else:
                logger.info(f"🎯 ROUTING DEBUG - No human message found")
            
            # Use context-aware routing if enabled
            if self.config.enable_context_aware_routing:
                logger.info("🎯 ROUTING DEBUG - Using context-aware routing")
                result = self._context_aware_route_to_agent(state)
                logger.info(f"🎯 ROUTING DEBUG - Context-aware result: {result}")
                return result
            else:
                logger.info("🎯 ROUTING DEBUG - Using keyword-only routing")
                result = self._keyword_only_routing(state)
                logger.info(f"🎯 ROUTING DEBUG - Keyword-only result: {result}")
                return result
                
        except Exception as e:
            self._routing_stats["routing_errors"] += 1
            logger.error(f"❌ ROUTING DEBUG - Error: {e}")
            logger.error(f"❌ ROUTING DEBUG - Using default agent: {self.config.default_agent}")
            return self.config.default_agent

    def _context_aware_route_to_agent(self, state: AssistantState) -> str:
        """Enhanced context-aware routing with detailed debugging"""
        try:
            messages = state.get("messages", [])
            current_agent = state.get("current_agent", self.config.default_agent)
            
            # 🔍 DEBUG: Log context analysis inputs
            recent_messages = messages[-self.config.conversation_history_limit:] if len(messages) > self.config.conversation_history_limit else messages
            logger.info(f"🎯 CONTEXT DEBUG - Analyzing {len(recent_messages)} recent messages")
            
            # Analyze conversation flow and context
            context_analysis = self._analyze_conversation_context(recent_messages, current_agent)
            logger.info(f"🎯 CONTEXT DEBUG - Context analysis: {context_analysis}")
            
            # Find the most recent human message
            last_human_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    last_human_message = message
                    break
            
            if not last_human_message:
                logger.info(f"🎯 CONTEXT DEBUG - No new human message, staying with: {current_agent}")
                return current_agent
            
            content = str(getattr(last_human_message, 'content', '')).lower().strip()
            if not content:
                logger.info(f"🎯 CONTEXT DEBUG - Empty content, staying with: {current_agent}")
                return current_agent
            
            # Make contextual routing decision
            routing_decision = self._make_contextual_routing_decision(
                content=content,
                context_analysis=context_analysis,
                current_agent=current_agent,
                recent_messages=recent_messages
            )
            
            # 🔍 DEBUG: Log detailed routing decision
            logger.info(f"🎯 CONTEXT DEBUG - Routing decision details:")
            logger.info(f"   Agent: {routing_decision['agent']}")
            logger.info(f"   Confidence: {routing_decision['confidence']:.2f}")
            logger.info(f"   Reason: {routing_decision['reason']}")
            logger.info(f"   Keyword scores: {routing_decision['keyword_scores']}")
            logger.info(f"   Context scores: {routing_decision['context_scores']}")
            
            if self.config.enable_routing_logs:
                logger.info(f"🎯 Context-aware routing: {routing_decision['agent']} "
                        f"(confidence: {routing_decision['confidence']:.2f}, "
                        f"reason: {routing_decision['reason']})")
            
            return routing_decision['agent']
            
        except Exception as e:
            logger.error(f"❌ CONTEXT DEBUG - Context-aware routing error: {e}")
            import traceback
            logger.error(f"❌ CONTEXT DEBUG - Traceback: {traceback.format_exc()}")
            return self.config.default_agent

    def _keyword_only_routing(self, state: AssistantState) -> str:
        """Your original keyword-only routing as fallback"""
        # This is your existing _enhanced_route_to_agent logic
        messages = state.get("messages", [])
        
        # Find the most recent human message
        last_human_message = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break

        if not last_human_message:
            return self.config.default_agent

        content = str(getattr(last_human_message, 'content', '')).lower().strip()
        if not content:
            return self.config.default_agent

        # Use configurable keyword matching with scoring
        agent_scores = {}
        for agent_name, keywords in self.config.routing_keywords.items():
            if agent_name in self.agents:
                score = sum(1 for keyword in keywords if keyword in content)
                if score > 0:
                    agent_scores[agent_name] = score

        # Route to agent with highest score
        if agent_scores:
            best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x])
            if self.config.enable_routing_logs:
                matched_keywords = [
                    kw for kw in self.config.routing_keywords[best_agent]
                    if kw in content
                ]
                logger.info(
                    f"🎯 Routing to {best_agent} agent "
                    f"(score: {agent_scores[best_agent]}, matched: {', '.join(matched_keywords[:3])})"
                )
            return best_agent

        return self.config.default_agent

    def _should_continue_to_tools(self, state: AssistantState) -> Literal["call_tool", "__end__"]:
        """Determine if the agent should call tools or end (Python 3.13.4 enhanced)"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
            if last_message.tool_calls:
                return "call_tool"
        return "__end__"

    def _smart_continuation_logic(self, state: AssistantState) -> str:
        """Smart logic for continuing after tool execution"""
        # Get the current agent from state, fallback to default
        current_agent = state.get("current_agent", self.config.default_agent)
        
        # Ensure the agent still exists
        if current_agent in self.agents:
            return current_agent
        
        # Fallback to default agent
        return self.config.default_agent
    
    async def process(
        self,
        state: AssistantState,
        runnable_config: dict[str, Any],
        timeout: float = 300.0,
    ) -> AssistantState:
        """
        Run the compiled LangGraph and return the updated state.

        Args:
            state:            Current assistant-state dict.
            runnable_config:  LangGraph runtime config
                            (e.g. {"configurable": {"thread_id": "..."} } ).
            timeout:          Max seconds before cancelling execution.
        """
        if not self.supervisor_graph:
            raise SupervisorError("Supervisor not initialised. Call initialize() first.")

        if not isinstance(state, dict):
            raise SupervisorError(f"Invalid state type: {type(state)}. Expected dict.")

        import asyncio
        result = await asyncio.wait_for(
            self.supervisor_graph.ainvoke(state, runnable_config),   # ✅ pass config
            timeout=timeout,
        )

        # Guarantee a dict-shaped result
        if not isinstance(result, dict):
            result = {
                "messages": [AIMessage(content=str(result))],
                "session_id": state.get("session_id", ""),
                "user_id": state.get("user_id", ""),
                "current_agent": state.get(
                    "current_agent", self.config.default_agent
                ),
            }
        return result
    
    def get_routing_statistics(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get routing statistics for monitoring and debugging"""
        return {
            "config": {
                "default_agent": self.config.default_agent,
                "max_replays": self.config.max_replays,
                "routing_logs_enabled": self.config.enable_routing_logs,
                "human_in_loop_enabled": self.config.enable_human_in_loop
            },
            "stats": dict(self._routing_stats),
            "agents_available": list(self.agents.keys()),
            "tools_count": len(self.tools),
            "graph_initialized": self.supervisor_graph is not None
        }
    
    def update_routing_keywords(self, agent_name: str, keywords: list[str]) -> None:  # Python 3.13.4 syntax
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

    async def stream_events(self, state: AssistantState, config_dict: dict[str, Any]):
        """
        Stream events from the supervisor graph (LangGraph 0.4.8 feature).
        """
        if not self.supervisor_graph:
            raise SupervisorError("Supervisor not initialized. Call initialize() first.")
        
        async for event in self.supervisor_graph.astream_events(state, config_dict, version="v1"):
            yield event

    def visualize_graph(self) -> str:
        """
        Generate a visual representation of the supervisor graph.
        """
        if not self.supervisor_graph:
            return "Graph not initialized"
        
        try:
            # Use LangGraph's built-in visualization if available
            return self.supervisor_graph.get_graph().draw_mermaid()
        except AttributeError:
            # Fallback to text representation
            return f"Supervisor Graph: {len(self.agents)} agents, {len(self.tools)} tools"
