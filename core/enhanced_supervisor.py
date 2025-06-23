# core/enhanced_supervisor.py - Enhanced Supervisor with Verification and Streaming

import logging
import asyncio
import time
import json
from typing import Any, Literal, Optional, List, Union, Self, Dict
from dataclasses import dataclass, field
from collections.abc import Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from core.enhanced_state import (
    EnhancedAssistantState, AgentCommunicationManager, ScratchpadManager, 
    StreamingManager, AgentMessage, MessageType, MessagePriority
)
from core.error_handling import ErrorHandler
from core.checkpointer import Checkpointer

logger = logging.getLogger("enhanced_supervisor")

@dataclass
class VerificationConfig:
    """Configuration for response verification"""
    enabled: bool = True
    check_content_safety: bool = True
    check_response_quality: bool = True
    check_agent_consensus: bool = False
    min_confidence_score: float = 0.7
    require_human_approval: bool = False
    timeout_seconds: float = 30.0

@dataclass
class StreamingConfig:
    """Configuration for streaming updates"""
    enabled: bool = True
    send_agent_activities: bool = True
    send_progress_updates: bool = True
    send_verification_status: bool = True
    update_interval_ms: int = 500
    max_events_per_session: int = 1000

class EnhancedSupervisor:
    """Enhanced supervisor with verification, streaming, and cross-agent communication"""
    
    def __init__(self, config=None, verification_config=None, streaming_config=None):
        self.config = config
        self.verification_config = verification_config or VerificationConfig()
        self.streaming_config = streaming_config or StreamingConfig()
        self.supervisor_graph = None
        self.agents: Dict[str, Any] = {}
        self.tools: List[Any] = []
        self._routing_stats = {
            "total_routes": 0,
            "routes_by_agent": {},
            "routing_errors": 0,
            "verifications_passed": 0,
            "verifications_failed": 0
        }
        self._active_streams: Dict[str, asyncio.Queue] = {}
    
    async def initialize(self, agents: Dict[str, Any], all_tools: List[Any], 
                        checkpointer: Optional[Checkpointer] = None) -> None:
        """Initialize enhanced supervisor with verification and streaming"""
        try:
            self.agents = agents
            self.tools = all_tools
            
            # Create enhanced workflow graph
            workflow = StateGraph(EnhancedAssistantState)
            
            # Add tool node
            tool_node = ToolNode(all_tools)
            workflow.add_node("call_tool", tool_node)
            
            # Add verification node
            workflow.add_node("verify_response", self._create_verification_node())
            
            # Add agent nodes with communication capabilities
            for agent_name, agent in agents.items():
                workflow.add_node(
                    agent_name,
                    self._create_enhanced_agent_node(agent, agent_name)
                )
            
            # Enhanced routing with verification
            workflow.add_conditional_edges(
                START,
                self._enhanced_route_to_agent,
                {**{agent_name: agent_name for agent_name in agents.keys()},
                 "__end__": END}
            )
            
            # Add verification after agent execution
            for agent_name in agents.keys():
                workflow.add_conditional_edges(
                    agent_name,
                    self._should_verify_or_continue,
                    {
                        "verify": "verify_response",
                        "call_tool": "call_tool",
                        "__end__": END
                    }
                )
            
            # Verification can end or require human approval
            workflow.add_conditional_edges(
                "verify_response",
                self._after_verification,
                {
                    "approved": "__end__",
                    "rejected": self._select_fallback_agent(),
                    "human_review": "__end__"  # Human review endpoint
                }
            )
            
            # Tool continuation with communication
            workflow.add_conditional_edges(
                "call_tool",
                self._smart_continuation_with_communication,
                {**{agent_name: agent_name for agent_name in agents.keys()},
                 "__end__": END}
            )
            
            # Compile with enhanced options
            compile_options = {
                "checkpointer": checkpointer,
                "debug": True
            }
            
            self.supervisor_graph = workflow.compile(**compile_options)
            
            # Initialize routing statistics
            for agent_name in agents.keys():
                self._routing_stats["routes_by_agent"][agent_name] = 0
            
            logger.info(f"✅ Enhanced supervisor initialized with verification and streaming")
            
        except Exception as e:
            logger.error(f"❌ Enhanced supervisor initialization failed: {e}")
            raise
    
    def _create_enhanced_agent_node(self, agent: Any, agent_name: str):
        """Create agent node with communication and streaming"""
        async def enhanced_agent_node(state: EnhancedAssistantState) -> EnhancedAssistantState:
            try:
                # Update streaming status
                if state.get('streaming_enabled', False):
                    state = StreamingManager.add_stream_event(
                        state, 'agent_started', 
                        {'agent': agent_name, 'message': f"Starting {agent_name} agent"}
                    )
                
                # Check for inter-agent messages
                incoming_messages = AgentCommunicationManager.get_messages_for_agent(state, agent_name)
                
                # Process incoming messages and add to context
                context_additions = []
                for msg in incoming_messages:
                    context_additions.append(f"Message from {msg.from_agent}: {msg.content}")
                    state = AgentCommunicationManager.mark_message_processed(state, msg.id)
                
                # Add context to current message if there are agent communications
                if context_additions:
                    messages = list(state.get("messages", []))
                    if messages and isinstance(messages[-1], HumanMessage):
                        original_content = messages[-1].content
                        enhanced_content = f"{original_content}\n\nAgent Context:\n" + "\n".join(context_additions)
                        messages[-1] = HumanMessage(content=enhanced_content)
                        state["messages"] = messages
                
                # Set current agent and execute
                state["current_agent"] = agent_name
                
                # Execute agent with circuit breaker protection
                from core.circuit_breaker import global_circuit_breaker
                result = await global_circuit_breaker.call_with_circuit_breaker(
                    f"agent_{agent_name}",
                    agent.ainvoke,
                    state
                )
                
                # Normalize result
                if not isinstance(result, dict):
                    result = {"messages": [AIMessage(content=str(result))]}
                
                # Ensure required fields
                result.setdefault("current_agent", agent_name)
                result.setdefault("session_id", state.get("session_id", ""))
                result.setdefault("user_id", state.get("user_id", ""))
                
                # Update streaming status
                if state.get('streaming_enabled', False):
                    result = StreamingManager.add_stream_event(
                        result, 'agent_completed',
                        {'agent': agent_name, 'message': f"Completed {agent_name} agent"}
                    )
                
                # Update routing stats
                self._routing_stats["routes_by_agent"][agent_name] += 1
                self._routing_stats["total_routes"] += 1
                
                return result
                
            except Exception as e:
                self._routing_stats["routing_errors"] += 1
                logger.error(f"❌ Enhanced agent {agent_name} execution failed: {e}")
                
                # Create error response with communication
                error_state = dict(state)
                error_state["messages"] = [
                    AIMessage(content="I encountered an issue. Let me try a different approach.")
                ]
                error_state["current_agent"] = agent_name
                
                return error_state
        
        return enhanced_agent_node
    
    def _create_verification_node(self):
        """Create verification node for response quality checking"""
        async def verify_response_node(state: EnhancedAssistantState) -> EnhancedAssistantState:
            try:
                if not self.verification_config.enabled:
                    return state
                
                # Update streaming
                if state.get('streaming_enabled', False):
                    state = StreamingManager.add_stream_event(
                        state, 'verification_started',
                        {'message': 'Starting response verification'}
                    )
                
                messages = state.get("messages", [])
                if not messages:
                    return state
                
                last_message = messages[-1]
                if not isinstance(last_message, AIMessage):
                    return state
                
                verification_result = await self._verify_response(last_message, state)
                
                # Store verification result
                verification_data = {
                    "timestamp": time.time(),
                    "agent": state.get("current_agent", "unknown"),
                    "passed": verification_result["passed"],
                    "confidence": verification_result["confidence"],
                    "issues": verification_result.get("issues", []),
                    "message_content": last_message.content[:200]
                }
                
                new_state = dict(state)
                new_state["pending_verification"] = verification_data
                
                if "verification_history" not in new_state:
                    new_state["verification_history"] = []
                new_state["verification_history"].append(verification_data)
                
                # Update stats
                if verification_result["passed"]:
                    self._routing_stats["verifications_passed"] += 1
                else:
                    self._routing_stats["verifications_failed"] += 1
                
                # Update streaming
                if state.get('streaming_enabled', False):
                    new_state = StreamingManager.add_stream_event(
                        new_state, 'verification_completed',
                        {
                            'passed': verification_result["passed"],
                            'confidence': verification_result["confidence"],
                            'message': f"Verification {'passed' if verification_result['passed'] else 'failed'}"
                        }
                    )
                
                return new_state
                
            except Exception as e:
                logger.error(f"❌ Response verification failed: {e}")
                return state
        
        return verify_response_node
    
    async def _verify_response(self, message: AIMessage, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Verify response quality and safety"""
        try:
            content = message.content
            issues = []
            confidence = 1.0
            
            # Basic content checks
            if not content or len(content.strip()) < 10:
                issues.append("Response too short")
                confidence -= 0.3
            
            if len(content) > 10000:
                issues.append("Response potentially too long")
                confidence -= 0.1
            
            # Check for error indicators
            error_indicators = [
                "I encountered an error",
                "I'm having trouble",
                "Something went wrong",
                "I can't access",
                "Failed to"
            ]
            
            for indicator in error_indicators:
                if indicator.lower() in content.lower():
                    issues.append(f"Error indicator found: {indicator}")
                    confidence -= 0.2
                    break
            
            # Check for completeness
            if content.lower().strip().endswith(("...", ".", "incomplete", "partial")):
                if "incomplete" in content.lower() or "partial" in content.lower():
                    issues.append("Response appears incomplete")
                    confidence -= 0.2
            
            # Quality scoring
            if self.verification_config.check_response_quality:
                # Check for structured response
                if any(marker in content for marker in ["##", "###", "- ", "1.", "2."]):
                    confidence += 0.1  # Bonus for structured content
                
                # Check for code blocks if it's a coding response
                current_agent = state.get("current_agent", "")
                if current_agent == "coder" and "```" in content:
                    confidence += 0.1  # Bonus for code formatting
            
            # Agent consensus check (if enabled)
            if self.verification_config.check_agent_consensus:
                # This could invoke another agent for a second opinion
                # For now, we'll simulate this
                pass
            
            # Final scoring
            confidence = max(0.0, min(1.0, confidence))
            passed = confidence >= self.verification_config.min_confidence_score and len(issues) == 0
            
            return {
                "passed": passed,
                "confidence": confidence,
                "issues": issues,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Verification process failed: {e}")
            return {"passed": False, "confidence": 0.0, "issues": [f"Verification error: {str(e)}"]}
    
    def _enhanced_route_to_agent(self, state: EnhancedAssistantState) -> str:
        """Enhanced routing with communication awareness"""
        try:
            # Check for urgent inter-agent messages that might affect routing
            agent_messages = state.get("agent_messages", [])
            urgent_messages = [msg for msg in agent_messages 
                             if msg.priority == MessagePriority.URGENT and not msg.processed]
            
            if urgent_messages:
                # Route to agent with urgent message
                return urgent_messages[0].to_agent
            
            # Use existing routing logic from original supervisor
            messages = state.get("messages", [])
            if not messages:
                return "chat"  # Default agent
            
            # Find last human message
            last_human_message = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    last_human_message = message
                    break
            
            if not last_human_message:
                return "chat"
            
            content = str(getattr(last_human_message, 'content', '')).lower().strip()
            
            # Enhanced keyword routing with communication context
            routing_keywords = {
                "coder": ["code", "python", "function", "debug", "programming", "script"],
                "web": ["search", "web", "find", "lookup", "research", "news"],
                "file_manager": ["file", "upload", "document", "analyze", "organize"],
                "project_management": ["project", "p6", "primavera", "schedule", "activity"]
            }
            
            # Score each agent
            agent_scores = {}
            for agent_name, keywords in routing_keywords.items():
                if agent_name in self.agents:
                    score = sum(1 for keyword in keywords if keyword in content)
                    if score > 0:
                        agent_scores[agent_name] = score
            
            # Route to highest scoring agent
            if agent_scores:
                best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x])
                logger.info(f"🎯 Enhanced routing to {best_agent} (score: {agent_scores[best_agent]})")
                return best_agent
            
            return "chat"  # Default fallback
            
        except Exception as e:
            logger.error(f"❌ Enhanced routing failed: {e}")
            return "chat"
    
    def _should_verify_or_continue(self, state: EnhancedAssistantState) -> Literal["verify", "call_tool", "__end__"]:
        """Determine if response should be verified or continue to tools"""
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        
        last_message = messages[-1]
        
        # Check if agent wants to use tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
            if last_message.tool_calls:
                return "call_tool"
        
        # Verify response if verification is enabled
        if self.verification_config.enabled:
            return "verify"
        
        return "__end__"
    
    def _after_verification(self, state: EnhancedAssistantState) -> Literal["approved", "rejected", "human_review"]:
        """Handle post-verification routing"""
        verification = state.get("pending_verification")
        if not verification:
            return "approved"
        
        if verification["passed"]:
            return "approved"
        
        # Check if human review is required
        if self.verification_config.require_human_approval:
            return "human_review"
        
        # Auto-retry with different agent
        return "rejected"
    
    def _select_fallback_agent(self) -> str:
        """Select fallback agent when verification fails"""
        # Simple fallback strategy - use chat agent
        return "chat"
    
    def _smart_continuation_with_communication(self, state: EnhancedAssistantState) -> str:
        """Smart continuation with inter-agent communication"""
        # Check for cross-agent requests in tool results
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                content = str(last_message.content).lower()
                
                # Check if tool result suggests using another agent
                if "file" in content and "analyze" in content:
                    return "file_manager"
                elif "search" in content or "web" in content:
                    return "web"
                elif "code" in content or "script" in content:
                    return "coder"
        
        # Continue with current agent
        current_agent = state.get("current_agent", "chat")
        return current_agent if current_agent in self.agents else "chat"
    
    async def process_with_streaming(
        self,
        state: EnhancedAssistantState,
        runnable_config: Dict[str, Any],
        stream_queue: Optional[asyncio.Queue] = None,
        timeout: float = 300.0
    ) -> EnhancedAssistantState:
        """Process with real-time streaming updates"""
        if not self.supervisor_graph:
            raise RuntimeError("Enhanced supervisor not initialized")
        
        # Enable streaming if queue provided
        if stream_queue and self.streaming_config.enabled:
            stream_id = f"stream_{int(time.time())}"
            state = StreamingManager.start_stream(state, stream_id)
            self._active_streams[stream_id] = stream_queue
        
        try:
            # Start streaming task if enabled
            streaming_task = None
            if stream_queue and state.get('streaming_enabled', False):
                streaming_task = asyncio.create_task(
                    self._stream_updates(state, stream_queue)
                )
            
            # Process through supervisor graph
            result = await asyncio.wait_for(
                self.supervisor_graph.ainvoke(state, runnable_config),
                timeout=timeout
            )
            
            # Clean up streaming
            if streaming_task:
                streaming_task.cancel()
                stream_id = state.get('current_stream_id')
                if stream_id and stream_id in self._active_streams:
                    del self._active_streams[stream_id]
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced processing failed: {e}")
            
            # Clean up streaming on error
            if stream_queue:
                stream_id = state.get('current_stream_id')
                if stream_id and stream_id in self._active_streams:
                    del self._active_streams[stream_id]
            
            raise
    
    async def _stream_updates(self, state: EnhancedAssistantState, stream_queue: asyncio.Queue):
        """Stream real-time updates"""
        try:
            last_event_count = 0
            
            while True:
                await asyncio.sleep(self.streaming_config.update_interval_ms / 1000.0)
                
                # Get new events
                events = state.get('stream_events', [])
                new_events = events[last_event_count:]
                
                # Send new events
                for event in new_events:
                    await stream_queue.put(event)
                
                last_event_count = len(events)
                
                # Check if streaming is still enabled
                if not state.get('streaming_enabled', False):
                    break
                    
        except asyncio.CancelledError:
            logger.info("Streaming task cancelled")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
    
    # Agent communication helpers
    async def send_agent_message(
        self,
        state: EnhancedAssistantState,
        from_agent: str,
        to_agent: str,
        content: str,
        message_type: MessageType = MessageType.NOTIFICATION,
        data: Optional[Dict] = None
    ) -> EnhancedAssistantState:
        """Send message between agents"""
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type=message_type,
            data=data
        )
        
        return AgentCommunicationManager.send_message(state, message)
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced routing and verification statistics"""
        base_stats = {
            "config": {
                "verification_enabled": self.verification_config.enabled,
                "streaming_enabled": self.streaming_config.enabled,
            },
            "routing": dict(self._routing_stats),
            "agents_available": list(self.agents.keys()),
            "tools_count": len(self.tools),
            "active_streams": len(self._active_streams)
        }
        
        return base_stats