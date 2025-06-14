# agents/agents.py - ✅ ENHANCED WITH PYTHON 3.13.4 COMPATIBILITY
import logging
import asyncio
import os
from typing import Optional, Union, Any
from collections.abc import Sequence  # Python 3.13.4 preferred import
from dataclasses import dataclass, field
from enum import Enum
from asyncio import TaskGroup  # Python 3.13.4 TaskGroup

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain.agents import create_tool_calling_agent, AgentExecutor

from config.settings import config
from config.llm_manager import llm_manager
from tools.file_tools import FileSystemTools
from core.state import AssistantState
from core.error_handling import ErrorHandler, with_error_handling
from core.circuit_breaker import global_circuit_breaker

logger = logging.getLogger("agents")

class AgentType(Enum):
    """Types of agents available in the system"""
    CHAT = "chat"
    CODER = "coder"
    WEB_SEARCH = "web"
    FILE_MANAGER = "file_manager"
    PROJECT_MANAGEMENT = "project_management"
    SUPERVISOR = "supervisor"

@dataclass
class AgentConfig:
    """Configuration for individual agents with Python 3.13.4 enhancements"""
    name: str
    agent_type: AgentType
    llm_node: str
    system_prompt: str
    tools_enabled: bool = True
    max_iterations: int = 10
    early_stopping_method: str = "generate"
    handle_parsing_errors: bool = True
    verbose: bool = False
    memory_enabled: bool = True
    custom_instructions: Optional[str] = None
    tool_names: list[str] = field(default_factory=list)  # Python 3.13.4 syntax
    
    def __post_init__(self):
        """Validate agent configuration"""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if not self.llm_node:
            raise ValueError("LLM node must be specified")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

class AgentFactory:
    """
    Factory for creating and managing agents with Python 3.13.4 enhancements.
    """
    
    def __init__(self):
        self.agents: dict[str, Any] = {}  # Python 3.13.4 syntax
        self.tools: dict[str, list[Any]] = {}  # Python 3.13.4 syntax
        self.file_tools = FileSystemTools()
        self._agent_configs = self._initialize_agent_configs()
        
    def _initialize_agent_configs(self) -> dict[str, AgentConfig]:  # Python 3.13.4 syntax
        """Initialize default agent configurations"""
        return {
            "chat": AgentConfig(
                name="chat",
                agent_type=AgentType.CHAT,
                llm_node="chat",
                system_prompt="""You are Mortey, a helpful AI assistant created to assist users with various tasks.

You are knowledgeable, friendly, and always aim to provide accurate and helpful responses. You can:
- Answer questions on a wide range of topics
- Help with problem-solving and analysis
- Provide explanations and tutorials
- Assist with creative tasks
- Offer suggestions and recommendations

Always be honest about your limitations and suggest alternative approaches when needed.
If you're unsure about something, say so rather than guessing.

Current date and time context will be provided when relevant.""",
                tools_enabled=False,
                max_iterations=5
            ),
            
            "coder": AgentConfig(
                name="coder",
                agent_type=AgentType.CODER,
                llm_node="coder",
                system_prompt="""You are Mortey's coding specialist, an expert programmer with deep knowledge across multiple programming languages and frameworks.

Your expertise includes:
- Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- Web development (React, Vue, Angular, Node.js)
- Backend development (FastAPI, Django, Express, Spring)
- Database design and optimization
- DevOps and deployment strategies
- Code review and optimization
- Debugging and troubleshooting
- Architecture and design patterns

You have access to file management tools to:
- Read and write code files
- Create project structures
- Organize and refactor codebases
- Search through code
- Backup and manage files

When helping with code:
1. Write clean, well-documented, and efficient code
2. Follow best practices and coding standards
3. Explain your reasoning and approach
4. Suggest improvements and optimizations
5. Use appropriate design patterns
6. Consider security and performance implications

Always test your code mentally and provide working examples when possible.""",
                tools_enabled=True,
                tool_names=["file_tools"],
                max_iterations=15
            ),
            
            "web": AgentConfig(
                name="web",
                agent_type=AgentType.WEB_SEARCH,
                llm_node="web",
                system_prompt="""You are Mortey's web research specialist, expert at finding and analyzing information from the internet.

Your capabilities include:
- Conducting comprehensive web searches
- Finding current news and information
- Researching topics across various domains
- Fact-checking and verification
- Finding specific resources and documentation
- Analyzing trends and developments

When conducting research:
1. Use specific and targeted search queries
2. Verify information from multiple sources when possible
3. Provide citations and source information
4. Distinguish between facts and opinions
5. Note the recency and relevance of information
6. Summarize findings clearly and concisely

Always be transparent about the sources of your information and any limitations in your search results.""",
                tools_enabled=True,
                tool_names=["web_search"],
                max_iterations=10
            ),
            
            "file_manager": AgentConfig(
                name="file_manager",
                agent_type=AgentType.FILE_MANAGER,
                llm_node="chat",
                system_prompt="""You are Mortey's file management specialist, expert at organizing, manipulating, and managing files and directories.

Your capabilities include:
- Creating and organizing file structures
- Reading, writing, and editing files
- Searching through file contents
- Converting between file formats
- Creating backups and managing versions
- Organizing workspaces efficiently
- Creating project templates

When managing files:
1. Always confirm destructive operations before proceeding
2. Create backups when modifying important files
3. Use clear and descriptive file naming conventions
4. Organize files logically by type and purpose
5. Respect file permissions and security
6. Provide clear feedback on operations performed

You prioritize data safety and organization efficiency.""",
                tools_enabled=True,
                tool_names=["file_tools"],
                max_iterations=12
            ),

            "project_management": AgentConfig(
                name="project_management",
                agent_type=AgentType.PROJECT_MANAGEMENT,
                llm_node="chat",
                system_prompt="""You are Mortey's project management specialist, expert at working with Primavera P6 project data.

Your expertise includes:
- Project portfolio analysis and reporting
- Schedule analysis and critical path identification  
- Resource utilization and allocation analysis
- Activity tracking and progress monitoring
- Data extraction and reporting from P6 systems
- Project performance metrics and KPIs

You have access to Primavera P6 REST API tools to:
- Search and filter projects across the enterprise
- Analyze project schedules and critical paths
- Track activity status and progress
- Examine resource allocations and utilization
- Export project data for further analysis
- Generate comprehensive project reports

When helping with project management:

1. **Context Awareness**: Remember the current project context across conversation turns
2. **Natural Language**: Translate business questions into appropriate P6 API queries
3. **Comprehensive Analysis**: Provide insights beyond raw data, including trends and recommendations
4. **Security First**: Operate in read-only mode, require confirmation for any write operations
5. **Performance**: Use efficient queries with appropriate filtering and field selection
6. **Error Handling**: Gracefully handle P6 connectivity issues and provide alternatives

Always explain your analysis methodology and highlight any limitations in the data or query scope.
When users ask about projects, help them navigate from high-level portfolio questions down to specific activity details.""",
                tools_enabled=True,
                tool_names=["p6_tools"],
                max_iterations=12
            )
        }
    
    async def initialize_agents(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Initialize all agents with enhanced error handling and concurrency"""
        try:
            # Initialize tools first
            await self._initialize_tools()
            
            # Use TaskGroup for concurrent agent initialization (Python 3.13.4)
            async with TaskGroup() as tg:
                tasks = {
                    name: tg.create_task(self._create_agent_async(name, config))
                    for name, config in self._agent_configs.items()
                }
            
            # Collect results
            for name, task in tasks.items():
                try:
                    self.agents[name] = task.result()
                    logger.info(f"✅ Agent '{name}' initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize agent '{name}': {e}")
                    # Create fallback agent
                    self.agents[name] = await self._create_fallback_agent(name)
            
            logger.info(f"✅ Agent factory initialized with {len(self.agents)} agents")
            return self.agents
            
        except Exception as e:
            logger.error(f"❌ Agent factory initialization failed: {e}")
            # Create minimal fallback agents
            return await self._create_fallback_agents()
    
    async def _initialize_tools(self) -> None:
        """Initialize tools for agents with enhanced error handling"""
        try:
            # File management tools
            file_tools = self.file_tools.get_tools()
            self.tools["file_tools"] = file_tools
            logger.info(f"✅ File tools initialized: {len(file_tools)} tools")
            
            # Web search tools (if available)
            try:
                web_tools = await self._initialize_web_tools()
                self.tools["web_search"] = web_tools
                logger.info(f"✅ Web search tools initialized: {len(web_tools)} tools")
            except Exception as e:
                logger.warning(f"⚠️ Web search tools not available: {e}")
                self.tools["web_search"] = []

            try:
                from tools.p6_tools import p6_tools_manager

                self.tools["p6_tools"] = [] 
                
                # Try auth key first, then username/password
                p6_auth_key = os.getenv("P6_AUTH_KEY")
                p6_database = os.getenv("P6_DATABASE", "PMDB")
                
                if p6_auth_key:
                    await p6_tools_manager.initialize_with_auth_key(p6_auth_key, p6_database)
                else:
                    p6_username = os.getenv("P6_USERNAME")
                    p6_password = os.getenv("P6_PASSWORD")
                    
                    if p6_username and p6_password:
                        await p6_tools_manager.initialize(p6_username, p6_password, p6_database)
                    else:
                        logger.warning("⚠️ P6 credentials not found, P6 tools unavailable")
                        self.tools["p6_tools"] = []
                        return
                
                p6_tools = p6_tools_manager.get_tools()   # now always safe
                self.tools["p6_tools"] = p6_tools
                logger.info("✅ P6 tools initialised: %s tools", len(p6_tools))
            except Exception as e:
                logger.warning("⚠️ P6 not ready (%s) – using stub tools", e)
                self.tools["p6_tools"] = p6_tools_manager.get_stub_tools()
            
        except Exception as e:
            logger.error(f"❌ Tool initialization failed: {e}")
            self.tools = {"file_tools": [], "web_search": []}
    
    async def _initialize_web_tools(self) -> list[Any]:  # Python 3.13.4 syntax
        """Initialize web search tools if available"""
        try:
            import os
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning("TAVILY_API_KEY not found, web search unavailable")
                return []
            
            search_tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )
            
            return [search_tool]
            
        except ImportError:
            logger.warning("Tavily search not available - install langchain-community")
            return []
        except Exception as e:
            logger.error(f"Web tools initialization failed: {e}")
            return []
    
    async def _create_agent_async(self, name: str, agent_config: AgentConfig) -> Any:
        """Create individual agent with 2025 unified approach"""
        try:
            # ✅ 2025: Use LLM manager for ALL providers (no special cases)
            llm = await self._get_agent_llm_unified(agent_config.llm_node)
            
            # Create prompt template
            prompt = await self._create_agent_prompt(agent_config)
            
            # Get tools for this agent
            agent_tools = self._get_agent_tools(agent_config)
            
            if agent_config.tools_enabled and agent_tools:
                # ✅ 2025: Unified tool agent creation for ALL providers
                agent = create_tool_calling_agent(llm, agent_tools, prompt)
                
                executor = AgentExecutor(
                    agent=agent,
                    tools=agent_tools,
                    max_iterations=agent_config.max_iterations,
                    verbose=agent_config.verbose,
                    return_intermediate_steps=True
                )
                
                return self._wrap_agent_with_protection(executor, name)
            else:
                # Simple conversational agent
                chain = prompt | llm
                return self._wrap_agent_with_protection(chain, name)
                
        except Exception as e:
            logger.error(f"Error creating agent {name}: {e}")
            raise

    async def _get_agent_llm_unified(self, llm_node: str) -> Any:
        """Get LLM instance using unified 2025 approach"""
        try:
            # ✅ 2025: Use LLM manager for ALL providers (OpenAI, Anthropic, etc.)
            model = await llm_manager.get_model(llm_node)
            
            # Test the model
            test_response = await llm_manager.generate_for_node(llm_node, "test")
            logger.debug(f"✅ LLM model {llm_node} working")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to get LLM for node {llm_node}: {e}")
            raise


    async def _create_openai_tool_agent(self, llm, agent_tools, prompt, agent_config):
        """Create OpenAI-optimized tool agent with MODERN format"""
        try:
            # ✅ USE MODERN TOOL CALLING (works with current OpenAI API)
            agent = create_tool_calling_agent(llm, agent_tools, prompt)
            
            return AgentExecutor(
                agent=agent,
                tools=agent_tools,
                max_iterations=agent_config.max_iterations,
                verbose=agent_config.verbose,
                return_intermediate_steps=True
            )
        except ImportError:
            # Fallback to generic tool agent
            return await self._create_generic_tool_agent(llm, agent_tools, prompt, agent_config)

    async def _create_generic_tool_agent(self, llm, agent_tools, prompt, agent_config):
        """Create generic tool agent (works with all providers)"""
        try:
            # Test tool binding
            test_llm = llm.bind_tools(agent_tools[:1])
            
            # Create tool-calling agent
            agent = create_tool_calling_agent(llm, agent_tools, prompt)
            
            return AgentExecutor(
                agent=agent,
                tools=agent_tools,
                max_iterations=agent_config.max_iterations,
                early_stopping_method=agent_config.early_stopping_method,
                handle_parsing_errors=agent_config.handle_parsing_errors,
                verbose=agent_config.verbose,
                return_intermediate_steps=True
            )
            
        except Exception as tool_error:
            logger.warning(f"⚠️ Tool binding failed: {tool_error}")
            # Fallback to simple conversational chain
            chain = prompt | llm
            return chain

    async def _get_agent_llm(self, llm_node: str) -> Any:
        """Get LLM instance for agent with provider awareness"""
        try:
            # Get node configuration to determine provider
            node_config = config.llm_config['nodes'].get(llm_node, {})
            provider = node_config.get('provider', 'anthropic')
            
            # 🔥 MODULAR: Handle different providers
            if provider == 'openai':
                # Use OpenAI directly for better function calling
                return await self._get_openai_model(llm_node, node_config)
            elif provider == 'anthropic':
                # Use Anthropic through LLM manager
                return await self._get_anthropic_model(llm_node)
            else:
                # Fallback to LLM manager for other providers
                return await llm_manager.get_model(llm_node)
                
        except Exception as e:
            logger.error(f"Failed to get LLM for node {llm_node}: {e}")
            raise

    async def _create_agent_prompt(self, agent_config: AgentConfig) -> ChatPromptTemplate:
        """Create 2025-compatible prompt template for agent"""
        try:
            # Base system message
            system_message = agent_config.system_prompt
            
            # Add custom instructions if provided
            if agent_config.custom_instructions:
                system_message += f"\n\nAdditional Instructions:\n{agent_config.custom_instructions}"
            
            # ✅ 2025 MODERN PROMPT FORMAT for tool-calling agents
            if agent_config.tools_enabled and agent_config.tool_names:
                tool_info = self._generate_tool_info(agent_config.tool_names)
                if tool_info != "No tools available":
                    system_message += f"\n\nAvailable Tools:\n{tool_info}"
                    
                    # Modern tool-calling agent prompt (2025)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        ("placeholder", "{chat_history}"),
                        ("human", "{input}"),
                        ("placeholder", "{agent_scratchpad}")
                    ])
                else:
                    # Simple conversational agent prompt
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        ("placeholder", "{messages}")
                    ])
            else:
                # Simple conversational agent prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_message),
                    ("placeholder", "{messages}")
                ])
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating prompt for {agent_config.name}: {e}")
            raise
    
    def _generate_tool_info(self, tool_names: list[str]) -> str:  # Python 3.13.4 syntax
        """Generate tool information for prompt"""
        tool_descriptions = []
        
        for tool_name in tool_names:
            if tool_name in self.tools:
                tools = self.tools[tool_name]
                tool_descriptions.append(f"\n{tool_name.replace('_', ' ').title()} ({len(tools)} tools available)")
                
                # Add brief description of each tool
                for tool in tools[:3]:  # Limit to first 3 tools to avoid prompt bloat
                    if hasattr(tool, 'description'):
                        tool_descriptions.append(f"  - {tool.name}: {tool.description[:100]}...")
        
        return "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
    
    def _get_agent_tools(self, agent_config: AgentConfig) -> list[Any]:  # Python 3.13.4 syntax
        """Get tools for specific agent"""
        if not agent_config.tools_enabled:
            return []
        
        agent_tools = []
        for tool_name in agent_config.tool_names:
            if tool_name in self.tools:
                agent_tools.extend(self.tools[tool_name])
        
        return agent_tools
    
    def _wrap_agent_with_protection(self, agent: Any, name: str) -> Any:
        """Wrap agent with protection while preserving ainvoke method"""
        
        # Get agent config to determine if it's a tool agent
        agent_config = self._agent_configs.get(name)
        
        class ProtectedAgent:
            def __init__(self, agent, name, agent_config):
                self.agent = agent
                self.name = name
                self.agent_config = agent_config
            
            async def ainvoke(self, state, config=None):
                """Preserve ainvoke method with proper state conversion and response extraction"""
                try:
                    # Convert state format based on agent type
                    if self.agent_config and self.agent_config.tools_enabled:
                        # Tool agents need input format
                        converted_state = self._convert_state_for_tool_agent(state)
                    else:
                        # Chat agents use messages directly
                        converted_state = state
                    
                    result = await self.agent.ainvoke(converted_state, config)
                    
                    # 🔥 ENHANCED: Better tool agent response extraction
                    if isinstance(result, dict):
                        # Check for tool agent output format
                        if "output" in result:
                            content = result["output"]
                            from langchain_core.messages import AIMessage
                            return {"messages": [AIMessage(content=content)]}
                        # Check for existing messages
                        elif "messages" in result:
                            return result
                        # Extract from intermediate steps if available
                        elif "intermediate_steps" in result:
                            # Get the final response from intermediate steps
                            steps = result["intermediate_steps"]
                            if steps and len(steps) > 0:
                                final_step = steps[-1]
                                if isinstance(final_step, tuple) and len(final_step) > 1:
                                    content = str(final_step[1])  # Action result
                                else:
                                    content = str(final_step)
                                from langchain_core.messages import AIMessage
                                return {"messages": [AIMessage(content=content)]}
                        # Fallback: convert entire result
                        else:
                            from langchain_core.messages import AIMessage
                            return {"messages": [AIMessage(content=str(result))]}
                    elif hasattr(result, 'content'):
                        return {"messages": [result]}
                    else:
                        from langchain_core.messages import AIMessage
                        return {"messages": [AIMessage(content=str(result))]}
                        
                except Exception as e:
                    logger.error(f"❌ Agent {self.name} execution failed: {e}")
                    logger.error(f"❌ Result type: {type(result) if 'result' in locals() else 'undefined'}")
                    logger.error(f"❌ Result content: {str(result)[:200] if 'result' in locals() else 'undefined'}")
                    from langchain_core.messages import AIMessage
                    return {
                        "messages": [AIMessage(content="I encountered an issue. Please try rephrasing your request.")]
                    }
            
            def _convert_state_for_tool_agent(self, state):
                """Convert AssistantState to tool agent input format"""
                messages = state.get("messages", [])
                
                # Extract last human message for input
                last_human_input = "Hello"
                chat_history = []
                
                for msg in messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if getattr(msg, 'type', '') == 'human':
                            last_human_input = msg.content
                        else:
                            chat_history.append(msg)
                
                return {
                    "input": last_human_input,
                    "chat_history": chat_history[:-1] if len(chat_history) > 1 else [],
                }
            
            def __getattr__(self, attr_name):
                """Delegate other methods to the wrapped agent"""
                return getattr(self.agent, attr_name)
        
        return ProtectedAgent(agent, name, agent_config)


    async def _process_chat_agent(self, agent: Any, state: AssistantState) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Process chat agent with simple conversation flow"""
        messages = state.get("messages", [])
        
        # Use circuit breaker protection
        result = await global_circuit_breaker.call_with_circuit_breaker(
            "chat_agent",
            agent.ainvoke,
            {"messages": messages}
        )
        
        if isinstance(result, AIMessage):
            return {"messages": [result]}
        else:
            return {"messages": [AIMessage(content=str(result))]}

    async def _process_tool_agent(self, agent: Any, state: AssistantState, context: str) -> dict[str, Any]:
        """Process tool-enabled agent with enhanced error handling"""
        messages = state.get("messages", [])
        
        # Extract the last human message for input
        last_human_message = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        
        if not last_human_message:
            return {
                "messages": [AIMessage(content="I need a message to respond to.")]
            }
        
        agent_input = {
            "input": last_human_message.content,
            "chat_history": messages[:-1] if len(messages) > 1 else []  # All except last message
        }
        
        try:
            # Use circuit breaker protection
            result = await global_circuit_breaker.call_with_circuit_breaker(
                f"{context}_agent",
                agent.ainvoke,
                agent_input
            )
            
            # Extract output properly
            if isinstance(result, dict) and "output" in result:
                output = result["output"]
                return {"messages": [AIMessage(content=output)]}
            else:
                return {"messages": [AIMessage(content=str(result))]}
                
        except Exception as e:
            logger.error(f"Tool agent {context} execution failed: {e}")
            return {
                "messages": [AIMessage(content=f"I encountered an issue while processing your request. Error: {str(e)}")]
            }
    
    async def _process_generic_agent(self, agent: Any, state: AssistantState) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Process generic agent with fallback handling"""
        try:
            result = await agent.ainvoke(state)
            
            if isinstance(result, dict):
                return result
            elif isinstance(result, AIMessage):
                return {"messages": [result]}
            else:
                return {"messages": [AIMessage(content=str(result))]}
                
        except Exception as e:
            logger.error(f"Generic agent processing failed: {e}")
            return {
                "messages": [AIMessage(content="I encountered an issue processing your request.")]
            }
    
    async def _create_fallback_agent(self, name: str) -> Any:
        """Create fallback agent with proper ainvoke method"""
        class FallbackAgent:
            def __init__(self, name):
                self.name = name
            
            async def ainvoke(self, state, config=None):
                return {
                    "messages": [AIMessage(content=f"The {self.name} agent is currently unavailable. Please try again later.")]
                }
        
        return FallbackAgent(name)

    async def _create_fallback_agents(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Create minimal fallback agents when initialization fails"""
        fallback_agents = {}
        
        for name in self._agent_configs.keys():
            fallback_agents[name] = await self._create_fallback_agent(name)
        
        logger.warning("⚠️ Using fallback agents due to initialization failure")
        return fallback_agents
    
    def get_all_tools(self) -> list[Any]:  # Python 3.13.4 syntax
        """Get all available tools across all agents"""
        all_tools = []
        for tool_list in self.tools.values():
            all_tools.extend(tool_list)
        return all_tools
    
    def get_agent_info(self, agent_name: str) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get information about a specific agent"""
        if agent_name not in self._agent_configs:
            return {"error": f"Agent {agent_name} not found"}
        
        config = self._agent_configs[agent_name]
        return {
            "name": config.name,
            "type": config.agent_type.value,
            "llm_node": config.llm_node,
            "tools_enabled": config.tools_enabled,
            "tool_count": len(self._get_agent_tools(config)),
            "max_iterations": config.max_iterations,
            "initialized": agent_name in self.agents
        }
    
    def get_all_agents_info(self) -> dict[str, dict[str, Any]]:  # Python 3.13.4 syntax
        """Get information about all agents"""
        return {
            name: self.get_agent_info(name)
            for name in self._agent_configs.keys()
        }

    # agents/agents.py - FIXED agent health checks
    async def health_check_agents(self) -> dict[str, bool]:
        """Optimized agent health check with intelligent caching"""
        
        # ✅ CACHE: Don't check agents that were recently healthy
        current_time = time.time()
        if not hasattr(self, '_agent_health_cache'):
            self._agent_health_cache = {}
            self._last_agent_check = {}
        
        results = {}
        agents_to_check = []
        
        # Check cache first (5 minute cache for healthy agents)
        for agent_name in self.agents.keys():
            last_check = self._last_agent_check.get(agent_name, 0)
            cached_result = self._agent_health_cache.get(agent_name)
            
            if cached_result is True and (current_time - last_check) < 300:  # 5 min cache
                results[agent_name] = True
                logger.debug(f"✅ Agent {agent_name} health: cached (healthy)")
            else:
                agents_to_check.append(agent_name)
        
        # Only check agents that need checking
        if not agents_to_check:
            return results
        
        logger.debug(f"🔍 Checking health for {len(agents_to_check)} agents...")
        
        # ✅ FIXED: Proper TaskGroup usage and exception handling
        task_exceptions = None
        
        try:
            async with asyncio.TaskGroup() as tg:
                # Create tasks with proper mapping
                tasks = {
                    agent_name: tg.create_task(self._ultra_lightweight_agent_check(agent_name))
                    for agent_name in agents_to_check
                }
            
            # ✅ FIXED: Tasks are completed when TaskGroup exits successfully
            # Collect results and update cache
            for agent_name, task in tasks.items():
                result = task.result()  # This is safe after TaskGroup exits
                results[agent_name] = result
                self._agent_health_cache[agent_name] = result
                self._last_agent_check[agent_name] = current_time
            
            return results
            
        except* Exception as eg:
            # ✅ FIXED: Store exceptions and handle outside the except* block
            task_exceptions = eg
        
        # ✅ FIXED: Handle exceptions outside except* block
        if task_exceptions:
            logger.error(f"❌ Agent health check failed: {len(task_exceptions.exceptions)} exceptions")
            # Mark unchecked agents as unhealthy
            for agent_name in agents_to_check:
                if agent_name not in results:
                    results[agent_name] = False
                    self._agent_health_cache[agent_name] = False
                    self._last_agent_check[agent_name] = current_time
        
        return results

    async def _ultra_lightweight_agent_check(self, agent_name: str) -> bool:
        """Ultra-lightweight agent check - no LLM calls"""
        try:
            if agent_name not in self.agents:
                return False
            
            agent = self.agents[agent_name]
            
            # ✅ LIGHTWEIGHT: Just check if agent has required structure
            has_ainvoke = hasattr(agent, 'ainvoke')
            has_config = agent_name in getattr(self, '_agent_configs', {})
            
            if not (has_ainvoke and has_config):
                return False
            
            # ✅ NO LLM CALLS: Skip expensive testing
            logger.debug(f"✅ Agent {agent_name}: structure check passed")
            return True
            
        except Exception as e:
            logger.debug(f"Agent {agent_name} lightweight check failed: {e}")
            return False

    async def _safe_health_check_agent(self, agent_name: str) -> bool:
        """Safe wrapper for agent health check that never raises exceptions"""
        try:
            return await self._health_check_agent(agent_name)
        except Exception as e:
            logger.debug(f"Agent health check failed for {agent_name}: {e}")
            return False

    async def _health_check_agent(self, agent_name: str) -> bool:
        """Enhanced health check with tool agent handling"""
        try:
            if agent_name not in self.agents:
                return False
            
            agent_config = self._agent_configs.get(agent_name)
            
            # 🔥 NEW: Special handling for tool agents
            if agent_config and agent_config.tools_enabled:
                return await self._health_check_tool_agent(agent_name, agent_config)
            else:
                return await self._health_check_simple_agent(agent_name)
                
        except Exception as e:
            logger.debug(f"Health check failed for {agent_name}: {e}")
            return False

    async def _health_check_tool_agent(self, agent_name: str, agent_config: AgentConfig) -> bool:
        """Simplified health check for tool agents"""
        try:
            # 🔥 STRATEGY: Just check if agent exists and has proper structure
            agent = self.agents[agent_name]
            
            # Validate agent has required methods
            if not hasattr(agent, 'ainvoke'):
                return False
            
            # 🔥 OPTIONAL: Quick LLM connectivity check instead of full agent test
            llm_node = agent_config.llm_node
            try:
                # Test the underlying LLM directly (faster)
                await asyncio.wait_for(
                    llm_manager.generate_for_node(llm_node, "test", override_max_tokens=5),
                    timeout=15.0
                )
                logger.debug(f"✅ Tool agent {agent_name} LLM connectivity verified")
                return True
            except Exception as e:
                logger.debug(f"⚠️ Tool agent {agent_name} LLM test failed: {e}")
                return False
                
        except Exception as e:
            logger.debug(f"Tool agent health check failed for {agent_name}: {e}")
            return False

    async def _health_check_simple_agent(self, agent_name: str) -> bool:
        """Health check for simple chat agents"""
        try:
            test_state = {
                "messages": [HumanMessage(content="hi")],
                "session_id": "health_check",
                "user_id": "system",
                "current_agent": agent_name
            }
            
            result = await asyncio.shield(
                asyncio.wait_for(
                    self.agents[agent_name].ainvoke(test_state),
                    timeout=10.0
                )
            )
            
            return isinstance(result, dict) and "messages" in result
            
        except Exception as e:
            logger.debug(f"Simple agent health check failed for {agent_name}: {e}")
            return False
    
    def add_custom_agent(self, name: str, agent_config: AgentConfig, agent_instance: Any) -> None:
        """Add custom agent to the factory"""
        self._agent_configs[name] = agent_config
        self.agents[name] = agent_instance
        logger.info(f"✅ Custom agent '{name}' added to factory")
    
    def remove_agent(self, name: str) -> bool:
        """Remove agent from factory"""
        if name in self.agents:
            del self.agents[name]
            if name in self._agent_configs:
                del self._agent_configs[name]
            logger.info(f"✅ Agent '{name}' removed from factory")
            return True
        return False
    
    def get_factory_statistics(self) -> dict[str, Any]:  # Python 3.13.4 syntax
        """Get comprehensive factory statistics"""
        total_tools = sum(len(tools) for tools in self.tools.values())
        
        return {
            "total_agents": len(self.agents),
            "total_tools": total_tools,
            "agents_by_type": {
                agent_type.value: sum(
                    1 for config in self._agent_configs.values()
                    if config.agent_type == agent_type
                )
                for agent_type in AgentType
            },
            "tools_by_category": {
                category: len(tools)
                for category, tools in self.tools.items()
            },
            "agent_names": list(self.agents.keys()),
            "initialization_complete": len(self.agents) == len(self._agent_configs)
        }

# Global agent factory instance
agent_factory = AgentFactory()

# Convenience functions
async def initialize_agents() -> dict[str, Any]:  # Python 3.13.4 syntax
    """Initialize all agents"""
    return await agent_factory.initialize_agents()

def get_agent(name: str) -> Optional[Any]:
    """Get specific agent"""
    return agent_factory.agents.get(name)

def get_all_agents() -> dict[str, Any]:  # Python 3.13.4 syntax
    """Get all agents"""
    return agent_factory.agents

def get_all_tools() -> list[Any]:  # Python 3.13.4 syntax
    """Get all tools"""
    return agent_factory.get_all_tools()

async def health_check_all_agents() -> dict[str, bool]:  # Python 3.13.4 syntax
    """Health check all agents"""
    return await agent_factory.health_check_agents()
