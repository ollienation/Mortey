# agents/agents.py - ✅ ENHANCED WITH DEPENDENCY INJECTION PATTERNS

# ✅ STANDARD LIBRARY IMPORTS
import os
import logging
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ✅ THIRD-PARTY IMPORTS
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent

# ✅ LOCAL IMPORTS (absolute paths)
from config.llm_manager import llm_manager
from tools.file_tools import FileSystemTools

# ✅ OPTIONAL THIRD-PARTY IMPORTS WITH ERROR HANDLING
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None

logger = logging.getLogger("agents")

@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for tool providers to ensure consistent interface"""
    
    def get_tools(self) -> List[BaseTool]:
        """Return list of tools provided by this provider"""
        ...
    
    def get_tool_metadata(self) -> Dict[str, Any]:
        """Return metadata about the tools"""
        ...

@dataclass
class AgentConfig:
    """Configuration for agent creation with dependency injection"""
    agent_name: str
    model_node: str
    tool_providers: List[str] = field(default_factory=list)
    custom_tools: List[BaseTool] = field(default_factory=list)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True

class ToolRegistry:
    """✅ ENHANCED: Central registry for tool providers with dependency injection"""
    
    def __init__(self):
        self._providers: Dict[str, ToolProvider] = {}
        self._tool_cache: Dict[str, List[BaseTool]] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        
    def register_provider(self, name: str, provider: ToolProvider) -> None:
        """Register a tool provider with the registry"""
        try:
            # Validate provider implements the protocol
            if not isinstance(provider, ToolProvider):
                raise ValueError(f"Provider {name} must implement ToolProvider protocol")
            
            self._providers[name] = provider
            # Clear caches when new provider is registered
            self._tool_cache.pop(name, None)
            self._metadata_cache.pop(name, None)
            
            logger.info(f"✅ Registered tool provider: {name}")
        except Exception as e:
            logger.error(f"❌ Failed to register tool provider {name}: {e}")
            raise
    
    def get_tools(self, provider_names: List[str]) -> List[BaseTool]:
        """Get tools from specified providers"""
        all_tools = []
        
        for provider_name in provider_names:
            try:
                # Check cache first
                if provider_name in self._tool_cache:
                    cached_tools = self._tool_cache[provider_name]
                    all_tools.extend(cached_tools)
                    logger.debug(f"Using cached tools for provider: {provider_name}")
                    continue
                
                # Get tools from provider
                if provider_name not in self._providers:
                    logger.warning(f"⚠️ Tool provider '{provider_name}' not found")
                    continue
                
                provider = self._providers[provider_name]
                provider_tools = provider.get_tools()
                
                # Cache the tools
                self._tool_cache[provider_name] = provider_tools
                all_tools.extend(provider_tools)
                
                logger.debug(f"Loaded {len(provider_tools)} tools from provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"❌ Error getting tools from provider {provider_name}: {e}")
                continue
        
        # Deduplicate tools by name (keep first occurrence)
        seen_names = set()
        deduplicated_tools = []
        
        for tool in all_tools:
            if tool.name not in seen_names:
                deduplicated_tools.append(tool)
                seen_names.add(tool.name)
            else:
                logger.debug(f"Skipping duplicate tool: {tool.name}")
        
        logger.info(f"✅ Collected {len(deduplicated_tools)} unique tools from {len(provider_names)} providers")
        return deduplicated_tools
    
    def get_provider_metadata(self, provider_name: str) -> Dict[str, Any]:
        """Get metadata for a specific provider"""
        if provider_name not in self._providers:
            return {}
        
        # Check cache first
        if provider_name in self._metadata_cache:
            return self._metadata_cache[provider_name]
        
        try:
            provider = self._providers[provider_name]
            metadata = provider.get_tool_metadata()
            self._metadata_cache[provider_name] = metadata
            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata for provider {provider_name}: {e}")
            return {}
    
    def list_providers(self) -> List[str]:
        """List all registered provider names"""
        return list(self._providers.keys())
    
    def clear_cache(self) -> None:
        """Clear all cached tools and metadata"""
        self._tool_cache.clear()
        self._metadata_cache.clear()
        logger.info("🧹 Tool registry cache cleared")

class ChatToolProvider:
    """✅ ENHANCED: Tool provider for chat agent with dependency injection"""
    
    def __init__(self, file_tools: FileSystemTools):
        self.file_tools = file_tools
    
    def get_tools(self) -> List[BaseTool]:
        """Get chat-specific tools"""
        @tool
        def list_workspace_files() -> str:
            """List all files in the workspace directory."""
            try:
                return self.file_tools.toolkit.run({"tool_input": "", "tool_name": "list_directory"})
            except Exception as e:
                logger.error(f"Error listing workspace files: {e}")
                return f"Error accessing workspace files: {str(e)}"
        
        @tool
        def get_workspace_summary() -> str:
            """Get a summary of the current workspace contents and structure."""
            try:
                # Get file listing
                files_result = self.file_tools.toolkit.run({"tool_input": "", "tool_name": "list_directory"})
                
                # Count different file types
                import os
                workspace_path = self.file_tools.workspace_dir
                file_types = {}
                total_files = 0
                
                for root, dirs, files in os.walk(workspace_path):
                    for file in files:
                        total_files += 1
                        ext = os.path.splitext(file)[1].lower() or "no_extension"
                        file_types[ext] = file_types.get(ext, 0) + 1
                
                summary = f"Workspace Summary:\n"
                summary += f"Total files: {total_files}\n"
                summary += f"File types: {dict(sorted(file_types.items()))}\n\n"
                summary += f"Directory listing:\n{files_result}"
                
                return summary
            except Exception as e:
                logger.error(f"Error getting workspace summary: {e}")
                return f"Error getting workspace summary: {str(e)}"
        
        return [list_workspace_files, get_workspace_summary]
    
    def get_tool_metadata(self) -> Dict[str, Any]:
        """Get metadata about chat tools"""
        return {
            "provider_name": "chat",
            "tool_count": 2,
            "description": "Basic workspace browsing tools for chat interactions",
            "capabilities": ["file_listing", "workspace_summary"],
            "dependencies": ["file_tools"]
        }

class CoderToolProvider:
    """✅ ENHANCED: Tool provider for coder agent"""
    
    def __init__(self, file_tools: FileSystemTools):
        self.file_tools = file_tools
    
    def get_tools(self) -> List[BaseTool]:
        """Get coding-specific tools"""
        try:
            base_tools = self.file_tools.get_tools()
            
            # Add coding-specific enhancements
            @tool
            def analyze_code_structure(file_path: str) -> str:
                """Analyze the structure of a code file (functions, classes, imports)."""
                try:
                    import ast
                    import os
                    
                    full_path = os.path.join(self.file_tools.workspace_dir, file_path)
                    if not os.path.exists(full_path):
                        return f"File not found: {file_path}"
                    
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    try:
                        tree = ast.parse(content)
                    except SyntaxError as e:
                        return f"Syntax error in {file_path}: {e}"
                    
                    analysis = f"Code structure analysis for {file_path}:\n\n"
                    
                    # Extract imports
                    imports = []
                    functions = []
                    classes = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(f"import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for alias in node.names:
                                imports.append(f"from {module} import {alias.name}")
                        elif isinstance(node, ast.FunctionDef):
                            args = [arg.arg for arg in node.args.args]
                            functions.append(f"{node.name}({', '.join(args)})")
                        elif isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                    
                    if imports:
                        analysis += f"Imports:\n" + "\n".join(f"  - {imp}" for imp in imports) + "\n\n"
                    
                    if classes:
                        analysis += f"Classes:\n" + "\n".join(f"  - {cls}" for cls in classes) + "\n\n"
                    
                    if functions:
                        analysis += f"Functions:\n" + "\n".join(f"  - {func}" for func in functions) + "\n\n"
                    
                    analysis += f"Total lines: {len(content.splitlines())}\n"
                    analysis += f"Total characters: {len(content)}"
                    
                    return analysis
                    
                except Exception as e:
                    return f"Error analyzing code structure: {str(e)}"
            
            return base_tools + [analyze_code_structure]
            
        except Exception as e:
            logger.error(f"Error getting coder tools: {e}")
            return []
    
    def get_tool_metadata(self) -> Dict[str, Any]:
        """Get metadata about coder tools"""
        base_count = len(self.file_tools.get_tools()) if self.file_tools else 0
        return {
            "provider_name": "coder",
            "tool_count": base_count + 1,
            "description": "Comprehensive file management and code analysis tools",
            "capabilities": ["file_operations", "code_analysis", "project_creation"],
            "dependencies": ["file_tools", "ast"]
        }

class WebToolProvider:
    """✅ ENHANCED: Tool provider for web agent"""
    
    def __init__(self):
        pass
    
    def get_tools(self) -> List[BaseTool]:
        """Get web-specific tools"""
        @tool
        def search_web(query: str, max_results: int = 5) -> str:
            """Search the web for information using Tavily."""
            if not TAVILY_AVAILABLE:
                return "Tavily search is not available. Please install the tavily-python package."
            
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
            
            try:
                client = TavilyClient(api_key=api_key)
                response = client.search(query=query, search_depth="advanced", max_results=max_results)
                
                # Format the results nicely
                results = response.get('results', [])
                if not results:
                    return "No search results found."
                
                formatted_results = f"Search results for '{query}':\n\n"
                for i, result in enumerate(results[:max_results], 1):
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    content = result.get('content', 'No content')[:200] + "..."
                    
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   URL: {url}\n"
                    formatted_results += f"   Summary: {content}\n\n"
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"Tavily search error: {e}")
                return f"Search error: {str(e)}"
        
        @tool
        def check_web_connectivity() -> str:
            """Check if web connectivity is available and working."""
            try:
                import urllib.request
                import socket
                
                # Test connectivity to a few reliable endpoints
                test_urls = [
                    "https://www.google.com",
                    "https://www.github.com",
                    "https://httpbin.org/status/200"
                ]
                
                results = []
                for url in test_urls:
                    try:
                        response = urllib.request.urlopen(url, timeout=5)
                        if response.getcode() == 200:
                            results.append(f"✅ {url} - OK")
                        else:
                            results.append(f"⚠️ {url} - Status {response.getcode()}")
                    except Exception as e:
                        results.append(f"❌ {url} - Error: {str(e)}")
                
                connectivity_report = "Web Connectivity Check:\n\n" + "\n".join(results)
                
                # Add Tavily API status
                if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
                    connectivity_report += "\n\n✅ Tavily API key configured"
                else:
                    connectivity_report += "\n\n❌ Tavily API not available or not configured"
                
                return connectivity_report
                
            except Exception as e:
                return f"Error checking web connectivity: {str(e)}"
        
        return [search_web, check_web_connectivity]
    
    def get_tool_metadata(self) -> Dict[str, Any]:
        """Get metadata about web tools"""
        return {
            "provider_name": "web",
            "tool_count": 2,
            "description": "Web search and connectivity tools",
            "capabilities": ["web_search", "connectivity_check"],
            "dependencies": ["tavily", "urllib"],
            "external_apis": ["tavily"]
        }

class AgentFactory:
    """
    ✅ ENHANCED: Agent factory with dependency injection and configuration-driven creation
    """
    
    def __init__(self, file_tools: Optional[FileSystemTools] = None):
        self.file_tools = file_tools or FileSystemTools()
        self.tool_registry = ToolRegistry()
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # Initialize and register tool providers
        self._initialize_tool_providers()
        self._setup_default_agent_configs()
    
    def _initialize_tool_providers(self):
        """Initialize and register all tool providers"""
        try:
            # Register tool providers with dependency injection
            self.tool_registry.register_provider("chat", ChatToolProvider(self.file_tools))
            self.tool_registry.register_provider("coder", CoderToolProvider(self.file_tools))
            self.tool_registry.register_provider("web", WebToolProvider())
            
            logger.info("✅ All tool providers initialized and registered")
        except Exception as e:
            logger.error(f"❌ Error initializing tool providers: {e}")
            raise
    
    def _setup_default_agent_configs(self):
        """Setup default configurations for agents"""
        self.agent_configs = {
            "chat": AgentConfig(
                agent_name="chat",
                model_node="chat",
                tool_providers=["chat"],
                description="General conversation and basic workspace browsing"
            ),
            "coder": AgentConfig(
                agent_name="coder",
                model_node="coder",
                tool_providers=["coder"],
                description="Code generation, analysis, and file management"
            ),
            "web": AgentConfig(
                agent_name="web",
                model_node="web",
                tool_providers=["web"],
                description="Web search and online information gathering"
            )
        }
    
    def _get_model(self, node_name: str):
        """Get model using the LLM manager with enhanced error handling"""
        try:
            return llm_manager._get_model(node_name)
        except Exception as e:
            logger.error(f"Failed to get model for node {node_name}: {e}")
            # Try fallback model if available
            try:
                fallback_node = "chat"  # Default fallback
                if node_name != fallback_node:
                    logger.warning(f"Attempting fallback model for {node_name} -> {fallback_node}")
                    return llm_manager._get_model(fallback_node)
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
            raise
    
    def create_agent(self, agent_name: str, config: Optional[AgentConfig] = None) -> Any:
        """✅ ENHANCED: Create agent using dependency injection and configuration"""
        try:
            # Use provided config or get default
            agent_config = config or self.agent_configs.get(agent_name)
            if not agent_config:
                raise ValueError(f"No configuration found for agent: {agent_name}")
            
            if not agent_config.enabled:
                raise ValueError(f"Agent {agent_name} is disabled")
            
            # Get model for agent
            model = self._get_model(agent_config.model_node)
            
            # Get tools from configured providers
            tools = self.tool_registry.get_tools(agent_config.tool_providers)
            
            # Add any custom tools
            if agent_config.custom_tools:
                tools.extend(agent_config.custom_tools)
                logger.debug(f"Added {len(agent_config.custom_tools)} custom tools to {agent_name}")
            
            # Create the agent
            agent = create_react_agent(
                model=model, 
                tools=tools,
                **agent_config.agent_kwargs
            )
            
            logger.info(f"✅ Created {agent_name} agent with {len(tools)} tools")
            return agent
            
        except Exception as e:
            logger.error(f"❌ Failed to create {agent_name} agent: {e}")
            raise
    
    def create_chat_agent(self):
        """Create chat agent using dependency injection"""
        return self.create_agent("chat")
    
    def create_coder_agent(self):
        """Create coder agent using dependency injection"""
        return self.create_agent("coder")
    
    def create_web_agent(self):
        """Create web agent using dependency injection"""
        return self.create_agent("web")
    
    def get_all_tools(self) -> List[BaseTool]:
        """✅ ENHANCED: Collect all tools from all providers"""
        try:
            all_provider_names = self.tool_registry.list_providers()
            return self.tool_registry.get_tools(all_provider_names)
        except Exception as e:
            logger.error(f"Failed to get all tools: {e}")
            return []
    
    def customize_agent_config(self, agent_name: str, **kwargs) -> None:
        """Customize agent configuration"""
        if agent_name not in self.agent_configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        config = self.agent_configs[agent_name]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Updated {agent_name} config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter for {agent_name}: {key}")
    
    def add_custom_tool_to_agent(self, agent_name: str, tool: BaseTool) -> None:
        """Add a custom tool to a specific agent"""
        if agent_name not in self.agent_configs:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        self.agent_configs[agent_name].custom_tools.append(tool)
        logger.info(f"Added custom tool '{tool.name}' to {agent_name} agent")
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive information about an agent"""
        if agent_name not in self.agent_configs:
            return {"error": f"Unknown agent: {agent_name}"}
        
        config = self.agent_configs[agent_name]
        
        # Get tool information
        tools = self.tool_registry.get_tools(config.tool_providers)
        tool_info = []
        
        for provider_name in config.tool_providers:
            metadata = self.tool_registry.get_provider_metadata(provider_name)
            tool_info.append(metadata)
        
        return {
            "agent_name": agent_name,
            "model_node": config.model_node,
            "description": config.description,
            "enabled": config.enabled,
            "tool_providers": config.tool_providers,
            "total_tools": len(tools),
            "custom_tools": len(config.custom_tools),
            "tool_provider_info": tool_info,
            "agent_kwargs": config.agent_kwargs
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "tool_registry": {
                "registered_providers": self.tool_registry.list_providers(),
                "cache_size": len(self.tool_registry._tool_cache)
            },
            "agents": {
                name: self.get_agent_info(name) 
                for name in self.agent_configs.keys()
            },
            "file_tools_workspace": self.file_tools.workspace_dir,
            "external_dependencies": {
                "tavily_available": TAVILY_AVAILABLE,
                "tavily_configured": bool(os.getenv("TAVILY_API_KEY"))
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check tool providers
        for provider_name in self.tool_registry.list_providers():
            try:
                tools = self.tool_registry.get_tools([provider_name])
                health_status["components"][f"tool_provider_{provider_name}"] = {
                    "status": "healthy",
                    "tool_count": len(tools)
                }
            except Exception as e:
                health_status["components"][f"tool_provider_{provider_name}"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        # Check agent configurations
        for agent_name, config in self.agent_configs.items():
            try:
                if config.enabled:
                    # Try to get model (without creating full agent)
                    model = self._get_model(config.model_node)
                    health_status["components"][f"agent_{agent_name}"] = {
                        "status": "healthy",
                        "model_node": config.model_node,
                        "enabled": True
                    }
                else:
                    health_status["components"][f"agent_{agent_name}"] = {
                        "status": "disabled",
                        "enabled": False
                    }
            except Exception as e:
                health_status["components"][f"agent_{agent_name}"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status
