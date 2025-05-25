import os
import asyncio
from typing import Dict, Any, List
from anthropic import Anthropic
from tavily import TavilyClient

from tools.file_tools import FileSystemTools

class CoderAgent:
    """Enhanced code generation agent with file system tools and web search"""
    
    def __init__(self, llm_service=None, workspace_dir: str = None):
        self.llm_service = llm_service
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # SIMPLIFIED: Use the modern file tools
        self.file_tools = FileSystemTools(workspace_dir)
        self.tools = self.file_tools.get_tools()
        
        # Initialize Tavily for web search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily_client = None
            print("⚠️ TAVILY_API_KEY not found - web search disabled for coder")
    
    async def generate_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code with file system access and web search"""
        
        user_input = state.get('user_input', '')
        
        # Update thinking state
        state['thinking_state'] = {
            'active_agent': 'CODER',
            'current_task': 'Analyzing coding request',
            'progress': 0.1,
            'details': f'Processing: {user_input}'
        }
        
        try:
            # Step 1: Determine what tools are needed
            needs_search = await self._should_search_web(user_input)
            needs_files = await self._should_use_files(user_input)
            
            # Step 2: Gather context
            search_context = ""
            file_context = ""
            
            if needs_search and self.tavily_client:
                state['thinking_state']['current_task'] = 'Searching for coding information'
                state['thinking_state']['progress'] = 0.3
                search_context = await self._search_coding_info(user_input)
            
            if needs_files:
                state['thinking_state']['current_task'] = 'Analyzing workspace files'
                state['thinking_state']['progress'] = 0.5
                file_context = await self._analyze_workspace(user_input)
            
            # Step 3: Generate response with all available tools
            state['thinking_state']['current_task'] = 'Generating code solution'
            state['thinking_state']['progress'] = 0.7
            
            response = await self._generate_code_with_tools(user_input, search_context, file_context)
            
            return {
                **state,
                'output_content': response,
                'output_type': 'code',
                'code_context': {
                    'language': 'python',
                    'request': user_input,
                    'web_search_used': bool(search_context),
                    'file_tools_used': bool(file_context),
                    'generated_at': 'now'
                },
                'thinking_state': {
                    'active_agent': 'CODER',
                    'current_task': 'Code generation complete',
                    'progress': 1.0,
                    'details': 'Ready for verification'
                }
            }
            
        except Exception as e:
            return {
                **state,
                'output_content': f"Sorry, I encountered an error: {str(e)}",
                'output_type': 'error'
            }
    
    # ADD BACK: Your existing methods from current coder_agent.py
    async def _should_search_web(self, user_input: str) -> bool:
        """Determine if web search would help with this coding request"""
        # Keywords that suggest web search would be helpful
        search_indicators = [
            'latest', 'newest', 'current', 'recent', 'updated',
            'best practices', 'how to', 'tutorial', 'example',
            'documentation', 'api', 'library', 'framework',
            'error', 'fix', 'solve', 'troubleshoot',
            'compare', 'difference', 'vs', 'alternative'
        ]
        
        # Check for specific technologies that might need current info
        tech_keywords = [
            'react', 'vue', 'angular', 'nextjs', 'svelte',
            'tensorflow', 'pytorch', 'scikit', 'pandas',
            'fastapi', 'django', 'flask', 'express',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp'
        ]
        
        user_lower = user_input.lower()
        
        # Search if user mentions search indicators or specific technologies
        has_search_indicator = any(indicator in user_lower for indicator in search_indicators)
        has_tech_keyword = any(tech in user_lower for tech in tech_keywords)
        
        # Also use Claude to make intelligent decision
        if not (has_search_indicator or has_tech_keyword):
            try:
                decision_prompt = f"""
                Analyze this coding request and determine if web search would be helpful:
                
                Request: {user_input}
                
                Web search would be helpful if the request involves:
                - Current best practices or recent changes
                - Specific library/framework documentation
                - Error troubleshooting
                - Comparing different approaches
                - Latest API usage examples
                
                Respond with only "YES" or "NO"
                """
                
                message = await asyncio.to_thread(
                    self.anthropic.messages.create,
                    model="claude-sonnet-4-20250514",
                    max_tokens=10,
                    messages=[{"role": "user", "content": decision_prompt}]
                )
                
                decision = message.content[0].text.strip().upper()
                return decision == "YES"
                
            except Exception as e:
                print(f"❌ Search decision error: {e}")
                return False
        
        return has_search_indicator or has_tech_keyword

    async def _search_coding_info(self, user_input: str) -> str:
        """Search for coding-related information using Tavily"""
        
        try:
            # Create targeted search query for coding
            search_query = await self._create_search_query(user_input)
            
            print(f"🔍 Searching for coding info: {search_query}")
            
            # Use Tavily to search for coding information
            search_response = await asyncio.to_thread(
                self.tavily_client.search,
                query=search_query,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                max_results=3  # Limit for focused results
            )
            
            # Process and format search results
            context = ""
            
            # Add AI-generated answer if available
            if search_response.get('answer'):
                context += f"## Search Summary\n{search_response['answer']}\n\n"
            
            # Add relevant search results
            if search_response.get('results'):
                context += "## Relevant Resources\n"
                for i, result in enumerate(search_response['results'][:2], 1):
                    context += f"{i}. **{result.get('title', 'Unknown')}**\n"
                    context += f"   URL: {result.get('url', '')}\n"
                    if result.get('content'):
                        # Limit content length
                        content = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                        context += f"   Content: {content}\n\n"
            
            return context
            
        except Exception as e:
            print(f"❌ Coding search error: {e}")
            return ""
    
    async def _create_search_query(self, user_input: str) -> str:
        """Create an optimized search query for coding information"""
        
        try:
            query_prompt = f"""
            Create an optimized web search query for this coding request:
            
            Request: {user_input}
            
            Create a search query that would find:
            - Documentation and examples
            - Best practices and tutorials
            - Recent solutions and approaches
            
            Return only the search query, optimized for finding coding information.
            """
            
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": query_prompt}]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            print(f"❌ Query creation error: {e}")
            # Fallback to original request with coding keywords
            return f"{user_input} programming tutorial example"
    
    # NEW: File system methods
    async def _should_use_files(self, user_input: str) -> bool:
        """Determine if file operations are needed"""
        file_indicators = [
            'create project', 'new project', 'analyze code', 'read file',
            'save to file', 'write file', 'project structure',
            'list files', 'workspace', 'directory'
        ]
        
        return any(indicator in user_input.lower() for indicator in file_indicators)
    
    async def _analyze_workspace(self, user_input: str) -> str:
        """Analyze workspace for relevant files"""
        try:
            # List files in workspace
            list_tool = next(tool for tool in self.tools if tool.name == "list_directory")
            file_list = list_tool.invoke({})
            
            context = f"**Workspace Contents:**\n{file_list}\n\n"
            
            return context
            
        except Exception as e:
            return f"Error analyzing workspace: {str(e)}"
    
    async def _generate_code_with_tools(self, user_input: str, search_context: str, file_context: str) -> str:
        """Generate code response with tool access and web search"""
        
        # Build tool descriptions for the prompt
        tool_descriptions = "\n".join([
            f"- **{tool.name}**: {tool.description}" 
            for tool in self.tools
        ])
        
        prompt = f"""
        You are an expert programmer with access to file system tools and web search. Help with this coding request:
        
        User Request: {user_input}
        
        Available File Tools:
        {tool_descriptions}
        
        Workspace Context:
        {file_context}
        
        Web Search Context:
        {search_context}
        
        Instructions:
        1. If the user wants to create files or projects, suggest using the appropriate tools
        2. Provide clear, working code with explanations
        3. If file operations are needed, explain what tools to use
        4. Use web search information when relevant
        5. Keep responses practical and focused on the user's request
        6. Mention the workspace directory: {self.file_tools.workspace_dir}
        
        Response should be helpful for voice interaction (concise but complete).
        """
        
        try:
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"I encountered an error generating code: {str(e)}"
