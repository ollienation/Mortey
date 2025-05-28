import os
import asyncio
from typing import Dict, Any, List
from config.llm_manager import llm_manager
from config.settings import config
from tools.file_tools import FileSystemTools
from tavily import TavilyClient

class CoderAgent:
    """Enhanced code generation agent with actual LangChain tool integration"""
    
    def __init__(self, llm_service=None, workspace_dir: str = None):
        self.llm_service = llm_service
        
        # Initialize file tools with LangChain toolkit
        self.file_tools = FileSystemTools(workspace_dir)
        self.tools = self.file_tools.get_tools()
        
        # Get specific tools for direct use
        self.write_tool = None
        self.read_tool = None
        self.list_tool = None
        
        for tool in self.tools:
            if tool.name == "write_file":
                self.write_tool = tool
            elif tool.name == "read_file":
                self.read_tool = tool
            elif tool.name == "list_directory":
                self.list_tool = tool
        
        # Initialize Tavily for web search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily_client = None
            print("⚠️ TAVILY_API_KEY not found - web search disabled for coder")
        
        print(f"🛠️ Coder agent initialized with {len(self.tools)} LangChain tools")
    
    async def generate_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code and actually save it using LangChain tools"""
        
        user_input = state.get('user_input', '')
        
        # Update thinking state
        state['thinking_state'] = {
            'active_agent': 'CODER',
            'current_task': 'Analyzing coding request',
            'progress': 0.1,
            'details': f'Processing: {user_input}'
        }
        
        try:
            # Step 1: Determine what's needed
            needs_search = await self._should_search_web(user_input)
            needs_file_save = self._should_save_file(user_input)
            
            # Step 2: Gather context
            search_context = ""
            file_context = ""
            
            if needs_search and self.tavily_client:
                state['thinking_state']['current_task'] = 'Searching for coding information'
                state['thinking_state']['progress'] = 0.3
                search_context = await self._search_coding_info(user_input)
            
            # Get workspace context
            if self.list_tool:
                try:
                    file_list = await asyncio.to_thread(self.list_tool.invoke, {})
                    file_context = f"Workspace files: {file_list}"
                except Exception as e:
                    file_context = f"Workspace access error: {str(e)}"
            
            # Step 3: Generate code
            state['thinking_state']['current_task'] = 'Generating code solution'
            state['thinking_state']['progress'] = 0.6
            
            code_response = await self._generate_code_response(user_input, search_context, file_context)
            
            # Step 4: Save file if requested
            if needs_file_save and self.write_tool:
                state['thinking_state']['current_task'] = 'Saving code file'
                state['thinking_state']['progress'] = 0.8
                
                filename = self._extract_filename(user_input)
                code_content = self._extract_code_from_response(code_response)
                
                if code_content:
                    temp_filename = f"temp_{filename}"
                    
                    try:
                        # Actually invoke the LangChain write_file tool
                        save_result = await asyncio.to_thread(
                            self.write_tool.invoke,
                            {
                                "file_path": temp_filename,
                                "text": code_content
                            }
                        )
                        
                        print(f"✅ File saved using LangChain tool: {temp_filename}")
                        
                        # Store file info for controller to rename after approval
                        state['temp_filename'] = temp_filename
                        state['final_filename'] = filename
                        state['file_saved'] = True
                        
                        # Update response to include save confirmation
                        code_response += f"\n\n✅ Code saved as temporary file: {temp_filename}"
                        code_response += f"\nWill be renamed to: {filename} after verification"
                        
                    except Exception as e:
                        print(f"❌ Error saving file with LangChain tool: {e}")
                        code_response += f"\n\n❌ Error saving file: {str(e)}"
            
            return {
                **state,
                'output_content': code_response,
                'output_type': 'code',
                'code_context': {
                    'language': 'python',
                    'request': user_input,
                    'web_search_used': bool(search_context),
                    'file_saved': needs_file_save,
                    'tools_used': len(self.tools)
                },
                'thinking_state': {
                    'active_agent': 'CODER',
                    'current_task': 'Code generation complete',
                    'progress': 1.0,
                    'details': 'Ready for verification'
                }
            }
            
        except Exception as e:
            print(f"❌ Coder agent error: {e}")
            return {
                **state,
                'output_content': f"Error generating code: {str(e)}",
                'output_type': 'error',
                'thinking_state': {
                    'active_agent': 'CODER',
                    'current_task': 'Error occurred',
                    'progress': 1.0,
                    'details': f'Error: {str(e)}'
                }
            }
    
    async def _generate_code_response(self, user_input: str, search_context: str, file_context: str) -> str:
        """Generate code response using LLM manager"""
        
        # Build context strings separately
        web_context = f"Web Search Context:\n{search_context}\n" if search_context else ""
        workspace_context = f"Workspace Context:\n{file_context}\n" if file_context else ""
        
        # Build comprehensive prompt
        prompt = f"""
        You are an expert programmer. Generate clean, working code for this request:
        
        User Request: {user_input}
        
        {web_context}
        
        {workspace_context}
        
        Available LangChain Tools:
        - write_file: Save code to files
        - read_file: Read existing files
        - list_directory: List workspace contents
        - create_project: Create project structures
        - analyze_code: Analyze code files
        
        Instructions:
        1. Provide complete, working code
        2. Include brief explanation of what the code does
        3. Add usage examples if helpful
        4. Do NOT use XML tags or simulate file operations
        5. Just provide the code and explanation
        
        Keep explanations concise since this may be spoken aloud.
        """
        try:
            response = await llm_manager.generate_for_node("coder", prompt)
            return response
        except Exception as e:
            return f"Error generating code response: {str(e)}"
    
    def _should_save_file(self, user_input: str) -> bool:
        """Determine if user wants to save code to a file"""
        save_indicators = [
            'save', 'create file', 'write to file', 'save as', 'create',
            'make a file', 'generate file', 'write file'
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in save_indicators)
    
    def _extract_filename(self, user_input: str) -> str:
        """Extract or generate filename from user request"""
        import re
        
        # Look for explicit filename in request
        patterns = [
            r'save.*?as\s+"([^"]+)"',
            r'save.*?as\s+([^\s]+)',
            r'create.*?file\s+"([^"]+)"',
            r'create.*?file\s+([^\s]+)',
            r'name.*?it\s+"([^"]+)"',
            r'call.*?it\s+"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                filename = match.group(1)
                # Ensure .py extension
                if not filename.endswith('.py'):
                    filename += '.py'
                return filename
        
        # Generate filename based on content
        user_lower = user_input.lower()
        if "gui" in user_lower or "tkinter" in user_lower:
            return "gui_app.py"
        elif "test" in user_lower:
            return "test_script.py"
        elif "web" in user_lower or "flask" in user_lower or "django" in user_lower:
            return "web_app.py"
        elif "api" in user_lower:
            return "api_server.py"
        elif "game" in user_lower:
            return "game.py"
        else:
            return "generated_code.py"
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract actual Python code from LLM response"""
        import re
        
        # Look for code blocks first (``````)
        code_patterns = [
            r'``````',
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code blocks, extract Python-like content
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see Python syntax
            if (line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#!', 'if __name__')) or
                'import ' in line or 'def ' in line):
                in_code = True
            
            if in_code:
                # Stop if we hit explanatory text
                if (line.strip() and 
                    not line.startswith((' ', '\t', '#')) and 
                    not any(line.strip().startswith(x) for x in [
                        'import', 'from', 'def', 'class', 'if', 'for', 'while', 
                        'try', 'with', 'async', 'return', 'print', 'app', 'root'
                    ]) and
                    any(word in line.lower() for word in [
                        'this code', 'the script', 'explanation', 'usage', 'run this'
                    ])):
                    break
                code_lines.append(line)
        
        extracted_code = '\n'.join(code_lines).strip()
        
        # Validate that we have actual code
        if len(extracted_code) > 20 and ('import ' in extracted_code or 'def ' in extracted_code):
            return extracted_code
        
        return ""
    
    async def _should_search_web(self, user_input: str) -> bool:
        """Determine if web search would help with this coding request"""
        
        # Keywords that suggest web search would be helpful
        search_indicators = [
            'latest', 'newest', 'current', 'recent', 'updated',
            'best practices', 'how to', 'tutorial', 'example',
            'documentation', 'api', 'library', 'framework',
            'error', 'fix', 'solve', 'troubleshoot'
        ]
        
        # Check for specific technologies that might need current info
        tech_keywords = [
            'react', 'vue', 'angular', 'nextjs', 'svelte',
            'tensorflow', 'pytorch', 'scikit', 'pandas',
            'fastapi', 'django', 'flask', 'express',
            'docker', 'kubernetes', 'aws', 'azure'
        ]
        
        user_lower = user_input.lower()
        has_search_indicator = any(indicator in user_lower for indicator in search_indicators)
        has_tech_keyword = any(tech in user_lower for tech in tech_keywords)
        
        # Use LLM for intelligent decision if not obvious
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
                
                decision = await llm_manager.generate_for_node("router", decision_prompt)
                return "YES" in decision.upper()
                
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
                max_results=3
            )
            
            # Process and format search results
            context = ""
            
            # Add AI-generated answer if available
            if search_response.get('answer'):
                context += f"Search Summary: {search_response['answer']}\n\n"
            
            # Add relevant search results
            if search_response.get('results'):
                context += "Relevant Resources:\n"
                for i, result in enumerate(search_response['results'][:2], 1):
                    context += f"{i}. {result.get('title', 'Unknown')}\n"
                    if result.get('content'):
                        # Limit content length
                        content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                        context += f"   {content}\n\n"
            
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
            
            query = await llm_manager.generate_for_node("router", query_prompt)
            return query.strip()
            
        except Exception as e:
            print(f"❌ Query creation error: {e}")
            # Fallback to original request with coding keywords
            return f"{user_input} programming tutorial example"
