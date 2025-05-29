import os
import asyncio
from typing import Dict, Any, List
from config.llm_manager import llm_manager
from config.settings import config
from tools.file_tools import FileSystemTools
from tavily import TavilyClient

class CoderAgent:
    """Enhanced code generation agent with controller feedback integration and smart search logic"""
    
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
        """Generate code with controller feedback integration and smart search logic"""
        
        user_input = state.get('user_input', '')
        controller_feedback = state.get('controller_feedback', '')
        loop_count = state.get('loop_count', 0)
        
        # Update thinking state
        state['thinking_state'] = {
            'active_agent': 'CODER',
            'current_task': 'Analyzing coding request',
            'progress': 0.1,
            'details': f'Processing: {user_input}'
        }
        
        try:
            # Step 1: Smart search decision
            needs_search = await self._should_search_web(user_input)
            needs_file_save = self._should_save_file(user_input)
            
            # Step 2: Gather context (only if needed)
            search_context = ""
            file_context = ""
            
            if needs_search and self.tavily_client:
                state['thinking_state']['current_task'] = 'Searching for relevant coding information'
                state['thinking_state']['progress'] = 0.3
                search_context = await self._search_coding_info(user_input)
            else:
                print("🔄 Skipping web search - not needed for this request")
            
            # Get workspace context
            if self.list_tool:
                try:
                    file_list = await asyncio.to_thread(self.list_tool.invoke, {})
                    file_context = f"Workspace files: {file_list}"
                except Exception as e:
                    file_context = f"Workspace access error: {str(e)}"
            
            # Step 3: Generate code with controller feedback
            state['thinking_state']['current_task'] = 'Generating code solution'
            state['thinking_state']['progress'] = 0.6
            
            code_response = await self._generate_code_response(
                user_input, search_context, file_context, controller_feedback, loop_count
            )
            
            # Step 4: Save file if requested
            if needs_file_save and self.write_tool:
                state['thinking_state']['current_task'] = 'Saving code file'
                state['thinking_state']['progress'] = 0.8
                
                filename = self._extract_filename(user_input)
                code_content = self._extract_code_from_response(code_response)
                
                if code_content:
                    temp_filename = f"temp_{filename}"
                    
                    try:
                        # Use LangChain write_file tool
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
                    'tools_used': len(self.tools),
                    'revision_attempt': loop_count
                },
                'thinking_state': {
                    'active_agent': 'CODER',
                    'current_task': 'Code generation complete',
                    'progress': 1.0,
                    'details': 'Ready for controller verification'
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
    
    async def _generate_code_response(self, user_input: str, search_context: str, 
                                    file_context: str, controller_feedback: str, 
                                    loop_count: int) -> str:
        """Generate code response with improved prompting and conditional context inclusion"""
        
        # Only include web context if it's actually relevant
        web_context = ""
        if search_context and len(search_context.strip()) > 50:
            web_context = f"Web Search Context:\n{search_context}\n"
        
        workspace_context = f"Workspace Context:\n{file_context}\n" if file_context else ""
        
        # Handle controller feedback for revisions
        if loop_count > 0 and controller_feedback:
            print(f"🔄 Controller feedback received: {controller_feedback}")
            
            prompt = f"""
You are an expert programmer. The controller reviewed your previous code and provided feedback.

Controller Feedback: "{controller_feedback}"

Original User Request: {user_input}

{web_context}
{workspace_context}

Please generate improved code that specifically addresses the controller's feedback.

Instructions:
1. Fix the issues mentioned in the controller feedback
2. Provide ONLY the code that should be saved to the file
3. Do NOT include file writing operations or save commands
4. Include proper imports and error handling
5. Add a brief explanation AFTER the code block

Format your response as:

[YOUR CODE HERE]

Brief explanation: [What the code does and how it addresses the feedback]

Keep explanations concise since this may be spoken aloud.
"""
        else:
            # Build regular first-attempt prompt
            prompt = f"""
You are an expert programmer. Generate clean, working code for this request:

User Request: {user_input}

{web_context}
{workspace_context}

Instructions:
1. Provide ONLY the code that should be saved to the file
2. Do NOT include file writing operations or save commands
3. Do NOT duplicate the code in explanations
4. Include necessary imports and proper error handling
5. Add a brief explanation AFTER the code block
6. The file will be automatically saved using LangChain tools

Format your response as:

[YOUR CODE HERE]

Brief explanation: [What the code does]

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
            'make a file', 'generate file', 'write file', 'filename', 'file name'
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in save_indicators)
    
    def _extract_filename(self, user_input: str) -> str:
        """Extract or generate filename from user request with improved patterns"""
        import re
        
        # Look for explicit filename in request with improved patterns
        patterns = [
            r'save.*?(?:as|with.*?name|file.*?name)\s+"([^"]+)"',
            r'save.*?(?:as|with.*?name|file.*?name)\s+([^\s]+)',
            r'file.*?name\s+"([^"]+)"',
            r'file.*?name\s+([^\s]+)',
            r'name.*?it\s+"([^"]+)"',
            r'call.*?it\s+"([^"]+)"',
            r'save.*?file\s+(\w+)',
            r'filename\s+(\w+)',
            r'call.*?(\w+)\.py',
            r'name.*?(\w+)\.py'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                filename = match.group(1).strip()
                # Don't automatically add .py if user didn't specify extension
                if '.' not in filename:
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
        """Extract actual Python code from LLM response with improved parsing"""
        import re
        
        # Look for code blocks first
        code_patterns = [
            r'``````',
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                return code
        
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
                    any(word in line.lower() for word in [
                        'this code', 'the script', 'explanation', 'usage', 'what this does',
                        'brief explanation', 'how it works'
                    ])):
                    break
                    
                code_lines.append(line)
        
        extracted_code = '\n'.join(code_lines).strip()
        
        # Validate that we have actual code
        if len(extracted_code) > 20 and ('import ' in extracted_code or 'def ' in extracted_code):
            return extracted_code
        
        return ""
    
    async def _should_search_web(self, user_input: str) -> bool:
        """Improved web search decision logic to avoid unnecessary searches"""
        
        # Keywords that suggest web search would be helpful
        search_indicators = [
            'latest', 'newest', 'current', 'recent', 'updated', 'new version',
            'best practices', 'tutorial', 'documentation', 'api reference',
            'how to integrate', 'connect to', 'authenticate with'
        ]
        
        # Technologies that might need current info
        tech_keywords = [
            'api', 'sdk', 'framework', 'library', 'package',
            'authentication', 'oauth', 'database connection',
            'deployment', 'cloud', 'aws', 'azure', 'docker'
        ]
        
        # GUI/Desktop app requests typically DON'T need web search
        gui_keywords = [
            'gui', 'window', 'button', 'tkinter', 'pyqt', 'kivy',
            'desktop app', 'interface', 'form', 'dialog',
            'colorful', 'widget', 'layout', 'menu'
        ]
        
        user_lower = user_input.lower()
        
        # Skip search for basic GUI requests
        if any(gui_word in user_lower for gui_word in gui_keywords):
            if not any(search_word in user_lower for search_word in search_indicators):
                print("🔄 Skipping web search for basic GUI request")
                return False
        
        # Skip search for simple, common programming tasks
        simple_tasks = [
            'create a', 'make a', 'write a', 'generate a',
            'simple', 'basic', 'hello world', 'calculator',
            'file reader', 'text editor', 'game'
        ]
        
        if any(simple in user_lower for simple in simple_tasks):
            if not any(search_word in user_lower for search_word in search_indicators):
                print("🔄 Skipping web search for simple coding task")
                return False
        
        # Only search if explicitly needed
        has_search_indicator = any(indicator in user_lower for indicator in search_indicators)
        has_tech_keyword = any(tech in user_lower for tech in tech_keywords)
        
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
            Create a focused web search query for this coding request:
            
            Request: {user_input}
            
            Guidelines:
            - Focus on technical implementation details
            - Include specific technology names (tkinter, PyQt, etc.)
            - Look for documentation, tutorials, or examples
            - Avoid generic terms like "optimization" or "best practices" unless specifically requested
            
            Examples:
            - "GUI with buttons" → "tkinter button click event python tutorial"
            - "connect to database" → "python database connection tutorial"
            - "web scraping" → "python web scraping requests beautifulsoup"
            
            Return only the search query, no explanation.
            """
            
            query = await llm_manager.generate_for_node("router", query_prompt)
            return query.strip()
            
        except Exception as e:
            print(f"❌ Query creation error: {e}")
            # Fallback to extracting key terms from user input
            key_terms = []
            user_lower = user_input.lower()
            
            if 'gui' in user_lower or 'button' in user_lower:
                key_terms.append('tkinter python gui')
            if 'web' in user_lower and 'browser' in user_lower:
                key_terms.append('python webbrowser module')
            if 'database' in user_lower:
                key_terms.append('python database connection')
                
            return ' '.join(key_terms) if key_terms else f"{user_input} python tutorial"