import os
import asyncio
from typing import Dict, Any
from config.llm_manager import llm_manager
from config.settings import config
from tools.file_tools import FileSystemTools

class ChatAgent:
    """Chat agent with read-only file browsing capabilities"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        
        # Initialize file tools for read-only operations
        self.file_tools = FileSystemTools()
        self.tools = self.file_tools.get_tools()
        
        # Get read-only tools
        self.read_tool = None
        self.list_tool = None
        
        for tool in self.tools:
            if tool.name == "read_file":
                self.read_tool = tool
            elif tool.name == "list_directory":
                self.list_tool = tool
        
        print("💬 Chat agent initialized with read-only file capabilities")
    
    async def chat(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat request with optional file browsing"""
        
        user_input = state.get('user_input', '')
        assistant_name = state.get('assistant_name', 'Mortey')
        
        # Check if this is a file browsing request
        if self._is_file_browsing_request(user_input):
            return await self._handle_file_browsing(state)
        
        # Regular chat processing
        return await self._handle_regular_chat(state, assistant_name)
    
    def _is_file_browsing_request(self, user_input: str) -> bool:
        """Determine if this is a file browsing request"""
        file_keywords = [
            'what files', 'which files', 'list files', 'show files',
            'files in workspace', 'workspace files', 'directory contents',
            'read file', 'show me', 'open file', 'file contents',
            'what\'s in the', 'browse files', 'file system'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in file_keywords)
    
    async def _handle_file_browsing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file browsing requests"""
        
        user_input = state.get('user_input', '')
        user_lower = user_input.lower()
        
        try:
            # Check if user wants to read a specific file
            if any(keyword in user_lower for keyword in ['read', 'open', 'show', 'contents']):
                return await self._read_file_request(state)
            
            # Otherwise, list workspace files
            return await self._list_workspace_files(state)
            
        except Exception as e:
            return {
                **state,
                'output_content': f"Sorry, I had trouble accessing the files: {str(e)}",
                'output_type': 'error'
            }
    
    async def _list_workspace_files(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """List files in the workspace"""
        
        try:
            if self.list_tool:
                # Use LangChain list_directory tool
                file_list_result = await asyncio.to_thread(self.list_tool.invoke, {})
                
                if file_list_result and file_list_result.strip():
                    files = [f.strip() for f in file_list_result.strip().split('\n') if f.strip()]
                    
                    if files:
                        response = f"Here are the files in your workspace:\n\n"
                        for i, file in enumerate(files, 1):
                            response += f"{i}. 📄 {file}\n"
                        
                        response += f"\nI can read any of these files for you if you'd like to see their contents!"
                    else:
                        response = "Your workspace directory is currently empty."
                else:
                    response = "Your workspace directory is currently empty."
            else:
                # Fallback to manual listing
                workspace_path = config.workspace_dir
                if workspace_path.exists():
                    files = [f.name for f in workspace_path.iterdir() if f.is_file()]
                    if files:
                        response = f"Here are the files in your workspace:\n\n"
                        for i, file in enumerate(sorted(files), 1):
                            response += f"{i}. 📄 {file}\n"
                        response += f"\nI can read any of these files for you!"
                    else:
                        response = "Your workspace directory is currently empty."
                else:
                    response = "I couldn't find your workspace directory."
            
            return {
                **state,
                'output_content': response,
                'output_type': 'file_list'
            }
            
        except Exception as e:
            return {
                **state,
                'output_content': f"I had trouble listing the files: {str(e)}",
                'output_type': 'error'
            }
    
    async def _read_file_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file reading requests"""
        
        user_input = state.get('user_input', '')
        
        # Try to extract filename from the request
        filename = self._extract_filename_from_request(user_input)
        
        if not filename:
            return {
                **state,
                'output_content': "Which file would you like me to read? Please specify the filename.",
                'output_type': 'chat'
            }
        
        try:
            if self.read_tool:
                # Use LangChain read_file tool
                file_content = await asyncio.to_thread(
                    self.read_tool.invoke, 
                    {"file_path": filename}
                )
                
                if file_content:
                    # Limit content length for chat display
                    if len(file_content) > 2000:
                        truncated_content = file_content[:2000] + "\n\n... (file truncated for display)"
                        response = f"Here's the content of **{filename}**:\n\n``````"
                    else:
                        response = f"Here's the content of **{filename}**:\n\n``````"
                else:
                    response = f"The file **{filename}** appears to be empty."
            else:
                response = "I don't have the ability to read files right now."
            
            return {
                **state,
                'output_content': response,
                'output_type': 'file_content'
            }
            
        except Exception as e:
            return {
                **state,
                'output_content': f"I couldn't read the file **{filename}**: {str(e)}",
                'output_type': 'error'
            }
    
    def _extract_filename_from_request(self, user_input: str) -> str:
        """Extract filename from user request"""
        import re
        
        # Look for common patterns
        patterns = [
            r'read\s+(?:file\s+)?["\']?([^"\']+)["\']?',
            r'open\s+(?:file\s+)?["\']?([^"\']+)["\']?',
            r'show\s+(?:me\s+)?(?:file\s+)?["\']?([^"\']+)["\']?',
            r'contents?\s+of\s+["\']?([^"\']+)["\']?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    async def _handle_regular_chat(self, state: Dict[str, Any], assistant_name: str) -> Dict[str, Any]:
        """Handle regular chat conversations"""
        
        user_input = state.get('user_input', '')
        
        prompt = f"""
        You are {assistant_name}, a helpful assistant with file browsing capabilities.
        
        User: {user_input}
        
        Guidelines:
        - Keep responses short and concise since this may be spoken aloud
        - Be friendly and helpful
        - If users ask about files, remind them you can list or read files in the workspace
        - Stay in character as an assistant
        """
        
        try:
            response_text = await llm_manager.generate_for_node("chat", prompt)
            
            return {
                **state,
                'output_content': response_text,
                'output_type': 'chat'
            }
            
        except Exception as e:
            return {
                **state,
                'output_content': "I'm having trouble right now. Please try again.",
                'output_type': 'error'
            }
