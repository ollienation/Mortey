import os
from typing import Dict, Any, List
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import tool
from config.settings import config

class FileSystemTools:
    """File system tools using official LangChain FileManagementToolkit"""
    
    def __init__(self, workspace_dir: str = None):
        # Use config-based workspace directory
        if workspace_dir is None:
            workspace_dir = str(config.workspace_dir)
        
        # Ensure workspace exists
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Use official FileManagementToolkit
        self.toolkit = FileManagementToolkit(
            root_dir=workspace_dir,
            selected_tools=["read_file", "write_file", "list_directory", "copy_file", "move_file"]
        )
        
        print(f"File tools initialized with workspace: {workspace_dir}")
    
    def get_tools(self) -> List:
        """Get all file system tools including custom ones"""
        # Get official LangChain tools
        official_tools = self.toolkit.get_tools()
        
        # Add custom tools using @tool decorator
        workspace = self.workspace_dir
        
        @tool
        def create_project(project_name: str, project_type: str = "python") -> str:
            """Create a new project structure.
            
            Args:
                project_name: Name of the project to create
                project_type: Type of project (python, web, etc.)
            """
            project_path = os.path.join(workspace, project_name)
            
            try:
                if os.path.exists(project_path):
                    return f"Project '{project_name}' already exists in workspace."
                
                os.makedirs(project_path)
                
                if project_type.lower() == "python":
                    # Create Python project structure
                    os.makedirs(os.path.join(project_path, "src"))
                    os.makedirs(os.path.join(project_path, "tests"))
                    
                    # Create basic files
                    with open(os.path.join(project_path, "README.md"), "w") as f:
                        f.write(f"# {project_name}\n\nA Python project created by Mortey.\n")
                    
                    with open(os.path.join(project_path, "requirements.txt"), "w") as f:
                        f.write("# Add your dependencies here\n")
                    
                    with open(os.path.join(project_path, "src", "__init__.py"), "w") as f:
                        f.write("")
                    
                    with open(os.path.join(project_path, "src", "main.py"), "w") as f:
                        f.write('def main():\n    print("Hello from Mortey!")\n\nif __name__ == "__main__":\n    main()\n')
                
                elif project_type.lower() == "web":
                    # Create web project structure
                    os.makedirs(os.path.join(project_path, "static"))
                    os.makedirs(os.path.join(project_path, "templates"))
                    
                    with open(os.path.join(project_path, "index.html"), "w") as f:
                        f.write(f"<!DOCTYPE html>\n<html>\n<head>\n    <title>{project_name}</title>\n</head>\n<body>\n    <h1>Welcome to {project_name}</h1>\n    <p>Created by Mortey</p>\n</body>\n</html>\n")
                
                return f"Created {project_type} project '{project_name}' successfully!"
                
            except Exception as e:
                return f"Error creating project: {str(e)}"
        
        @tool
        def analyze_code(file_path: str) -> str:
            """Analyze a code file and provide insights.
            
            Args:
                file_path: Path to the code file to analyze (relative to workspace)
            """
            full_path = os.path.join(workspace, file_path)
            
            try:
                if not os.path.exists(full_path):
                    return f"File not found: {file_path}"
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic code analysis
                lines = content.split('\n')
                total_lines = len(lines)
                non_empty_lines = len([line for line in lines if line.strip()])
                
                # Count functions and classes
                functions = len([line for line in lines if line.strip().startswith('def ')])
                classes = len([line for line in lines if line.strip().startswith('class ')])
                
                # Detect imports
                imports = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
                
                analysis = f"""Code Analysis for {file_path}:

File Statistics:
- Total lines: {total_lines}
- Non-empty lines: {non_empty_lines}
- Functions: {functions}
- Classes: {classes}

Imports ({len(imports)}):
{chr(10).join(f"  - {imp}" for imp in imports[:10])}
{'  - ... and more' if len(imports) > 10 else ''}

Quick Review:
- File size: {'Small' if total_lines < 50 else 'Medium' if total_lines < 200 else 'Large'}
- Structure: {'Well-organized' if functions > 0 or classes > 0 else 'Script-style'}
"""
                
                return analysis
                
            except Exception as e:
                return f"Error analyzing code: {str(e)}"
        
        # Combine official tools with custom tools
        return official_tools + [create_project, analyze_code]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available tools"""
        tools = self.get_tools()
        descriptions = {}
        
        for tool in tools:
            descriptions[tool.name] = tool.description
        
        return descriptions
