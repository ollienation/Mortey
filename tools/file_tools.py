import os
import json
import shutil
import time
import hashlib
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit

from config.settings import config

logger = logging.getLogger("file_tools")

class FileSystemTools:
    """
    File system tools, duh
    """
    
    def __init__(self, workspace_dir: str = None):
        # Use config-based workspace directory if not specified
        if workspace_dir is None:
            workspace_dir = str(config.workspace_dir)
            
        # Ensure workspace exists
        self.workspace_dir = workspace_dir
        try:
            os.makedirs(workspace_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create workspace directory {workspace_dir}: {e}")
            raise
        
        # Use official FileManagementToolkit for base functionality
        try:
            self.toolkit = FileManagementToolkit(
                root_dir=workspace_dir, 
                selected_tools=["read_file", "write_file", "list_directory", "copy_file", "move_file"]
            )
            logger.info(f"✅ File tools initialized with workspace: {workspace_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize FileManagementToolkit: {e}")
            raise
    
    def get_tools(self) -> List:
        """Get all file system tools including custom enhanced tools"""
        try:
            # Get base LangChain tools
            base_tools = self.toolkit.get_tools()
            
            # Add custom enhanced tools
            workspace = self.workspace_dir
        
            @tool
            def create_project(project_name: str, project_type: str = "python") -> str:
                """
                Create a new project structure.
                
                Args:
                    project_name: Name of the project to create
                    project_type: Type of project (python, web, data, etc.)
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
                        os.makedirs(os.path.join(project_path, "static", "css"))
                        os.makedirs(os.path.join(project_path, "static", "js"))
                        os.makedirs(os.path.join(project_path, "templates"))
                        
                        # Create basic files
                        with open(os.path.join(project_path, "index.html"), "w") as f:
                            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Web Project</title>
    <link rel="stylesheet" href="static/css/style.css">
</head>
<body>
    <h1>Hello from Mortey!</h1>
    <script src="static/js/script.js"></script>
</body>
</html>""")
                    
                        with open(os.path.join(project_path, "static", "css", "style.css"), "w") as f:
                            f.write("body {\n    font-family: Arial, sans-serif;\n    margin: 2rem;\n}\n")
                        
                        with open(os.path.join(project_path, "static", "js", "script.js"), "w") as f:
                            f.write('console.log("Hello from Mortey!");\n')
                            
                    elif project_type.lower() == "data":
                        # Create data analysis project structure
                        os.makedirs(os.path.join(project_path, "data"))
                        os.makedirs(os.path.join(project_path, "notebooks"))
                        os.makedirs(os.path.join(project_path, "scripts"))
                        os.makedirs(os.path.join(project_path, "output"))
                        
                        # Create basic files
                        with open(os.path.join(project_path, "README.md"), "w") as f:
                            f.write(f"# {project_name}\n\nA data analysis project created by Mortey.\n")
                        
                        with open(os.path.join(project_path, "notebooks", "analysis.ipynb"), "w") as f:
                            f.write('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
                        
                        with open(os.path.join(project_path, "scripts", "process_data.py"), "w") as f:
                            f.write('import pandas as pd\n\ndef process_data(input_file, output_file):\n    """Process data from input file and save to output file."""\n    # Add your data processing code here\n    print(f"Processing {input_file} and saving to {output_file}")\n\nif __name__ == "__main__":\n    process_data("../data/input.csv", "../output/processed.csv")\n')
                    else:
                        # Generic project structure
                        os.makedirs(os.path.join(project_path, "docs"))
                        
                        with open(os.path.join(project_path, "README.md"), "w") as f:
                            f.write(f"# {project_name}\n\nA {project_type} project created by Mortey.\n")
                    
                    return f"✅ Successfully created {project_type} project: {project_name}"
                    
                except Exception as e:
                    return f"❌ Error creating project '{project_name}': {str(e)}"
            
            @tool
            def search_in_files(query: str, file_extension: str = None) -> str:
                """
                Search for text across all files in the workspace.
                
                Args:
                    query: Text to search for
                    file_extension: Optional file extension filter (e.g., '.py', '.txt')
                """
                try:
                    results = []
                    query = query.lower()
                    
                    for root, _, files in os.walk(workspace):
                        for file in files:
                            if file_extension and not file.endswith(file_extension):
                                continue
                                
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, workspace)
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    
                                if query in content.lower():
                                    line_number = None
                                    context = ""
                                    
                                    # Find line number and context
                                    for i, line in enumerate(content.split('\n')):
                                        if query in line.lower():
                                            line_number = i + 1
                                            context = line.strip()
                                            break
                                            
                                    results.append(f"📄 {rel_path}:{line_number} - {context}")
                            except Exception:
                                # Skip files that can't be read as text
                                continue
                    
                    if results:
                        return f"Found '{query}' in {len(results)} files:\n" + "\n".join(results[:15]) + ("\n... and more results" if len(results) > 15 else "")
                    else:
                        return f"No matches found for '{query}'"
                        
                except Exception as e:
                    return f"❌ Error searching files: {str(e)}"
            
            @tool
            def backup_file(filename: str) -> str:
                """
                Create a timestamped backup of a file.
                
                Args:
                    filename: Name of the file to backup
                """
                try:
                    file_path = os.path.join(workspace, filename)
                    if not os.path.exists(file_path):
                        return f"File '{filename}' not found in workspace."
                    
                    # Create backup with timestamp
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    backup_name = f"{filename}.{timestamp}.bak"
                    backup_path = os.path.join(workspace, backup_name)
                    
                    shutil.copy2(file_path, backup_path)
                    
                    return f"✅ Backup created: {backup_name}"
                    
                except Exception as e:
                    return f"❌ Error creating backup: {str(e)}"
            
            @tool
            def file_info(filename: str) -> str:
                """
                Get detailed information about a file.
                
                Args:
                    filename: Name of the file to get information about
                """
                try:
                    file_path = os.path.join(workspace, filename)
                    if not os.path.exists(file_path):
                        return f"File '{filename}' not found in workspace."
                    
                    stat = os.stat(file_path)
                    size_bytes = stat.st_size
                    
                    # Calculate human-readable size
                    for unit in ['B', 'KB', 'MB', 'GB']:
                        if size_bytes < 1024 or unit == 'GB':
                            break
                        size_bytes /= 1024
                    
                    size_str = f"{size_bytes:.2f} {unit}"
                    
                    # Calculate MD5 hash for small files
                    md5_hash = "Not calculated (file too large)"
                    if stat.st_size < 10_000_000:  # 10MB limit
                        with open(file_path, 'rb') as f:
                            md5_hash = hashlib.md5(f.read()).hexdigest()
                    
                    # Get file extension and mime type guess
                    _, ext = os.path.splitext(filename)
                    mime_type = "Unknown"
                    
                    # Simple mime type mapping
                    mime_map = {
                        '.txt': 'text/plain',
                        '.py': 'text/x-python',
                        '.js': 'application/javascript',
                        '.html': 'text/html',
                        '.css': 'text/css',
                        '.json': 'application/json',
                        '.xml': 'application/xml',
                        '.csv': 'text/csv',
                        '.md': 'text/markdown',
                        '.pdf': 'application/pdf',
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif'
                    }
                    
                    if ext.lower() in mime_map:
                        mime_type = mime_map[ext.lower()]
                    
                    # Format creation and modification times
                    ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime))
                    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                    
                    info = f"""📄 File Information for {filename}:
                
📊 Basic Properties:
- Size: {size_str} ({stat.st_size} bytes)
- Type: {mime_type}
- Extension: {ext if ext else 'None'}

⏰ Timestamps:
- Created: {ctime}
- Modified: {mtime}

🔒 Permissions:
- Mode: {stat.st_mode & 0o777:o}
- Owner ID: {stat.st_uid}

🔍 Identification:
- MD5 Hash: {md5_hash}
- inode: {stat.st_ino}
"""
                    
                    return info
                    
                except Exception as e:
                    return f"❌ Error getting file info: {str(e)}"
            
            @tool
            def organize_workspace() -> str:
                """
                Organize files in the workspace by type into subdirectories.
                """
                try:
                    # Define file type categories
                    categories = {
                        "documents": [".txt", ".pdf", ".doc", ".docx", ".md", ".rtf"],
                        "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".php", ".go", ".rb"],
                        "data": [".csv", ".json", ".xml", ".xlsx", ".xls", ".db", ".sql"],
                        "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
                        "archives": [".zip", ".tar", ".gz", ".7z", ".rar"]
                    }
                    
                    # Create directories if they don't exist
                    for category in categories:
                        os.makedirs(os.path.join(workspace, category), exist_ok=True)
                    
                    # Track moved files
                    moved_files = {}
                    
                    # Get all files in the workspace root
                    files = [f for f in os.listdir(workspace) if os.path.isfile(os.path.join(workspace, f))]
                    
                    for file in files:
                        _, ext = os.path.splitext(file)
                        source_path = os.path.join(workspace, file)
                        
                        # Find category for this file
                        target_category = None
                        for category, extensions in categories.items():
                            if ext.lower() in extensions:
                                target_category = category
                                break
                        
                        # Skip if no category or already in a category directory
                        if target_category is None or file == "README.md":
                            continue
                        
                        # Move file to appropriate directory
                        target_path = os.path.join(workspace, target_category, file)
                        
                        # Handle duplicate filenames
                        if os.path.exists(target_path):
                            base, ext = os.path.splitext(file)
                            target_path = os.path.join(workspace, target_category, f"{base}_{str(uuid.uuid4())[:8]}{ext}")
                        
                        shutil.move(source_path, target_path)
                        
                        # Track the move
                        if target_category not in moved_files:
                            moved_files[target_category] = []
                        moved_files[target_category].append(file)
                    
                    # Build report
                    if not moved_files:
                        return "No files needed organization."
                    
                    report = ["✅ Workspace organized:"]
                    for category, files in moved_files.items():
                        report.append(f"\n📁 {category.capitalize()} ({len(files)} files):")
                        for file in files[:5]:
                            report.append(f"  - {file}")
                        if len(files) > 5:
                            report.append(f"  - ... and {len(files) - 5} more")
                    
                    return "\n".join(report)
                    
                except Exception as e:
                    return f"❌ Error organizing workspace: {str(e)}"
            
            @tool
            def convert_file_format(filename: str, output_format: str) -> str:
                """
                Convert a file from one format to another.
                
                Args:
                    filename: Name of the file to convert
                    output_format: Desired output format (json, yaml, csv, txt, md)
                """
                try:
                    file_path = os.path.join(workspace, filename)
                    if not os.path.exists(file_path):
                        return f"File '{filename}' not found in workspace."
                    
                    # Get base name without extension
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}.{output_format.lower()}"
                    output_path = os.path.join(workspace, output_filename)
                    
                    # Check for unsupported conversions
                    _, ext = os.path.splitext(filename)
                    ext = ext.lower().lstrip('.')
                    
                    # Handle JSON <-> YAML conversion
                    if (ext == 'json' and output_format.lower() == 'yaml') or (ext == 'yaml' and output_format.lower() == 'json'):
                        try:
                            import yaml
                            
                            with open(file_path, 'r') as f:
                                if ext == 'json':
                                    data = json.load(f)
                                    with open(output_path, 'w') as out:
                                        yaml.dump(data, out, default_flow_style=False)
                                else:  # yaml to json
                                    data = yaml.safe_load(f)
                                    with open(output_path, 'w') as out:
                                        json.dump(data, out, indent=2)
                            
                            return f"✅ Converted {filename} to {output_filename}"
                        except Exception as e:
                            return f"❌ Error during conversion: {str(e)}"
                    
                    # Handle CSV <-> JSON conversion
                    elif (ext == 'csv' and output_format.lower() == 'json') or (ext == 'json' and output_format.lower() == 'csv'):
                        try:
                            import csv
                            
                            if ext == 'csv':
                                # CSV to JSON
                                data = []
                                with open(file_path, 'r', newline='') as f:
                                    reader = csv.DictReader(f)
                                    for row in reader:
                                        data.append(dict(row))
                                
                                with open(output_path, 'w') as out:
                                    json.dump(data, out, indent=2)
                            else:
                                # JSON to CSV
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                
                                if not isinstance(data, list):
                                    return "❌ JSON must contain a list of objects to convert to CSV"
                                
                                # Get all possible keys as fieldnames
                                fieldnames = set()
                                for item in data:
                                    if isinstance(item, dict):
                                        fieldnames.update(item.keys())
                                
                                with open(output_path, 'w', newline='') as out:
                                    writer = csv.DictWriter(out, fieldnames=list(fieldnames))
                                    writer.writeheader()
                                    for item in data:
                                        if isinstance(item, dict):
                                            writer.writerow(item)
                            
                            return f"✅ Converted {filename} to {output_filename}"
                        except Exception as e:
                            return f"❌ Error during conversion: {str(e)}"
                    
                    # Handle Markdown <-> TXT conversion
                    elif (ext == 'md' and output_format.lower() == 'txt') or (ext == 'txt' and output_format.lower() == 'md'):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            
                            with open(output_path, 'w') as out:
                                out.write(content)
                            
                            return f"✅ Converted {filename} to {output_filename} (plain text conversion)"
                        except Exception as e:
                            return f"❌ Error during conversion: {str(e)}"
                    
                    else:
                        return f"❌ Unsupported conversion: {ext} to {output_format}"
                    
                except Exception as e:
                    return f"❌ Error converting file: {str(e)}"
            
            @tool
            def summarize_file(filename: str, max_length: int = 500) -> str:
                """
                Generate a summary of a text file's contents.
                
                Args:
                    filename: Name of the file to summarize
                    max_length: Maximum length of the summary in characters
                """
                try:
                    file_path = os.path.join(workspace, filename)
                    if not os.path.exists(file_path):
                        return f"File '{filename}' not found in workspace."
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Get file stats
                    word_count = len(content.split())
                    line_count = len(content.splitlines())
                    char_count = len(content)
                    
                    # Extract beginning and end of file
                    beginning = content[:max_length // 2].strip()
                    ending = content[-(max_length // 2):].strip()
                    
                    # Create the summary
                    summary = f"""📄 Summary of {filename}:

📊 Statistics:
- Word count: {word_count}
- Line count: {line_count}
- Character count: {char_count}

📝 Content preview:
Begin:
{beginning}
...
End:
{ending}
"""
                    
                    return summary
                    
                except Exception as e:
                    return f"❌ Error summarizing file: {str(e)}"
            
            # Combine base tools with enhanced tools
            enhanced_tools = [
                create_project,
                search_in_files,
                backup_file,
                file_info,
                organize_workspace,
                convert_file_format,
                summarize_file
            ]

            logger.info(f"✅ File tools ready: {len(base_tools)} base + {len(enhanced_tools)} enhanced")
            return base_tools + enhanced_tools
            
        except Exception as e:
            logger.error(f"Failed to get file tools: {e}")
            return []
