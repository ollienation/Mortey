# tools/file_tools.py - ✅ ENHANCED WITH PYTHON 3.13.4 COMPATIBILITY
import os
import json
import shutil
import time
import hashlib
import uuid
import logging
import asyncio
from typing import Optional, Union, Any
from collections.abc import Sequence  # Python 3.13.4 preferred import
from pathlib import Path
from asyncio import TaskGroup  # Python 3.13.4 TaskGroup
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit

from config.settings import config
from core.error_handling import ErrorHandler
from core.circuit_breaker import global_circuit_breaker

logger = logging.getLogger("file_tools")

class FileOperation(Enum):
    """File operation types for better tracking"""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    SEARCH = "search"
    ORGANIZE = "organize"

@dataclass
class FileSecurityConfig:
    """Security configuration for file operations"""
    allowed_extensions: set[str] = field(default_factory=lambda: {  # Python 3.13.4 syntax
        ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml", 
        ".csv", ".xml", ".log", ".cfg", ".ini", ".toml", ".sql", ".sh", ".bat"
    })
    max_file_size_mb: int = 50
    max_files_per_operation: int = 100
    blocked_paths: set[str] = field(default_factory=lambda: {  # Python 3.13.4 syntax
        "/etc", "/sys", "/proc", "/dev", "C:\\Windows", "C:\\System32"
    })
    
    def is_extension_allowed(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        _, ext = os.path.splitext(filename.lower())
        return ext in self.allowed_extensions or not ext  # Allow extensionless files
    
    def is_path_safe(self, path: str) -> bool:
        """Check if path is safe for operations"""
        abs_path = os.path.abspath(path)
        return not any(abs_path.startswith(blocked) for blocked in self.blocked_paths)

@dataclass
class FileOperationResult:
    """Result of a file operation with enhanced metadata"""
    success: bool
    message: str
    operation: FileOperation
    files_affected: list[str] = field(default_factory=list)  # Python 3.13.4 syntax
    bytes_processed: int = 0
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)  # Python 3.13.4 syntax
    metadata: dict[str, any] = field(default_factory=dict)  # Python 3.13.4 syntax

class FileSystemTools:
    """
    Enhanced file system tools with Python 3.13.4 features and security
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        # Use config-based workspace directory if not specified
        if workspace_dir is None:
            workspace_dir = str(config.workspace_dir)
            
        # Ensure workspace exists
        self.workspace_dir = Path(workspace_dir)
        self.security_config = FileSecurityConfig()
        self._operation_stats: dict[str, int] = {}  # Python 3.13.4 syntax
        
        try:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Workspace directory ensured: {self.workspace_dir}")
        except Exception as e:
            logger.error(f"Failed to create workspace directory {workspace_dir}: {e}")
            raise
        
        # Use official FileManagementToolkit for base functionality
        try:
            self.toolkit = FileManagementToolkit(
                root_dir=str(self.workspace_dir), 
                selected_tools=["read_file", "write_file", "list_directory", "copy_file", "move_file"]
            )
            logger.info(f"✅ File tools initialized with workspace: {self.workspace_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize FileManagementToolkit: {e}")
            raise
    
    def _validate_file_operation(self, filename: str, operation: FileOperation) -> tuple[bool, str]:
        """Validate file operation for security and constraints"""
        # Check file extension
        if not self.security_config.is_extension_allowed(filename):
            return False, f"File extension not allowed: {filename}"
        
        # Check path safety
        file_path = self.workspace_dir / filename
        if not self.security_config.is_path_safe(str(file_path)):
            return False, f"Path not safe for operations: {filename}"
        
        # Check file size for read operations
        if operation in [FileOperation.READ, FileOperation.COPY] and file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.security_config.max_file_size_mb:
                return False, f"File too large: {size_mb:.1f}MB (max: {self.security_config.max_file_size_mb}MB)"
        
        return True, "Valid"
    
    def _record_operation(self, operation: FileOperation, success: bool = True):
        """Record operation statistics"""
        key = f"{operation.value}_{'success' if success else 'failure'}"
        self._operation_stats[key] = self._operation_stats.get(key, 0) + 1
    
    def get_tools(self) -> list[any]:  # Python 3.13.4 syntax
        """Get all file system tools including custom enhanced tools"""
        try:
            # Get base LangChain tools
            base_tools = self.toolkit.get_tools()
            
            # Add custom enhanced tools with security and Python 3.13.4 features
            workspace = self.workspace_dir
            security_config = self.security_config
        
            @tool
            def create_project(project_name: str, project_type: str = "python") -> str:
                """
                Create a new project structure with enhanced templates.
                
                Args:
                    project_name: Name of the project to create
                    project_type: Type of project (python, web, data, api, ml)
                """
                start_time = time.time()
                
                # Validate project name
                if not project_name.replace("_", "").replace("-", "").isalnum():
                    return "❌ Project name must contain only letters, numbers, hyphens, and underscores"
                
                project_path = workspace / project_name
                
                try:
                    if project_path.exists():
                        return f"Project '{project_name}' already exists in workspace."
                    
                    project_path.mkdir(parents=True)
                    files_created = []
                    
                    # Enhanced project templates using match-case (Python 3.13.4)
                    match project_type.lower():
                        case "python":
                            files_created = self._create_python_project(project_path, project_name)
                        case "web":
                            files_created = self._create_web_project(project_path, project_name)
                        case "data":
                            files_created = self._create_data_project(project_path, project_name)
                        case "api":
                            files_created = self._create_api_project(project_path, project_name)
                        case "ml":
                            files_created = self._create_ml_project(project_path, project_name)
                        case _:
                            files_created = self._create_generic_project(project_path, project_name, project_type)
                    
                    duration = (time.time() - start_time) * 1000
                    self._record_operation(FileOperation.CREATE, True)
                    
                    return f"✅ Successfully created {project_type} project: {project_name}\n📁 Created {len(files_created)} files in {duration:.1f}ms"
                    
                except Exception as e:
                    self._record_operation(FileOperation.CREATE, False)
                    return f"❌ Error creating project '{project_name}': {str(e)}"
            
            @tool
            def search_in_files(
                query: str, 
                file_extension: Optional[str] = None,
                case_sensitive: bool = False,
                max_results: int = 20
            ) -> str:
                """
                Search for text across all files in the workspace with enhanced options.
                
                Args:
                    query: Text to search for
                    file_extension: Optional file extension filter (e.g., '.py', '.txt')
                    case_sensitive: Whether to perform case-sensitive search
                    max_results: Maximum number of results to return
                """
                start_time = time.time()
                
                try:
                    results = []
                    search_query = query if case_sensitive else query.lower()
                    files_searched = 0
                    
                    for file_path in workspace.rglob("*"):
                        if not file_path.is_file():
                            continue
                        
                        # Apply extension filter
                        if file_extension and not file_path.name.endswith(file_extension):
                            continue
                        
                        # Security check
                        if not security_config.is_extension_allowed(file_path.name):
                            continue
                        
                        files_searched += 1
                        rel_path = file_path.relative_to(workspace)
                        
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            search_content = content if case_sensitive else content.lower()
                            
                            if search_query in search_content:
                                # Find line number and context
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    line_search = line if case_sensitive else line.lower()
                                    if search_query in line_search:
                                        context = line.strip()[:100]
                                        results.append(f"📄 {rel_path}:{i+1} - {context}")
                                        
                                        if len(results) >= max_results:
                                            break
                            
                            if len(results) >= max_results:
                                break
                                
                        except (UnicodeDecodeError, PermissionError):
                            # Skip files that can't be read as text
                            continue
                    
                    duration = (time.time() - start_time) * 1000
                    self._record_operation(FileOperation.SEARCH, True)
                    
                    if results:
                        result_summary = f"Found '{query}' in {len(results)} locations (searched {files_searched} files in {duration:.1f}ms):\n"
                        result_summary += "\n".join(results)
                        if len(results) >= max_results:
                            result_summary += f"\n... (limited to {max_results} results)"
                        return result_summary
                    else:
                        return f"No matches found for '{query}' (searched {files_searched} files in {duration:.1f}ms)"
                        
                except Exception as e:
                    self._record_operation(FileOperation.SEARCH, False)
                    return f"❌ Error searching files: {str(e)}"
            
            @tool
            def backup_file(filename: str, backup_location: str = "backups") -> str:
                """
                Create a timestamped backup of a file with enhanced options.
                
                Args:
                    filename: Name of the file to backup
                    backup_location: Subdirectory for backups (default: 'backups')
                """
                start_time = time.time()
                
                try:
                    file_path = workspace / filename
                    if not file_path.exists():
                        return f"File '{filename}' not found in workspace."
                    
                    # Validate file
                    is_valid, message = self._validate_file_operation(filename, FileOperation.COPY)
                    if not is_valid:
                        return f"❌ {message}"
                    
                    # Create backup directory
                    backup_dir = workspace / backup_location
                    backup_dir.mkdir(exist_ok=True)
                    
                    # Create backup with timestamp and hash
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
                    backup_name = f"{file_path.stem}_{timestamp}_{file_hash}{file_path.suffix}.bak"
                    backup_path = backup_dir / backup_name
                    
                    shutil.copy2(file_path, backup_path)
                    
                    duration = (time.time() - start_time) * 1000
                    file_size = file_path.stat().st_size
                    self._record_operation(FileOperation.COPY, True)
                    
                    return f"✅ Backup created: {backup_location}/{backup_name} ({file_size} bytes in {duration:.1f}ms)"
                    
                except Exception as e:
                    self._record_operation(FileOperation.COPY, False)
                    return f"❌ Error creating backup: {str(e)}"
            
            @tool
            def file_info(filename: str, include_content_analysis: bool = False) -> str:
                """
                Get detailed information about a file with enhanced analysis.
                
                Args:
                    filename: Name of the file to get information about
                    include_content_analysis: Whether to include content analysis
                """
                try:
                    file_path = workspace / filename
                    if not file_path.exists():
                        return f"File '{filename}' not found in workspace."
                    
                    stat = file_path.stat()
                    size_bytes = stat.st_size
                    
                    # Calculate human-readable size
                    for unit in ['B', 'KB', 'MB', 'GB']:
                        if size_bytes < 1024 or unit == 'GB':
                            break
                        size_bytes /= 1024
                    
                    size_str = f"{size_bytes:.2f} {unit}"
                    
                    # Calculate hashes for small files
                    md5_hash = "Not calculated (file too large)"
                    sha256_hash = "Not calculated (file too large)"
                    
                    if stat.st_size < 10_000_000:  # 10MB limit
                        file_bytes = file_path.read_bytes()
                        md5_hash = hashlib.md5(file_bytes).hexdigest()
                        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
                    
                    # Enhanced mime type detection
                    mime_type = self._detect_mime_type(file_path)
                    
                    # Format timestamps
                    ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime))
                    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                    
                    info = f"""📄 File Information for {filename}:

📊 Basic Properties:
- Size: {size_str} ({stat.st_size:,} bytes)
- Type: {mime_type}
- Extension: {file_path.suffix if file_path.suffix else 'None'}

⏰ Timestamps:
- Created: {ctime}
- Modified: {mtime}
- Age: {self._format_age(time.time() - stat.st_mtime)}

🔒 Security:
- Permissions: {oct(stat.st_mode)[-3:]}
- Owner ID: {stat.st_uid}
- Safe for operations: {security_config.is_extension_allowed(filename)}

🔍 Identification:
- MD5: {md5_hash}
- SHA256: {sha256_hash[:16]}...
- Inode: {stat.st_ino}
"""
                    
                    # Add content analysis if requested
                    if include_content_analysis and stat.st_size < 1_000_000:  # 1MB limit
                        try:
                            content_info = self._analyze_file_content(file_path)
                            info += f"\n📝 Content Analysis:\n{content_info}"
                        except Exception as e:
                            info += f"\n📝 Content Analysis: Failed ({e})"
                    
                    return info
                    
                except Exception as e:
                    return f"❌ Error getting file info: {str(e)}"
            
            @tool
            async def organize_workspace_async(
                dry_run: bool = False,
                create_date_folders: bool = False
            ) -> str:
                """
                Organize files in the workspace by type with async processing.
                
                Args:
                    dry_run: If True, show what would be done without actually moving files
                    create_date_folders: If True, organize by date within categories
                """
                start_time = time.time()
                
                try:
                    # Enhanced file categorization
                    categories = {
                        "documents": [".txt", ".pdf", ".doc", ".docx", ".md", ".rtf", ".odt"],
                        "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".php", ".go", ".rb", ".rs", ".ts"],
                        "data": [".csv", ".json", ".xml", ".xlsx", ".xls", ".db", ".sql", ".parquet"],
                        "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico"],
                        "archives": [".zip", ".tar", ".gz", ".7z", ".rar", ".bz2"],
                        "config": [".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"],
                        "logs": [".log", ".out", ".err"],
                        "media": [".mp4", ".avi", ".mov", ".mp3", ".wav", ".flac"]
                    }
                    
                    # Get all files in workspace root
                    files = [f for f in workspace.iterdir() if f.is_file()]
                    
                    if not files:
                        return "No files found to organize."
                    
                    # Use TaskGroup for concurrent processing (Python 3.13.4)
                    moved_files: dict[str, list[str]] = {}  # Python 3.13.4 syntax
                    
                    async with TaskGroup() as tg:
                        tasks = []
                        for file_path in files:
                            if file_path.name.startswith('.') or file_path.name == "README.md":
                                continue  # Skip hidden files and README
                            
                            task = tg.create_task(
                                self._organize_file_async(
                                    file_path, categories, dry_run, create_date_folders
                                )
                            )
                            tasks.append((file_path, task))
                    
                    # Collect results
                    for file_path, task in tasks:
                        result = task.result()
                        if result:
                            category, new_path = result
                            if category not in moved_files:
                                moved_files[category] = []
                            moved_files[category].append(file_path.name)
                    
                    duration = (time.time() - start_time) * 1000
                    self._record_operation(FileOperation.ORGANIZE, True)
                    
                    # Build report
                    if not moved_files:
                        return f"No files needed organization (processed {len(files)} files in {duration:.1f}ms)."
                    
                    action = "Would move" if dry_run else "Moved"
                    report = [f"✅ Workspace organization {'preview' if dry_run else 'complete'} ({duration:.1f}ms):"]
                    
                    for category, file_list in moved_files.items():
                        report.append(f"\n📁 {category.capitalize()} ({len(file_list)} files):")
                        for filename in file_list[:5]:
                            report.append(f"  - {action}: {filename}")
                        if len(file_list) > 5:
                            report.append(f"  - ... and {len(file_list) - 5} more")
                    
                    if dry_run:
                        report.append(f"\n💡 Run with dry_run=False to actually move files")
                    
                    return "\n".join(report)
                    
                except Exception as e:
                    self._record_operation(FileOperation.ORGANIZE, False)
                    return f"❌ Error organizing workspace: {str(e)}"
            
            @tool
            def convert_file_format(
                filename: str, 
                output_format: str,
                preserve_original: bool = True
            ) -> str:
                """
                Convert a file from one format to another with enhanced support.
                
                Args:
                    filename: Name of the file to convert
                    output_format: Desired output format (json, yaml, csv, txt, md, xml)
                    preserve_original: Whether to keep the original file
                """
                start_time = time.time()
                
                try:
                    file_path = workspace / filename
                    if not file_path.exists():
                        return f"File '{filename}' not found in workspace."
                    
                    # Validate file operation
                    is_valid, message = self._validate_file_operation(filename, FileOperation.READ)
                    if not is_valid:
                        return f"❌ {message}"
                    
                    # Get base name and create output filename
                    base_name = file_path.stem
                    output_filename = f"{base_name}.{output_format.lower()}"
                    output_path = workspace / output_filename
                    
                    # Get file extension
                    input_ext = file_path.suffix.lower().lstrip('.')
                    output_ext = output_format.lower()
                    
                    # Enhanced conversion logic using match-case (Python 3.13.4)
                    conversion_result = self._perform_file_conversion(
                        file_path, output_path, input_ext, output_ext
                    )
                    
                    if not conversion_result.success:
                        return conversion_result.message
                    
                    # Optionally remove original file
                    if not preserve_original and output_path.exists():
                        file_path.unlink()
                    
                    duration = (time.time() - start_time) * 1000
                    file_size = output_path.stat().st_size if output_path.exists() else 0
                    
                    return f"✅ Converted {filename} to {output_filename} ({file_size:,} bytes in {duration:.1f}ms)"
                    
                except Exception as e:
                    return f"❌ Error converting file: {str(e)}"
            
            @tool
            def get_workspace_statistics() -> str:
                """
                Get comprehensive statistics about the workspace.
                """
                try:
                    stats = self._calculate_workspace_stats()
                    
                    report = f"""📊 Workspace Statistics:

📁 Files & Directories:
- Total files: {stats['total_files']:,}
- Total directories: {stats['total_dirs']:,}
- Total size: {stats['total_size_formatted']}
- Largest file: {stats['largest_file']} ({stats['largest_size_formatted']})

📄 File Types:
"""
                    
                    for ext, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
                        report += f"- {ext or 'No extension'}: {count:,} files\n"
                    
                    if len(stats['file_types']) > 10:
                        report += f"- ... and {len(stats['file_types']) - 10} more types\n"
                    
                    report += f"""
🔧 Operations Performed:
"""
                    for operation, count in self._operation_stats.items():
                        report += f"- {operation.replace('_', ' ').title()}: {count:,}\n"
                    
                    report += f"""
🔒 Security Status:
- Safe files: {stats['safe_files']:,}
- Blocked files: {stats['blocked_files']:,}
- Large files (>{security_config.max_file_size_mb}MB): {stats['large_files']:,}
"""
                    
                    return report
                    
                except Exception as e:
                    return f"❌ Error getting workspace statistics: {str(e)}"
            
            # Combine base tools with enhanced tools
            enhanced_tools = [
                create_project,
                search_in_files,
                backup_file,
                file_info,
                organize_workspace_async,
                convert_file_format,
                get_workspace_statistics
            ]

            logger.info(f"✅ File tools ready: {len(base_tools)} base + {len(enhanced_tools)} enhanced")
            return base_tools + enhanced_tools
            
        except Exception as e:
            logger.error(f"Failed to get file tools: {e}")
            return []
    
    def _create_python_project(self, project_path: Path, project_name: str) -> list[str]:
        """Create Python project structure with modern practices"""
        files_created = []
        
        # Create directory structure
        dirs = ["src", "tests", "docs", ".github/workflows"]
        for dir_name in dirs:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create files with enhanced templates
        files = {
            "README.md": f"""# {project_name}

A Python project created by Mortey Assistant.

## Installation

pip install -r requirements.txt

## Usage

from src.{project_name.lower().replace('-', '_')} import main
main()

## Development

Install development dependencies

pip install -r requirements-dev.txt
Run tests

pytest
Run linting

black src/ tests/
flake8 src/ tests/

""",
            "requirements.txt": """# Core dependencies
# Add your project dependencies here

# Example:
# requests>=2.28.0
# pydantic>=1.10.0
""",
            "requirements-dev.txt": """# Development dependencies
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
pre-commit>=2.20.0
""",
            "pyproject.toml": f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "A Python project created by Mortey"
authors = [{{name = "Your Name", email = "your.email@example.com"}}]
license = {{text = "MIT"}}
readme = "README.md"
requires-python = ">=3.9"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
""",
            "src/__init__.py": "",
            f"src/{project_name.lower().replace('-', '_')}.py": f'''"""
{project_name} - Main module
"""

def main():
    """Main function for {project_name}"""
    print("Hello from {project_name}!")
    return "Success"

if __name__ == "__main__":
    main()
''',
            "tests/__init__.py": "",
            f"tests/test_{project_name.lower().replace('-', '_')}.py": f'''"""
Tests for {project_name}
"""
import pytest
from src.{project_name.lower().replace('-', '_')} import main

def test_main():
    """Test main function"""
    result = main()
    assert result == "Success"
''',
            ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""",
            ".github/workflows/ci.yml": f"""name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest
    
    - name: Run linting
      run: |
        black --check src/ tests/
        flake8 src/ tests/
"""
        }
        
        # Write all files
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(file_path)
        
        return files_created
    
    def _create_web_project(self, project_path: Path, project_name: str) -> list[str]:
        """Create modern web project structure"""
        files_created = []
        
        # Create directory structure
        dirs = ["src", "public", "assets/css", "assets/js", "assets/images"]
        for dir_name in dirs:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        files = {
            "index.html": f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <h1>Welcome to {project_name}</h1>
    </header>
    
    <main>
        <section class="hero">
            <h2>Built with Mortey Assistant</h2>
            <p>A modern web project template</p>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2025 {project_name}. Created with Mortey.</p>
    </footer>
    
    <script src="assets/js/main.js"></script>
</body>
</html>""",
            "assets/css/style.css": """/* Modern CSS Reset */
*, *::before, *::after {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem 0;
    text-align: center;
}

main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.hero {
    text-align: center;
    padding: 4rem 0;
}

.hero h2 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

footer {
    background: #f8f9fa;
    text-align: center;
    padding: 2rem;
    margin-top: 4rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .hero h2 {
        font-size: 2rem;
    }
    
    main {
        padding: 1rem;
    }
}
""",
            "assets/js/main.js": """// Modern JavaScript for the web project
console.log('Welcome to your new web project!');

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Add fade-in animation
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });
    
    document.querySelectorAll('section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
});
""",
            "package.json": f"""{{
  "name": "{project_name.lower().replace('_', '-')}",
  "version": "1.0.0",
  "description": "A web project created by Mortey",
  "main": "index.html",
  "scripts": {{
    "dev": "python -m http.server 8000",
    "build": "echo 'Build process here'",
    "lint": "echo 'Linting process here'"
  }},
  "keywords": ["web", "html", "css", "javascript"],
  "author": "Your Name",
  "license": "MIT"
}}
""",
            "README.md": f"""# {project_name}

A modern web project created by Mortey Assistant.

## Getting Started

1. Open `index.html` in your browser
2. Or run a local server:

python -m http.server 8000

3. Visit `http://localhost:8000`

## Project Structure

{project_name}/
├── index.html # Main HTML file
├── assets/
│ ├── css/
│ │ └── style.css # Styles
│ ├── js/
│ │ └── main.js # JavaScript
│ └── images/ # Images
├── src/ # Source files
└── public/ # Public assets

## Features

- Modern CSS with responsive design
- Smooth scrolling and animations
- Clean, semantic HTML structure
- ES6+ JavaScript

## Development

- Edit files in the project directory
- Refresh browser to see changes
- Use browser dev tools for debugging
"""
        }
        
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(file_path)
        
        return files_created
    
    def _create_data_project(self, project_path: Path, project_name: str) -> list[str]:
        """Create data science project structure"""
        files_created = []
        
        dirs = ["data/raw", "data/processed", "notebooks", "src", "reports", "models"]
        for dir_name in dirs:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        files = {
            "README.md": f"""# {project_name}

    A data science project created by Mortey Assistant.

    ## Project Structure

    {project_name}/
    ├── data/
    │ ├── raw/ # Raw, immutable data
    │ └── processed/ # Cleaned, processed data
    ├── notebooks/ # Jupyter notebooks
    ├── src/ # Source code
    ├── reports/ # Generated reports
    ├── models/ # Trained models
    └── requirements.txt # Dependencies

    ## Getting Started

    1. Install dependencies:

    pip install -r requirements.txt

    2. Start Jupyter:

    jupyter lab

    3. Open `notebooks/01_exploration.ipynb`
    """,
        "requirements.txt": """# Data Science Stack
    pandas>=1.5.0
    numpy>=1.24.0
    matplotlib>=3.6.0
    seaborn>=0.12.0
    scikit-learn>=1.2.0
    jupyter>=1.0.0
    jupyterlab>=3.5.0

    # Optional but recommended
    plotly>=5.12.0
    scipy>=1.10.0
    statsmodels>=0.13.0
    """,
        "notebooks/01_exploration.ipynb": """{
    "cells": [
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
    "# Data Exploration\\n",
    "\\n",
    "Initial exploration of the dataset."
    ]
    },
    {
    "cell_type": "code",
    "execution_count": null,
    "metadata": {},
    "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Set style\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "\\n",
    "print('Data exploration notebook ready!')"
    ]
    }
    ],
    "metadata": {
    "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
    },
    "language_info": {
    "name": "python",
    "version": "3.11.0"
    }
    },
    "nbformat": 4,
    "nbformat_minor": 4
    }""",
        "src/data_processing.py": f'''"""
    Data processing utilities for {project_name}
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    def load_raw_data(filename: str) -> pd.DataFrame:
    """Load raw data from data/raw directory"""
    data_path = Path(__file__).parent.parent / "data" / "raw" / filename
    
    if filename.endswith('.csv'):
        return pd.read_csv(data_path)
    elif filename.endswith('.json'):
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {{filename}}")

    def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to data/processed directory"""
    output_path = Path(__file__).parent.parent / "data" / "processed" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if filename.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {{filename}}")

    def basic_data_info(df: pd.DataFrame) -> dict:
    """Get basic information about a DataFrame"""
    return {{
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum()
    }}

    if __name__ == "__main__":
    print("Data processing utilities loaded")
    ''',
        ".gitignore": """# Data files
    data/raw/*
    !data/raw/.gitkeep
    data/processed/*
    !data/processed/.gitkeep

    # Models
    models/*
    !models/.gitkeep

    # Jupyter
    .ipynb_checkpoints/
    *.ipynb

    # Python
    __pycache__/
    *.pyc
    *.pyo
    *.pyd
    .Python
    env/
    venv/

    # Reports
    reports/*.pdf
    reports/*.html
    """
    }
    
        # Create .gitkeep files
        for keep_dir in ["data/raw", "data/processed", "models"]:
            (project_path / keep_dir / ".gitkeep").touch()
            files_created.append(f"{keep_dir}/.gitkeep")
        
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(file_path)

        return files_created
    
    def _create_api_project(self, project_path: Path, project_name: str) -> list[str]:
        """Create FastAPI project structure"""
        files_created = []
        
        dirs = ["app", "app/api", "app/models", "app/services", "tests"]
        for dir_name in dirs:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        files = {
            "requirements.txt": """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0
sqlalchemy>=2.0.0
alembic>=1.13.0
pytest>=7.4.0
httpx>=0.25.0
""",
         "app/__init__.py": "",
         "app/main.py": f'''"""
{project_name} FastAPI Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Create FastAPI instance
app = FastAPI(
 title="{project_name}",
 description="API created by Mortey Assistant",
 version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],  # Configure appropriately for production
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
 status: str
 message: str

class ItemCreate(BaseModel):
 name: str
 description: str = None

class ItemResponse(BaseModel):
 id: int
 name: str
 description: str = None

# In-memory storage (use database in production)
items_db = []
next_id = 1

@app.get("/", response_model=HealthResponse)
async def root():
 """Root endpoint"""
 return HealthResponse(status="ok", message="Welcome to {project_name} API")

@app.get("/health", response_model=HealthResponse)
async def health_check():
 """Health check endpoint"""
 return HealthResponse(status="healthy", message="API is running")

@app.post("/items/", response_model=ItemResponse)
async def create_item(item: ItemCreate):
 """Create a new item"""
 global next_id
 new_item = ItemResponse(
     id=next_id,
     name=item.name,
     description=item.description
 )
 items_db.append(new_item)
 next_id += 1
 return new_item

@app.get("/items/", response_model=list[ItemResponse])
async def list_items():
 """List all items"""
 return items_db

@app.get("/items/{{item_id}}", response_model=ItemResponse)
async def get_item(item_id: int):
 """Get item by ID"""
 for item in items_db:
     if item.id == item_id:
         return item
 raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
 uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
''',
         "app/models/__init__.py": "",
         "app/api/__init__.py": "",
         "app/services/__init__.py": "",
         "tests/__init__.py": "",
         "tests/test_main.py": f'''"""
Tests for {project_name} API
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
 """Test root endpoint"""
 response = client.get("/")
 assert response.status_code == 200
 data = response.json()
 assert data["status"] == "ok"

def test_health_check():
 """Test health check endpoint"""
 response = client.get("/health")
 assert response.status_code == 200
 data = response.json()
 assert data["status"] == "healthy"

def test_create_item():
 """Test item creation"""
 item_data = {{"name": "Test Item", "description": "A test item"}}
 response = client.post("/items/", json=item_data)
 assert response.status_code == 200
 data = response.json()
 assert data["name"] == "Test Item"
 assert data["id"] == 1

def test_list_items():
 """Test listing items"""
 response = client.get("/items/")
 assert response.status_code == 200
 data = response.json()
 assert isinstance(data, list)
''',
         "Dockerfile": f"""FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
         "docker-compose.yml": f"""version: '3.8'

services:
api:
 build: .
 ports:
   - "8000:8000"
 environment:
   - ENVIRONMENT=development
 volumes:
   - ./app:/app/app
 command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Add database service when needed
# db:
#   image: postgres:15
#   environment:
#     POSTGRES_DB: {project_name.lower()}
#     POSTGRES_USER: user
#     POSTGRES_PASSWORD: password
#   ports:
#     - "5432:5432"
""",
         "README.md": f"""# {project_name} API

A FastAPI application created by Mortey Assistant.

## Features

- FastAPI with automatic OpenAPI documentation
- Pydantic models for request/response validation
- CORS middleware configured
- Basic CRUD operations
- Docker support
- Test suite with pytest

## Quick Start

1. Install dependencies:

pip install -r requirements.txt

2. Run the development server:

uvicorn app.main:app --reload

3. Visit the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker

Build and run with Docker Compose

docker-compose up --build
Or build and run manually

docker build -t {project_name.lower()}-api .
docker run -p 8000:8000 {project_name.lower()}-api

## Testing

pytest

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /items/` - Create item
- `GET /items/` - List items
- `GET /items/{{id}}` - Get item by ID
"""
        }
        
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(file_path)
        
        return files_created
    
    def _create_ml_project(self, project_path: Path, project_name: str) -> list[str]:
        """Create machine learning project structure"""
        files_created = []
        
        dirs = [
            "data/raw", "data/processed", "data/external",
            "models", "notebooks", "src/data", "src/features", 
            "src/models", "src/visualization", "reports/figures"
        ]
        for dir_name in dirs:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        files = {
            "requirements.txt": """# Core ML stack
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0

# Deep Learning (optional)
# torch>=1.13.0
# tensorflow>=2.11.0

# MLOps
mlflow>=2.1.0
optuna>=3.1.0

# Utilities
click>=8.1.0
python-dotenv>=1.0.0
pyyaml>=6.0
""",
            "src/__init__.py": "",
            "src/data/make_dataset.py": f'''"""
Data ingestion and preprocessing for {project_name}
"""
import pandas as pd
import numpy as np
from pathlib import Path
import click

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Load raw data
    df = pd.read_csv(input_filepath)
    
    # Basic preprocessing
    df_processed = preprocess_data(df)
    
    # Save processed data
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    logger.info(f'processed data saved to {{output_path}}')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data preprocessing"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    
    return df

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
''',
            "src/models/train_model.py": f'''"""
Model training script for {project_name}
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
import click

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(input_filepath, model_filepath):
    """Train a machine learning model"""
    
    # Load processed data
    df = pd.read_csv(input_filepath)
    
    # Prepare features and target (modify as needed)
    # This is a template - adjust for your specific problem
    X = df.drop('target', axis=1)  # Assuming 'target' column exists
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = Path(model_filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        
        print(f"Model trained with accuracy: {{accuracy:.4f}}")
        print(f"Model saved to: {{model_path}}")
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
''',
            "notebooks/01_data_exploration.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration\\n",
    "\\n",
    "Exploratory Data Analysis for the ML project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Set up plotting\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "%matplotlib inline\\n",
    "\\n",
    "print('Data exploration environment ready!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load your dataset here\\n",
    "# df = pd.read_csv('../data/raw/your_dataset.csv')\\n",
    "# df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}""",
            "MLproject": f"""name: {project_name}

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_file: path
      model_output: path
    command: "python src/models/train_model.py {{data_file}} {{model_output}}"
""",
            "README.md": f"""# {project_name}

A machine learning project created by Mortey Assistant.

## Project Organization

├── README.md <- The top-level README
├── data
│ ├── external <- Data from third party sources
│ ├── interim <- Intermediate data that has been transformed
│ ├── processed <- The final, canonical data sets for modeling
│ └── raw <- The original, immutable data dump
│
├── models <- Trained and serialized models, model predictions
│
├── notebooks <- Jupyter notebooks for exploration and analysis
│
├── reports <- Generated analysis as HTML, PDF, LaTeX, etc.
│ └── figures <- Generated graphics and figures
│
├── requirements.txt <- Requirements file for reproducing the environment
│
├── src <- Source code for use in this project
│ ├── init.py <- Makes src a Python module
│ │
│ ├── data <- Scripts to download or generate data
│ │ └── make_dataset.py
│ │
│ ├── features <- Scripts to turn raw data into features for modeling
│ │ └── build_features.py
│ │
│ ├── models <- Scripts to train models and make predictions
│ │ ├── predict_model.py
│ │ └── train_model.py
│ │
│ └── visualization <- Scripts to create exploratory and results oriented visualizations
│ └── visualize.py

## Getting Started

1. Install dependencies:

pip install -r requirements.txt

2. Start MLflow tracking server:

mlflow ui

3. Run data processing:

python src/data/make_dataset.py data/raw/dataset.csv data/processed/dataset.csv

4. Train model:

python src/models/train_model.py data/processed/dataset.csv models/model.pkl

## MLOps Features

- **MLflow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization
- **Reproducible**: Environment and dependency management
- **Modular**: Separate scripts for different pipeline stages

## Notebooks

- `01_data_exploration.ipynb`: Initial data exploration
- `02_feature_engineering.ipynb`: Feature creation and selection
- `03_model_training.ipynb`: Model training and evaluation
"""
     }
        
        # Create .gitkeep files for empty directories
        for keep_dir in ["data/raw", "data/processed", "data/external", "models", "reports/figures"]:
            (project_path / keep_dir / ".gitkeep").touch()
            files_created.append(f"{keep_dir}/.gitkeep")
        
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(file_path)
        
        return files_created

    def _create_generic_project(self, project_path: Path, project_name: str, project_type: str) -> list[str]:
        """Create generic project structure"""
        files_created = []
        
        dirs = ["src", "docs", "tests"]
        for dir_name in dirs:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        files = {
         "README.md": f"""# {project_name}

A {project_type} project created by Mortey Assistant.

## Getting Started

1. Navigate to the project directory:

cd {project_name}

2. Start developing your {project_type} project!

## Project Structure

- `src/` - Source code
- `docs/` - Documentation
- `tests/` - Test files
""",
         "src/main.py": f'''"""
Main module for {project_name}
"""

def main():
 """Main function"""
 print("Hello from {project_name}!")
 print(f"This is a {project_type} project created by Mortey Assistant")

if __name__ == "__main__":
 main()
''',
         ".gitignore": """# General
*.log
*.tmp
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
build/
dist/
"""
    }
     
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_created.append(file_path)

        return files_created

    async def _organize_file_async(
        self, 
        file_path: Path, 
        categories: dict[str, list[str]],  # Python 3.13.4 syntax
        dry_run: bool,
        create_date_folders: bool
    ) -> Optional[tuple[str, Path]]:  # Python 3.13.4 syntax
        """Organize a single file asynchronously"""
        await asyncio.sleep(0)  # Yield control
        
        try:
            _, ext = file_path.suffix, file_path.suffix.lower()
            
            # Find category for this file
            target_category = None
            for category, extensions in categories.items():
                if ext in extensions:
                    target_category = category
                    break
            
            # Skip if no category or special files
            if target_category is None or file_path.name.startswith('.') or file_path.name == "README.md":
                return None
            
            # Create target directory structure
            if create_date_folders:
                file_date = time.strftime('%Y-%m', time.localtime(file_path.stat().st_mtime))
                target_dir = self.workspace_dir / target_category / file_date
            else:
                target_dir = self.workspace_dir / target_category
            
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle duplicate filenames
            target_path = target_dir / file_path.name
            if target_path.exists() and not dry_run:
                base_name = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{base_name}_{counter}{suffix}"
                    counter += 1
            
            # Move file (or simulate in dry run)
            if not dry_run:
                shutil.move(str(file_path), str(target_path))
            
            return target_category, target_path
            
        except Exception as e:
            logger.error(f"Error organizing file {file_path}: {e}")
            return None
    
    def _perform_file_conversion(
        self, 
        input_path: Path, 
        output_path: Path, 
        input_ext: str, 
        output_ext: str
    ) -> FileOperationResult:
        """Perform file format conversion using match-case (Python 3.13.4)"""
        start_time = time.time()
        
        try:
            # Enhanced conversion logic using match-case (Python 3.13.4)
            match (input_ext, output_ext):
                case ("json", "yaml") | ("yaml", "json"):
                    return self._convert_json_yaml(input_path, output_path, input_ext == "json")
                
                case ("csv", "json") | ("json", "csv"):
                    return self._convert_csv_json(input_path, output_path, input_ext == "csv")
                
                case ("md", "txt") | ("txt", "md"):
                    return self._convert_text_formats(input_path, output_path)
                
                case ("xml", "json") | ("json", "xml"):
                    return self._convert_xml_json(input_path, output_path, input_ext == "xml")
                
                case ("yaml", "toml") | ("toml", "yaml"):
                    return self._convert_yaml_toml(input_path, output_path, input_ext == "yaml")
                
                case _:
                    return FileOperationResult(
                        success=False,
                        message=f"Unsupported conversion: {input_ext} to {output_ext}",
                        operation=FileOperation.WRITE
                    )
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return FileOperationResult(
                success=False,
                message=f"Conversion failed: {str(e)}",
                operation=FileOperation.WRITE,
                duration_ms=duration
            )
    
    def _convert_json_yaml(self, input_path: Path, output_path: Path, json_to_yaml: bool) -> FileOperationResult:
        """Convert between JSON and YAML formats"""
        try:
            import yaml
            
            with open(input_path, 'r', encoding='utf-8') as f:
                if json_to_yaml:
                    data = json.load(f)
                    with open(output_path, 'w', encoding='utf-8') as out:
                        yaml.dump(data, out, default_flow_style=False, allow_unicode=True)
                else:  # yaml to json
                    data = yaml.safe_load(f)
                    with open(output_path, 'w', encoding='utf-8') as out:
                        json.dump(data, out, indent=2, ensure_ascii=False)
            
            return FileOperationResult(
                success=True,
                message="Conversion successful",
                operation=FileOperation.WRITE,
                files_affected=[str(output_path)],
                bytes_processed=output_path.stat().st_size
            )
            
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"JSON/YAML conversion failed: {str(e)}",
                operation=FileOperation.WRITE
            )
    
    def _convert_csv_json(self, input_path: Path, output_path: Path, csv_to_json: bool) -> FileOperationResult:
        """Convert between CSV and JSON formats"""
        try:
            import csv
            
            if csv_to_json:
                # CSV to JSON
                data = []
                with open(input_path, 'r', newline='', encoding='utf-8') as f:
                    # Auto-detect delimiter
                    sample = f.read(1024)
                    f.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row in reader:
                        # Convert numeric strings to numbers where possible
                        converted_row = {}
                        for key, value in row.items():
                            try:
                                # Try integer conversion
                                if '.' not in value:
                                    converted_row[key] = int(value)
                                else:
                                    converted_row[key] = float(value)
                            except (ValueError, TypeError):
                                converted_row[key] = value
                        data.append(converted_row)
                
                with open(output_path, 'w', encoding='utf-8') as out:
                    json.dump(data, out, indent=2, ensure_ascii=False)
            else:
                # JSON to CSV
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    raise ValueError("JSON must contain a list of objects to convert to CSV")
                
                if not data:
                    raise ValueError("JSON list is empty")
                
                # Get all possible keys as fieldnames
                fieldnames = set()
                for item in data:
                    if isinstance(item, dict):
                        fieldnames.update(item.keys())
                
                with open(output_path, 'w', newline='', encoding='utf-8') as out:
                    writer = csv.DictWriter(out, fieldnames=list(fieldnames))
                    writer.writeheader()
                    for item in data:
                        if isinstance(item, dict):
                            writer.writerow(item)
            
            return FileOperationResult(
                success=True,
                message="CSV/JSON conversion successful",
                operation=FileOperation.WRITE,
                files_affected=[str(output_path)],
                bytes_processed=output_path.stat().st_size
            )
            
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"CSV/JSON conversion failed: {str(e)}",
                operation=FileOperation.WRITE
            )
    
    def _convert_text_formats(self, input_path: Path, output_path: Path) -> FileOperationResult:
        """Convert between text formats (MD, TXT)"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            with open(output_path, 'w', encoding='utf-8') as out:
                out.write(content)
            
            return FileOperationResult(
                success=True,
                message="Text format conversion successful",
                operation=FileOperation.WRITE,
                files_affected=[str(output_path)],
                bytes_processed=output_path.stat().st_size
            )
            
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Text conversion failed: {str(e)}",
                operation=FileOperation.WRITE
            )
    
    def _convert_xml_json(self, input_path: Path, output_path: Path, xml_to_json: bool) -> FileOperationResult:
        """Convert between XML and JSON formats"""
        try:
            import xml.etree.ElementTree as ET
            
            if xml_to_json:
                # XML to JSON
                tree = ET.parse(input_path)
                root = tree.getroot()
                
                def xml_to_dict(element):
                    result = {}
                    
                    # Add attributes
                    if element.attrib:
                        result['@attributes'] = element.attrib
                    
                    # Add text content
                    if element.text and element.text.strip():
                        if len(element) == 0:  # No children
                            return element.text.strip()
                        else:
                            result['#text'] = element.text.strip()
                    
                    # Add children
                    for child in element:
                        child_data = xml_to_dict(child)
                        if child.tag in result:
                            # Convert to list if multiple children with same tag
                            if not isinstance(result[child.tag], list):
                                result[child.tag] = [result[child.tag]]
                            result[child.tag].append(child_data)
                        else:
                            result[child.tag] = child_data
                    
                    return result
                
                data = {root.tag: xml_to_dict(root)}
                
                with open(output_path, 'w', encoding='utf-8') as out:
                    json.dump(data, out, indent=2, ensure_ascii=False)
            else:
                # JSON to XML (simplified)
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                def dict_to_xml(tag, d):
                    elem = ET.Element(tag)
                    if isinstance(d, dict):
                        for key, val in d.items():
                            if key.startswith('@'):
                                continue  # Skip attributes for now
                            child = dict_to_xml(key, val)
                            elem.append(child)
                    else:
                        elem.text = str(d)
                    return elem
                
                if isinstance(data, dict) and len(data) == 1:
                    root_tag, root_data = next(iter(data.items()))
                    root = dict_to_xml(root_tag, root_data)
                else:
                    root = dict_to_xml('root', data)
                
                tree = ET.ElementTree(root)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            return FileOperationResult(
                success=True,
                message="XML/JSON conversion successful",
                operation=FileOperation.WRITE,
                files_affected=[str(output_path)],
                bytes_processed=output_path.stat().st_size
            )
            
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"XML/JSON conversion failed: {str(e)}",
                operation=FileOperation.WRITE
            )
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Enhanced MIME type detection"""
        # Enhanced mime type mapping
        mime_map = {
            '.txt': 'text/plain',
            '.py': 'text/x-python',
            '.js': 'application/javascript',
            '.ts': 'application/typescript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.csv': 'text/csv',
            '.md': 'text/markdown',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.toml': 'application/toml',
            '.ini': 'text/plain',
            '.cfg': 'text/plain',
            '.conf': 'text/plain',
            '.log': 'text/plain',
            '.sql': 'application/sql',
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
            '.ico': 'image/x-icon',
            '.zip': 'application/zip',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip',
            '.7z': 'application/x-7z-compressed',
            '.rar': 'application/vnd.rar',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac'
        }
        
        ext = file_path.suffix.lower()
        return mime_map.get(ext, 'application/octet-stream')
    
    def _format_age(self, seconds: float) -> str:
        """Format age in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds // 60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours"
        else:
            days = int(seconds // 86400)
            return f"{days} day{'s' if days != 1 else ''}"
    
    def _analyze_file_content(self, file_path: Path) -> str:
        """Analyze file content for additional insights"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Basic content analysis
            lines = content.split('\n')
            words = content.split()
            chars = len(content)
            
            # Language detection for code files
            language = "Unknown"
            if file_path.suffix.lower() in ['.py', '.js', '.html', '.css', '.sql']:
                language = {
                    '.py': 'Python',
                    '.js': 'JavaScript',
                    '.html': 'HTML',
                    '.css': 'CSS',
                    '.sql': 'SQL'
                }.get(file_path.suffix.lower(), 'Unknown')
            
            # Check for common patterns
            patterns = []
            if 'import ' in content or 'from ' in content:
                patterns.append("Contains imports")
            if 'def ' in content or 'function ' in content:
                patterns.append("Contains functions")
            if 'class ' in content:
                patterns.append("Contains classes")
            if any(word in content.lower() for word in ['todo', 'fixme', 'hack', 'bug']):
                patterns.append("Contains TODO/FIXME comments")
            
            analysis = f"""- Lines: {len(lines):,}
- Words: {len(words):,}
- Characters: {chars:,}
- Language: {language}
- Encoding: UTF-8
- Empty lines: {sum(1 for line in lines if not line.strip()):,}"""
            
            if patterns:
                analysis += f"\n- Patterns: {', '.join(patterns)}"
            
            return analysis
            
        except Exception as e:
            return f"Content analysis failed: {str(e)}"
    
    def _calculate_workspace_stats(self) -> dict[str, any]:  # Python 3.13.4 syntax
        """Calculate comprehensive workspace statistics"""
        stats = {
            'total_files': 0,
            'total_dirs': 0,
            'total_size': 0,
            'file_types': {},
            'largest_file': '',
            'largest_size': 0,
            'safe_files': 0,
            'blocked_files': 0,
            'large_files': 0
        }
        
        try:
            for item in self.workspace_dir.rglob("*"):
                if item.is_file():
                    stats['total_files'] += 1
                    size = item.stat().st_size
                    stats['total_size'] += size
                    
                    # Track largest file
                    if size > stats['largest_size']:
                        stats['largest_size'] = size
                        stats['largest_file'] = str(item.relative_to(self.workspace_dir))
                    
                    # Track file types
                    ext = item.suffix.lower() or 'No extension'
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                    
                    # Security analysis
                    if self.security_config.is_extension_allowed(item.name):
                        stats['safe_files'] += 1
                    else:
                        stats['blocked_files'] += 1
                    
                    # Large file detection
                    if size > (self.security_config.max_file_size_mb * 1024 * 1024):
                        stats['large_files'] += 1
                        
                elif item.is_dir():
                    stats['total_dirs'] += 1
            
            # Format total size
            size_bytes = stats['total_size']
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024 or unit == 'TB':
                    break
                size_bytes /= 1024
            stats['total_size_formatted'] = f"{size_bytes:.2f} {unit}"
            
            # Format largest file size
            largest_bytes = stats['largest_size']
            for unit in ['B', 'KB', 'MB', 'GB']:
                if largest_bytes < 1024 or unit == 'GB':
                    break
                largest_bytes /= 1024
            stats['largest_size_formatted'] = f"{largest_bytes:.2f} {unit}"
            
        except Exception as e:
            logger.error(f"Error calculating workspace stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def get_operation_statistics(self) -> dict[str, any]:  # Python 3.13.4 syntax
        """Get file operation statistics"""
        return {
            "workspace_path": str(self.workspace_dir),
            "security_config": {
                "allowed_extensions": list(self.security_config.allowed_extensions),
                "max_file_size_mb": self.security_config.max_file_size_mb,
                "max_files_per_operation": self.security_config.max_files_per_operation
            },
            "operation_stats": dict(self._operation_stats),
            "toolkit_available": hasattr(self, 'toolkit') and self.toolkit is not None
        }

# Global file tools instance
file_tools = FileSystemTools()
