# core/file_processing_pipeline.py - Comprehensive File Processing Pipeline

import logging
import asyncio
import os
import time
import mimetypes
import hashlib
from typing import Optional, List, Dict, Any, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.enhanced_state import (
    EnhancedAssistantState, FileProcessingManager, ScratchpadManager,
    AgentCommunicationManager, AgentMessage, MessageType
)
from core.error_handling import ErrorHandler, with_error_handling
from tools.file_tools import FileSystemTools

logger = logging.getLogger("file_processing_pipeline")

class ProcessingStep(Enum):
    """File processing steps"""
    UPLOAD = "upload"
    VALIDATION = "validation"
    CONTENT_EXTRACTION = "content_extraction"
    ANALYSIS = "analysis"
    INDEXING = "indexing"
    COMPLETION = "completion"

class FileType(Enum):
    """Supported file types"""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    IMAGE = "image"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"

@dataclass
class ProcessingResult:
    """Result of a processing step"""
    step: ProcessingStep
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FileMetadata:
    """Comprehensive file metadata"""
    filename: str = ""
    file_size: int = 0
    file_type: FileType = FileType.UNKNOWN
    mime_type: str = ""
    md5_hash: str = ""
    sha256_hash: str = ""
    upload_timestamp: float = field(default_factory=time.time)
    processing_agent: str = ""
    tags: List[str] = field(default_factory=list)

class FileProcessingPipeline:
    """Comprehensive file processing pipeline with multi-format support"""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.file_tools = FileSystemTools(workspace_dir)
        self.supported_extensions = {
            # Text files
            '.txt': FileType.TEXT,
            '.md': FileType.TEXT,
            '.rtf': FileType.TEXT,
            
            # Code files
            '.py': FileType.CODE,
            '.js': FileType.CODE,
            '.html': FileType.CODE,
            '.css': FileType.CODE,
            '.java': FileType.CODE,
            '.cpp': FileType.CODE,
            '.c': FileType.CODE,
            
            # Data files
            '.json': FileType.DATA,
            '.csv': FileType.DATA,
            '.xml': FileType.DATA,
            '.yaml': FileType.DATA,
            '.yml': FileType.DATA,
            
            # Document files
            '.pdf': FileType.DOCUMENT,
            '.doc': FileType.DOCUMENT,
            '.docx': FileType.DOCUMENT,
            
            # Archive files
            '.zip': FileType.ARCHIVE,
            '.tar': FileType.ARCHIVE,
            '.gz': FileType.ARCHIVE,
        }
        
        self.processing_stats = {
            "total_files": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "files_by_type": {},
            "avg_processing_time": 0.0
        }
    
    async def process_file_upload(
        self,
        state: EnhancedAssistantState,
        file_path: str,
        requesting_agent: str = "file_manager",
        notify_agents: Optional[List[str]] = None
    ) -> tuple[EnhancedAssistantState, str, Dict[str, Any]]:
        """Process uploaded file through complete pipeline"""
        
        start_time = time.time()
        
        try:
            # Step 1: Add to processing queue
            state, file_id = FileProcessingManager.add_file_to_queue(
                state, 
                os.path.basename(file_path),
                total_steps=6
            )
            
            # Step 2: Upload and validation
            state, upload_result = await self._process_upload_step(
                state, file_id, file_path, requesting_agent
            )
            
            if not upload_result.success:
                return state, file_id, {"error": "Upload failed", "details": upload_result.errors}
            
            # Step 3: Content extraction
            state, extraction_result = await self._process_extraction_step(
                state, file_id, upload_result.data["file_metadata"]
            )
            
            # Step 4: Analysis
            state, analysis_result = await self._process_analysis_step(
                state, file_id, extraction_result.data
            )
            
            # Step 5: Indexing and storage
            state, indexing_result = await self._process_indexing_step(
                state, file_id, analysis_result.data
            )
            
            # Step 6: Completion and notification
            state = await self._process_completion_step(
                state, file_id, notify_agents, requesting_agent
            )
            
            # Update stats
            self.processing_stats["total_files"] += 1
            self.processing_stats["successful_processing"] += 1
            
            processing_time = time.time() - start_time
            total_time = (self.processing_stats["avg_processing_time"] * 
                         (self.processing_stats["total_files"] - 1) + processing_time)
            self.processing_stats["avg_processing_time"] = total_time / self.processing_stats["total_files"]
            
            # Compile final results
            final_results = {
                "file_id": file_id,
                "processing_time": processing_time,
                "upload": upload_result.data,
                "extraction": extraction_result.data,
                "analysis": analysis_result.data,
                "indexing": indexing_result.data
            }
            
            logger.info(f"✅ File processing completed: {file_id} in {processing_time:.2f}s")
            
            return state, file_id, final_results
            
        except Exception as e:
            self.processing_stats["total_files"] += 1
            self.processing_stats["failed_processing"] += 1
            
            logger.error(f"❌ File processing failed: {e}")
            return state, file_id or "unknown", {"error": str(e)}
    
    async def _process_upload_step(
        self, 
        state: EnhancedAssistantState, 
        file_id: str, 
        file_path: str,
        requesting_agent: str
    ) -> tuple[EnhancedAssistantState, ProcessingResult]:
        """Process file upload and validation step"""
        
        try:
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 10.0, "Uploading and validating file"
            )
            
            # Check file exists
            if not os.path.exists(file_path):
                return state, ProcessingResult(
                    step=ProcessingStep.UPLOAD,
                    success=False,
                    errors=["File not found"]
                )
            
            # Generate file metadata
            file_metadata = await self._generate_file_metadata(file_path, requesting_agent)
            
            # Validate file
            validation_errors = await self._validate_file(file_path, file_metadata)
            if validation_errors:
                return state, ProcessingResult(
                    step=ProcessingStep.UPLOAD,
                    success=False,
                    errors=validation_errors
                )
            
            # Copy to workspace
            workspace_path = await self._copy_to_workspace(file_path, file_metadata.filename)
            
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 20.0, "File uploaded successfully"
            )
            
            # Store in scratchpad
            state = ScratchpadManager.set_data(
                state,
                f"file_metadata_{file_id}",
                file_metadata,
                requesting_agent,
                tags=["file_upload", "metadata"]
            )
            
            return state, ProcessingResult(
                step=ProcessingStep.UPLOAD,
                success=True,
                data={
                    "file_metadata": file_metadata,
                    "workspace_path": workspace_path
                }
            )
            
        except Exception as e:
            return state, ProcessingResult(
                step=ProcessingStep.UPLOAD,
                success=False,
                errors=[f"Upload failed: {str(e)}"]
            )
    
    async def _process_extraction_step(
        self,
        state: EnhancedAssistantState,
        file_id: str,
        file_metadata: FileMetadata
    ) -> tuple[EnhancedAssistantState, ProcessingResult]:
        """Process content extraction step"""
        
        try:
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 40.0, "Extracting content"
            )
            
            extracted_content = {}
            
            # Extract based on file type
            if file_metadata.file_type == FileType.TEXT:
                extracted_content = await self._extract_text_content(file_metadata.filename)
            elif file_metadata.file_type == FileType.CODE:
                extracted_content = await self._extract_code_content(file_metadata.filename)
            elif file_metadata.file_type == FileType.DATA:
                extracted_content = await self._extract_data_content(file_metadata.filename)
            elif file_metadata.file_type == FileType.DOCUMENT:
                extracted_content = await self._extract_document_content(file_metadata.filename)
            else:
                extracted_content = {"type": "unsupported", "message": "Content extraction not supported for this file type"}
            
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 50.0, "Content extracted"
            )
            
            return state, ProcessingResult(
                step=ProcessingStep.CONTENT_EXTRACTION,
                success=True,
                data={
                    "extracted_content": extracted_content,
                    "content_summary": self._generate_content_summary(extracted_content)
                }
            )
            
        except Exception as e:
            return state, ProcessingResult(
                step=ProcessingStep.CONTENT_EXTRACTION,
                success=False,
                errors=[f"Content extraction failed: {str(e)}"]
            )
    
    async def _process_analysis_step(
        self,
        state: EnhancedAssistantState,
        file_id: str,
        extraction_data: Dict[str, Any]
    ) -> tuple[EnhancedAssistantState, ProcessingResult]:
        """Process content analysis step"""
        
        try:
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 70.0, "Analyzing content"
            )
            
            extracted_content = extraction_data.get("extracted_content", {})
            
            analysis_results = {
                "word_count": 0,
                "line_count": 0,
                "character_count": 0,
                "key_terms": [],
                "structure_analysis": {},
                "quality_score": 0.0
            }
            
            # Perform analysis based on content type
            if "text" in extracted_content:
                text = extracted_content["text"]
                analysis_results.update({
                    "word_count": len(text.split()),
                    "line_count": len(text.split('\n')),
                    "character_count": len(text),
                    "key_terms": self._extract_key_terms(text),
                    "readability_score": self._calculate_readability(text)
                })
            
            if "code" in extracted_content:
                code = extracted_content["code"]
                analysis_results.update({
                    "functions": extracted_content.get("functions", []),
                    "classes": extracted_content.get("classes", []),
                    "imports": extracted_content.get("imports", []),
                    "complexity_score": self._calculate_code_complexity(code)
                })
            
            if "data" in extracted_content:
                data = extracted_content["data"]
                analysis_results.update({
                    "data_structure": self._analyze_data_structure(data),
                    "data_quality": self._assess_data_quality(data)
                })
            
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 80.0, "Analysis completed"
            )
            
            return state, ProcessingResult(
                step=ProcessingStep.ANALYSIS,
                success=True,
                data={"analysis_results": analysis_results}
            )
            
        except Exception as e:
            return state, ProcessingResult(
                step=ProcessingStep.ANALYSIS,
                success=False,
                errors=[f"Analysis failed: {str(e)}"]
            )
    
    async def _process_indexing_step(
        self,
        state: EnhancedAssistantState,
        file_id: str,
        analysis_data: Dict[str, Any]
    ) -> tuple[EnhancedAssistantState, ProcessingResult]:
        """Process indexing and storage step"""
        
        try:
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 90.0, "Indexing and storing results"
            )
            
            analysis_results = analysis_data.get("analysis_results", {})
            
            # Create searchable index
            index_data = {
                "file_id": file_id,
                "searchable_content": self._create_searchable_content(analysis_results),
                "tags": self._generate_auto_tags(analysis_results),
                "metadata_index": self._create_metadata_index(analysis_results)
            }
            
            # Store in scratchpad for future searches
            state = ScratchpadManager.set_data(
                state,
                f"file_index_{file_id}",
                index_data,
                "file_manager",
                tags=["file_index", "searchable"],
                expires_in=86400  # 24 hours
            )
            
            # Update progress
            state = FileProcessingManager.update_file_progress(
                state, file_id, 95.0, "Indexing completed"
            )
            
            return state, ProcessingResult(
                step=ProcessingStep.INDEXING,
                success=True,
                data={"index_data": index_data}
            )
            
        except Exception as e:
            return state, ProcessingResult(
                step=ProcessingStep.INDEXING,
                success=False,
                errors=[f"Indexing failed: {str(e)}"]
            )
    
    async def _process_completion_step(
        self,
        state: EnhancedAssistantState,
        file_id: str,
        notify_agents: Optional[List[str]],
        requesting_agent: str
    ) -> EnhancedAssistantState:
        """Process completion and notification step"""
        
        try:
            # Update progress to completion
            state = FileProcessingManager.update_file_progress(
                state, file_id, 100.0, "Processing completed"
            )
            
            # Send notifications to other agents if requested
            if notify_agents:
                for agent in notify_agents:
                    if agent != requesting_agent:
                        message = AgentMessage(
                            from_agent="file_manager",
                            to_agent=agent,
                            message_type=MessageType.NOTIFICATION,
                            content=f"File processing completed: {file_id}",
                            data={"file_id": file_id, "status": "completed"}
                        )
                        state = AgentCommunicationManager.send_message(state, message)
            
            logger.info(f"✅ File processing pipeline completed for: {file_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Completion step failed: {e}")
            return state
    
    # Helper methods for content extraction and analysis
    
    async def _generate_file_metadata(self, file_path: str, processing_agent: str) -> FileMetadata:
        """Generate comprehensive file metadata"""
        file_stat = os.stat(file_path)
        filename = os.path.basename(file_path)
        
        # Determine file type
        extension = Path(file_path).suffix.lower()
        file_type = self.supported_extensions.get(extension, FileType.UNKNOWN)
        
        # Generate hashes
        with open(file_path, 'rb') as f:
            file_content = f.read()
            md5_hash = hashlib.md5(file_content).hexdigest()
            sha256_hash = hashlib.sha256(file_content).hexdigest()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return FileMetadata(
            filename=filename,
            file_size=file_stat.st_size,
            file_type=file_type,
            mime_type=mime_type or "application/octet-stream",
            md5_hash=md5_hash,
            sha256_hash=sha256_hash,
            processing_agent=processing_agent
        )
    
    async def _validate_file(self, file_path: str, metadata: FileMetadata) -> List[str]:
        """Validate uploaded file"""
        errors = []
        
        # Size validation (50MB limit)
        if metadata.file_size > 50 * 1024 * 1024:
            errors.append("File size exceeds 50MB limit")
        
        # Type validation
        if metadata.file_type == FileType.UNKNOWN:
            errors.append("Unsupported file type")
        
        # Security validation
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr']
        if any(metadata.filename.lower().endswith(ext) for ext in dangerous_extensions):
            errors.append("Potentially dangerous file type")
        
        return errors
    
    async def _copy_to_workspace(self, source_path: str, filename: str) -> str:
        """Copy file to workspace"""
        import shutil
        from config.settings import config
        
        workspace_path = config.workspace_dir / filename
        shutil.copy2(source_path, workspace_path)
        return str(workspace_path)
    
    async def _extract_text_content(self, filename: str) -> Dict[str, Any]:
        """Extract content from text files"""
        from config.settings import config
        
        file_path = config.workspace_dir / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return {
                "type": "text",
                "text": text,
                "encoding": "utf-8",
                "preview": text[:500] + "..." if len(text) > 500 else text
            }
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            
            return {
                "type": "text",
                "text": text,
                "encoding": "latin-1",
                "preview": text[:500] + "..." if len(text) > 500 else text
            }
    
    async def _extract_code_content(self, filename: str) -> Dict[str, Any]:
        """Extract content from code files"""
        text_content = await self._extract_text_content(filename)
        code = text_content["text"]
        
        # Basic code analysis
        functions = []
        classes = []
        imports = []
        
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                functions.append(line)
            elif line.startswith('class '):
                classes.append(line)
            elif line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        return {
            "type": "code",
            "code": code,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "language": self._detect_language(filename),
            "preview": code[:500] + "..." if len(code) > 500 else code
        }
    
    async def _extract_data_content(self, filename: str) -> Dict[str, Any]:
        """Extract content from data files"""
        from config.settings import config
        
        file_path = config.workspace_dir / filename
        extension = Path(filename).suffix.lower()
        
        try:
            if extension == '.json':
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return {
                    "type": "data",
                    "format": "json",
                    "data": data,
                    "preview": str(data)[:500] + "..." if len(str(data)) > 500 else str(data)
                }
            
            elif extension in ['.csv']:
                import pandas as pd
                df = pd.read_csv(file_path)
                return {
                    "type": "data",
                    "format": "csv",
                    "data": df.to_dict(),
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "preview": df.head().to_string()
                }
            
            elif extension in ['.yaml', '.yml']:
                import yaml
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                return {
                    "type": "data",
                    "format": "yaml",
                    "data": data,
                    "preview": str(data)[:500] + "..." if len(str(data)) > 500 else str(data)
                }
            
            else:
                # Fallback to text extraction
                return await self._extract_text_content(filename)
                
        except Exception as e:
            return {
                "type": "error",
                "error": f"Data extraction failed: {str(e)}"
            }
    
    async def _extract_document_content(self, filename: str) -> Dict[str, Any]:
        """Extract content from document files"""
        # For now, return placeholder - would need specific libraries for PDF, Word, etc.
        return {
            "type": "document",
            "message": "Document content extraction requires additional libraries",
            "filename": filename
        }
    
    def _generate_content_summary(self, content: Dict[str, Any]) -> str:
        """Generate content summary"""
        content_type = content.get("type", "unknown")
        
        if content_type == "text":
            text = content.get("text", "")
            return f"Text file with {len(text.split())} words, {len(text.split('\n'))} lines"
        elif content_type == "code":
            functions = len(content.get("functions", []))
            classes = len(content.get("classes", []))
            return f"Code file with {functions} functions, {classes} classes"
        elif content_type == "data":
            data_format = content.get("format", "unknown")
            return f"Data file in {data_format} format"
        else:
            return f"File of type {content_type}"
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple implementation - could be enhanced with NLP
        words = text.lower().split()
        # Filter out common words and return frequent terms
        common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        key_words = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Count frequency and return top terms
        from collections import Counter
        word_counts = Counter(key_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        # Simple readability metric
        words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        if sentences == 0:
            return 0.0
        
        avg_words_per_sentence = words / sentences
        # Simple scale: shorter sentences = higher readability
        return max(0.0, min(10.0, 10.0 - (avg_words_per_sentence / 5.0)))
    
    def _calculate_code_complexity(self, code: str) -> float:
        """Calculate code complexity score"""
        # Simple complexity metric based on control structures
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally']
        complexity = 1  # Base complexity
        
        for keyword in control_keywords:
            complexity += code.count(keyword)
        
        return min(10.0, complexity / 10.0)  # Normalize to 0-10 scale
    
    def _analyze_data_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze data structure"""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:10],  # First 10 keys
                "key_count": len(data.keys())
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample_items": data[:3] if len(data) > 0 else []
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100]
            }
    
    def _assess_data_quality(self, data: Any) -> Dict[str, Any]:
        """Assess data quality"""
        return {
            "completeness": 0.8,  # Placeholder
            "consistency": 0.9,   # Placeholder
            "validity": 0.85      # Placeholder
        }
    
    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename"""
        extension_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        extension = Path(filename).suffix.lower()
        return extension_mapping.get(extension, 'unknown')
    
    def _create_searchable_content(self, analysis_results: Dict[str, Any]) -> str:
        """Create searchable content index"""
        searchable_parts = []
        
        if "key_terms" in analysis_results:
            searchable_parts.extend(analysis_results["key_terms"])
        
        if "functions" in analysis_results:
            searchable_parts.extend(analysis_results["functions"])
        
        if "classes" in analysis_results:
            searchable_parts.extend(analysis_results["classes"])
        
        return " ".join(searchable_parts)
    
    def _generate_auto_tags(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate automatic tags based on analysis"""
        tags = []
        
        if "functions" in analysis_results and analysis_results["functions"]:
            tags.append("code")
            tags.append("functions")
        
        if "data_structure" in analysis_results:
            tags.append("data")
        
        if "readability_score" in analysis_results:
            tags.append("text")
        
        return tags
    
    def _create_metadata_index(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata index for searching"""
        return {
            "word_count": analysis_results.get("word_count", 0),
            "complexity": analysis_results.get("complexity_score", 0),
            "quality": analysis_results.get("quality_score", 0),
            "tags": self._generate_auto_tags(analysis_results)
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing pipeline statistics"""
        return dict(self.processing_stats)

# Global file processing pipeline
file_processing_pipeline = FileProcessingPipeline()
