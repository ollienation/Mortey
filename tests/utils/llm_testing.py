# tests/utils/llm_testing.py - Enhanced LLM Testing Utilities
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, AIMessage
from sentence_transformers import SentenceTransformer

@dataclass
class LLMResponseValidation:
    """Validation results for LLM responses."""
    is_valid: bool
    confidence_score: float
    validation_errors: List[str]
    semantic_similarity: Optional[float] = None
    structure_compliance: bool = True
    
class LLMResponseValidator:
    """Validate LLM responses using semantic and structural checks."""
    
    def __init__(self):
        # Load semantic similarity model for response validation
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def validate_response_structure(
        self, 
        response: AIMessage, 
        expected_structure: Dict[str, Any]
    ) -> LLMResponseValidation:
        """Validate response structure and semantic content."""
        errors = []
        
        # Check tool calls if expected
        if expected_structure.get("has_tool_calls", False):
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                errors.append("Expected tool calls but none found")
        
        # Check content patterns
        if patterns := expected_structure.get("content_patterns", []):
            content = response.content.lower()
            for pattern in patterns:
                if not re.search(pattern, content):
                    errors.append(f"Content pattern not found: {pattern}")
        
        # Semantic similarity check
        similarity_score = None
        if reference_text := expected_structure.get("reference_text"):
            similarity_score = self._calculate_similarity(
                response.content, 
                reference_text
            )
            
            min_similarity = expected_structure.get("min_similarity", 0.7)
            if similarity_score < min_similarity:
                errors.append(f"Semantic similarity too low: {similarity_score:.3f}")
        
        return LLMResponseValidation(
            is_valid=len(errors) == 0,
            confidence_score=1.0 - (len(errors) * 0.2),
            validation_errors=errors,
            semantic_similarity=similarity_score,
            structure_compliance=len(errors) == 0
        )
