# Security Controller with Modern Interrupt Pattern
# June 2025 - Production Ready

import asyncio
import os
import re
import hashlib
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from langgraph.types import interrupt
from config.llm_manager import llm_manager
from config.settings import config
from core.state import AssistantState

# Setup secure logging
logger = logging.getLogger("controller")

class ApprovalDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"

@dataclass
class SecurityCheck:
    passed: bool
    reason: str
    confidence: float
    requires_human_review: bool = False

class Controller:
    """
    Modern security controller using LangGraph interrupt patterns.
    
    Key improvements for June 2025:
    - Uses interrupt() instead of force-approval bypass
    - Proper human-in-the-loop patterns
    - Production-ready error handling
    - Rate limiting and session tracking
    - Semaphore-based concurrency control
    """

    def __init__(self):
        # Security configurations
        self.MAX_CONTENT_LENGTH = 50000
        self.MAX_SESSION_REQUESTS = 100
        self.REQUEST_WINDOW_SECONDS = 3600
        
        # Critical security patterns (only block truly dangerous operations)
        self.critical_patterns = [
            r'rm\s+-rf\s+/', # Only block root deletion
            r'sudo\s+rm\s+-rf', # Sudo root deletion
            r'format\s+c:', # Format C drive
            r'DROP\s+TABLE.*WHERE.*1=1', # Dangerous SQL
            r'del\s+/[sq]', # Windows system deletion
        ]
        
        # Session tracking for rate limiting
        self.session_requests = {}
        self.session_timestamps = {}
        
        # Concurrency control
        self.MAX_CONCURRENT_REQUESTS = 5
        self._request_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        
        logger.info("🛡️ Security controller initialized with interrupt patterns")

    async def verify_and_approve(self, state: AssistantState) -> AssistantState:
        """
        Modern verification using interrupt patterns instead of force-approval.
        This replaces the dangerous force-approval logic with proper human-in-the-loop.
        """
        # Apply semaphore for concurrency control
        async with self._request_semaphore:
            session_id = state.get('session_id', 'default')
            output_content = state.get('output_content', '')
            output_type = state.get('output_type', 'text')
            
            try:
                # 1. Input validation
                validation_check = self._validate_input(output_content)
                if not validation_check.passed:
                    return self._create_rejection_response(state, validation_check.reason)
                
                # 2. Rate limiting check
                rate_check = self._check_rate_limits(session_id)
                if not rate_check.passed:
                    return self._create_rejection_response(state, rate_check.reason)
                
                # 3. Critical security pattern check
                security_check = self._check_critical_patterns(output_content)
                if not security_check.passed:
                    return self._create_rejection_response(state, security_check.reason)
                
                # 4. AI-powered content analysis
                ai_check = await self._ai_safety_analysis(output_content, output_type)
                
                # 5. Decide on approval flow
                if ai_check.requires_human_review:
                    # Use interrupt for human review instead of force-approval
                    return await self._request_human_approval(state, ai_check.reason)
                elif not ai_check.passed:
                    return self._create_rejection_response(state, ai_check.reason)
                else:
                    # Auto-approve safe content
                    return self._create_approval_response(state)
            except Exception as e:
                error_message = self._secure_error_handling(e, "verify_and_approve")
                return self._create_rejection_response(state, error_message)

    async def _request_human_approval(self, state: AssistantState, reason: str) -> AssistantState:
        """
        Request human approval using LangGraph interrupt pattern.
        This is the replacement for force-approval bypass.
        """
        approval_payload = {
            "content": state.get('output_content', ''),
            "type": state.get('output_type', 'text'),
            "reason": reason,
            "session_id": state.get('session_id', ''),
            "timestamp": time.time()
        }
        
        # Use LangGraph interrupt to pause for human review
        try:
            human_response = interrupt({
                "message": f"🔍 Human review required: {reason}",
                "content": approval_payload['content'][:500],
                "type": approval_payload['type'],
                "reason": reason
            })
            
            # Process human response
            decision = human_response.get('decision', 'reject').lower()
            if decision == 'approve':
                return self._create_approval_response(state)
            elif decision == 'edit':
                edited_content = human_response.get('edited_content', state.get('output_content', ''))
                return self._create_approval_response({
                    **state,
                    'output_content': edited_content
                })
            else:  # reject
                rejection_reason = human_response.get('reason', 'Rejected by human reviewer')
                return self._create_rejection_response(state, rejection_reason)
        except Exception as e:
            logger.error(f"Error in human approval process: {e}")
            return self._create_rejection_response(state, "Human approval process failed")

    def _validate_input(self, content: str) -> SecurityCheck:
        """Comprehensive input validation"""
        # Length validation
        if len(content) > self.MAX_CONTENT_LENGTH:
            return SecurityCheck(
                passed=False,
                reason="Content too long for security review",
                confidence=1.0
            )
            
        # Character encoding validation
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            return SecurityCheck(
                passed=False,
                reason="Invalid character encoding detected",
                confidence=1.0
            )
            
        # Null byte injection check
        if '\x00' in content:
            return SecurityCheck(
                passed=False,
                reason="Invalid characters detected",
                confidence=1.0
            )
            
        return SecurityCheck(
            passed=True,
            reason="Input validation passed",
            confidence=1.0
        )
        
    def _check_rate_limits(self, session_id: str) -> SecurityCheck:
        """Rate limiting to prevent abuse"""
        current_time = time.time()
        
        # Initialize session tracking
        if session_id not in self.session_requests:
            self.session_requests[session_id] = 0
            self.session_timestamps[session_id] = []
        
        # Clean old timestamps
        cutoff_time = current_time - self.REQUEST_WINDOW_SECONDS
        self.session_timestamps[session_id] = [
            ts for ts in self.session_timestamps[session_id] if ts > cutoff_time
        ]
        
        # Check rate limit
        if len(self.session_timestamps[session_id]) >= self.MAX_SESSION_REQUESTS:
            return SecurityCheck(
                passed=False,
                reason="Rate limit exceeded. Please wait before trying again.",
                confidence=1.0
            )
            
        # Update tracking
        self.session_timestamps[session_id].append(current_time)
        self.session_requests[session_id] += 1
        
        return SecurityCheck(
            passed=True,
            reason="Rate limiting check passed",
            confidence=1.0
        )
        
    def _check_critical_patterns(self, content: str) -> SecurityCheck:
        """Check for truly dangerous patterns that should always be blocked"""
        content_lower = content.lower()
        
        for pattern in self.critical_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                logger.warning(f"Critical security pattern detected: {pattern}")
                return SecurityCheck(
                    passed=False,
                    reason="Content contains potentially dangerous operations",
                    confidence=0.95
                )
                
        return SecurityCheck(
            passed=True,
            reason="Critical pattern check passed",
            confidence=1.0
        )
    
    async def _ai_safety_analysis(self, content: str, output_type: str) -> SecurityCheck:
        """AI-powered safety analysis with context awareness"""
        # Truncate content for analysis
        safe_content = content[:2000] if len(content) > 2000 else content
        
        safety_prompt = f"""
        Analyze this AI-generated content for security and appropriateness:
        
        Content Type: {output_type}
        Content: {safe_content}
        
        Evaluate:
        1. Is this content safe and appropriate for the user?
        2. Does it contain any security risks if executed?
        3. Are there any harmful instructions or malicious patterns?
        4. Would this benefit from human review?
        
        Consider context - legitimate programming tasks should be approved even if they use 
        system functions, as long as they serve the user's reasonable request.
        
        Respond with JSON:
        {{
          "safe": true/false,
          "requires_human_review": true/false, 
          "confidence": 0.0-1.0,
          "reasoning": "explanation of the decision"
        }}
        """
        
        try:
            response_text = await llm_manager.generate_for_node("controller", safety_prompt)
            
            # Parse JSON response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    safety_data = json.loads(response_text[json_start:json_end])
                else:
                    raise ValueError("No valid JSON found")
            except (json.JSONDecodeError, ValueError):
                logger.warning("AI safety check returned invalid JSON")
                return SecurityCheck(
                    passed=False,
                    reason="Safety analysis inconclusive",
                    confidence=0.3,
                    requires_human_review=True
                )
            
            return SecurityCheck(
                passed=safety_data.get('safe', False),
                reason=safety_data.get('reasoning', 'AI safety analysis'),
                confidence=safety_data.get('confidence', 0.8),
                requires_human_review=safety_data.get('requires_human_review', False)
            )
        
        except Exception as e:
            error_message = self._secure_error_handling(e, "ai_safety_analysis")
            return SecurityCheck(
                passed=False,
                reason=error_message,
                confidence=0.3,
                requires_human_review=True
            )
            
    def _create_approval_response(self, state: AssistantState) -> AssistantState:
        """Create approval response"""
        return {
            **state,
            'requires_approval': False,
            'approval_context': {
                'status': 'approved',
                'timestamp': time.time(),
                'method': 'automated_approval'
            }
        }
        
    def _create_rejection_response(self, state: AssistantState, reason: str) -> AssistantState:
        """Create rejection response"""
        return {
            **state,
            'output_content': f"I cannot fulfill this request. {reason}",
            'output_type': 'security_message',
            'requires_approval': False,
            'approval_context': {
                'status': 'rejected',
                'reason': reason,
                'timestamp': time.time()
            }
        }
        
    def _secure_error_handling(self, error: Exception, context: str) -> str:
        """Secure error handling that doesn't leak sensitive information"""
        # Hash the error for internal tracking
        error_hash = hashlib.sha256(str(error).encode()).hexdigest()[:8]
        
        # Log detailed error internally
        logger.error(f"Controller error [{error_hash}] in {context}: {str(error)}")
        
        # Return generic message to user
        return f"Security check encountered an issue (ref: {error_hash})"

# Global instance
controller = Controller()