import asyncio
import os
import re
from typing import Dict, Any, List
from anthropic import Anthropic
from dataclasses import dataclass
from enum import Enum

class VerificationResult(Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    BLOCKED = "blocked"
    HUMAN_REVIEW = "human_review"

@dataclass
class SafetyCheck:
    passed: bool
    reason: str
    confidence: float
    action: VerificationResult

class ControllerAgent:
    """Safety and quality controller with loop protection"""
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Safety patterns to detect
        self.dangerous_patterns = [
            r'rm\s+-rf',
            r'sudo\s+rm',
            r'format\s+c:',
            r'del\s+/[sq]',
            r'DROP\s+TABLE',
            r'DELETE\s+FROM.*WHERE\s+1=1',
            r'exec\(',
            r'eval\(',
            r'__import__',
            r'subprocess\.',
            r'os\.system',
        ]
        
        # Sensitive operation keywords
        self.sensitive_keywords = [
            'delete', 'remove', 'install', 'uninstall', 'format',
            'execute', 'run', 'sudo', 'admin', 'root', 'password'
        ]
        
        # Loop tracking per session
        self.session_loops = {}
        
    async def verify_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main verification method with comprehensive safety checks"""
        
        session_id = state.get('session_id', 'default')
        user_input = state.get('user_input', '')
        output_content = state.get('output_content', '')
        output_type = state.get('output_type', 'text')
        
        # Update thinking state
        state['thinking_state'] = {
            'active_agent': 'CONTROLLER',
            'current_task': 'Verifying output safety and quality',
            'progress': 0.2,
            'details': 'Running safety checks...'
        }
        
        # 1. Loop Protection
        loop_result = self._check_loops(session_id, state)
        if loop_result.action == VerificationResult.BLOCKED:
            return self._create_blocked_response(state, loop_result.reason)
        
        # 2. Pattern-based Safety Check (Fast Local Check)
        pattern_result = await self._pattern_safety_check(output_content, user_input)
        if not pattern_result.passed:
            if pattern_result.action == VerificationResult.BLOCKED:
                return self._create_blocked_response(state, pattern_result.reason)
        
        # 3. Content-based Safety Check (AI-powered)
        content_result = await self._ai_safety_check(output_content, user_input, output_type)
        if not content_result.passed:
            if content_result.action == VerificationResult.BLOCKED:
                return self._create_blocked_response(state, content_result.reason)
            elif content_result.action == VerificationResult.NEEDS_REVISION:
                return self._create_revision_response(state, content_result.reason)
        
        # 4. Human Review Check
        if self._requires_human_review(user_input, output_content):
            return self._create_human_review_response(state)
        
        # 5. Quality Check
        quality_result = await self._quality_check(output_content, user_input, output_type)
        if not quality_result.passed:
            return self._create_revision_response(state, quality_result.reason)
        
        # All checks passed
        return self._create_approved_response(state)
    
    def _check_loops(self, session_id: str, state: Dict[str, Any]) -> SafetyCheck:
        """Prevent infinite loops between agents"""
        
        loop_count = state.get('loop_count', 0) + 1
        max_loops = state.get('max_loops', 3)
        
        # Track loops per session
        if session_id not in self.session_loops:
            self.session_loops[session_id] = []
        
        self.session_loops[session_id].append({
            'user_input': state.get('user_input', ''),
            'agent': state.get('current_agent', 'unknown'),
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # Check for excessive loops
        if loop_count >= max_loops:
            return SafetyCheck(
                passed=False,
                reason=f"Maximum revision attempts ({max_loops}) reached. Please rephrase your request.",
                confidence=1.0,
                action=VerificationResult.BLOCKED
            )
        
        # Check for rapid repeated requests (potential loop)
        recent_requests = [
            req for req in self.session_loops[session_id]
            if asyncio.get_event_loop().time() - req['timestamp'] < 60  # Last minute
        ]
        
        if len(recent_requests) > 10:
            return SafetyCheck(
                passed=False,
                reason="Too many rapid requests. Please wait a moment before continuing.",
                confidence=0.9,
                action=VerificationResult.BLOCKED
            )
        
        return SafetyCheck(
            passed=True,
            reason="Loop check passed",
            confidence=1.0,
            action=VerificationResult.APPROVED
        )
    
    async def _pattern_safety_check(self, content: str, user_input: str) -> SafetyCheck:
        """Fast pattern-based safety check using regex"""
        
        # Check for dangerous command patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return SafetyCheck(
                    passed=False,
                    reason=f"Detected potentially dangerous command pattern: {pattern}",
                    confidence=0.95,
                    action=VerificationResult.BLOCKED
                )
        
        # Check for sensitive operations
        sensitive_count = sum(
            1 for keyword in self.sensitive_keywords
            if keyword.lower() in user_input.lower()
        )
        
        if sensitive_count >= 2:  # Multiple sensitive keywords
            return SafetyCheck(
                passed=False,
                reason="Request contains multiple sensitive operations",
                confidence=0.8,
                action=VerificationResult.HUMAN_REVIEW
            )
        
        return SafetyCheck(
            passed=True,
            reason="Pattern check passed",
            confidence=0.9,
            action=VerificationResult.APPROVED
        )
    
    async def _ai_safety_check(self, content: str, user_input: str, output_type: str) -> SafetyCheck:
        """AI-powered safety analysis using Claude"""
        
        safety_prompt = f"""
        Analyze this AI assistant output for safety and appropriateness:
        
        User Request: {user_input}
        Output Type: {output_type}
        Assistant Output: {content[:2000]}
        
        Check for:
        1. Harmful or dangerous instructions
        2. Privacy violations or data exposure
        3. Malicious code or commands
        4. Inappropriate content
        5. Misinformation or false claims
        
        Respond with JSON format:
        {{
            "safe": true/false,
            "confidence": 0.0-1.0,
            "issues": ["list of specific issues"],
            "recommendation": "approve/revise/block"
        }}
        """
        
        try:
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": safety_prompt}]
            )
            
            # Parse JSON response
            import json
            response_text = message.content[0].text
            
            # Extract JSON from response (handle markdown formatting)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                safety_data = json.loads(response_text[json_start:json_end])
            else:
                # Fallback if JSON parsing fails
                return SafetyCheck(
                    passed=True,
                    reason="Safety check completed (parsing issue)",
                    confidence=0.5,
                    action=VerificationResult.APPROVED
                )
            
            if not safety_data.get('safe', True):
                action = VerificationResult.BLOCKED
                if safety_data.get('recommendation') == 'revise':
                    action = VerificationResult.NEEDS_REVISION
                
                return SafetyCheck(
                    passed=False,
                    reason=f"Safety issues detected: {', '.join(safety_data.get('issues', []))}",
                    confidence=safety_data.get('confidence', 0.8),
                    action=action
                )
            
            return SafetyCheck(
                passed=True,
                reason="AI safety check passed",
                confidence=safety_data.get('confidence', 0.9),
                action=VerificationResult.APPROVED
            )
            
        except Exception as e:
            # If safety check fails, err on the side of caution
            return SafetyCheck(
                passed=False,
                reason=f"Safety check failed: {str(e)}",
                confidence=0.3,
                action=VerificationResult.HUMAN_REVIEW
            )
    
    async def _quality_check(self, content: str, user_input: str, output_type: str) -> SafetyCheck:
        """Check output quality and relevance"""
        
        # Basic quality checks
        if len(content.strip()) < 10:
            return SafetyCheck(
                passed=False,
                reason="Output too short or empty",
                confidence=0.9,
                action=VerificationResult.NEEDS_REVISION
            )
        
        # Check for code quality if it's a code output
        if output_type == "code":
            if "```" not in content:
                return SafetyCheck(
                    passed=False,
                    reason="Code output doesn't appear to contain proper code formatting",
                    confidence=0.7,
                    action=VerificationResult.NEEDS_REVISION
                )
        
        return SafetyCheck(
            passed=True,
            reason="Quality check passed",
            confidence=0.8,
            action=VerificationResult.APPROVED
        )
    
    def _requires_human_review(self, user_input: str, content: str) -> bool:
        """Determine if human review is required"""
        
        # Check for sensitive operations
        sensitive_in_input = any(
            keyword in user_input.lower() 
            for keyword in ['delete', 'remove', 'install', 'execute', 'run']
        )
        
        # Check for system-level operations in output
        system_operations = any(
            term in content.lower()
            for term in ['sudo', 'admin', 'root', 'system', 'registry']
        )
        
        return sensitive_in_input and system_operations
    
    def _create_approved_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create approved response"""
        return {
            **state,
            'verification_result': VerificationResult.APPROVED.value,
            'verification_required': False,
            'thinking_state': {
                'active_agent': 'CONTROLLER',
                'current_task': 'Verification complete',
                'progress': 1.0,
                'details': 'Output approved and ready for user'
            }
        }
    
    def _create_blocked_response(self, state: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Create blocked response"""
        return {
            **state,
            'verification_result': VerificationResult.BLOCKED.value,
            'verification_required': False,
            'output_content': f"I cannot fulfill this request. Reason: {reason}",
            'output_type': 'safety_message',
            'thinking_state': {
                'active_agent': 'CONTROLLER',
                'current_task': 'Request blocked for safety',
                'progress': 1.0,
                'details': f'Blocked: {reason}'
            }
        }
    
    def _create_revision_response(self, state: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Create revision request response"""
        return {
            **state,
            'verification_result': VerificationResult.NEEDS_REVISION.value,
            'verification_required': True,
            'loop_count': state.get('loop_count', 0) + 1,
            'revision_reason': reason,
            'thinking_state': {
                'active_agent': 'CONTROLLER',
                'current_task': 'Requesting revision',
                'progress': 0.7,
                'details': f'Needs revision: {reason}'
            }
        }
    
    def _create_human_review_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create human review request response"""
        return {
            **state,
            'verification_result': VerificationResult.HUMAN_REVIEW.value,
            'verification_required': True,
            'output_content': 'This request requires human review before proceeding. Please confirm you want to continue.',
            'output_type': 'human_review_request',
            'thinking_state': {
                'active_agent': 'CONTROLLER',
                'current_task': 'Awaiting human review',
                'progress': 0.5,
                'details': 'Sensitive operation detected - human approval required'
            }
        }
