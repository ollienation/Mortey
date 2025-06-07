# tests/unit/test_circuit_breaker_integration.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from core.circuit_breaker import AdvancedCircuitBreaker
from core.error_handling import ErrorHandler, ErrorType

class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with LLM providers."""
    
    @pytest.mark.asyncio
    async def test_llm_provider_failover(
        self, 
        circuit_breaker, 
        mock_llm_providers
    ):
        """Test automatic failover between LLM providers."""
        # Configure primary provider to fail
        mock_llm_providers["anthropic"].side_effect = ConnectionError("API unavailable")
        mock_llm_providers["openai"].return_value = "Fallback response"
        
        # Test provider failover
        with patch('config.llm_manager.llm_manager') as mock_manager:
            mock_manager.generate_for_node = self._create_failover_generator(
                mock_llm_providers
            )
            
            result = await circuit_breaker.call_with_circuit_breaker(
                "anthropic",
                mock_manager.generate_for_node,
                "chat",
                "test prompt"
            )
        
        # Verify fallback occurred
        assert result == "Fallback response"
        assert circuit_breaker.get_circuit_status("anthropic")["state"] == "open"
    
    async def test_recovery_after_circuit_open(
        self, 
        circuit_breaker, 
        failing_service
    ):
        """Test service recovery after circuit breaker opens."""
        # Trigger circuit breaker opening
        for _ in range(5):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call_with_circuit_breaker(
                    "test_service",
                    failing_service,
                    should_fail=True
                )
        
        assert circuit_breaker.get_circuit_status("test_service")["state"] == "open"
        
        # Wait for recovery period
        await asyncio.sleep(1.1)  # Slightly longer than recovery timeout
        
        # Test successful call after recovery
        result = await circuit_breaker.call_with_circuit_breaker(
            "test_service",
            failing_service,
            should_fail=False
        )
        
        assert "Success" in result
        assert circuit_breaker.get_circuit_status("test_service")["state"] == "closed"
