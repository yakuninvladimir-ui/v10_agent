"""
Unit Tests for VLLMClient Module

Tests verify critical invariants:
1. Thinking traces are isolated from payload
2. Role-specific configurations are applied correctly
3. Error handling returns appropriate status codes
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from v10_agent.llm_client import (
    VLLMClient,
    AgentRole,
    RoleConfig,
    ParsedResponse,
    ROLE_CONFIGS,
)


class TestThinkingIsolation:
    """Test that thinking traces are properly isolated from payload."""
    
    def test_thinking_tags_stripped_from_content(self):
        """
        Test case: Content contains <think>...</think> tags with reasoning.
        
        Input: "<think>Модель рассуждает о цели уровня...</think>{"action": "ACTION1"}"
        Expected: 
          - payload = {"action": "ACTION1"}
          - reasoning_trace contains "Модель рассуждает о цели уровня..."
        """
        client = VLLMClient()
        
        # Mock vLLM response with thinking tags leaked into content
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "<think>Модель рассуждает о цели уровня и пытается найти паттерны...</think>{\"action\": \"ACTION1\", \"confidence\": 0.85}",
                    "reasoning_content": None
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 200
            }
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        # Verify status
        assert result.status == "OK"
        
        # Verify payload is clean JSON without thinking tags
        assert result.payload == {"action": "ACTION1", "confidence": 0.85}
        
        # Verify reasoning_trace contains the extracted thinking
        assert result.reasoning_trace is not None
        assert "Модель рассуждает о цели уровня" in result.reasoning_trace
        
        # Verify usage statistics
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["reasoning_tokens"] == 200
    
    def test_reasoning_content_field_isolated(self):
        """
        Test case: vLLM returns separate reasoning_content field.
        
        Input: 
          - content = "{\"result\": \"success\"}"
          - reasoning_content = "Длинные рассуждения модели о подходе..."
        Expected:
          - payload = {"result": "success"}
          - reasoning_trace = "Длинные рассуждения модели о подходе..."
        """
        client = VLLMClient()
        
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "{\"result\": \"success\", \"steps\": 3}",
                    "reasoning_content": "Длинные рассуждения модели о том, как лучше решить эту задачу, анализируя пространственные отношения..."
                }
            }],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 80,
                "reasoning_tokens": 300
            }
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        # Verify status
        assert result.status == "OK"
        
        # Verify payload is clean JSON
        assert result.payload == {"result": "success", "steps": 3}
        
        # Verify reasoning_trace contains the reasoning_content
        assert result.reasoning_trace is not None
        assert "Длинные рассуждения модели" in result.reasoning_trace
        
        # Verify payload does NOT contain reasoning content
        payload_str = json.dumps(result.payload)
        assert "Длинные рассуждения" not in payload_str
    
    def test_both_thinking_tags_and_reasoning_content(self):
        """
        Test case: Both <think> tags AND reasoning_content field present.
        
        Expected: Both should be extracted and combined in reasoning_trace
        """
        client = VLLMClient()
        
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "<think>Первое рассуждение...</think>{\"action\": \"test\"}",
                    "reasoning_content": "Второе рассуждение из поля reasoning_content"
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        assert result.status == "OK"
        assert result.payload == {"action": "test"}
        
        # Both thinking sources should be in reasoning_trace
        assert result.reasoning_trace is not None
        assert "Первое рассуждение" in result.reasoning_trace
        assert "Второе рассуждение из поля reasoning_content" in result.reasoning_trace
    
    def test_markdown_json_wrapper_removed(self):
        """Test that ```json ... ``` wrappers are properly removed."""
        client = VLLMClient()
        
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"wrapped\": \"json\", \"value\": 42}\n```",
                    "reasoning_content": None
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        assert result.status == "OK"
        assert result.payload == {"wrapped": "json", "value": 42}
    
    def test_generic_markdown_wrapper_removed(self):
        """Test that generic ``` ... ``` wrappers are properly removed."""
        client = VLLMClient()
        
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "```\n{\"generic\": \"wrapper\"}\n```",
                    "reasoning_content": None
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        assert result.status == "OK"
        assert result.payload == {"generic": "wrapper"}


class TestRoleConfiguration:
    """Test role-specific LLM configurations."""
    
    def test_coder_role_has_thinking_disabled(self):
        """
        Test that CODER role has enable_thinking=False.
        
        This is CRITICAL for ISO-2 compliance: Coder should not waste tokens
        on reasoning about level goals.
        """
        coder_config = ROLE_CONFIGS[AgentRole.CODER]
        
        assert coder_config.enable_thinking is False
        assert coder_config.temperature == 0.1  # Near-deterministic
        assert coder_config.reasoning_budget_tokens == 0
    
    def test_solver_budget_is_32k(self):
        """
        Test that SOLVER role has 32k reasoning budget.
        
        Solver needs extended budget for deep Brusentsov logic reasoning.
        """
        solver_config = ROLE_CONFIGS[AgentRole.SOLVER]
        
        assert solver_config.enable_thinking is True
        assert solver_config.reasoning_budget_tokens == 32000
        assert solver_config.temperature == 0.4  # Balanced for reasoning
    
    def test_explorer_role_has_high_creativity(self):
        """
        Test that EXPLORER role has higher temperature for creativity.
        
        Explorer needs creativity for spatial search and pattern discovery.
        """
        explorer_config = ROLE_CONFIGS[AgentRole.EXPLORER]
        
        assert explorer_config.enable_thinking is True
        assert explorer_config.reasoning_budget_tokens == 16000
        assert explorer_config.temperature == 0.6  # Higher for creativity
    
    @patch('v10_agent.llm_client.requests.post')
    def test_generate_request_includes_role_config(self, mock_post):
        """
        Test that generate() method includes correct role config in request.
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "{\"test\": \"ok\"}",
                    "reasoning_content": None
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = VLLMClient(base_url="http://test:1234")
        
        # Test CODER request
        messages = [{"role": "user", "content": "test"}]
        json_schema = {"type": "object"}
        
        client.generate(role=AgentRole.CODER, messages=messages, json_schema=json_schema)
        
        # Verify request was made
        assert mock_post.called
        call_args = mock_post.call_args
        
        # Check request body
        request_body = call_args[1]['json']
        assert request_body["temperature"] == 0.1  # CODER temperature
        assert request_body["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    
    @patch('v10_agent.llm_client.requests.post')
    def test_generate_solver_enables_thinking(self, mock_post):
        """
        Test that SOLVER role enables thinking in request.
        """
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "{\"test\": \"ok\"}",
                    "reasoning_content": None
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = VLLMClient()
        messages = [{"role": "user", "content": "test"}]
        json_schema = {"type": "object"}
        
        client.generate(role=AgentRole.SOLVER, messages=messages, json_schema=json_schema)
        
        request_body = mock_post.call_args[1]['json']
        assert request_body["temperature"] == 0.4  # SOLVER temperature
        assert request_body["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_parse_error_on_invalid_json(self):
        """Test PARSE_ERROR status when JSON parsing fails."""
        client = VLLMClient()
        
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON at all {{{",
                    "reasoning_content": None
                }
            }],
            "usage": {}
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        assert result.status == "PARSE_ERROR"
        assert result.payload is None
        assert result.error_message is not None
        assert "JSON parse failed" in result.error_message
    
    def test_parse_error_on_non_dict_json(self):
        """Test PARSE_ERROR when JSON is not a dict."""
        client = VLLMClient()
        
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "[\"array\", \"not\", \"dict\"]",
                    "reasoning_content": None
                }
            }],
            "usage": {}
        }
        
        result = client._isolate_and_parse_thinking(mock_response_data)
        
        assert result.status == "PARSE_ERROR"
        assert result.payload is None
        assert "Expected dict" in result.error_message
    
    def test_timeout_status(self):
        """Test TIMEOUT status from factory method."""
        result = ParsedResponse.timeout("Request exceeded 120s")
        
        assert result.status == "TIMEOUT"
        assert result.payload is None
        assert result.reasoning_trace is None
        assert "timeout" in result.error_message.lower()
    
    def test_connection_error_status(self):
        """Test CONNECTION_ERROR status from factory method."""
        result = ParsedResponse.connection_error("Connection refused")
        
        assert result.status == "CONNECTION_ERROR"
        assert result.payload is None
        assert "connection" in result.error_message.lower()
    
    @patch('v10_agent.llm_client.requests.post')
    def test_network_timeout_returns_timeout_status(self, mock_post):
        """Test that network timeout returns TIMEOUT status."""
        from requests.exceptions import Timeout
        
        mock_post.side_effect = Timeout("Request timed out")
        
        client = VLLMClient(timeout=1.0)
        messages = [{"role": "user", "content": "test"}]
        json_schema = {"type": "object"}
        
        result = client.generate(role=AgentRole.SOLVER, messages=messages, json_schema=json_schema)
        
        assert result.status == "TIMEOUT"
        assert result.payload is None
    
    @patch('v10_agent.llm_client.requests.post')
    def test_connection_error_returns_connection_error_status(self, mock_post):
        """Test that connection error returns CONNECTION_ERROR status."""
        from requests.exceptions import ConnectionError
        
        mock_post.side_effect = ConnectionError("Connection refused")
        
        client = VLLMClient()
        messages = [{"role": "user", "content": "test"}]
        json_schema = {"type": "object"}
        
        result = client.generate(role=AgentRole.SOLVER, messages=messages, json_schema=json_schema)
        
        assert result.status == "CONNECTION_ERROR"
        assert result.payload is None


class TestParsedResponseFactories:
    """Test ParsedResponse factory methods."""
    
    def test_ok_factory(self):
        """Test ok() factory creates valid response."""
        payload = {"action": "test"}
        reasoning = "Some reasoning"
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "reasoning_tokens": 200}
        
        result = ParsedResponse.ok(payload, reasoning, usage)
        
        assert result.status == "OK"
        assert result.payload == payload
        assert result.reasoning_trace == reasoning
        assert result.usage == usage
        assert result.error_message is None
    
    def test_parse_error_factory(self):
        """Test parse_error() factory creates valid error response."""
        result = ParsedResponse.parse_error(raw_content="invalid", error="Bad JSON")
        
        assert result.status == "PARSE_ERROR"
        assert result.payload is None
        assert result.reasoning_trace is None
        assert "Bad JSON" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
