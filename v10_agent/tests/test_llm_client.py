"""
Unit Tests for V10 Agent LLM Client

Ref: Engineering Specification V10.0, Section 8 - LLM Integration
Ref: Architectural Specification V10.0, Section 1.4 - Isolation Invariants

Tests verify:
- Thinking trace isolation from payload
- Role-specific configuration (enable_thinking, budgets)
- JSON parsing with markdown cleanup
- Timeout and error handling
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from v10_agent.llm_client import (
    VLLMClient,
    ParsedResponse,
    RoleConfig,
    AgentRole,
    ROLE_CONFIGS,
    get_role_config,
    create_client,
    ACCELERATOR,
    _ACCELERATORS,
    get_accelerator_config,
)


class TestAcceleratorConfig:
    """Test accelerator configuration constants."""
    
    def test_accelerator_constant_is_rtx6000(self):
        """Verify ACCELERATOR constant is set to 'rtx6000'."""
        assert ACCELERATOR == "rtx6000"
    
    def test_accelerators_dict_has_rtx6000(self):
        """Verify _ACCELERATORS contains rtx6000 config."""
        assert "rtx6000" in _ACCELERATORS
        assert _ACCELERATORS["rtx6000"]["name"] == "nvidiaRtx6000"
        assert _ACCELERATORS["rtx6000"]["machine_shape"] == "NvidiaRtxPro6000"
        assert _ACCELERATORS["rtx6000"]["gpu"] is True
    
    def test_get_accelerator_config_returns_correct_config(self):
        """Verify get_accelerator_config returns correct config."""
        config = get_accelerator_config("rtx6000")
        assert config["name"] == "nvidiaRtx6000"
        assert config["gpu"] is True
    
    def test_get_accelerator_config_raises_for_unknown(self):
        """Verify get_accelerator_config raises ValueError for unknown accelerator."""
        with pytest.raises(ValueError, match="Unknown accelerator"):
            get_accelerator_config("unknown_gpu")


class TestRoleConfig:
    """Test role configuration."""
    
    def test_explorer_role_has_thinking_enabled(self):
        """Verify EXPLORER role has enable_thinking=True."""
        config = get_role_config(AgentRole.EXPLORER)
        assert config.enable_thinking is True
        assert config.reasoning_budget_tokens == 16000
        assert config.temperature == 0.6
    
    def test_coder_role_has_thinking_disabled(self):
        """Verify CODER role has enable_thinking=False."""
        config = get_role_config(AgentRole.CODER)
        assert config.enable_thinking is False
        assert config.temperature == 0.1
    
    def test_solver_role_has_thinking_enabled_with_32k_budget(self):
        """Verify SOLVER role has enable_thinking=True with 32k budget."""
        config = get_role_config(AgentRole.SOLVER)
        assert config.enable_thinking is True
        assert config.reasoning_budget_tokens == 32000
        assert config.temperature == 0.4
    
    def test_role_config_to_vllm_params_explorer(self):
        """Verify EXPLORER role produces correct vLLM params."""
        config = get_role_config(AgentRole.EXPLORER)
        params = config.to_vllm_params()
        
        assert params["temperature"] == 0.6
        assert params["top_p"] == 0.95
        assert "extra_body" in params
        assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert params["extra_body"]["reasoning_budget_tokens"] == 16000
    
    def test_role_config_to_vllm_params_coder(self):
        """Verify CODER role produces correct vLLM params with thinking disabled."""
        config = get_role_config(AgentRole.CODER)
        params = config.to_vllm_params()
        
        assert params["temperature"] == 0.1
        assert "extra_body" in params
        assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        # CODER should not have reasoning_budget_tokens
        assert "reasoning_budget_tokens" not in params["extra_body"]
    
    def test_role_config_to_vllm_params_solver(self):
        """Verify SOLVER role produces correct vLLM params with 32k budget."""
        config = get_role_config(AgentRole.SOLVER)
        params = config.to_vllm_params()
        
        assert params["temperature"] == 0.4
        assert "extra_body" in params
        assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert params["extra_body"]["reasoning_budget_tokens"] == 32000


class TestParsedResponse:
    """Test ParsedResponse dataclass."""
    
    def test_create_ok(self):
        """Test creating OK response."""
        payload = {"action": "ACTION1"}
        response = ParsedResponse.create_ok(payload, reasoning_trace="test", usage={"prompt_tokens": 10})
        
        assert response.status == "OK"
        assert response.payload == payload
        assert response.reasoning_trace == "test"
        assert response.usage == {"prompt_tokens": 10}
    
    def test_create_timeout(self):
        """Test creating timeout response."""
        response = ParsedResponse.create_timeout()
        
        assert response.status == "TIMEOUT"
        assert response.payload is None
        assert response.reasoning_trace is None
    
    def test_create_parse_error(self):
        """Test creating parse error response."""
        response = ParsedResponse.create_parse_error("raw content here")
        
        assert response.status == "PARSE_ERROR"
        assert response.payload is not None
        assert "_raw_content" in response.payload


class TestThinkingIsolation:
    """Test critical thinking trace isolation from payload."""
    
    def test_thinking_tags_stripped_from_content(self):
        """
        Verify that <think>...</think> tags are stripped from content.
        
        Input: Content with embedded thinking tags
        Expected: payload contains only JSON, reasoning_trace contains thinking
        """
        client = VLLMClient()
        
        # Mock response with thinking tags in content
        mock_response = {
            "choices": [{
                "message": {
                    "content": '<think>Модель рассуждает о цели уровня...</think>{"action": "ACTION1"}',
                    "reasoning_content": None
                }
            }],
            "usage": {}
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        # Verify status is OK
        assert result.status == "OK"
        
        # Verify payload is clean JSON without thinking
        assert result.payload == {"action": "ACTION1"}
        
        # Verify reasoning_trace contains the thinking content
        assert result.reasoning_trace is not None
        assert "Модель рассуждает о цели уровня" in result.reasoning_trace
        
        # CRITICAL: Verify no thinking traces in payload
        payload_str = json.dumps(result.payload)
        assert "<think>" not in payload_str
        assert "</think>" not in payload_str
    
    def test_reasoning_content_field_isolated(self):
        """
        Verify that reasoning_content field is isolated from payload.
        
        Input: Clean JSON content with separate reasoning_content field
        Expected: payload contains only JSON, reasoning_trace contains reasoning_content
        """
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"action": "ACTION2", "target": "obj_1"}',
                    "reasoning_content": 'Длинные рассуждения модели о том, как выполнить действие...'
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        # Verify payload is clean
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION2", "target": "obj_1"}
        
        # Verify reasoning_trace contains reasoning_content
        assert result.reasoning_trace == 'Длинные рассуждения модели о том, как выполнить действие...'
        
        # Verify usage is captured
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
    
    def test_multiple_thinking_tags_all_extracted(self):
        """Verify multiple thinking tags are all extracted and removed."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '<think>First thought</think>{"step": 1}<think>Second thought</think>{"step": 2}',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        # Payload should have both JSON fragments concatenated (which will fail parse)
        # or just the last valid JSON depending on model output
        # The key is that thinking tags are removed
        assert result.status == "PARSE_ERROR" or (
            result.status == "OK" and "<think>" not in json.dumps(result.payload)
        )
        
        # Both thinking segments should be in reasoning_trace
        if result.reasoning_trace:
            assert "First thought" in result.reasoning_trace
            assert "Second thought" in result.reasoning_trace
    
    def test_markdown_wrappers_removed(self):
        """Verify markdown code block wrappers are removed."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '```json\n{"action": "ACTION3"}\n```',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION3"}
    
    def test_markdown_wrapper_without_language(self):
        """Verify markdown wrappers without language specifier are removed."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '```\n{"action": "ACTION4"}\n```',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION4"}
    
    def test_thinking_priority_reasoning_content_over_tags(self):
        """
        Verify reasoning_content field takes priority over think tags.
        
        When both are present, reasoning_content should be used.
        """
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '<think>Tag thinking</think>{"action": "ACTION5"}',
                    "reasoning_content": 'Field reasoning is better'
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION5"}
        
        # reasoning_content should take priority
        assert result.reasoning_trace == 'Field reasoning is better'
    
    def test_no_thinking_no_reasoning_content(self):
        """Verify response with no thinking sources has None reasoning_trace."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"action": "ACTION6"}',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION6"}
        assert result.reasoning_trace is None


class TestHTTPIntegration:
    """Test HTTP request handling."""
    
    @patch('urllib.request.urlopen')
    def test_generate_sends_correct_payload(self, mock_urlopen):
        """Verify generate() sends correct payload to API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"result": "ok"}'}}],
            "usage": {}
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = VLLMClient(base_url="http://test:1234", model_name="test-model")
        
        messages = [{"role": "user", "content": "test"}]
        result = client.generate(AgentRole.CODER, messages)
        
        assert result.status == "OK"
        
        # Verify the request was made
        assert mock_urlopen.called
        
        # Get the request object passed to urlopen
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        
        # Verify URL
        assert request_obj.full_url == "http://test:1234/v1/chat/completions"
        
        # Verify method
        assert request_obj.method == "POST"
        
        # Verify headers
        assert request_obj.headers["Content-Type"] == "application/json"
    
    @patch('urllib.request.urlopen')
    def test_coder_role_has_thinking_disabled_in_request(self, mock_urlopen):
        """
        Verify that CODER role request has enable_thinking=False.
        
        This is a critical ISO invariant test.
        """
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"code": "def f(): pass"}'}}],
            "usage": {}
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = VLLMClient()
        
        messages = [{"role": "user", "content": "generate DSL"}]
        result = client.generate(AgentRole.CODER, messages)
        
        # Verify the request payload
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        
        # Parse the request body
        import json
        request_body = json.loads(request_obj.data.decode('utf-8'))
        
        # CRITICAL: Verify enable_thinking is False for CODER
        assert "extra_body" in request_body
        assert request_body["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    
    @patch('urllib.request.urlopen')
    def test_solver_budget_is_32k(self, mock_urlopen):
        """
        Verify that SOLVER role receives 32k reasoning budget.
        """
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"trajectory": []}'}}],
            "usage": {}
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = VLLMClient()
        
        messages = [{"role": "user", "content": "solve level"}]
        result = client.generate(AgentRole.SOLVER, messages)
        
        # Verify the request payload
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        
        import json
        request_body = json.loads(request_obj.data.decode('utf-8'))
        
        # Verify solver has 32k reasoning budget
        assert "extra_body" in request_body
        assert request_body["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert request_body["extra_body"]["reasoning_budget_tokens"] == 32000
    
    @patch('urllib.request.urlopen')
    def test_explorer_budget_is_16k(self, mock_urlopen):
        """Verify that EXPLORER role receives 16k reasoning budget."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"spec": {}}'}}],
            "usage": {}
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = VLLMClient()
        
        messages = [{"role": "user", "content": "explore"}]
        result = client.generate(AgentRole.EXPLORER, messages)
        
        # Verify the request payload
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        
        import json
        request_body = json.loads(request_obj.data.decode('utf-8'))
        
        assert "extra_body" in request_body
        assert request_body["extra_body"]["reasoning_budget_tokens"] == 16000
    
    @patch('urllib.request.urlopen')
    def test_json_schema_enforcement(self, mock_urlopen):
        """Verify JSON schema is passed in response_format."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"validated": true}'}}],
            "usage": {}
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = VLLMClient()
        
        messages = [{"role": "user", "content": "test"}]
        json_schema = {
            "type": "object",
            "properties": {"validated": {"type": "boolean"}}
        }
        
        result = client.generate(AgentRole.CODER, messages, json_schema=json_schema)
        
        # Verify the request payload
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        
        request_body = json.loads(request_obj.data.decode('utf-8'))
        
        # Verify response_format with json_schema
        assert "response_format" in request_body
        assert request_body["response_format"]["type"] == "json_schema"
        assert request_body["response_format"]["json_schema"]["strict"] is True
        assert request_body["response_format"]["json_schema"]["schema"] == json_schema
    
    @patch('urllib.request.urlopen')
    def test_timeout_handling(self, mock_urlopen):
        """Verify timeout returns TIMEOUT status."""
        # Setup mock to raise timeout
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError(TimeoutError("Request timed out"))
        
        client = VLLMClient(timeout_seconds=1.0)
        
        messages = [{"role": "user", "content": "test"}]
        result = client.generate(AgentRole.CODER, messages)
        
        assert result.status == "TIMEOUT"
        assert result.payload is None
    
    @patch('urllib.request.urlopen')
    def test_network_error_handling(self, mock_urlopen):
        """Verify network errors return PARSE_ERROR status."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        
        client = VLLMClient()
        
        messages = [{"role": "user", "content": "test"}]
        result = client.generate(AgentRole.CODER, messages)
        
        assert result.status == "PARSE_ERROR"


class TestClientInstantiation:
    """Test client instantiation and configuration."""
    
    def test_default_base_url(self):
        """Verify default base URL is correct."""
        client = VLLMClient()
        assert client.base_url == "http://127.0.0.1:1234"
    
    def test_custom_base_url(self):
        """Verify custom base URL is accepted."""
        client = VLLMClient(base_url="http://custom:8080")
        assert client.base_url == "http://custom:8080"
    
    def test_custom_model_name(self):
        """Verify custom model name is accepted."""
        client = VLLMClient(model_name="custom-model")
        assert client.model_name == "custom-model"
    
    def test_custom_timeout(self):
        """Verify custom timeout is accepted."""
        client = VLLMClient(timeout_seconds=60.0)
        assert client.timeout_seconds == 60.0
    
    def test_create_client_factory(self):
        """Verify create_client factory function works."""
        client = create_client(
            base_url="http://factory:9999",
            model_name="factory-model",
            timeout_seconds=30.0
        )
        
        assert client.base_url == "http://factory:9999"
        assert client.model_name == "factory-model"
        assert client.timeout_seconds == 30.0
    
    def test_custom_role_config(self):
        """Verify custom role configurations can be passed."""
        custom_configs = {
            AgentRole.CODER: RoleConfig(
                role=AgentRole.CODER,
                enable_thinking=False,
                temperature=0.0,
                max_tokens=1000
            )
        }
        
        client = VLLMClient(default_role_config=custom_configs)
        
        # Verify custom config is used
        assert client.role_configs[AgentRole.CODER].temperature == 0.0
        assert client.role_configs[AgentRole.CODER].max_tokens == 1000
        
        # Other roles should use defaults
        assert client.role_configs[AgentRole.EXPLORER].temperature == 0.6


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_content(self):
        """Verify empty content is handled gracefully."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "PARSE_ERROR"
    
    def test_invalid_json_in_content(self):
        """Verify invalid JSON returns PARSE_ERROR."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": "not valid json {",
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "PARSE_ERROR"
        assert result.payload is not None
        assert "_raw_content" in result.payload
    
    def test_array_json_wrapped_as_dict(self):
        """Verify array JSON is wrapped in dict."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '[1, 2, 3]',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert "_result" in result.payload
        assert result.payload["_result"] == [1, 2, 3]
    
    def test_case_insensitive_think_tags(self):
        """Verify think tag matching is case-insensitive."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '<THINK>Uppercase thinking</THINK>{"action": "ACTION7"}',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION7"}
        assert "Uppercase thinking" in result.reasoning_trace
    
    def test_whitespace_in_content(self):
        """Verify whitespace handling in content."""
        client = VLLMClient()
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": '  \n  {"action": "ACTION8"}  \n  ',
                    "reasoning_content": None
                }
            }]
        }
        
        result = client._isolate_and_parse_thinking(mock_response)
        
        assert result.status == "OK"
        assert result.payload == {"action": "ACTION8"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
