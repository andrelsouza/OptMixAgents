"""Tests for OpenAIClient — tool format translation and response parsing."""

import json

import pytest

from optmix.core.llm import OpenAIClient, ToolCall, create_llm_client


class TestToolTranslation:
    """Test Anthropic → OpenAI tool format translation."""

    def test_tool_schema_translation(self):
        anthropic_tool = {
            "name": "validate_data",
            "description": "Validate a dataset",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        }

        result = OpenAIClient._translate_tool(anthropic_tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "validate_data"
        assert result["function"]["description"] == "Validate a dataset"
        assert result["function"]["parameters"]["type"] == "object"
        assert "path" in result["function"]["parameters"]["properties"]

    def test_tool_with_no_description(self):
        tool = {"name": "noop", "input_schema": {"type": "object"}}
        result = OpenAIClient._translate_tool(tool)
        assert result["function"]["description"] == ""


class TestMessageTranslation:
    """Test Anthropic → OpenAI message format translation."""

    def test_plain_text_passthrough(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = OpenAIClient._translate_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_tool_result_translation(self):
        """Anthropic tool_result → OpenAI role:tool format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Data is valid.",
                    }
                ],
            }
        ]
        result = OpenAIClient._translate_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "Data is valid."

    def test_assistant_tool_use_translation(self):
        """Anthropic tool_use blocks → OpenAI tool_calls format."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me validate."},
                    {
                        "type": "tool_use",
                        "id": "call_abc",
                        "name": "validate_data",
                        "input": {"path": "data.csv"},
                    },
                ],
            }
        ]
        result = OpenAIClient._translate_messages(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me validate."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "validate_data"
        assert json.loads(tc["function"]["arguments"]) == {"path": "data.csv"}

    def test_multiple_tool_results(self):
        """Multiple tool_result blocks produce multiple role:tool messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "Result 1"},
                    {"type": "tool_result", "tool_use_id": "c2", "content": "Result 2"},
                ],
            }
        ]
        result = OpenAIClient._translate_messages(messages)
        assert len(result) == 2
        assert result[0]["tool_call_id"] == "c1"
        assert result[1]["tool_call_id"] == "c2"

    def test_mixed_conversation(self):
        """Full conversation with text, tool_use, and tool_result messages."""
        messages = [
            {"role": "user", "content": "Check my data"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "validate_data", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "OK"},
                ],
            },
            {"role": "assistant", "content": "Your data looks good!"},
        ]
        result = OpenAIClient._translate_messages(messages)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "assistant"


class TestResponseParsing:
    """Test OpenAI response parsing using mock response objects."""

    def _make_response(self, content="Hello", tool_calls=None, finish_reason="stop"):
        """Create a mock OpenAI ChatCompletion response."""

        class _Usage:
            prompt_tokens = 100
            completion_tokens = 50

        class _Function:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class _ToolCall:
            def __init__(self, id, function):
                self.id = id
                self.type = "function"
                self.function = function

        class _Message:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
                self.role = "assistant"

        class _Choice:
            def __init__(self, message, finish_reason):
                self.message = message
                self.finish_reason = finish_reason
                self.index = 0

        oai_tcs = None
        if tool_calls:
            oai_tcs = [
                _ToolCall(tc["id"], _Function(tc["name"], json.dumps(tc["input"])))
                for tc in tool_calls
            ]

        class _Response:
            choices = [_Choice(_Message(content, oai_tcs), finish_reason)]
            usage = _Usage()

        return _Response()

    def test_parse_text_response(self):
        resp = self._make_response(content="Hello world")
        # Use a dummy client to call _parse_response
        client = OpenAIClient.__new__(OpenAIClient)
        result = client._parse_response(resp)
        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert result.usage["input_tokens"] == 100

    def test_parse_tool_call_response(self):
        resp = self._make_response(
            content=None,
            tool_calls=[
                {"id": "call_x", "name": "validate_data", "input": {"path": "test.csv"}},
            ],
            finish_reason="tool_calls",
        )
        client = OpenAIClient.__new__(OpenAIClient)
        result = client._parse_response(resp)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "validate_data"
        assert result.tool_calls[0].input == {"path": "test.csv"}
        assert result.stop_reason == "tool_use"

    def test_parse_usage(self):
        resp = self._make_response()
        client = OpenAIClient.__new__(OpenAIClient)
        result = client._parse_response(resp)
        assert result.usage == {"input_tokens": 100, "output_tokens": 50}


class TestCreateLLMClientOpenAI:
    """Test factory function with OpenAI provider."""

    def test_create_openai_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        # This will fail on import if openai package isn't installed,
        # but we can test that the factory routes correctly
        try:
            client = create_llm_client(provider="openai")
            assert client.provider_name() == "openai"
        except ImportError:
            pytest.skip("openai package not installed")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(provider="gemini")
