"""The HTTP client is exercised with a fake transport, so these tests stay offline."""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm import LLMError, OpenAIChatLLM


def test_missing_api_key_does_not_call_the_network():
    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"network was called: {request.url}")

    llm = OpenAIChatLLM(
        api_key="",
        base_url="https://example.invalid/v1",
        model="fake",
        client=httpx.Client(transport=httpx.MockTransport(explode)),
    )
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        llm.complete("system", "farmer message")


def test_complete_posts_chat_completions_and_reads_the_text():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"findings": []}'}}]},
        )

    llm = OpenAIChatLLM(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="fake-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.complete("be precise", "the cow is sick") == '{"findings": []}'
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "fake-model"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][1]["content"] == "the cow is sick"


def test_json_mode_can_be_turned_off():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert "response_format" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = OpenAIChatLLM(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="fake-model",
        json_mode=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.complete("system", "message") == "{}"


def test_provider_http_error_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    llm = OpenAIChatLLM(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="fake-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match="LLM request failed"):
        llm.complete("system", "message")


def test_content_blocks_are_joined():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": [{"type": "text", "text": '{"findings": []}'}]}}]},
        )

    llm = OpenAIChatLLM(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="fake-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.complete("system", "message") == '{"findings": []}'
