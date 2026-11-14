import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.config import (
    ConfigurationError,
    load_deepseek_config,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
    DeepSeekJsonClient,
    TokenTrackingJsonClient,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    TokenUsage,
)


BASE_URL = "LLM_DEEPSEEK_OFFICIAL_BASE_URL"
API_KEY = "LLM_DEEPSEEK_OFFICIAL_API_KEY"
API_STYLE = "LLM_DEEPSEEK_OFFICIAL_API_STYLE"
MODEL = "LLM_DEEPSEEK_OFFICIAL_DEEPSEEK_V4_FLASH_MODEL"


def test_config_loads_only_required_deepseek_backend_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"{BASE_URL}=https://api.deepseek.com",
                f"{API_KEY}=secret-for-test-only",
                f"{API_STYLE}=openai",
                f"{MODEL}=deepseek-v4-flash",
            ]
        ),
        encoding="utf-8",
    )

    config = load_deepseek_config(env_file=env_file, environ={})

    assert config.base_url == "https://api.deepseek.com"
    assert config.api_style == "openai"
    assert config.model == "deepseek-v4-flash"


def test_config_reports_missing_variable_name_without_other_values(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"{BASE_URL}=https://api.deepseek.com",
                f"{API_STYLE}=openai",
                f"{MODEL}=deepseek-v4-flash",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as error:
        load_deepseek_config(env_file=env_file, environ={})

    assert API_KEY in str(error.value)
    assert "https://api.deepseek.com" not in str(error.value)


def test_config_rejects_non_openai_api_style(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"{BASE_URL}=https://api.deepseek.com",
                f"{API_KEY}=secret-for-test-only",
                f"{API_STYLE}=gemini",
                f"{MODEL}=deepseek-v4-flash",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_deepseek_config(env_file=env_file, environ={})


def test_deepseek_client_disables_sdk_retries_and_sets_no_generation_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_kwargs: dict = {}
    request_kwargs: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"research_questions": []}'),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            constructor_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai_module = SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    client = DeepSeekJsonClient(
        base_url="https://api.deepseek.com",
        api_key="secret-for-test-only",
        model="deepseek-v4-flash",
    )
    client.create_json_completion(
        [{"role": "user", "content": "请输出 JSON。"}]
    )

    assert constructor_kwargs["max_retries"] == 0
    assert set(request_kwargs) == {"model", "messages", "response_format"}
    assert request_kwargs["response_format"] == {"type": "json_object"}


def test_token_tracking_client_sums_only_returned_api_usage() -> None:
    responses = iter(
        [
            CompletionResult(
                content="{}",
                finish_reason="stop",
                token_usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            ),
            CompletionResult(
                content="{}",
                finish_reason="stop",
                token_usage=TokenUsage(
                    prompt_tokens=20,
                    completion_tokens=7,
                    total_tokens=27,
                ),
            ),
        ]
    )

    class FakeClient:
        def create_json_completion(
            self,
            messages: list[dict[str, str]],
        ) -> CompletionResult:
            return next(responses)

    client = TokenTrackingJsonClient(client=FakeClient())
    client.create_json_completion([])
    client.create_json_completion([])

    assert client.token_usage == TokenUsage(
        prompt_tokens=30,
        completion_tokens=12,
        total_tokens=42,
    )
