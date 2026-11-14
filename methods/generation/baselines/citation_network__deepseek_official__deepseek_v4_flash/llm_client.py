from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol

from .contracts import TokenUsage


@dataclass(frozen=True)
class CompletionResult:
    content: str
    finish_reason: str
    token_usage: TokenUsage


class TransientNetworkError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class JsonCompletionClient(Protocol):
    def create_json_completion(
        self,
        messages: list[dict[str, str]],
    ) -> CompletionResult: ...


class DeepSeekJsonClient:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self._model = model

    def create_json_completion(
        self,
        messages: list[dict[str, str]],
    ) -> CompletionResult:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            if _is_transient_error(error):
                response = getattr(error, "response", None)
                headers = getattr(response, "headers", None)
                raise TransientNetworkError(
                    type(error).__name__,
                    retry_after=_parse_retry_after(headers),
                ) from error
            raise
        choice = response.choices[0]
        usage = response.usage
        return CompletionResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            token_usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )


def _parse_retry_after(headers: object | None) -> float | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
        return seconds if seconds >= 0 else None
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(raw))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _is_transient_error(error: Exception) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError
    except ImportError:
        return False
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    status_code = getattr(error, "status_code", None)
    return status_code in {408, 429} or (
        isinstance(status_code, int) and status_code >= 500
    )
