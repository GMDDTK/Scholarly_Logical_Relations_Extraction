from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import (
    CitationIdentification,
    CitationReviewResult,
    StageMetrics,
    TokenUsage,
)
from .llm_client import JsonCompletionClient, TransientNetworkError


PROMPT_ROOT = Path(__file__).with_name("prompts")
COMMON_RULES = (PROMPT_ROOT / "common_rules.md").read_text(encoding="utf-8").strip()
IDENTIFICATION_PROMPT = (
    COMMON_RULES
    + "\n\n"
    + (PROMPT_ROOT / "identification.md").read_text(encoding="utf-8").strip()
)
REVIEW_PROMPT = (
    COMMON_RULES
    + "\n\n"
    + (PROMPT_ROOT / "review.md").read_text(encoding="utf-8").strip()
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class AgentExecution(Generic[T]):
    result: T
    metrics: StageMetrics


class AgentRetriesExhausted(RuntimeError):
    def __init__(self, message: str, *, metrics: StageMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


class ContentRetriesExhausted(AgentRetriesExhausted):
    pass


class NetworkRetriesExhausted(AgentRetriesExhausted):
    pass


class _StructuredAgent(Generic[T]):
    output_model: type[T]
    system_prompt: str

    def __init__(
        self,
        *,
        client: JsonCompletionClient,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._sleep = sleep

    def execute(
        self,
        *,
        user_message: str,
        validator: Callable[[T], None] | None = None,
    ) -> AgentExecution[T]:
        started = time.perf_counter()
        usage = TokenUsage.zero()
        last_content_error: Exception | None = None
        last_network_error: NetworkRetriesExhausted | None = None
        for content_attempt in range(3):
            try:
                completion = self._complete_with_network_retries(user_message)
            except NetworkRetriesExhausted as error:
                last_network_error = error
                if content_attempt < 2:
                    continue
                raise NetworkRetriesExhausted(
                    str(error),
                    metrics=StageMetrics(
                        elapsed_seconds=time.perf_counter() - started,
                        token_usage=usage.plus(error.metrics.token_usage),
                    ),
                ) from error
            usage = usage.plus(completion.token_usage)
            try:
                if completion.finish_reason != "stop":
                    raise ValueError(
                        f"completion finish_reason is {completion.finish_reason!r}"
                    )
                payload = json.loads(completion.content)
                result = self.output_model.model_validate(payload)
                if validator is not None:
                    validator(result)
                return AgentExecution(
                    result=result,
                    metrics=StageMetrics(
                        elapsed_seconds=time.perf_counter() - started,
                        token_usage=usage,
                    ),
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                last_content_error = error
        if last_network_error is not None and last_content_error is None:
            raise NetworkRetriesExhausted(
                str(last_network_error),
                metrics=StageMetrics(
                    elapsed_seconds=time.perf_counter() - started,
                    token_usage=usage,
                ),
            ) from last_network_error
        raise ContentRetriesExhausted(
            f"structured content remained invalid after 3 attempts: {last_content_error}",
            metrics=StageMetrics(
                elapsed_seconds=time.perf_counter() - started,
                token_usage=usage,
            ),
        )

    def _complete_with_network_retries(self, user_message: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        for network_attempt in range(3):
            try:
                return self._client.create_json_completion(messages)
            except TransientNetworkError as error:
                if network_attempt == 2:
                    raise NetworkRetriesExhausted(
                        "network retries exhausted",
                        metrics=StageMetrics(
                            elapsed_seconds=0,
                            token_usage=TokenUsage.zero(),
                        ),
                    ) from error
                delay = error.retry_after
                if delay is None:
                    delay = (2.0, 4.0)[network_attempt]
                self._sleep(delay)
        raise AssertionError("unreachable")


class CitationIdentificationAgent(_StructuredAgent[CitationIdentification]):
    output_model = CitationIdentification
    system_prompt = IDENTIFICATION_PROMPT

    def identify(
        self,
        *,
        user_message: str,
        validator: Callable[[CitationIdentification], None] | None = None,
    ) -> AgentExecution[CitationIdentification]:
        return self.execute(user_message=user_message, validator=validator)


class CitationReviewAgent(_StructuredAgent[CitationReviewResult]):
    output_model = CitationReviewResult
    system_prompt = REVIEW_PROMPT

    def review(
        self,
        *,
        user_message: str,
        validator: Callable[[CitationReviewResult], None] | None = None,
    ) -> AgentExecution[CitationReviewResult]:
        return self.execute(user_message=user_message, validator=validator)
