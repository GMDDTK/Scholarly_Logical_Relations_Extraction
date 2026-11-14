import json
import time
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from ..contracts.paper_profile import PaperProfileContent
from ..contracts.profile_review import ProfileReviewIssue
from ..contracts.runtime import PaperParserRunResult, StageMetrics, TokenUsage
from ..llm_client import JsonCompletionClient, TransientNetworkError
from ..prompting import (
    build_initial_user_message,
    build_revision_user_message,
    build_system_prompt,
)


class PaperParserExecutionError(RuntimeError):
    def __init__(self, message: str, *, metrics: StageMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


class NetworkRetriesExhausted(PaperParserExecutionError):
    pass


class ContentRetriesExhausted(PaperParserExecutionError):
    pass


class _NetworkAttemptsExhausted(RuntimeError):
    pass


class PaperParserAgent:
    def __init__(
        self,
        *,
        client: JsonCompletionClient,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self._sleep = sleep
        self._clock = clock

    def initial_generation(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
    ) -> PaperParserRunResult:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt("initial_generation"),
            },
            {
                "role": "user",
                "content": build_initial_user_message(
                    title=title,
                    authors=authors,
                    article_full_text=article_full_text,
                ),
            },
        ]
        return self._execute(messages)

    def targeted_revision(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
        current_profile: PaperProfileContent,
        review_issues: Sequence[ProfileReviewIssue],
    ) -> PaperParserRunResult:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt("targeted_revision"),
            },
            {
                "role": "user",
                "content": build_revision_user_message(
                    title=title,
                    authors=authors,
                    article_full_text=article_full_text,
                    current_profile=current_profile,
                    review_issues=review_issues,
                ),
            },
        ]
        return self._execute(messages)

    def _execute(
        self,
        messages: list[dict[str, str]],
    ) -> PaperParserRunResult:
        started = self._clock()
        total_usage = TokenUsage.zero()
        last_content_error: Exception | None = None

        for _content_attempt in range(3):
            try:
                completion = self._request_with_network_retries(messages)
            except _NetworkAttemptsExhausted as error:
                raise NetworkRetriesExhausted(
                    "network request failed after three attempts",
                    metrics=StageMetrics(
                        elapsed_seconds=max(0.0, self._clock() - started),
                        token_usage=total_usage,
                    ),
                ) from error
            total_usage = total_usage.plus(completion.token_usage)
            try:
                if completion.finish_reason != "stop":
                    raise ValueError(
                        f"unexpected finish_reason: {completion.finish_reason!r}"
                    )
                raw_profile = json.loads(completion.content)
                profile = PaperProfileContent.model_validate(raw_profile)
            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                TypeError,
            ) as error:
                last_content_error = error
                continue

            return PaperParserRunResult(
                profile=profile,
                metrics=StageMetrics(
                    elapsed_seconds=max(0.0, self._clock() - started),
                    token_usage=total_usage,
                ),
            )

        raise ContentRetriesExhausted(
            "paper parser content failed validation after three attempts",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, self._clock() - started),
                token_usage=total_usage,
            ),
        ) from last_content_error

    def _request_with_network_retries(
        self,
        messages: list[dict[str, str]],
    ):
        for network_attempt in range(3):
            try:
                return self._client.create_json_completion(messages)
            except TransientNetworkError as error:
                if network_attempt == 2:
                    raise _NetworkAttemptsExhausted(
                        "network request failed after three attempts"
                    ) from error
                default_delay = (2.0, 4.0)[network_attempt]
                delay = (
                    error.retry_after
                    if error.retry_after is not None
                    else default_delay
                )
                self._sleep(delay)
        raise AssertionError("unreachable")
