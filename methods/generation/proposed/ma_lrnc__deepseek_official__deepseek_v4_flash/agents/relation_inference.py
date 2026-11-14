import json
import time
from collections.abc import Callable
from collections.abc import Sequence

from pydantic import TypeAdapter, ValidationError

from ..contracts.paper_metadata import PaperMetadata
from ..contracts.paper_profile import PaperProfileContent
from ..contracts.relation_record import RelationRecord
from ..contracts.relation_review import RelationReviewIssue
from ..contracts.runtime import RelationInferenceRunResult, StageMetrics, TokenUsage
from ..llm_client import JsonCompletionClient, TransientNetworkError
from ..prompting import (
    build_relation_inference_system_prompt,
    build_relation_inference_user_message,
    build_relation_revision_system_prompt,
    build_relation_revision_user_message,
)
from ..relation_validation import (
    RelationHandoffError,
    validate_relation_handoff,
    validate_relation_revision_handoff,
)


_RELATION_ADAPTER = TypeAdapter(RelationRecord)


class RelationInferenceExecutionError(RuntimeError):
    def __init__(self, message: str, *, metrics: StageMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


class RelationInferenceNetworkRetriesExhausted(
    RelationInferenceExecutionError
):
    pass


class RelationInferenceContentRetriesExhausted(
    RelationInferenceExecutionError
):
    pass


class _NetworkAttemptsExhausted(RuntimeError):
    pass


class RelationInferenceAgent:
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

    def initial_inference(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
    ) -> RelationInferenceRunResult:
        messages = [
            {
                "role": "system",
                "content": build_relation_inference_system_prompt(),
            },
            {
                "role": "user",
                "content": build_relation_inference_user_message(
                    paper_a_metadata=paper_a_metadata,
                    paper_a_profile=paper_a_profile,
                    paper_a_full_text=paper_a_full_text,
                    paper_b_metadata=paper_b_metadata,
                    paper_b_profile=paper_b_profile,
                    paper_b_full_text=paper_b_full_text,
                ),
            },
        ]
        return self._execute(
            messages=messages,
            paper_a_metadata=paper_a_metadata,
            paper_b_metadata=paper_b_metadata,
        )

    def targeted_revision(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
        current_relation: RelationRecord,
        review_issues: Sequence[RelationReviewIssue],
    ) -> RelationInferenceRunResult:
        messages = [
            {
                "role": "system",
                "content": build_relation_revision_system_prompt(),
            },
            {
                "role": "user",
                "content": build_relation_revision_user_message(
                    paper_a_metadata=paper_a_metadata,
                    paper_a_profile=paper_a_profile,
                    paper_a_full_text=paper_a_full_text,
                    paper_b_metadata=paper_b_metadata,
                    paper_b_profile=paper_b_profile,
                    paper_b_full_text=paper_b_full_text,
                    current_relation=current_relation,
                    review_issues=review_issues,
                ),
            },
        ]
        return self._execute(
            messages=messages,
            paper_a_metadata=paper_a_metadata,
            paper_b_metadata=paper_b_metadata,
            current_relation=current_relation,
        )

    def _execute(
        self,
        *,
        messages: list[dict[str, str]],
        paper_a_metadata: PaperMetadata,
        paper_b_metadata: PaperMetadata,
        current_relation: RelationRecord | None = None,
    ) -> RelationInferenceRunResult:
        started = self._clock()
        total_usage = TokenUsage.zero()
        last_content_error: Exception | None = None

        for _content_attempt in range(3):
            try:
                completion = self._request_with_network_retries(messages)
            except _NetworkAttemptsExhausted as error:
                raise RelationInferenceNetworkRetriesExhausted(
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
                raw_record = json.loads(completion.content)
                record = _RELATION_ADAPTER.validate_python(raw_record)
                if current_relation is None:
                    validate_relation_handoff(
                        record,
                        paper_a_metadata,
                        paper_b_metadata,
                    )
                else:
                    validate_relation_revision_handoff(
                        record,
                        current_relation,
                        paper_a_metadata,
                        paper_b_metadata,
                    )
            except (
                json.JSONDecodeError,
                ValidationError,
                RelationHandoffError,
                ValueError,
                TypeError,
            ) as error:
                last_content_error = error
                continue

            return RelationInferenceRunResult(
                record=record,
                metrics=StageMetrics(
                    elapsed_seconds=max(0.0, self._clock() - started),
                    token_usage=total_usage,
                ),
            )

        raise RelationInferenceContentRetriesExhausted(
            "relation inference content failed validation after three attempts",
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
