import json
import time
from collections.abc import Callable, Sequence

from pydantic import TypeAdapter, ValidationError

from ..contracts.paper_metadata import PaperMetadata
from ..contracts.paper_profile import PaperProfileContent
from ..contracts.profile_review import ProfileReviewResult
from ..contracts.relation_record import RelationRecord
from ..contracts.relation_review import RelationReviewIssue, RelationReviewResult
from ..contracts.runtime import (
    ProfileReviewerRunResult,
    RelationReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from ..llm_client import JsonCompletionClient, TransientNetworkError
from ..prompting import (
    build_profile_review_system_prompt,
    build_profile_review_user_message,
    build_relation_post_revision_review_system_prompt,
    build_relation_post_revision_review_user_message,
    build_relation_review_system_prompt,
    build_relation_review_user_message,
)
from ..relation_review_validation import (
    RelationReviewFieldError,
    validate_relation_review_fields,
)
from ..review_validation import (
    ProfileReviewLocationError,
    validate_profile_review_locations,
)


_REVIEW_ADAPTER = TypeAdapter(ProfileReviewResult)
_RELATION_REVIEW_ADAPTER = TypeAdapter(RelationReviewResult)


class ProfileReviewerExecutionError(RuntimeError):
    def __init__(self, message: str, *, metrics: StageMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


class ProfileReviewNetworkRetriesExhausted(ProfileReviewerExecutionError):
    pass


class ProfileReviewContentRetriesExhausted(ProfileReviewerExecutionError):
    pass


class RelationReviewExecutionError(RuntimeError):
    def __init__(self, message: str, *, metrics: StageMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


class RelationReviewNetworkRetriesExhausted(RelationReviewExecutionError):
    pass


class RelationReviewContentRetriesExhausted(RelationReviewExecutionError):
    pass


class _NetworkAttemptsExhausted(RuntimeError):
    pass


class EvidenceReviewerAgent:
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

    def review_profile(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
        current_profile: PaperProfileContent,
    ) -> ProfileReviewerRunResult:
        messages = [
            {
                "role": "system",
                "content": build_profile_review_system_prompt(),
            },
            {
                "role": "user",
                "content": build_profile_review_user_message(
                    title=title,
                    authors=authors,
                    article_full_text=article_full_text,
                    current_profile=current_profile,
                ),
            },
        ]
        return self._execute(messages, current_profile)

    def review_relation(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
        current_relation: RelationRecord,
    ) -> RelationReviewerRunResult:
        messages = [
            {
                "role": "system",
                "content": build_relation_review_system_prompt(),
            },
            {
                "role": "user",
                "content": build_relation_review_user_message(
                    paper_a_metadata=paper_a_metadata,
                    paper_a_profile=paper_a_profile,
                    paper_a_full_text=paper_a_full_text,
                    paper_b_metadata=paper_b_metadata,
                    paper_b_profile=paper_b_profile,
                    paper_b_full_text=paper_b_full_text,
                    current_relation=current_relation,
                ),
            },
        ]
        return self._execute_relation_review(messages, current_relation)

    def review_relation_after_revision(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
        previous_review_issues: Sequence[RelationReviewIssue],
        revised_relation: RelationRecord,
    ) -> RelationReviewerRunResult:
        messages = [
            {
                "role": "system",
                "content": build_relation_post_revision_review_system_prompt(),
            },
            {
                "role": "user",
                "content": build_relation_post_revision_review_user_message(
                    paper_a_metadata=paper_a_metadata,
                    paper_a_profile=paper_a_profile,
                    paper_a_full_text=paper_a_full_text,
                    paper_b_metadata=paper_b_metadata,
                    paper_b_profile=paper_b_profile,
                    paper_b_full_text=paper_b_full_text,
                    previous_review_issues=previous_review_issues,
                    revised_relation=revised_relation,
                ),
            },
        ]
        return self._execute_relation_review(messages, revised_relation)

    def _execute(
        self,
        messages: list[dict[str, str]],
        current_profile: PaperProfileContent,
    ) -> ProfileReviewerRunResult:
        started = self._clock()
        total_usage = TokenUsage.zero()
        last_content_error: Exception | None = None

        for _content_attempt in range(3):
            try:
                completion = self._request_with_network_retries(messages)
            except _NetworkAttemptsExhausted as error:
                raise ProfileReviewNetworkRetriesExhausted(
                    "profile review network request failed after three attempts",
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
                raw_review = json.loads(completion.content)
                review = _REVIEW_ADAPTER.validate_python(raw_review)
                validate_profile_review_locations(review, current_profile)
            except (
                json.JSONDecodeError,
                ValidationError,
                ProfileReviewLocationError,
                ValueError,
                TypeError,
            ) as error:
                last_content_error = error
                continue

            return ProfileReviewerRunResult(
                review=review,
                metrics=StageMetrics(
                    elapsed_seconds=max(0.0, self._clock() - started),
                    token_usage=total_usage,
                ),
            )

        raise ProfileReviewContentRetriesExhausted(
            "profile review content failed validation after three attempts",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, self._clock() - started),
                token_usage=total_usage,
            ),
        ) from last_content_error

    def _execute_relation_review(
        self,
        messages: list[dict[str, str]],
        current_relation: RelationRecord,
    ) -> RelationReviewerRunResult:
        started = self._clock()
        total_usage = TokenUsage.zero()
        last_content_error: Exception | None = None

        for _content_attempt in range(3):
            try:
                completion = self._request_with_network_retries(messages)
            except _NetworkAttemptsExhausted as error:
                raise RelationReviewNetworkRetriesExhausted(
                    "relation review network request failed after three attempts",
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
                raw_review = json.loads(completion.content)
                review = _RELATION_REVIEW_ADAPTER.validate_python(raw_review)
                validate_relation_review_fields(review, current_relation)
            except (
                json.JSONDecodeError,
                ValidationError,
                RelationReviewFieldError,
                ValueError,
                TypeError,
            ) as error:
                last_content_error = error
                continue

            return RelationReviewerRunResult(
                review=review,
                metrics=StageMetrics(
                    elapsed_seconds=max(0.0, self._clock() - started),
                    token_usage=total_usage,
                ),
            )

        raise RelationReviewContentRetriesExhausted(
            "relation review content failed validation after three attempts",
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
