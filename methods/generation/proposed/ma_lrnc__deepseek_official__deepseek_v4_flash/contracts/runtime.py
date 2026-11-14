from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .paper_profile import PaperProfileContent
from .profile_review import ProfileReviewResult
from .relation_record import RelationRecord
from .relation_review import RelationReviewResult


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def total_must_match_parts(self) -> "TokenUsage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self

    @classmethod
    def zero(cls) -> "TokenUsage":
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def plus(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class StageMetrics(StrictModel):
    elapsed_seconds: float = Field(ge=0)
    token_usage: TokenUsage

    @classmethod
    def zero(cls) -> "StageMetrics":
        return cls(
            elapsed_seconds=0.0,
            token_usage=TokenUsage.zero(),
        )

    def plus(self, other: "StageMetrics") -> "StageMetrics":
        return StageMetrics(
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
            token_usage=self.token_usage.plus(other.token_usage),
        )


PaperProfileStage = Literal[
    "paper_profile_initial_generation",
    "paper_profile_targeted_revision",
]

RelationInferenceStage = Literal[
    "relation_initial_inference",
    "relation_targeted_revision",
]


class PaperParserRunResult(StrictModel):
    profile: PaperProfileContent
    metrics: StageMetrics


class ProfileReviewerRunResult(StrictModel):
    review: ProfileReviewResult
    metrics: StageMetrics


class RelationInferenceRunResult(StrictModel):
    record: RelationRecord
    metrics: StageMetrics


class RelationReviewerRunResult(StrictModel):
    review: RelationReviewResult
    metrics: StageMetrics


class PaperProfileCheckpoint(StrictModel):
    checkpoint_version: Literal[1]
    stage: PaperProfileStage
    business_key: str = Field(pattern=r"^P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result: PaperProfileContent
    metrics: StageMetrics


class ProfileReviewCheckpoint(StrictModel):
    checkpoint_version: Literal[1]
    stage: Literal["paper_profile_review"]
    business_key: str = Field(pattern=r"^P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result: ProfileReviewResult
    metrics: StageMetrics


class RelationInferenceCheckpoint(StrictModel):
    checkpoint_version: Literal[1]
    stage: RelationInferenceStage
    business_key: str = Field(pattern=r"^P\d{3}__P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result: RelationRecord
    metrics: StageMetrics


class RelationReviewCheckpoint(StrictModel):
    checkpoint_version: Literal[1]
    stage: Literal["relation_initial_review", "relation_post_revision_review"]
    business_key: str = Field(pattern=r"^P\d{3}__P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result: RelationReviewResult
    metrics: StageMetrics


FailureKind = Literal[
    "input_invalid",
    "network_exhausted",
    "content_exhausted",
    "internal_error",
]


class PaperParserFailureRecord(StrictModel):
    failure_record_version: Literal[1]
    stage: PaperProfileStage
    business_key: str = Field(pattern=r"^P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    failure_kind: FailureKind
    metrics: StageMetrics


class ProfileReviewerFailureRecord(StrictModel):
    failure_record_version: Literal[1]
    stage: Literal["paper_profile_review"]
    business_key: str = Field(pattern=r"^P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    failure_kind: FailureKind
    metrics: StageMetrics


class RelationInferenceFailureRecord(StrictModel):
    failure_record_version: Literal[1]
    stage: RelationInferenceStage
    business_key: str = Field(pattern=r"^P\d{3}__P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    failure_kind: FailureKind
    metrics: StageMetrics


class RelationReviewerFailureRecord(StrictModel):
    failure_record_version: Literal[1]
    stage: Literal["relation_initial_review", "relation_post_revision_review"]
    business_key: str = Field(pattern=r"^P\d{3}__P\d{3}$")
    task_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    failure_kind: FailureKind
    metrics: StageMetrics
