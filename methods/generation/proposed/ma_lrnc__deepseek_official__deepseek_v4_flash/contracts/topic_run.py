from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .paper_profile import NonEmptyText
from .runtime import StageMetrics


class PaperRunCounts(StrictModel):
    expected: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    input_failed: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_must_cover_expected_papers(self) -> "PaperRunCounts":
        if self.completed + self.failed + self.input_failed != self.expected:
            raise ValueError("paper counts must sum to expected")
        return self


class PairRunCounts(StrictModel):
    expected: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    blocked: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_must_cover_expected_pairs(self) -> "PairRunCounts":
        if self.completed + self.failed + self.blocked != self.expected:
            raise ValueError("pair counts must sum to expected")
        return self


class TopicRunFailure(StrictModel):
    scope: Literal["input", "paper", "pair"]
    business_key: NonEmptyText
    code: NonEmptyText
    message: NonEmptyText


class TopicConsumptionSummary(StrictModel):
    current_run: StageMetrics
    completed_result_chain: StageMetrics


class TopicRunSummary(StrictModel):
    status: Literal["relations_complete", "incomplete"]
    topic: NonEmptyText
    papers: PaperRunCounts
    pairs: PairRunCounts
    consumption: TopicConsumptionSummary
    failures: list[TopicRunFailure]

    @model_validator(mode="after")
    def status_must_match_counts(self) -> "TopicRunSummary":
        complete = (
            self.papers.failed == 0
            and self.papers.input_failed == 0
            and self.pairs.failed == 0
            and self.pairs.blocked == 0
        )
        expected_status = "relations_complete" if complete else "incomplete"
        if self.status != expected_status:
            raise ValueError("status does not match paper and pair counts")
        return self
