from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator

from .base import StrictModel
from .paper_metadata import NonEmptyText


class RelationRecordField(StrEnum):
    RECORD_TYPE = "record_type"
    RELATION_TYPE = "relation_type"
    DIRECTION = "direction"
    RELATION_EXPLANATION = "relation_explanation"
    PAPER_A_EVIDENCE_QUOTES = "paper_a_evidence_quotes"
    PAPER_B_EVIDENCE_QUOTES = "paper_b_evidence_quotes"
    NO_RELATION_REASON = "no_relation_reason"


class RelationReviewIssue(StrictModel):
    target_fields: list[RelationRecordField] = Field(min_length=1)
    issue_description: NonEmptyText
    paper_a_review_evidence_quotes: list[NonEmptyText] = Field(max_length=3)
    paper_b_review_evidence_quotes: list[NonEmptyText] = Field(max_length=3)
    revision_guidance: NonEmptyText

    @field_validator(
        "target_fields",
        "paper_a_review_evidence_quotes",
        "paper_b_review_evidence_quotes",
    )
    @classmethod
    def list_values_must_be_unique(cls, value: list) -> list:
        if len(value) != len(set(value)):
            raise ValueError("list values must not contain duplicates")
        return value


class RelationReviewPassed(StrictModel):
    status: Literal["pass"]


class RelationReviewNeedsRevision(StrictModel):
    status: Literal["needs_revision"]
    issues: list[RelationReviewIssue] = Field(min_length=1)


RelationReviewResult = Annotated[
    RelationReviewPassed | RelationReviewNeedsRevision,
    Field(discriminator="status"),
]
