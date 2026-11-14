from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from .base import StrictModel


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ProfileField(StrEnum):
    RESEARCH_QUESTIONS = "research_questions"
    THEORIES_AND_VIEWPOINTS = "theories_and_viewpoints"
    METHODS_AND_MODELS = "methods_and_models"
    DATA_AND_CONTEXTS = "data_and_contexts"
    MAIN_FINDINGS = "main_findings"
    LIMITATIONS_AND_BOUNDARY_CONDITIONS = "limitations_and_boundary_conditions"


class ProfileTargetLocation(StrictModel):
    field: ProfileField
    item_index: int | None = Field(default=None, ge=0)


class ProfileReviewIssue(StrictModel):
    target_locations: list[ProfileTargetLocation] = Field(min_length=1)
    issue_description: NonEmptyText
    review_evidence_quotes: list[NonEmptyText] = Field(max_length=3)
    revision_guidance: NonEmptyText

    @field_validator("review_evidence_quotes")
    @classmethod
    def review_quotes_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("review_evidence_quotes must not contain duplicates")
        return value

    @model_validator(mode="after")
    def target_locations_must_be_unique(self) -> "ProfileReviewIssue":
        keys = [(location.field, location.item_index) for location in self.target_locations]
        if len(keys) != len(set(keys)):
            raise ValueError("target_locations must not contain duplicates")
        return self


class ProfileReviewPassed(StrictModel):
    status: Literal["pass"]


class ProfileReviewNeedsRevision(StrictModel):
    status: Literal["needs_revision"]
    issues: list[ProfileReviewIssue] = Field(min_length=1)


ProfileReviewResult = Annotated[
    ProfileReviewPassed | ProfileReviewNeedsRevision,
    Field(discriminator="status"),
]

