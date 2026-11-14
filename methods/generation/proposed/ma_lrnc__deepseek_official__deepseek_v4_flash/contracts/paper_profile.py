from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator

from .base import StrictModel


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PaperAction(StrEnum):
    PROPOSES = "proposes"
    ADOPTS = "adopts"
    DEVELOPS = "develops"
    MODIFIES = "modifies"
    APPLIES = "applies"
    TESTS = "tests"
    CHALLENGES = "challenges"


class ProfileItem(StrictModel):
    statement: NonEmptyText
    evidence_quotes: list[NonEmptyText] = Field(min_length=1, max_length=3)

    @field_validator("evidence_quotes")
    @classmethod
    def evidence_quotes_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_quotes must not contain duplicates")
        return value


class AttributedProfileItem(ProfileItem):
    paper_actions: list[PaperAction] = Field(min_length=1, max_length=7)

    @field_validator("paper_actions")
    @classmethod
    def paper_actions_must_be_unique(
        cls, value: list[PaperAction]
    ) -> list[PaperAction]:
        if len(value) != len(set(value)):
            raise ValueError("paper_actions must not contain duplicates")
        return value


class PaperProfileContent(StrictModel):
    research_questions: list[ProfileItem]
    theories_and_viewpoints: list[AttributedProfileItem]
    methods_and_models: list[AttributedProfileItem]
    data_and_contexts: list[ProfileItem]
    main_findings: list[ProfileItem]
    limitations_and_boundary_conditions: list[ProfileItem]

