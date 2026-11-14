from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .base import StrictModel
from .paper_metadata import NonEmptyText, PaperId


class RelationType(StrEnum):
    CARRIES_FORWARD = "承接"
    EXTENDS = "拓展"
    CORROBORATES = "印证"
    REVISES = "修正"
    CHALLENGES = "挑战"
    COMPLEMENTS = "互补"


class RelationDirection(StrEnum):
    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"
    SYMMETRIC = "symmetric"


class PairRecordBase(StrictModel):
    paper_a_id: PaperId
    paper_b_id: PaperId

    @model_validator(mode="after")
    def paper_ids_must_be_distinct_and_ordered(self) -> "PairRecordBase":
        if self.paper_a_id >= self.paper_b_id:
            raise ValueError("paper_a_id must be less than paper_b_id")
        return self


class RelatedPairRecord(PairRecordBase):
    record_type: Literal["related"]
    relation_type: RelationType
    direction: RelationDirection
    relation_explanation: NonEmptyText
    paper_a_evidence_quotes: list[NonEmptyText] = Field(
        min_length=1,
        max_length=3,
    )
    paper_b_evidence_quotes: list[NonEmptyText] = Field(
        min_length=1,
        max_length=3,
    )

    @field_validator(
        "paper_a_evidence_quotes",
        "paper_b_evidence_quotes",
    )
    @classmethod
    def evidence_quotes_must_be_unique(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence quotes must not contain duplicates")
        return value

    @model_validator(mode="after")
    def relation_type_and_direction_must_match(self) -> "RelatedPairRecord":
        if self.relation_type is RelationType.COMPLEMENTS:
            if self.direction is not RelationDirection.SYMMETRIC:
                raise ValueError("互补 must use symmetric direction")
        elif self.direction is RelationDirection.SYMMETRIC:
            raise ValueError("only 互补 may use symmetric direction")
        return self


class UnrelatedPairRecord(PairRecordBase):
    record_type: Literal["unrelated"]
    no_relation_reason: NonEmptyText


RelationRecord = Annotated[
    RelatedPairRecord | UnrelatedPairRecord,
    Field(discriminator="record_type"),
]
