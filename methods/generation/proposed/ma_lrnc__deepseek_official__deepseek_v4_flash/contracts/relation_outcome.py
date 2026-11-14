from typing import Annotated, Literal

from pydantic import Field

from .base import StrictModel
from .relation_record import RelationRecord
from .relation_review import RelationReviewIssue


class AcceptedRelationOutcome(StrictModel):
    review_status: Literal["accepted"]
    relation_record: RelationRecord


class ManualReviewRelationOutcome(StrictModel):
    review_status: Literal["manual_review_required"]
    relation_record: RelationRecord
    outstanding_issues: list[RelationReviewIssue] = Field(min_length=1)


RelationOutcome = Annotated[
    AcceptedRelationOutcome | ManualReviewRelationOutcome,
    Field(discriminator="review_status"),
]
