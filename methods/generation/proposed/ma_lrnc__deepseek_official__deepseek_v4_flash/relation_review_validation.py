from collections.abc import Sequence

from .contracts.relation_record import RelationRecord
from .contracts.relation_review import (
    RelationReviewIssue,
    RelationReviewNeedsRevision,
    RelationReviewResult,
)


class RelationReviewFieldError(ValueError):
    pass


def validate_relation_review_fields(
    review: RelationReviewResult,
    current_relation: RelationRecord,
) -> None:
    if not isinstance(review, RelationReviewNeedsRevision):
        return
    validate_relation_review_issue_fields(review.issues, current_relation)


def validate_relation_review_issue_fields(
    issues: Sequence[RelationReviewIssue],
    current_relation: RelationRecord,
) -> None:
    if not issues:
        raise RelationReviewFieldError(
            "relation review issues must contain at least one issue"
        )
    current_fields = set(current_relation.model_fields_set)
    for issue in issues:
        missing = [
            field.value
            for field in issue.target_fields
            if field.value not in current_fields
        ]
        if missing:
            raise RelationReviewFieldError(
                "relation review target fields do not exist on current record: "
                + ", ".join(missing)
            )
