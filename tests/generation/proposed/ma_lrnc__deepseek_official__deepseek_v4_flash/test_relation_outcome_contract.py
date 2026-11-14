import pytest
from pydantic import TypeAdapter, ValidationError

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_outcome import (
    AcceptedRelationOutcome,
    ManualReviewRelationOutcome,
    RelationOutcome,
)


def unrelated_record() -> dict[str, object]:
    return {
        "record_type": "unrelated",
        "paper_a_id": "P001",
        "paper_b_id": "P002",
        "no_relation_reason": "两篇论文没有形成可命名的知识动作。",
    }


def issue() -> dict[str, object]:
    return {
        "target_fields": ["no_relation_reason"],
        "issue_description": "无关系理由不充分。",
        "paper_a_review_evidence_quotes": [],
        "paper_b_review_evidence_quotes": [],
        "revision_guidance": "补充排除六类关系的依据。",
    }


def test_relation_outcome_uses_review_status_discriminator() -> None:
    adapter = TypeAdapter(RelationOutcome)

    accepted = adapter.validate_python(
        {
            "review_status": "accepted",
            "relation_record": unrelated_record(),
        }
    )
    manual = adapter.validate_python(
        {
            "review_status": "manual_review_required",
            "relation_record": unrelated_record(),
            "outstanding_issues": [issue()],
        }
    )

    assert isinstance(accepted, AcceptedRelationOutcome)
    assert isinstance(manual, ManualReviewRelationOutcome)


def test_accepted_outcome_cannot_carry_outstanding_issues() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(RelationOutcome).validate_python(
            {
                "review_status": "accepted",
                "relation_record": unrelated_record(),
                "outstanding_issues": [issue()],
            }
        )


def test_manual_review_outcome_requires_nonempty_outstanding_issues() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(RelationOutcome).validate_python(
            {
                "review_status": "manual_review_required",
                "relation_record": unrelated_record(),
                "outstanding_issues": [],
            }
        )


def test_relation_outcome_rejects_unknown_review_status() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(RelationOutcome).validate_python(
            {
                "review_status": "pending",
                "relation_record": unrelated_record(),
            }
        )


def test_relation_outcome_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(RelationOutcome).validate_python(
            {
                "review_status": "accepted",
                "relation_record": unrelated_record(),
                "confidence": 0.9,
            }
        )
