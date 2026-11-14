import pytest
from pydantic import TypeAdapter, ValidationError

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    RelationRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_review import (
    RelationReviewIssue,
    RelationReviewResult,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.relation_review_validation import (
    RelationReviewFieldError,
    validate_relation_review_fields,
)


REVIEW_ADAPTER = TypeAdapter(RelationReviewResult)
RELATION_ADAPTER = TypeAdapter(RelationRecord)


def unrelated_record():
    return RELATION_ADAPTER.validate_python(
        {
            "record_type": "unrelated",
            "paper_a_id": "P001",
            "paper_b_id": "P002",
            "no_relation_reason": "两篇论文只有宽泛主题相似。",
        }
    )


def test_relation_review_contract_has_only_pass_or_structured_issues() -> None:
    passed = REVIEW_ADAPTER.validate_python({"status": "pass"})
    needs_revision = REVIEW_ADAPTER.validate_python(
        {
            "status": "needs_revision",
            "issues": [
                {
                    "target_fields": ["record_type", "no_relation_reason"],
                    "issue_description": "遗漏了一项具体的方法承接关系。",
                    "paper_a_review_evidence_quotes": ["论文A提出该方法。"],
                    "paper_b_review_evidence_quotes": ["论文B采用该方法。"],
                    "revision_guidance": "重新判断关系是否存在。",
                }
            ],
        }
    )

    assert passed.status == "pass"
    assert needs_revision.issues[0].target_fields == [
        "record_type",
        "no_relation_reason",
    ]

    with pytest.raises(ValidationError):
        REVIEW_ADAPTER.validate_python(
            {"status": "needs_revision", "issues": []}
        )


def test_relation_review_issue_rejects_duplicates_and_unapproved_fields() -> None:
    payload = {
        "target_fields": ["record_type", "record_type"],
        "issue_description": "重复字段。",
        "paper_a_review_evidence_quotes": [],
        "paper_b_review_evidence_quotes": [],
        "revision_guidance": "去重。",
    }
    with pytest.raises(ValidationError):
        RelationReviewIssue.model_validate(payload)

    payload["target_fields"] = ["record_type"]
    payload["confidence"] = 0.9
    with pytest.raises(ValidationError):
        RelationReviewIssue.model_validate(payload)


def test_relation_review_fields_must_exist_on_current_record_branch() -> None:
    valid = REVIEW_ADAPTER.validate_python(
        {
            "status": "needs_revision",
            "issues": [
                {
                    "target_fields": ["record_type", "no_relation_reason"],
                    "issue_description": "无关系理由可能遗漏实质联系。",
                    "paper_a_review_evidence_quotes": [],
                    "paper_b_review_evidence_quotes": [],
                    "revision_guidance": "重新比较两篇论文。",
                }
            ],
        }
    )
    validate_relation_review_fields(valid, unrelated_record())

    invalid = REVIEW_ADAPTER.validate_python(
        {
            "status": "needs_revision",
            "issues": [
                {
                    "target_fields": ["relation_type"],
                    "issue_description": "当前分支不存在这个字段。",
                    "paper_a_review_evidence_quotes": [],
                    "paper_b_review_evidence_quotes": [],
                    "revision_guidance": "修正目标字段。",
                }
            ],
        }
    )
    with pytest.raises(RelationReviewFieldError):
        validate_relation_review_fields(invalid, unrelated_record())
