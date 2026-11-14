import pytest
from pydantic import ValidationError

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.profile_review import (
    ProfileReviewIssue,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    PaperProfileCheckpoint,
    StageMetrics,
    TokenUsage,
)


def valid_profile_data() -> dict:
    return {
        "research_questions": [
            {
                "statement": "本文研究学生信息素养的影响因素。",
                "evidence_quotes": ["本文旨在研究学生信息素养的影响因素。"],
            }
        ],
        "theories_and_viewpoints": [
            {
                "statement": "本文采用计划行为理论作为分析基础。",
                "evidence_quotes": ["本文采用计划行为理论作为分析基础。"],
                "paper_actions": ["adopts", "applies"],
            }
        ],
        "methods_and_models": [],
        "data_and_contexts": [],
        "main_findings": [],
        "limitations_and_boundary_conditions": [],
    }


def test_paper_profile_accepts_complete_strict_contract() -> None:
    profile = PaperProfileContent.model_validate(valid_profile_data())

    assert profile.theories_and_viewpoints[0].paper_actions == ["adopts", "applies"]


def test_paper_profile_rejects_missing_top_level_field() -> None:
    data = valid_profile_data()
    del data["main_findings"]

    with pytest.raises(ValidationError):
        PaperProfileContent.model_validate(data)


def test_paper_profile_rejects_extra_field_and_duplicate_evidence() -> None:
    data = valid_profile_data()
    data["unexpected"] = []

    with pytest.raises(ValidationError):
        PaperProfileContent.model_validate(data)

    data = valid_profile_data()
    data["research_questions"][0]["evidence_quotes"] = ["同一句。", "同一句。"]

    with pytest.raises(ValidationError):
        PaperProfileContent.model_validate(data)


def test_attributed_item_requires_nonempty_unique_actions() -> None:
    data = valid_profile_data()
    data["theories_and_viewpoints"][0]["paper_actions"] = []

    with pytest.raises(ValidationError):
        PaperProfileContent.model_validate(data)

    data["theories_and_viewpoints"][0]["paper_actions"] = ["applies", "applies"]

    with pytest.raises(ValidationError):
        PaperProfileContent.model_validate(data)


def test_profile_review_issue_allows_open_description_but_validates_locations() -> None:
    issue = ProfileReviewIssue.model_validate(
        {
            "target_locations": [
                {"field": "methods_and_models", "item_index": 0},
                {"field": "main_findings", "item_index": None},
            ],
            "issue_description": "条目归属和缺失问题需要同时修正。",
            "review_evidence_quotes": [],
            "revision_guidance": "核对全文后修正归属并补充遗漏发现。",
        }
    )

    assert len(issue.target_locations) == 2

    with pytest.raises(ValidationError):
        ProfileReviewIssue.model_validate(
            {
                "target_locations": [
                    {"field": "methods_and_models", "item_index": 0},
                    {"field": "methods_and_models", "item_index": 0},
                ],
                "issue_description": "重复位置。",
                "review_evidence_quotes": [],
                "revision_guidance": "去重。",
            }
        )


def test_checkpoint_contract_is_strict_and_metrics_are_consistent() -> None:
    profile = PaperProfileContent.model_validate(valid_profile_data())
    checkpoint = PaperProfileCheckpoint(
        checkpoint_version=1,
        stage="paper_profile_initial_generation",
        business_key="P001",
        task_fingerprint="sha256:" + "a" * 64,
        result=profile,
        metrics=StageMetrics(
            elapsed_seconds=1.25,
            token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        ),
    )

    assert checkpoint.metrics.token_usage.total_tokens == 150

    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=149)

