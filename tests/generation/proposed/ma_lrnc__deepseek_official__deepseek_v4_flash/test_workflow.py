import json
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.paper_parser import (
    NetworkRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    PaperParserRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.profile_review import (
    ProfileReviewIssue,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.result_paths import (
    build_topic_result_paths,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.workflow import (
    run_initial_profile_generation,
    run_targeted_profile_revision,
)


def valid_profile() -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[
            {
                "statement": "本文研究信息素养。",
                "evidence_quotes": ["本文研究信息素养。"],
            }
        ],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


class SuccessfulAgent:
    def __init__(self) -> None:
        self.calls = 0

    def initial_generation(self, **kwargs) -> PaperParserRunResult:
        self.calls += 1
        return PaperParserRunResult(
            profile=valid_profile(),
            metrics=StageMetrics(
                elapsed_seconds=2.0,
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                ),
            ),
        )


class FailingAgent:
    def initial_generation(self, **kwargs) -> PaperParserRunResult:
        raise NetworkRetriesExhausted(
            "network failed",
            metrics=StageMetrics(
                elapsed_seconds=6.0,
                token_usage=TokenUsage.zero(),
            ),
        )


class SuccessfulRevisionAgent:
    def __init__(self) -> None:
        self.calls = 0

    def targeted_revision(self, **kwargs) -> PaperParserRunResult:
        self.calls += 1
        return PaperParserRunResult(
            profile=valid_profile(),
            metrics=StageMetrics(
                elapsed_seconds=1.0,
                token_usage=TokenUsage(
                    prompt_tokens=120,
                    completion_tokens=30,
                    total_tokens=150,
                ),
            ),
        )


def test_workflow_reuses_only_valid_matching_checkpoint(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    temp_root = tmp_path / "resume"
    first_agent = SuccessfulAgent()

    first = run_initial_profile_generation(
        paper_id="P001",
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        model="deepseek-v4-flash",
        agent=first_agent,
        result_root=result_root,
        temp_root=temp_root,
    )
    assert first.reused is False
    assert first_agent.calls == 1

    second_agent = SuccessfulAgent()
    second = run_initial_profile_generation(
        paper_id="P001",
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        model="deepseek-v4-flash",
        agent=second_agent,
        result_root=result_root,
        temp_root=temp_root,
    )
    assert second.reused is True
    assert second_agent.calls == 0


def test_workflow_does_not_reuse_same_paper_id_from_another_topic(
    tmp_path: Path,
) -> None:
    information_literacy = build_topic_result_paths(
        topic="信息素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "resume" / "unit",
    )
    data_literacy = build_topic_result_paths(
        topic="数据素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "resume" / "unit",
    )
    information_agent = SuccessfulAgent()
    data_agent = SuccessfulAgent()

    run_initial_profile_generation(
        paper_id="P016",
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        model="deepseek-v4-flash",
        agent=information_agent,
        result_root=information_literacy.result_root,
        temp_root=information_literacy.temp_root,
    )
    data_result = run_initial_profile_generation(
        paper_id="P016",
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        model="deepseek-v4-flash",
        agent=data_agent,
        result_root=data_literacy.result_root,
        temp_root=data_literacy.temp_root,
    )

    assert information_agent.calls == 1
    assert data_agent.calls == 1
    assert data_result.reused is False
    assert (
        information_literacy.checkpoint_root
        / "paper_profile_initial_generation"
        / "P016.json"
    ).is_file()
    assert (
        data_literacy.checkpoint_root
        / "paper_profile_initial_generation"
        / "P016.json"
    ).is_file()


def test_workflow_writes_minimal_failure_record_not_success_checkpoint(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"

    with pytest.raises(NetworkRetriesExhausted):
        run_initial_profile_generation(
            paper_id="P001",
            title="测试论文",
            authors=["张三"],
            article_full_text="本文研究信息素养。",
            model="deepseek-v4-flash",
            agent=FailingAgent(),
            result_root=result_root,
            temp_root=tmp_path / "resume",
        )

    assert not (
        result_root
        / "checkpoints"
        / "paper_profile_initial_generation"
        / "P001.json"
    ).exists()
    failure_path = (
        result_root
        / "failures"
        / "paper_profile_initial_generation"
        / "P001.json"
    )
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "network_exhausted"
    assert set(payload) == {
        "failure_record_version",
        "stage",
        "business_key",
        "task_fingerprint",
        "failure_kind",
        "metrics",
    }


def test_targeted_revision_has_independent_checkpoint_and_reuse(
    tmp_path: Path,
) -> None:
    issue = ProfileReviewIssue(
        target_locations=[{"field": "research_questions", "item_index": None}],
        issue_description="缺少研究问题。",
        review_evidence_quotes=[],
        revision_guidance="根据全文补充研究问题。",
    )
    current_profile = PaperProfileContent(
        research_questions=[],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )
    result_root = tmp_path / "results"
    first_agent = SuccessfulRevisionAgent()

    first = run_targeted_profile_revision(
        paper_id="P001",
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=current_profile,
        review_issues=[issue],
        model="deepseek-v4-flash",
        agent=first_agent,
        result_root=result_root,
        temp_root=tmp_path / "resume",
    )
    assert first.reused is False
    assert first.checkpoint.stage == "paper_profile_targeted_revision"

    second_agent = SuccessfulRevisionAgent()
    second = run_targeted_profile_revision(
        paper_id="P001",
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=current_profile,
        review_issues=[issue],
        model="deepseek-v4-flash",
        agent=second_agent,
        result_root=result_root,
        temp_root=tmp_path / "resume",
    )
    assert second.reused is True
    assert second_agent.calls == 0
