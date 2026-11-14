from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.evidence_reviewer import (
    ProfileReviewNetworkRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.profile_review import (
    ProfileReviewPassed,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    ProfileReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.workflow import (
    run_profile_review,
)


def profile() -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


class SuccessfulReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review_profile(self, **kwargs) -> ProfileReviewerRunResult:
        self.calls += 1
        return ProfileReviewerRunResult(
            review=ProfileReviewPassed(status="pass"),
            metrics=StageMetrics(
                elapsed_seconds=2.0,
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            ),
        )


class FailingReviewer:
    def review_profile(self, **kwargs) -> ProfileReviewerRunResult:
        raise ProfileReviewNetworkRetriesExhausted(
            "network failed",
            metrics=StageMetrics(
                elapsed_seconds=6.0,
                token_usage=TokenUsage.zero(),
            ),
        )


def test_profile_review_workflow_writes_and_reuses_independent_checkpoint(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"
    first_agent = SuccessfulReviewer()

    first = run_profile_review(
        paper_id="P001",
        title="测试论文",
        authors=["张三"],
        article_full_text="正文。",
        current_profile=profile(),
        model="deepseek-v4-flash",
        agent=first_agent,
        result_root=result_root,
        temp_root=tmp_path / "resume",
    )
    assert first.reused is False
    assert first.checkpoint.stage == "paper_profile_review"

    second_agent = SuccessfulReviewer()
    second = run_profile_review(
        paper_id="P001",
        title="测试论文",
        authors=["张三"],
        article_full_text="正文。",
        current_profile=profile(),
        model="deepseek-v4-flash",
        agent=second_agent,
        result_root=result_root,
        temp_root=tmp_path / "resume",
    )

    assert second.reused is True
    assert second_agent.calls == 0


def test_profile_review_failure_is_not_a_success_checkpoint(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"

    with pytest.raises(ProfileReviewNetworkRetriesExhausted):
        run_profile_review(
            paper_id="P001",
            title="测试论文",
            authors=["张三"],
            article_full_text="正文。",
            current_profile=profile(),
            model="deepseek-v4-flash",
            agent=FailingReviewer(),
            result_root=result_root,
            temp_root=tmp_path / "resume",
        )

    assert not (
        result_root / "checkpoints" / "paper_profile_review" / "P001.json"
    ).exists()
    assert (
        result_root / "failures" / "paper_profile_review" / "P001.json"
    ).exists()
