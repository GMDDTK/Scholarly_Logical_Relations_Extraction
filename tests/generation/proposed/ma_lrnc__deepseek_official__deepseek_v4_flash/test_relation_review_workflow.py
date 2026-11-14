from datetime import date
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.evidence_reviewer import (
    RelationReviewNetworkRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    UnrelatedPairRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_review import (
    RelationReviewPassed,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    RelationReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.workflow import (
    run_initial_relation_review,
)


def metadata(paper_id: str) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}论文",
        authors=["张三"],
        publication_date=date(2021, 1, 1),
    )


def empty_profile() -> PaperProfileContent:
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

    def review_relation(self, **kwargs) -> RelationReviewerRunResult:
        self.calls += 1
        return RelationReviewerRunResult(
            review=RelationReviewPassed(status="pass"),
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
    def review_relation(self, **kwargs) -> RelationReviewerRunResult:
        raise RelationReviewNetworkRetriesExhausted(
            "network failed",
            metrics=StageMetrics(
                elapsed_seconds=6.0,
                token_usage=TokenUsage.zero(),
            ),
        )


def workflow_kwargs(tmp_path: Path) -> dict:
    return {
        "paper_a_metadata": metadata("P001"),
        "paper_a_profile": empty_profile(),
        "paper_a_full_text": "A全文。",
        "paper_b_metadata": metadata("P002"),
        "paper_b_profile": empty_profile(),
        "paper_b_full_text": "B全文。",
        "current_relation": UnrelatedPairRecord(
            record_type="unrelated",
            paper_a_id="P001",
            paper_b_id="P002",
            no_relation_reason="两篇论文只有主题相似。",
        ),
        "model": "deepseek-v4-flash",
        "result_root": tmp_path / "results",
        "temp_root": tmp_path / "resume",
    }


def test_relation_review_workflow_writes_and_reuses_pair_checkpoint(
    tmp_path: Path,
) -> None:
    kwargs = workflow_kwargs(tmp_path)
    first_agent = SuccessfulReviewer()
    first = run_initial_relation_review(agent=first_agent, **kwargs)

    assert first.reused is False
    assert first.checkpoint.stage == "relation_initial_review"
    assert first.checkpoint.business_key == "P001__P002"

    second_agent = SuccessfulReviewer()
    second = run_initial_relation_review(agent=second_agent, **kwargs)

    assert second.reused is True
    assert second_agent.calls == 0


def test_relation_review_failure_is_not_a_success_checkpoint(
    tmp_path: Path,
) -> None:
    kwargs = workflow_kwargs(tmp_path)

    with pytest.raises(RelationReviewNetworkRetriesExhausted):
        run_initial_relation_review(agent=FailingReviewer(), **kwargs)

    result_root = kwargs["result_root"]
    assert not (
        result_root
        / "checkpoints"
        / "relation_initial_review"
        / "P001__P002.json"
    ).exists()
    assert (
        result_root
        / "failures"
        / "relation_initial_review"
        / "P001__P002.json"
    ).exists()
