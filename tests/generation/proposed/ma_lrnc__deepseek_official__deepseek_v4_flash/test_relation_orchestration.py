from pathlib import Path

import pytest
from pydantic import TypeAdapter

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.evidence_reviewer import (
    RelationReviewNetworkRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_outcome import (
    AcceptedRelationOutcome,
    ManualReviewRelationOutcome,
    RelationOutcome,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    RelatedPairRecord,
    RelationDirection,
    RelationType,
    UnrelatedPairRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_review import (
    RelationRecordField,
    RelationReviewIssue,
    RelationReviewNeedsRevision,
    RelationReviewPassed,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    RelationInferenceRunResult,
    RelationReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.orchestration import (
    run_relation_orchestration,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.result_paths import (
    build_topic_result_paths,
)


def metrics() -> StageMetrics:
    return StageMetrics(
        elapsed_seconds=1.0,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def paper_metadata(
    paper_id: str,
    publication_date: str,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}论文",
        authors=["张三"],
        publication_date=publication_date,
    )


def profile() -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[
            {
                "statement": "研究信息素养。",
                "evidence_quotes": ["研究信息素养。"],
            }
        ],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


def initial_record() -> UnrelatedPairRecord:
    return UnrelatedPairRecord(
        record_type="unrelated",
        paper_a_id="P001",
        paper_b_id="P002",
        no_relation_reason="尚无足够证据表明存在六类关系。",
    )


def revised_record() -> RelatedPairRecord:
    return RelatedPairRecord(
        record_type="related",
        paper_a_id="P001",
        paper_b_id="P002",
        relation_type=RelationType.EXTENDS,
        direction=RelationDirection.A_TO_B,
        relation_explanation="A 在 B 的基础上增加了新的研究变量。",
        paper_a_evidence_quotes=["A 增加了新的研究变量。"],
        paper_b_evidence_quotes=["B 提出了原始研究框架。"],
    )


def initial_issue() -> RelationReviewIssue:
    return RelationReviewIssue(
        target_fields=[RelationRecordField.NO_RELATION_REASON],
        issue_description="无关系判断遗漏了可能的拓展关系。",
        paper_a_review_evidence_quotes=["A 增加了新的研究变量。"],
        paper_b_review_evidence_quotes=["B 提出了原始研究框架。"],
        revision_guidance="重新判断是否构成拓展。",
    )


def outstanding_issue() -> RelationReviewIssue:
    return RelationReviewIssue(
        target_fields=[RelationRecordField.RELATION_EXPLANATION],
        issue_description="拓展机制解释仍不充分。",
        paper_a_review_evidence_quotes=["A 增加了新的研究变量。"],
        paper_b_review_evidence_quotes=["B 提出了原始研究框架。"],
        revision_guidance="人工核对两项变量是否属于实质新增。",
    )


def result_paths(tmp_path: Path):
    return build_topic_result_paths(
        topic="信息素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "resume" / "unit",
    )


class FakeRelationAgent:
    def __init__(self) -> None:
        self.initial_calls = 0
        self.revision_calls = 0

    def initial_inference(self, **kwargs: object) -> RelationInferenceRunResult:
        self.initial_calls += 1
        return RelationInferenceRunResult(
            record=initial_record(),
            metrics=metrics(),
        )

    def targeted_revision(
        self,
        **kwargs: object,
    ) -> RelationInferenceRunResult:
        self.revision_calls += 1
        return RelationInferenceRunResult(
            record=revised_record(),
            metrics=metrics(),
        )


class FakeRelationReviewer:
    def __init__(
        self,
        *,
        initial_needs_revision: bool,
        post_needs_revision: bool = False,
    ) -> None:
        self.initial_needs_revision = initial_needs_revision
        self.post_needs_revision = post_needs_revision
        self.initial_calls = 0
        self.post_calls = 0

    def review_relation(self, **kwargs: object) -> RelationReviewerRunResult:
        self.initial_calls += 1
        review = (
            RelationReviewNeedsRevision(
                status="needs_revision",
                issues=[initial_issue()],
            )
            if self.initial_needs_revision
            else RelationReviewPassed(status="pass")
        )
        return RelationReviewerRunResult(review=review, metrics=metrics())

    def review_relation_after_revision(
        self,
        **kwargs: object,
    ) -> RelationReviewerRunResult:
        self.post_calls += 1
        review = (
            RelationReviewNeedsRevision(
                status="needs_revision",
                issues=[outstanding_issue()],
            )
            if self.post_needs_revision
            else RelationReviewPassed(status="pass")
        )
        return RelationReviewerRunResult(review=review, metrics=metrics())


class FailingRelationReviewer(FakeRelationReviewer):
    def review_relation(self, **kwargs: object) -> RelationReviewerRunResult:
        raise RelationReviewNetworkRetriesExhausted(
            "network failed",
            metrics=metrics(),
        )


def pair_inputs() -> dict[str, object]:
    return {
        "paper_a_metadata": paper_metadata("P001", "2020-01-02"),
        "paper_a_profile": profile(),
        "paper_a_full_text": "A 全文。",
        "paper_b_metadata": paper_metadata("P002", "2019-01-02"),
        "paper_b_profile": profile(),
        "paper_b_full_text": "B 全文。",
        "model": "deepseek-v4-flash",
    }


def test_initial_review_passes_and_accepts_initial_relation(
    tmp_path: Path,
) -> None:
    agent = FakeRelationAgent()
    reviewer = FakeRelationReviewer(initial_needs_revision=False)

    result = run_relation_orchestration(
        relation_agent=agent,
        reviewer_agent=reviewer,
        paths=result_paths(tmp_path),
        **pair_inputs(),
    )

    assert isinstance(result.outcome, AcceptedRelationOutcome)
    assert result.outcome.relation_record == initial_record()
    assert result.outcome_reused is False
    assert result.metrics == StageMetrics(
        elapsed_seconds=2.0,
        token_usage=TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
    )
    assert agent.initial_calls == 1
    assert agent.revision_calls == 0
    assert reviewer.initial_calls == 1
    assert reviewer.post_calls == 0
    assert not (
        result_paths(tmp_path).checkpoint_root
        / "relation_targeted_revision"
        / "P001__P002.json"
    ).exists()
    assert TypeAdapter(RelationOutcome).validate_json(
        result.outcome_path.read_text(encoding="utf-8")
    ) == result.outcome


def test_post_revision_passes_and_accepts_revised_relation(
    tmp_path: Path,
) -> None:
    agent = FakeRelationAgent()
    reviewer = FakeRelationReviewer(initial_needs_revision=True)

    result = run_relation_orchestration(
        relation_agent=agent,
        reviewer_agent=reviewer,
        paths=result_paths(tmp_path),
        **pair_inputs(),
    )

    assert isinstance(result.outcome, AcceptedRelationOutcome)
    assert result.outcome.relation_record == revised_record()
    assert result.metrics == StageMetrics(
        elapsed_seconds=4.0,
        token_usage=TokenUsage(
            prompt_tokens=40,
            completion_tokens=20,
            total_tokens=60,
        ),
    )
    assert agent.revision_calls == 1
    assert reviewer.post_calls == 1


def test_post_revision_issues_produce_manual_review_outcome_with_only_current_issues(
    tmp_path: Path,
) -> None:
    reviewer = FakeRelationReviewer(
        initial_needs_revision=True,
        post_needs_revision=True,
    )
    result = run_relation_orchestration(
        relation_agent=FakeRelationAgent(),
        reviewer_agent=reviewer,
        paths=result_paths(tmp_path),
        **pair_inputs(),
    )

    assert isinstance(result.outcome, ManualReviewRelationOutcome)
    assert result.outcome.relation_record == revised_record()
    assert result.outcome.outstanding_issues == [outstanding_issue()]
    assert initial_issue() not in result.outcome.outstanding_issues
    assert reviewer.post_calls == 1


def test_rerun_reuses_all_stage_checkpoints_and_matching_outcome(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    run_relation_orchestration(
        relation_agent=FakeRelationAgent(),
        reviewer_agent=FakeRelationReviewer(initial_needs_revision=True),
        paths=paths,
        **pair_inputs(),
    )
    rerun_agent = FakeRelationAgent()
    rerun_reviewer = FakeRelationReviewer(initial_needs_revision=True)

    result = run_relation_orchestration(
        relation_agent=rerun_agent,
        reviewer_agent=rerun_reviewer,
        paths=paths,
        **pair_inputs(),
    )

    assert result.outcome_reused is True
    assert rerun_agent.initial_calls == 0
    assert rerun_agent.revision_calls == 0
    assert rerun_reviewer.initial_calls == 0
    assert rerun_reviewer.post_calls == 0


def test_mismatching_outcome_is_replaced_after_stage_validation(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    run_relation_orchestration(
        relation_agent=FakeRelationAgent(),
        reviewer_agent=FakeRelationReviewer(initial_needs_revision=False),
        paths=paths,
        **pair_inputs(),
    )
    outcome_path = paths.relation_outcome_root / "P001__P002.json"
    stale = AcceptedRelationOutcome(
        review_status="accepted",
        relation_record=UnrelatedPairRecord(
            record_type="unrelated",
            paper_a_id="P001",
            paper_b_id="P002",
            no_relation_reason="过期的关系判断。",
        ),
    )
    outcome_path.write_bytes(
        TypeAdapter(RelationOutcome).dump_json(stale, indent=2)
    )
    rerun_agent = FakeRelationAgent()
    rerun_reviewer = FakeRelationReviewer(initial_needs_revision=False)

    result = run_relation_orchestration(
        relation_agent=rerun_agent,
        reviewer_agent=rerun_reviewer,
        paths=paths,
        **pair_inputs(),
    )

    assert result.outcome_reused is False
    assert result.outcome.relation_record == initial_record()
    assert rerun_agent.initial_calls == 0
    assert rerun_reviewer.initial_calls == 0


def test_corrupt_outcome_is_replaced_after_stage_validation(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    run_relation_orchestration(
        relation_agent=FakeRelationAgent(),
        reviewer_agent=FakeRelationReviewer(initial_needs_revision=False),
        paths=paths,
        **pair_inputs(),
    )
    outcome_path = paths.relation_outcome_root / "P001__P002.json"
    outcome_path.write_text("{broken", encoding="utf-8")
    rerun_agent = FakeRelationAgent()
    rerun_reviewer = FakeRelationReviewer(initial_needs_revision=False)

    result = run_relation_orchestration(
        relation_agent=rerun_agent,
        reviewer_agent=rerun_reviewer,
        paths=paths,
        **pair_inputs(),
    )

    assert result.outcome_reused is False
    assert TypeAdapter(RelationOutcome).validate_json(
        outcome_path.read_text(encoding="utf-8")
    ) == result.outcome
    assert rerun_agent.initial_calls == 0
    assert rerun_reviewer.initial_calls == 0


def test_failed_required_stage_does_not_create_relation_outcome(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)

    with pytest.raises(RelationReviewNetworkRetriesExhausted):
        run_relation_orchestration(
            relation_agent=FakeRelationAgent(),
            reviewer_agent=FailingRelationReviewer(
                initial_needs_revision=False
            ),
            paths=paths,
            **pair_inputs(),
        )

    assert not (
        paths.relation_outcome_root / "P001__P002.json"
    ).exists()
    assert (
        paths.failure_root
        / "relation_initial_review"
        / "P001__P002.json"
    ).is_file()
