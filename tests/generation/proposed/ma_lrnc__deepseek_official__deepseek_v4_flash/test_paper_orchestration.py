import json
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.evidence_reviewer import (
    ProfileReviewNetworkRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.profile_review import (
    ProfileReviewIssue,
    ProfileReviewNeedsRevision,
    ProfileReviewPassed,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    PaperParserRunResult,
    ProfileReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.orchestration import (
    run_paper_profile_orchestration,
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


def profile(statement: str) -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[
            {
                "statement": statement,
                "evidence_quotes": [statement],
            }
        ],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


def metadata() -> PaperMetadata:
    return PaperMetadata(
        paper_id="P016",
        title="测试论文",
        authors=["张三"],
        publication_date="2020-01-02",
    )


def result_paths(tmp_path: Path):
    return build_topic_result_paths(
        topic="信息素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "resume" / "unit",
    )


class FakeParserAgent:
    def __init__(
        self,
        *,
        initial_profile: PaperProfileContent,
        revised_profile: PaperProfileContent,
    ) -> None:
        self.initial_profile = initial_profile
        self.revised_profile = revised_profile
        self.initial_calls = 0
        self.revision_calls = 0

    def initial_generation(self, **kwargs: object) -> PaperParserRunResult:
        self.initial_calls += 1
        return PaperParserRunResult(
            profile=self.initial_profile,
            metrics=metrics(),
        )

    def targeted_revision(self, **kwargs: object) -> PaperParserRunResult:
        self.revision_calls += 1
        return PaperParserRunResult(
            profile=self.revised_profile,
            metrics=metrics(),
        )


class PassingReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review_profile(self, **kwargs: object) -> ProfileReviewerRunResult:
        self.calls += 1
        return ProfileReviewerRunResult(
            review=ProfileReviewPassed(status="pass"),
            metrics=metrics(),
        )


class RevisionReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review_profile(self, **kwargs: object) -> ProfileReviewerRunResult:
        self.calls += 1
        return ProfileReviewerRunResult(
            review=ProfileReviewNeedsRevision(
                status="needs_revision",
                issues=[
                    ProfileReviewIssue(
                        target_locations=[
                            {
                                "field": "research_questions",
                                "item_index": 0,
                            }
                        ],
                        issue_description="研究问题表述需要修订。",
                        review_evidence_quotes=["修订证据。"],
                        revision_guidance="按证据修订研究问题。",
                    )
                ],
            ),
            metrics=metrics(),
        )


class FailingReviewer:
    def review_profile(self, **kwargs: object) -> ProfileReviewerRunResult:
        raise ProfileReviewNetworkRetriesExhausted(
            "network failed",
            metrics=metrics(),
        )


def test_pass_branch_materializes_initial_profile_without_revision(
    tmp_path: Path,
) -> None:
    initial = profile("初始画像。")
    parser = FakeParserAgent(
        initial_profile=initial,
        revised_profile=profile("不应使用的修订画像。"),
    )
    reviewer = PassingReviewer()
    paths = result_paths(tmp_path)

    outcome = run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=parser,
        reviewer_agent=reviewer,
        paths=paths,
    )

    assert outcome.profile == initial
    assert outcome.source_stage == "paper_profile_initial_generation"
    assert outcome.final_profile_reused is False
    assert outcome.metrics == StageMetrics(
        elapsed_seconds=2.0,
        token_usage=TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
    )
    assert parser.initial_calls == 1
    assert parser.revision_calls == 0
    assert reviewer.calls == 1
    persisted = PaperProfileContent.model_validate_json(
        outcome.final_profile_path.read_text(encoding="utf-8")
    )
    assert persisted == initial
    assert not (
        paths.checkpoint_root
        / "paper_profile_targeted_revision"
        / "P016.json"
    ).exists()


def test_revision_branch_runs_exactly_one_revision_and_materializes_it(
    tmp_path: Path,
) -> None:
    revised = profile("修订后画像。")
    parser = FakeParserAgent(
        initial_profile=profile("初始画像。"),
        revised_profile=revised,
    )
    reviewer = RevisionReviewer()

    outcome = run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=parser,
        reviewer_agent=reviewer,
        paths=result_paths(tmp_path),
    )

    assert outcome.profile == revised
    assert outcome.source_stage == "paper_profile_targeted_revision"
    assert outcome.metrics == StageMetrics(
        elapsed_seconds=3.0,
        token_usage=TokenUsage(
            prompt_tokens=30,
            completion_tokens=15,
            total_tokens=45,
        ),
    )
    assert parser.initial_calls == 1
    assert parser.revision_calls == 1
    assert reviewer.calls == 1


def test_rerun_reuses_all_stage_checkpoints_and_matching_final_profile(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    initial_parser = FakeParserAgent(
        initial_profile=profile("初始画像。"),
        revised_profile=profile("修订后画像。"),
    )
    run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=initial_parser,
        reviewer_agent=RevisionReviewer(),
        paths=paths,
    )
    rerun_parser = FakeParserAgent(
        initial_profile=profile("不应调用。"),
        revised_profile=profile("不应调用。"),
    )
    rerun_reviewer = RevisionReviewer()

    outcome = run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=rerun_parser,
        reviewer_agent=rerun_reviewer,
        paths=paths,
    )

    assert outcome.final_profile_reused is True
    assert outcome.profile == profile("修订后画像。")
    assert rerun_parser.initial_calls == 0
    assert rerun_parser.revision_calls == 0
    assert rerun_reviewer.calls == 0


def test_schema_valid_but_mismatching_final_profile_is_replaced(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    parser = FakeParserAgent(
        initial_profile=profile("权威画像。"),
        revised_profile=profile("不应使用。"),
    )
    run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=parser,
        reviewer_agent=PassingReviewer(),
        paths=paths,
    )
    final_path = paths.final_profile_root / "P016.json"
    final_path.write_text(
        profile("过期画像。").model_dump_json(indent=2),
        encoding="utf-8",
    )
    rerun_parser = FakeParserAgent(
        initial_profile=profile("不应调用。"),
        revised_profile=profile("不应调用。"),
    )

    outcome = run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=rerun_parser,
        reviewer_agent=PassingReviewer(),
        paths=paths,
    )

    assert outcome.final_profile_reused is False
    assert PaperProfileContent.model_validate_json(
        final_path.read_text(encoding="utf-8")
    ) == profile("权威画像。")
    assert rerun_parser.initial_calls == 0


def test_corrupt_final_profile_is_replaced_after_upstream_validation(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=FakeParserAgent(
            initial_profile=profile("权威画像。"),
            revised_profile=profile("不应使用。"),
        ),
        reviewer_agent=PassingReviewer(),
        paths=paths,
    )
    final_path = paths.final_profile_root / "P016.json"
    final_path.write_text("{broken", encoding="utf-8")
    rerun_parser = FakeParserAgent(
        initial_profile=profile("不应调用。"),
        revised_profile=profile("不应调用。"),
    )

    outcome = run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=rerun_parser,
        reviewer_agent=PassingReviewer(),
        paths=paths,
    )

    assert outcome.final_profile_reused is False
    assert PaperProfileContent.model_validate_json(
        final_path.read_text(encoding="utf-8")
    ) == profile("权威画像。")
    assert rerun_parser.initial_calls == 0


def test_failed_required_stage_does_not_create_final_profile(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)

    with pytest.raises(ProfileReviewNetworkRetriesExhausted):
        run_paper_profile_orchestration(
            paper_metadata=metadata(),
            article_full_text="论文全文。",
            model="deepseek-v4-flash",
            parser_agent=FakeParserAgent(
                initial_profile=profile("初始画像。"),
                revised_profile=profile("修订画像。"),
            ),
            reviewer_agent=FailingReviewer(),
            paths=paths,
        )

    assert not (paths.final_profile_root / "P016.json").exists()
    assert (
        paths.failure_root
        / "paper_profile_review"
        / "P016.json"
    ).is_file()


def test_final_profile_is_plain_paper_profile_without_wrapper_fields(
    tmp_path: Path,
) -> None:
    paths = result_paths(tmp_path)
    run_paper_profile_orchestration(
        paper_metadata=metadata(),
        article_full_text="论文全文。",
        model="deepseek-v4-flash",
        parser_agent=FakeParserAgent(
            initial_profile=profile("初始画像。"),
            revised_profile=profile("修订画像。"),
        ),
        reviewer_agent=PassingReviewer(),
        paths=paths,
    )

    payload = json.loads(
        (paths.final_profile_root / "P016.json").read_text(encoding="utf-8")
    )
    assert set(payload) == {
        "research_questions",
        "theories_and_viewpoints",
        "methods_and_models",
        "data_and_contexts",
        "main_findings",
        "limitations_and_boundary_conditions",
    }
