from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from .checkpointing import (
    load_matching_final_paper_profile,
    load_matching_relation_outcome,
    write_final_paper_profile,
    write_relation_outcome,
)
from .contracts.paper_metadata import PaperMetadata
from .contracts.paper_profile import PaperProfileContent
from .contracts.profile_review import ProfileReviewIssue
from .contracts.relation_outcome import (
    AcceptedRelationOutcome,
    ManualReviewRelationOutcome,
    RelationOutcome,
)
from .contracts.relation_record import RelationRecord
from .contracts.relation_review import RelationReviewIssue
from .contracts.runtime import (
    PaperParserRunResult,
    RelationInferenceRunResult,
    RelationReviewerRunResult,
    StageMetrics,
)
from .result_paths import TopicResultPaths
from .workflow import (
    INITIAL_STAGE,
    REVISION_STAGE,
    PostRevisionRelationReviewer,
    ProfileReviewer,
    RelationReviewer,
    run_initial_profile_generation,
    run_initial_relation_inference,
    run_initial_relation_review,
    run_post_revision_relation_review,
    run_profile_review,
    run_targeted_relation_revision,
    run_targeted_profile_revision,
)


FinalProfileSourceStage = Literal[
    "paper_profile_initial_generation",
    "paper_profile_targeted_revision",
]


class PaperProfileAgent(Protocol):
    def initial_generation(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
    ) -> PaperParserRunResult: ...

    def targeted_revision(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
        current_profile: PaperProfileContent,
        review_issues: Sequence[ProfileReviewIssue],
    ) -> PaperParserRunResult: ...


class RelationAgent(Protocol):
    def initial_inference(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
    ) -> RelationInferenceRunResult: ...

    def targeted_revision(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
        current_relation: RelationRecord,
        review_issues: Sequence[RelationReviewIssue],
    ) -> RelationInferenceRunResult: ...


class RelationReviewerAgent(
    RelationReviewer,
    PostRevisionRelationReviewer,
    Protocol,
):
    pass


@dataclass(frozen=True)
class PaperProfileOutcome:
    paper_id: str
    profile: PaperProfileContent
    source_stage: FinalProfileSourceStage
    final_profile_path: Path
    final_profile_reused: bool
    metrics: StageMetrics


@dataclass(frozen=True)
class RelationPairOutcome:
    pair_key: str
    outcome: RelationOutcome
    outcome_path: Path
    outcome_reused: bool
    metrics: StageMetrics


def run_paper_profile_orchestration(
    *,
    paper_metadata: PaperMetadata,
    article_full_text: str,
    model: str,
    parser_agent: PaperProfileAgent,
    reviewer_agent: ProfileReviewer,
    paths: TopicResultPaths,
) -> PaperProfileOutcome:
    initial = run_initial_profile_generation(
        paper_id=paper_metadata.paper_id,
        title=paper_metadata.title,
        authors=paper_metadata.authors,
        article_full_text=article_full_text,
        model=model,
        agent=parser_agent,
        result_root=paths.result_root,
        temp_root=paths.temp_root,
    )
    initial_profile = initial.checkpoint.result

    review = run_profile_review(
        paper_id=paper_metadata.paper_id,
        title=paper_metadata.title,
        authors=paper_metadata.authors,
        article_full_text=article_full_text,
        current_profile=initial_profile,
        model=model,
        agent=reviewer_agent,
        result_root=paths.result_root,
        temp_root=paths.temp_root,
    )
    review_result = review.checkpoint.result
    metrics = initial.checkpoint.metrics.plus(review.checkpoint.metrics)

    if review_result.status == "pass":
        final_profile = initial_profile
        source_stage: FinalProfileSourceStage = INITIAL_STAGE
    else:
        revision = run_targeted_profile_revision(
            paper_id=paper_metadata.paper_id,
            title=paper_metadata.title,
            authors=paper_metadata.authors,
            article_full_text=article_full_text,
            current_profile=initial_profile,
            review_issues=review_result.issues,
            model=model,
            agent=parser_agent,
            result_root=paths.result_root,
            temp_root=paths.temp_root,
        )
        final_profile = revision.checkpoint.result
        source_stage = REVISION_STAGE
        metrics = metrics.plus(revision.checkpoint.metrics)

    final_profile_path = (
        paths.final_profile_root / f"{paper_metadata.paper_id}.json"
    )
    persisted = load_matching_final_paper_profile(
        path=final_profile_path,
        expected_profile=final_profile,
    )
    if persisted is not None:
        return PaperProfileOutcome(
            paper_id=paper_metadata.paper_id,
            profile=persisted,
            source_stage=source_stage,
            final_profile_path=final_profile_path,
            final_profile_reused=True,
            metrics=metrics,
        )

    write_final_paper_profile(
        profile=final_profile,
        destination=final_profile_path,
        temp_root=paths.temp_root,
        paper_id=paper_metadata.paper_id,
    )
    return PaperProfileOutcome(
        paper_id=paper_metadata.paper_id,
        profile=final_profile,
        source_stage=source_stage,
        final_profile_path=final_profile_path,
        final_profile_reused=False,
        metrics=metrics,
    )


def run_relation_orchestration(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    model: str,
    relation_agent: RelationAgent,
    reviewer_agent: RelationReviewerAgent,
    paths: TopicResultPaths,
) -> RelationPairOutcome:
    pair_inputs = {
        "paper_a_metadata": paper_a_metadata,
        "paper_a_profile": paper_a_profile,
        "paper_a_full_text": paper_a_full_text,
        "paper_b_metadata": paper_b_metadata,
        "paper_b_profile": paper_b_profile,
        "paper_b_full_text": paper_b_full_text,
        "model": model,
        "result_root": paths.result_root,
        "temp_root": paths.temp_root,
    }
    initial = run_initial_relation_inference(
        agent=relation_agent,
        **pair_inputs,
    )
    initial_record = initial.checkpoint.result

    initial_review = run_initial_relation_review(
        current_relation=initial_record,
        agent=reviewer_agent,
        **pair_inputs,
    )
    initial_review_result = initial_review.checkpoint.result
    metrics = initial.checkpoint.metrics.plus(
        initial_review.checkpoint.metrics
    )

    if initial_review_result.status == "pass":
        outcome: RelationOutcome = AcceptedRelationOutcome(
            review_status="accepted",
            relation_record=initial_record,
        )
    else:
        revision = run_targeted_relation_revision(
            current_relation=initial_record,
            review_issues=initial_review_result.issues,
            agent=relation_agent,
            **pair_inputs,
        )
        revised_record = revision.checkpoint.result
        post_review = run_post_revision_relation_review(
            previous_review_issues=initial_review_result.issues,
            revised_relation=revised_record,
            agent=reviewer_agent,
            **pair_inputs,
        )
        post_review_result = post_review.checkpoint.result
        metrics = (
            metrics.plus(revision.checkpoint.metrics)
            .plus(post_review.checkpoint.metrics)
        )
        if post_review_result.status == "pass":
            outcome = AcceptedRelationOutcome(
                review_status="accepted",
                relation_record=revised_record,
            )
        else:
            outcome = ManualReviewRelationOutcome(
                review_status="manual_review_required",
                relation_record=revised_record,
                outstanding_issues=post_review_result.issues,
            )

    pair_key = f"{paper_a_metadata.paper_id}__{paper_b_metadata.paper_id}"
    outcome_path = paths.relation_outcome_root / f"{pair_key}.json"
    persisted = load_matching_relation_outcome(
        path=outcome_path,
        expected_outcome=outcome,
    )
    if persisted is not None:
        return RelationPairOutcome(
            pair_key=pair_key,
            outcome=persisted,
            outcome_path=outcome_path,
            outcome_reused=True,
            metrics=metrics,
        )

    write_relation_outcome(
        outcome=outcome,
        destination=outcome_path,
        temp_root=paths.temp_root,
        pair_key=pair_key,
    )
    return RelationPairOutcome(
        pair_key=pair_key,
        outcome=outcome,
        outcome_path=outcome_path,
        outcome_reused=False,
        metrics=metrics,
    )
