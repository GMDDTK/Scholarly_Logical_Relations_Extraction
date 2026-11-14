import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import TypeAdapter

from .agents.evidence_reviewer import (
    ProfileReviewContentRetriesExhausted,
    ProfileReviewNetworkRetriesExhausted,
    RelationReviewContentRetriesExhausted,
    RelationReviewNetworkRetriesExhausted,
)
from .agents.paper_parser import (
    ContentRetriesExhausted,
    NetworkRetriesExhausted,
)
from .agents.relation_inference import (
    RelationInferenceContentRetriesExhausted,
    RelationInferenceNetworkRetriesExhausted,
)
from .checkpointing import (
    compute_task_fingerprint,
    load_valid_paper_profile_checkpoint,
    load_valid_profile_review_checkpoint,
    load_valid_relation_inference_checkpoint,
    load_valid_relation_review_checkpoint,
    write_failure_record,
    write_paper_profile_checkpoint,
    write_profile_review_checkpoint,
    write_relation_inference_checkpoint,
    write_relation_review_checkpoint,
)
from .config import EXPERIMENT_UNIT
from .contracts.paper_metadata import PaperMetadata
from .contracts.paper_profile import PaperProfileContent
from .contracts.profile_review import ProfileReviewIssue, ProfileReviewResult
from .contracts.relation_record import RelationRecord
from .contracts.relation_review import RelationReviewIssue, RelationReviewResult
from .contracts.runtime import (
    PaperParserFailureRecord,
    PaperParserRunResult,
    PaperProfileCheckpoint,
    ProfileReviewCheckpoint,
    ProfileReviewerFailureRecord,
    ProfileReviewerRunResult,
    RelationInferenceCheckpoint,
    RelationInferenceFailureRecord,
    RelationInferenceRunResult,
    RelationReviewCheckpoint,
    RelationReviewerFailureRecord,
    RelationReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from .prompting import (
    build_profile_review_system_prompt,
    build_relation_post_revision_review_system_prompt,
    build_relation_post_revision_review_user_message,
    build_relation_inference_system_prompt,
    build_relation_inference_user_message,
    build_relation_revision_system_prompt,
    build_relation_revision_user_message,
    build_relation_review_system_prompt,
    build_relation_review_user_message,
    build_system_prompt,
)
from .relation_validation import (
    validate_relation_handoff,
    validate_relation_pair_inputs,
)


INITIAL_STAGE = "paper_profile_initial_generation"
REVISION_STAGE = "paper_profile_targeted_revision"
PROFILE_REVIEW_STAGE = "paper_profile_review"
RELATION_INITIAL_STAGE = "relation_initial_inference"
RELATION_INITIAL_REVIEW_STAGE = "relation_initial_review"
RELATION_REVISION_STAGE = "relation_targeted_revision"
RELATION_POST_REVISION_REVIEW_STAGE = "relation_post_revision_review"


class InitialGenerationAgent(Protocol):
    def initial_generation(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
    ) -> PaperParserRunResult: ...


class TargetedRevisionAgent(Protocol):
    def targeted_revision(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
        current_profile: PaperProfileContent,
        review_issues: Sequence[ProfileReviewIssue],
    ) -> PaperParserRunResult: ...


class ProfileReviewer(Protocol):
    def review_profile(
        self,
        *,
        title: str,
        authors: Sequence[str],
        article_full_text: str,
        current_profile: PaperProfileContent,
    ) -> ProfileReviewerRunResult: ...


class InitialRelationInferenceAgent(Protocol):
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


class TargetedRelationRevisionAgent(Protocol):
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


class RelationReviewer(Protocol):
    def review_relation(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
        current_relation: RelationRecord,
    ) -> RelationReviewerRunResult: ...


class PostRevisionRelationReviewer(Protocol):
    def review_relation_after_revision(
        self,
        *,
        paper_a_metadata: PaperMetadata,
        paper_a_profile: PaperProfileContent,
        paper_a_full_text: str,
        paper_b_metadata: PaperMetadata,
        paper_b_profile: PaperProfileContent,
        paper_b_full_text: str,
        previous_review_issues: Sequence[RelationReviewIssue],
        revised_relation: RelationRecord,
    ) -> RelationReviewerRunResult: ...


@dataclass(frozen=True)
class InitialProfileWorkflowResult:
    checkpoint: PaperProfileCheckpoint
    reused: bool


@dataclass(frozen=True)
class ProfileReviewWorkflowResult:
    checkpoint: ProfileReviewCheckpoint
    reused: bool


@dataclass(frozen=True)
class RelationInferenceWorkflowResult:
    checkpoint: RelationInferenceCheckpoint
    reused: bool


@dataclass(frozen=True)
class RelationReviewWorkflowResult:
    checkpoint: RelationReviewCheckpoint
    reused: bool


def run_initial_profile_generation(
    *,
    paper_id: str,
    title: str,
    authors: Sequence[str],
    article_full_text: str,
    model: str,
    agent: InitialGenerationAgent,
    result_root: Path,
    temp_root: Path,
) -> InitialProfileWorkflowResult:
    if re.fullmatch(r"P\d{3}", paper_id) is None:
        raise ValueError("paper_id must match P followed by three digits")

    system_prompt = build_system_prompt("initial_generation")
    fingerprint = compute_task_fingerprint(
        stage=INITIAL_STAGE,
        business_key=paper_id,
        dynamic_input={
            "title": title,
            "authors": list(authors),
            "article_full_text": article_full_text,
        },
        system_prompt=system_prompt,
        output_schema=PaperProfileContent.model_json_schema(),
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root / "checkpoints" / INITIAL_STAGE / f"{paper_id}.json"
    )
    failure_path = result_root / "failures" / INITIAL_STAGE / f"{paper_id}.json"
    loaded = load_valid_paper_profile_checkpoint(
        path=checkpoint_path,
        expected_stage=INITIAL_STAGE,
        expected_business_key=paper_id,
        expected_fingerprint=fingerprint,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return InitialProfileWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.initial_generation(
            title=title,
            authors=authors,
            article_full_text=article_full_text,
        )
    except NetworkRetriesExhausted as error:
        _persist_failure(
            stage=INITIAL_STAGE,
            failure_kind="network_exhausted",
            metrics=error.metrics,
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ContentRetriesExhausted as error:
        _persist_failure(
            stage=INITIAL_STAGE,
            failure_kind="content_exhausted",
            metrics=error.metrics,
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_failure(
            stage=INITIAL_STAGE,
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_failure(
            stage=INITIAL_STAGE,
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = PaperProfileCheckpoint(
        checkpoint_version=1,
        stage=INITIAL_STAGE,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        result=run_result.profile,
        metrics=run_result.metrics,
    )
    write_paper_profile_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return InitialProfileWorkflowResult(checkpoint=checkpoint, reused=False)


def run_targeted_profile_revision(
    *,
    paper_id: str,
    title: str,
    authors: Sequence[str],
    article_full_text: str,
    current_profile: PaperProfileContent,
    review_issues: Sequence[ProfileReviewIssue],
    model: str,
    agent: TargetedRevisionAgent,
    result_root: Path,
    temp_root: Path,
) -> InitialProfileWorkflowResult:
    if re.fullmatch(r"P\d{3}", paper_id) is None:
        raise ValueError("paper_id must match P followed by three digits")
    if not review_issues:
        raise ValueError("review_issues must contain at least one issue")

    system_prompt = build_system_prompt("targeted_revision")
    fingerprint = compute_task_fingerprint(
        stage=REVISION_STAGE,
        business_key=paper_id,
        dynamic_input={
            "title": title,
            "authors": list(authors),
            "article_full_text": article_full_text,
            "current_profile": current_profile.model_dump(mode="json"),
            "review_issues": [
                issue.model_dump(mode="json") for issue in review_issues
            ],
        },
        system_prompt=system_prompt,
        output_schema=PaperProfileContent.model_json_schema(),
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root / "checkpoints" / REVISION_STAGE / f"{paper_id}.json"
    )
    failure_path = result_root / "failures" / REVISION_STAGE / f"{paper_id}.json"
    loaded = load_valid_paper_profile_checkpoint(
        path=checkpoint_path,
        expected_stage=REVISION_STAGE,
        expected_business_key=paper_id,
        expected_fingerprint=fingerprint,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return InitialProfileWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.targeted_revision(
            title=title,
            authors=authors,
            article_full_text=article_full_text,
            current_profile=current_profile,
            review_issues=review_issues,
        )
    except NetworkRetriesExhausted as error:
        _persist_failure(
            stage=REVISION_STAGE,
            failure_kind="network_exhausted",
            metrics=error.metrics,
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ContentRetriesExhausted as error:
        _persist_failure(
            stage=REVISION_STAGE,
            failure_kind="content_exhausted",
            metrics=error.metrics,
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_failure(
            stage=REVISION_STAGE,
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_failure(
            stage=REVISION_STAGE,
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = PaperProfileCheckpoint(
        checkpoint_version=1,
        stage=REVISION_STAGE,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        result=run_result.profile,
        metrics=run_result.metrics,
    )
    write_paper_profile_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return InitialProfileWorkflowResult(checkpoint=checkpoint, reused=False)


def run_profile_review(
    *,
    paper_id: str,
    title: str,
    authors: Sequence[str],
    article_full_text: str,
    current_profile: PaperProfileContent,
    model: str,
    agent: ProfileReviewer,
    result_root: Path,
    temp_root: Path,
) -> ProfileReviewWorkflowResult:
    if re.fullmatch(r"P\d{3}", paper_id) is None:
        raise ValueError("paper_id must match P followed by three digits")

    system_prompt = build_profile_review_system_prompt()
    fingerprint = compute_task_fingerprint(
        stage=PROFILE_REVIEW_STAGE,
        business_key=paper_id,
        dynamic_input={
            "title": title,
            "authors": list(authors),
            "article_full_text": article_full_text,
            "current_profile": current_profile.model_dump(mode="json"),
        },
        system_prompt=system_prompt,
        output_schema=TypeAdapter(ProfileReviewResult).json_schema(),
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root / "checkpoints" / PROFILE_REVIEW_STAGE / f"{paper_id}.json"
    )
    failure_path = (
        result_root / "failures" / PROFILE_REVIEW_STAGE / f"{paper_id}.json"
    )
    loaded = load_valid_profile_review_checkpoint(
        path=checkpoint_path,
        expected_business_key=paper_id,
        expected_fingerprint=fingerprint,
        current_profile=current_profile,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return ProfileReviewWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.review_profile(
            title=title,
            authors=authors,
            article_full_text=article_full_text,
            current_profile=current_profile,
        )
    except ProfileReviewNetworkRetriesExhausted as error:
        _persist_profile_review_failure(
            failure_kind="network_exhausted",
            metrics=error.metrics,
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ProfileReviewContentRetriesExhausted as error:
        _persist_profile_review_failure(
            failure_kind="content_exhausted",
            metrics=error.metrics,
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_profile_review_failure(
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_profile_review_failure(
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            paper_id=paper_id,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = ProfileReviewCheckpoint(
        checkpoint_version=1,
        stage=PROFILE_REVIEW_STAGE,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        result=run_result.review,
        metrics=run_result.metrics,
    )
    write_profile_review_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return ProfileReviewWorkflowResult(checkpoint=checkpoint, reused=False)


def run_initial_relation_inference(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    model: str,
    agent: InitialRelationInferenceAgent,
    result_root: Path,
    temp_root: Path,
) -> RelationInferenceWorkflowResult:
    validate_relation_pair_inputs(paper_a_metadata, paper_b_metadata)
    user_message = build_relation_inference_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
    )
    business_key = (
        f"{paper_a_metadata.paper_id}__{paper_b_metadata.paper_id}"
    )
    system_prompt = build_relation_inference_system_prompt()
    output_schema = TypeAdapter(RelationRecord).json_schema()
    fingerprint = compute_task_fingerprint(
        stage=RELATION_INITIAL_STAGE,
        business_key=business_key,
        dynamic_input={"user_message": user_message},
        system_prompt=system_prompt,
        output_schema=output_schema,
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root
        / "checkpoints"
        / RELATION_INITIAL_STAGE
        / f"{business_key}.json"
    )
    failure_path = (
        result_root
        / "failures"
        / RELATION_INITIAL_STAGE
        / f"{business_key}.json"
    )
    loaded = load_valid_relation_inference_checkpoint(
        path=checkpoint_path,
        expected_stage=RELATION_INITIAL_STAGE,
        expected_business_key=business_key,
        expected_fingerprint=fingerprint,
        paper_a_metadata=paper_a_metadata,
        paper_b_metadata=paper_b_metadata,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return RelationInferenceWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.initial_inference(
            paper_a_metadata=paper_a_metadata,
            paper_a_profile=paper_a_profile,
            paper_a_full_text=paper_a_full_text,
            paper_b_metadata=paper_b_metadata,
            paper_b_profile=paper_b_profile,
            paper_b_full_text=paper_b_full_text,
        )
    except RelationInferenceNetworkRetriesExhausted as error:
        _persist_relation_inference_failure(
            stage=RELATION_INITIAL_STAGE,
            failure_kind="network_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except RelationInferenceContentRetriesExhausted as error:
        _persist_relation_inference_failure(
            stage=RELATION_INITIAL_STAGE,
            failure_kind="content_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_relation_inference_failure(
            stage=RELATION_INITIAL_STAGE,
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_relation_inference_failure(
            stage=RELATION_INITIAL_STAGE,
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = RelationInferenceCheckpoint(
        checkpoint_version=1,
        stage=RELATION_INITIAL_STAGE,
        business_key=business_key,
        task_fingerprint=fingerprint,
        result=run_result.record,
        metrics=run_result.metrics,
    )
    write_relation_inference_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return RelationInferenceWorkflowResult(checkpoint=checkpoint, reused=False)


def run_targeted_relation_revision(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    current_relation: RelationRecord,
    review_issues: Sequence[RelationReviewIssue],
    model: str,
    agent: TargetedRelationRevisionAgent,
    result_root: Path,
    temp_root: Path,
) -> RelationInferenceWorkflowResult:
    validate_relation_pair_inputs(paper_a_metadata, paper_b_metadata)
    validate_relation_handoff(
        current_relation,
        paper_a_metadata,
        paper_b_metadata,
    )
    user_message = build_relation_revision_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
        current_relation=current_relation,
        review_issues=review_issues,
    )
    business_key = (
        f"{paper_a_metadata.paper_id}__{paper_b_metadata.paper_id}"
    )
    system_prompt = build_relation_revision_system_prompt()
    output_schema = TypeAdapter(RelationRecord).json_schema()
    fingerprint = compute_task_fingerprint(
        stage=RELATION_REVISION_STAGE,
        business_key=business_key,
        dynamic_input={"user_message": user_message},
        system_prompt=system_prompt,
        output_schema=output_schema,
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root
        / "checkpoints"
        / RELATION_REVISION_STAGE
        / f"{business_key}.json"
    )
    failure_path = (
        result_root
        / "failures"
        / RELATION_REVISION_STAGE
        / f"{business_key}.json"
    )
    loaded = load_valid_relation_inference_checkpoint(
        path=checkpoint_path,
        expected_stage=RELATION_REVISION_STAGE,
        expected_business_key=business_key,
        expected_fingerprint=fingerprint,
        paper_a_metadata=paper_a_metadata,
        paper_b_metadata=paper_b_metadata,
        current_relation=current_relation,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return RelationInferenceWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.targeted_revision(
            paper_a_metadata=paper_a_metadata,
            paper_a_profile=paper_a_profile,
            paper_a_full_text=paper_a_full_text,
            paper_b_metadata=paper_b_metadata,
            paper_b_profile=paper_b_profile,
            paper_b_full_text=paper_b_full_text,
            current_relation=current_relation,
            review_issues=review_issues,
        )
    except RelationInferenceNetworkRetriesExhausted as error:
        _persist_relation_inference_failure(
            stage=RELATION_REVISION_STAGE,
            failure_kind="network_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except RelationInferenceContentRetriesExhausted as error:
        _persist_relation_inference_failure(
            stage=RELATION_REVISION_STAGE,
            failure_kind="content_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_relation_inference_failure(
            stage=RELATION_REVISION_STAGE,
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_relation_inference_failure(
            stage=RELATION_REVISION_STAGE,
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = RelationInferenceCheckpoint(
        checkpoint_version=1,
        stage=RELATION_REVISION_STAGE,
        business_key=business_key,
        task_fingerprint=fingerprint,
        result=run_result.record,
        metrics=run_result.metrics,
    )
    write_relation_inference_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return RelationInferenceWorkflowResult(checkpoint=checkpoint, reused=False)


def run_initial_relation_review(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    current_relation: RelationRecord,
    model: str,
    agent: RelationReviewer,
    result_root: Path,
    temp_root: Path,
) -> RelationReviewWorkflowResult:
    validate_relation_pair_inputs(paper_a_metadata, paper_b_metadata)
    validate_relation_handoff(
        current_relation,
        paper_a_metadata,
        paper_b_metadata,
    )
    user_message = build_relation_review_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
        current_relation=current_relation,
    )
    business_key = (
        f"{paper_a_metadata.paper_id}__{paper_b_metadata.paper_id}"
    )
    system_prompt = build_relation_review_system_prompt()
    output_schema = TypeAdapter(RelationReviewResult).json_schema()
    fingerprint = compute_task_fingerprint(
        stage=RELATION_INITIAL_REVIEW_STAGE,
        business_key=business_key,
        dynamic_input={"user_message": user_message},
        system_prompt=system_prompt,
        output_schema=output_schema,
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root
        / "checkpoints"
        / RELATION_INITIAL_REVIEW_STAGE
        / f"{business_key}.json"
    )
    failure_path = (
        result_root
        / "failures"
        / RELATION_INITIAL_REVIEW_STAGE
        / f"{business_key}.json"
    )
    loaded = load_valid_relation_review_checkpoint(
        path=checkpoint_path,
        expected_stage=RELATION_INITIAL_REVIEW_STAGE,
        expected_business_key=business_key,
        expected_fingerprint=fingerprint,
        current_relation=current_relation,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return RelationReviewWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.review_relation(
            paper_a_metadata=paper_a_metadata,
            paper_a_profile=paper_a_profile,
            paper_a_full_text=paper_a_full_text,
            paper_b_metadata=paper_b_metadata,
            paper_b_profile=paper_b_profile,
            paper_b_full_text=paper_b_full_text,
            current_relation=current_relation,
        )
    except RelationReviewNetworkRetriesExhausted as error:
        _persist_relation_review_failure(
            stage=RELATION_INITIAL_REVIEW_STAGE,
            failure_kind="network_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except RelationReviewContentRetriesExhausted as error:
        _persist_relation_review_failure(
            stage=RELATION_INITIAL_REVIEW_STAGE,
            failure_kind="content_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_relation_review_failure(
            stage=RELATION_INITIAL_REVIEW_STAGE,
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_relation_review_failure(
            stage=RELATION_INITIAL_REVIEW_STAGE,
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = RelationReviewCheckpoint(
        checkpoint_version=1,
        stage=RELATION_INITIAL_REVIEW_STAGE,
        business_key=business_key,
        task_fingerprint=fingerprint,
        result=run_result.review,
        metrics=run_result.metrics,
    )
    write_relation_review_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return RelationReviewWorkflowResult(checkpoint=checkpoint, reused=False)


def run_post_revision_relation_review(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    previous_review_issues: Sequence[RelationReviewIssue],
    revised_relation: RelationRecord,
    model: str,
    agent: PostRevisionRelationReviewer,
    result_root: Path,
    temp_root: Path,
) -> RelationReviewWorkflowResult:
    validate_relation_pair_inputs(paper_a_metadata, paper_b_metadata)
    validate_relation_handoff(
        revised_relation,
        paper_a_metadata,
        paper_b_metadata,
    )
    user_message = build_relation_post_revision_review_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
        previous_review_issues=previous_review_issues,
        revised_relation=revised_relation,
    )
    business_key = (
        f"{paper_a_metadata.paper_id}__{paper_b_metadata.paper_id}"
    )
    system_prompt = build_relation_post_revision_review_system_prompt()
    output_schema = TypeAdapter(RelationReviewResult).json_schema()
    fingerprint = compute_task_fingerprint(
        stage=RELATION_POST_REVISION_REVIEW_STAGE,
        business_key=business_key,
        dynamic_input={"user_message": user_message},
        system_prompt=system_prompt,
        output_schema=output_schema,
        experiment_unit=EXPERIMENT_UNIT,
        model=model,
    )
    checkpoint_path = (
        result_root
        / "checkpoints"
        / RELATION_POST_REVISION_REVIEW_STAGE
        / f"{business_key}.json"
    )
    failure_path = (
        result_root
        / "failures"
        / RELATION_POST_REVISION_REVIEW_STAGE
        / f"{business_key}.json"
    )
    loaded = load_valid_relation_review_checkpoint(
        path=checkpoint_path,
        expected_stage=RELATION_POST_REVISION_REVIEW_STAGE,
        expected_business_key=business_key,
        expected_fingerprint=fingerprint,
        current_relation=revised_relation,
    )
    if loaded is not None:
        failure_path.unlink(missing_ok=True)
        return RelationReviewWorkflowResult(checkpoint=loaded, reused=True)

    started = time.perf_counter()
    try:
        run_result = agent.review_relation_after_revision(
            paper_a_metadata=paper_a_metadata,
            paper_a_profile=paper_a_profile,
            paper_a_full_text=paper_a_full_text,
            paper_b_metadata=paper_b_metadata,
            paper_b_profile=paper_b_profile,
            paper_b_full_text=paper_b_full_text,
            previous_review_issues=previous_review_issues,
            revised_relation=revised_relation,
        )
    except RelationReviewNetworkRetriesExhausted as error:
        _persist_relation_review_failure(
            stage=RELATION_POST_REVISION_REVIEW_STAGE,
            failure_kind="network_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except RelationReviewContentRetriesExhausted as error:
        _persist_relation_review_failure(
            stage=RELATION_POST_REVISION_REVIEW_STAGE,
            failure_kind="content_exhausted",
            metrics=error.metrics,
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except ValueError:
        _persist_relation_review_failure(
            stage=RELATION_POST_REVISION_REVIEW_STAGE,
            failure_kind="input_invalid",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise
    except Exception:
        _persist_relation_review_failure(
            stage=RELATION_POST_REVISION_REVIEW_STAGE,
            failure_kind="internal_error",
            metrics=StageMetrics(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                token_usage=TokenUsage.zero(),
            ),
            business_key=business_key,
            fingerprint=fingerprint,
            failure_path=failure_path,
            temp_root=temp_root,
        )
        raise

    checkpoint = RelationReviewCheckpoint(
        checkpoint_version=1,
        stage=RELATION_POST_REVISION_REVIEW_STAGE,
        business_key=business_key,
        task_fingerprint=fingerprint,
        result=run_result.review,
        metrics=run_result.metrics,
    )
    write_relation_review_checkpoint(
        checkpoint=checkpoint,
        destination=checkpoint_path,
        temp_root=temp_root,
    )
    failure_path.unlink(missing_ok=True)
    return RelationReviewWorkflowResult(checkpoint=checkpoint, reused=False)


def _persist_failure(
    *,
    stage: str,
    failure_kind: str,
    metrics: StageMetrics,
    paper_id: str,
    fingerprint: str,
    failure_path: Path,
    temp_root: Path,
) -> None:
    failure = PaperParserFailureRecord(
        failure_record_version=1,
        stage=stage,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        failure_kind=failure_kind,
        metrics=metrics,
    )
    write_failure_record(
        failure=failure,
        destination=failure_path,
        temp_root=temp_root,
    )


def _persist_profile_review_failure(
    *,
    failure_kind: str,
    metrics: StageMetrics,
    paper_id: str,
    fingerprint: str,
    failure_path: Path,
    temp_root: Path,
) -> None:
    failure = ProfileReviewerFailureRecord(
        failure_record_version=1,
        stage=PROFILE_REVIEW_STAGE,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        failure_kind=failure_kind,
        metrics=metrics,
    )
    write_failure_record(
        failure=failure,
        destination=failure_path,
        temp_root=temp_root,
    )


def _persist_relation_inference_failure(
    *,
    stage: str,
    failure_kind: str,
    metrics: StageMetrics,
    business_key: str,
    fingerprint: str,
    failure_path: Path,
    temp_root: Path,
) -> None:
    failure = RelationInferenceFailureRecord(
        failure_record_version=1,
        stage=stage,
        business_key=business_key,
        task_fingerprint=fingerprint,
        failure_kind=failure_kind,
        metrics=metrics,
    )
    write_failure_record(
        failure=failure,
        destination=failure_path,
        temp_root=temp_root,
    )


def _persist_relation_review_failure(
    *,
    stage: str,
    failure_kind: str,
    metrics: StageMetrics,
    business_key: str,
    fingerprint: str,
    failure_path: Path,
    temp_root: Path,
) -> None:
    failure = RelationReviewerFailureRecord(
        failure_record_version=1,
        stage=stage,
        business_key=business_key,
        task_fingerprint=fingerprint,
        failure_kind=failure_kind,
        metrics=metrics,
    )
    write_failure_record(
        failure=failure,
        destination=failure_path,
        temp_root=temp_root,
    )
