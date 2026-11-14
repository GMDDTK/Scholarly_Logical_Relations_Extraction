from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import EXPERIMENT_UNIT
from .agents import (
    AgentRetriesExhausted,
    CitationIdentificationAgent,
    CitationReviewAgent,
    ContentRetriesExhausted,
    IDENTIFICATION_PROMPT,
    NetworkRetriesExhausted,
    REVIEW_PROMPT,
)
from .checkpointing import (
    compute_fingerprint,
    load_valid_identification_checkpoint,
    load_valid_outcome,
    load_valid_review_checkpoint,
    write_model_atomically,
)
from .contracts import (
    CitationEvidence,
    CitationFailureRecord,
    CitationIdentification,
    CitationPaperOutcome,
    CitationReviewResult,
    CitedPaperEvidence,
    IdentificationCheckpoint,
    ReferenceDocument,
    ReviewCheckpoint,
    StageMetrics,
    TokenUsage,
    TopicInputSet,
)
from .llm_client import JsonCompletionClient
from .prompting import build_identification_message, build_review_message
from .result_paths import CitationResultPaths


ProcessStatus = Literal["accepted", "skipped", "needs_retry", "failed"]


@dataclass(frozen=True)
class PaperProcessResult:
    paper_id: str
    status: ProcessStatus
    outcome: CitationPaperOutcome | None
    message: str | None = None
    current_run_metrics: StageMetrics = field(
        default_factory=lambda: StageMetrics(
            elapsed_seconds=0,
            token_usage=TokenUsage.zero(),
        )
    )


def paper_input_fingerprint(
    *,
    topic_input: TopicInputSet,
    document: ReferenceDocument,
    model: str,
) -> str:
    return compute_fingerprint(
        {
            "contract_version": 1,
            "reference_ingestion_version": 1,
            "experiment_unit": EXPERIMENT_UNIT,
            "backend": "DEEPSEEK_OFFICIAL",
            "model": model,
            "request_config": {"response_format": "json_object"},
            "topic": topic_input.topic,
            "paper_id": document.paper.paper_id,
            "catalog_sha256": topic_input.catalog_sha256,
            "reference_workbook_sha256": document.workbook_sha256,
            "reference_lines": [line.model_dump(mode="json") for line in document.lines],
            "identification_prompt": IDENTIFICATION_PROMPT,
            "review_prompt": REVIEW_PROMPT,
            "identification_schema": CitationIdentification.model_json_schema(),
            "review_schema": CitationReviewResult.model_json_schema(),
        }
    )


def process_paper(
    *,
    topic_input: TopicInputSet,
    document: ReferenceDocument,
    client: JsonCompletionClient,
    paths: CitationResultPaths,
    model: str,
) -> PaperProcessResult:
    paper_id = document.paper.paper_id
    fingerprint = paper_input_fingerprint(
        topic_input=topic_input,
        document=document,
        model=model,
    )
    existing = load_valid_outcome(
        path=paths.outcome(paper_id),
        topic=topic_input.topic,
        paper_id=paper_id,
        input_fingerprint=fingerprint,
    )
    if existing is not None:
        return PaperProcessResult(
            paper_id=paper_id,
            status="skipped",
            outcome=existing,
        )

    if not document.lines:
        outcome = CitationPaperOutcome(
            topic=topic_input.topic,
            paper_id=paper_id,
            input_fingerprint=fingerprint,
            has_references=False,
            cited_papers=[],
        )
        _commit_outcome(outcome=outcome, paths=paths)
        return PaperProcessResult(paper_id=paper_id, status="accepted", outcome=outcome)

    identification_fingerprint = compute_fingerprint(
        {"stage": "citation_identification", "paper_fingerprint": fingerprint}
    )
    identification_checkpoint = load_valid_identification_checkpoint(
        path=paths.identification_checkpoint(paper_id),
        business_key=paper_id,
        task_fingerprint=identification_fingerprint,
    )
    if identification_checkpoint is not None:
        try:
            _validate_identification(
                result=identification_checkpoint.result,
                topic_input=topic_input,
                document=document,
            )
        except ValueError:
            identification_checkpoint = None
    current_metrics = _zero_metrics()
    if identification_checkpoint is not None:
        identification = identification_checkpoint.result
    else:
        try:
            identification_execution = CitationIdentificationAgent(
                client=client
            ).identify(
                user_message=build_identification_message(
                    topic_input=topic_input,
                    document=document,
                ),
                validator=lambda result: _validate_identification(
                    result=result,
                    topic_input=topic_input,
                    document=document,
                ),
            )
            identification = identification_execution.result
        except AgentRetriesExhausted as error:
            return _record_agent_failure(
                paper_id=paper_id,
                stage="citation_identification",
                fingerprint=identification_fingerprint,
                error=error,
                paths=paths,
            )
        except Exception as error:
            return _record_unexpected_failure(
                paper_id=paper_id,
                stage="citation_identification",
                fingerprint=identification_fingerprint,
                error=error,
                metrics=current_metrics,
                paths=paths,
            )
        write_model_atomically(
            model=IdentificationCheckpoint(
                business_key=paper_id,
                task_fingerprint=identification_fingerprint,
                result=identification,
                metrics=identification_execution.metrics,
            ),
            destination=paths.identification_checkpoint(paper_id),
            temp_root=paths.temp_root,
            business_key=paper_id,
        )
        current_metrics = _plus_metrics(
            current_metrics,
            identification_execution.metrics,
        )

    review_fingerprint = compute_fingerprint(
        {
            "stage": "citation_review",
            "paper_fingerprint": fingerprint,
            "identification": identification.model_dump(mode="json"),
        }
    )
    review_checkpoint = load_valid_review_checkpoint(
        path=paths.review_checkpoint(paper_id),
        business_key=paper_id,
        task_fingerprint=review_fingerprint,
    )
    if review_checkpoint is not None:
        try:
            _validate_review(result=review_checkpoint.result, document=document)
        except ValueError:
            review_checkpoint = None
    if review_checkpoint is not None and review_checkpoint.result.status == "pass":
        review = review_checkpoint.result
        review_metrics = review_checkpoint.metrics
    else:
        try:
            review_execution = CitationReviewAgent(client=client).review(
                user_message=build_review_message(
                    topic_input=topic_input,
                    document=document,
                    identification=identification,
                ),
                validator=lambda result: _validate_review(
                    result=result,
                    document=document,
                ),
            )
            review = review_execution.result
            review_metrics = review_execution.metrics
        except AgentRetriesExhausted as error:
            failed = _record_agent_failure(
                paper_id=paper_id,
                stage="citation_review",
                fingerprint=review_fingerprint,
                error=error,
                paths=paths,
            )
            return PaperProcessResult(
                paper_id=failed.paper_id,
                status=failed.status,
                outcome=None,
                message=failed.message,
                current_run_metrics=_plus_metrics(current_metrics, error.metrics),
            )
        except Exception as error:
            return _record_unexpected_failure(
                paper_id=paper_id,
                stage="citation_review",
                fingerprint=review_fingerprint,
                error=error,
                metrics=current_metrics,
                paths=paths,
            )
        write_model_atomically(
            model=ReviewCheckpoint(
                business_key=paper_id,
                task_fingerprint=review_fingerprint,
                result=review,
                metrics=review_execution.metrics,
            ),
            destination=paths.review_checkpoint(paper_id),
            temp_root=paths.temp_root,
            business_key=paper_id,
        )
        current_metrics = _plus_metrics(
            current_metrics,
            review_execution.metrics,
        )

    if review.status == "needs_retry":
        paths.identification_checkpoint(paper_id).unlink(missing_ok=True)
        paths.review_checkpoint(paper_id).unlink(missing_ok=True)
        failure = CitationFailureRecord(
            stage="citation_review",
            business_key=paper_id,
            task_fingerprint=review_fingerprint,
            failure_kind="review_needs_retry",
            message="Citation reviewer requested a fresh identification on the next run",
            metrics=current_metrics,
            review_result=review,
        )
        write_model_atomically(
            model=failure,
            destination=paths.failure("citation_review", paper_id),
            temp_root=paths.temp_root,
            business_key=paper_id,
        )
        return PaperProcessResult(
            paper_id=paper_id,
            status="needs_retry",
            outcome=None,
            message=failure.message,
            current_run_metrics=current_metrics,
        )

    line_by_number = {line.row_number: line for line in document.lines}
    outcome = CitationPaperOutcome(
        topic=topic_input.topic,
        paper_id=paper_id,
        input_fingerprint=fingerprint,
        has_references=True,
        cited_papers=[
            CitedPaperEvidence(
                cited_paper_id=item.cited_paper_id,
                evidence=[
                    CitationEvidence(
                        row_number=row_number,
                        raw_text=line_by_number[row_number].raw_text,
                    )
                    for row_number in item.reference_row_numbers
                ],
            )
            for item in identification.cited_papers
        ],
    )
    _commit_outcome(outcome=outcome, paths=paths)
    return PaperProcessResult(
        paper_id=paper_id,
        status="accepted",
        outcome=outcome,
        current_run_metrics=current_metrics,
    )


def _validate_identification(
    *,
    result: CitationIdentification,
    topic_input: TopicInputSet,
    document: ReferenceDocument,
) -> None:
    valid_ids = {paper.paper_id for paper in topic_input.papers}
    valid_rows = {line.row_number for line in document.lines}
    for item in result.cited_papers:
        if item.cited_paper_id not in valid_ids:
            raise ValueError(f"unknown target paper ID {item.cited_paper_id}")
        if item.cited_paper_id == document.paper.paper_id:
            raise ValueError("self-citation is outside the network contract")
        if not set(item.reference_row_numbers).issubset(valid_rows):
            raise ValueError("identification contains an unknown reference row")


def _validate_review(
    *,
    result: CitationReviewResult,
    document: ReferenceDocument,
) -> None:
    valid_rows = {line.row_number for line in document.lines}
    for issue in result.issues:
        if not set(issue.reference_row_numbers).issubset(valid_rows):
            raise ValueError("review contains an unknown reference row")


def _record_agent_failure(
    *,
    paper_id: str,
    stage: Literal["citation_identification", "citation_review"],
    fingerprint: str,
    error: AgentRetriesExhausted,
    paths: CitationResultPaths,
) -> PaperProcessResult:
    kind = "content_exhausted" if isinstance(error, ContentRetriesExhausted) else "network_exhausted"
    failure = CitationFailureRecord(
        stage=stage,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        failure_kind=kind,
        message=str(error),
        metrics=error.metrics,
    )
    write_model_atomically(
        model=failure,
        destination=paths.failure(stage, paper_id),
        temp_root=paths.temp_root,
        business_key=paper_id,
    )
    return PaperProcessResult(
        paper_id=paper_id,
        status="failed",
        outcome=None,
        message=str(error),
        current_run_metrics=error.metrics,
    )


def _record_unexpected_failure(
    *,
    paper_id: str,
    stage: Literal["citation_identification", "citation_review"],
    fingerprint: str,
    error: Exception,
    metrics: StageMetrics,
    paths: CitationResultPaths,
) -> PaperProcessResult:
    message = f"{type(error).__name__}: {error}"
    failure = CitationFailureRecord(
        stage=stage,
        business_key=paper_id,
        task_fingerprint=fingerprint,
        failure_kind="unexpected_error",
        message=message,
        metrics=metrics,
    )
    write_model_atomically(
        model=failure,
        destination=paths.failure(stage, paper_id),
        temp_root=paths.temp_root,
        business_key=paper_id,
    )
    return PaperProcessResult(
        paper_id=paper_id,
        status="failed",
        outcome=None,
        message=message,
        current_run_metrics=metrics,
    )


def _commit_outcome(*, outcome: CitationPaperOutcome, paths: CitationResultPaths) -> None:
    write_model_atomically(
        model=outcome,
        destination=paths.outcome(outcome.paper_id),
        temp_root=paths.temp_root,
        business_key=outcome.paper_id,
    )


def _zero_metrics() -> StageMetrics:
    return StageMetrics(elapsed_seconds=0, token_usage=TokenUsage.zero())


def _plus_metrics(left: StageMetrics, right: StageMetrics) -> StageMetrics:
    return StageMetrics(
        elapsed_seconds=left.elapsed_seconds + right.elapsed_seconds,
        token_usage=left.token_usage.plus(right.token_usage),
    )


def completed_result_chain_metrics(
    *,
    topic_input: TopicInputSet,
    paths: CitationResultPaths,
    model: str,
) -> StageMetrics:
    total = _zero_metrics()
    for document in topic_input.reference_documents:
        paper_id = document.paper.paper_id
        fingerprint = paper_input_fingerprint(
            topic_input=topic_input,
            document=document,
            model=model,
        )
        outcome = load_valid_outcome(
            path=paths.outcome(paper_id),
            topic=topic_input.topic,
            paper_id=paper_id,
            input_fingerprint=fingerprint,
        )
        if outcome is None or not document.lines:
            continue
        identification_fingerprint = compute_fingerprint(
            {"stage": "citation_identification", "paper_fingerprint": fingerprint}
        )
        identification = load_valid_identification_checkpoint(
            path=paths.identification_checkpoint(paper_id),
            business_key=paper_id,
            task_fingerprint=identification_fingerprint,
        )
        if identification is None:
            continue
        review_fingerprint = compute_fingerprint(
            {
                "stage": "citation_review",
                "paper_fingerprint": fingerprint,
                "identification": identification.result.model_dump(mode="json"),
            }
        )
        review = load_valid_review_checkpoint(
            path=paths.review_checkpoint(paper_id),
            business_key=paper_id,
            task_fingerprint=review_fingerprint,
        )
        if review is None or review.result.status != "pass":
            continue
        total = _plus_metrics(total, identification.metrics)
        total = _plus_metrics(total, review.metrics)
    return total
