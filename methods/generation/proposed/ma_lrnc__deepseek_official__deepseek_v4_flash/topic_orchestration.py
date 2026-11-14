from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from pydantic import ValidationError

from .contracts.paper_metadata import PaperMetadata
from .contracts.paper_profile import PaperProfileContent
from .contracts.topic_run import (
    PairRunCounts,
    PaperRunCounts,
    TopicRunFailure,
    TopicRunSummary,
    TopicConsumptionSummary,
)
from .contracts.runtime import StageMetrics, TokenUsage
from .orchestration import (
    PaperProfileAgent,
    PaperProfileOutcome,
    RelationAgent,
    RelationPairOutcome,
    RelationReviewerAgent,
    run_paper_profile_orchestration,
    run_relation_orchestration,
)
from .pdf_preprocessing import write_pdf_markdown
from .result_paths import TopicResultPaths
from .topic_input import TopicInputSet
from .workflow import ProfileReviewer


PdfWriter = Callable[..., str]
ProfileRunner = Callable[..., PaperProfileOutcome]
RelationRunner = Callable[..., RelationPairOutcome]


class TopicReviewerAgent(
    ProfileReviewer,
    RelationReviewerAgent,
    Protocol,
):
    pass


@dataclass(frozen=True)
class _CompletedPaper:
    metadata: PaperMetadata
    full_text: str
    profile: PaperProfileContent


def run_topic_orchestration(
    *,
    topic_input: TopicInputSet,
    processed_root: Path,
    model: str,
    parser_agent: PaperProfileAgent,
    reviewer_agent: TopicReviewerAgent,
    relation_agent: RelationAgent,
    paths: TopicResultPaths,
    pdf_writer: PdfWriter = write_pdf_markdown,
    profile_runner: ProfileRunner = run_paper_profile_orchestration,
    relation_runner: RelationRunner = run_relation_orchestration,
    current_run_token_usage: Callable[[], TokenUsage] | None = None,
    started_at: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TopicRunSummary:
    run_started_at = clock() if started_at is None else started_at
    if topic_input.topic != paths.topic:
        raise ValueError("topic input and result paths must use the same topic")

    failures = [
        TopicRunFailure(
            scope="input",
            business_key=failure.pdf_path.name,
            code=failure.code,
            message=failure.message,
        )
        for failure in topic_input.failures
    ]
    completed_papers: dict[str, _CompletedPaper] = {}
    completed_result_metrics = StageMetrics.zero()

    for match in topic_input.matches:
        paper_id = match.metadata.paper_id
        paper_root = processed_root / topic_input.topic / paper_id
        metadata_path = paper_root / "metadata.json"
        full_text_path = paper_root / "full_text.md"
        try:
            _ensure_matching_metadata(
                metadata=match.metadata,
                destination=metadata_path,
                temp_root=paths.temp_root / "preprocessing",
            )
            full_text = _load_or_extract_full_text(
                input_pdf=match.pdf_path,
                output_markdown=full_text_path,
                pdf_writer=pdf_writer,
            )
            profile_outcome = profile_runner(
                paper_metadata=match.metadata,
                article_full_text=full_text,
                model=model,
                parser_agent=parser_agent,
                reviewer_agent=reviewer_agent,
                paths=paths,
            )
        except Exception as error:
            failures.append(
                TopicRunFailure(
                    scope="paper",
                    business_key=paper_id,
                    code="paper_processing_failed",
                    message=_safe_error_message(error),
                )
            )
            continue
        completed_papers[paper_id] = _CompletedPaper(
            metadata=match.metadata,
            full_text=full_text,
            profile=profile_outcome.profile,
        )
        completed_result_metrics = completed_result_metrics.plus(
            profile_outcome.metrics
        )

    completed_pairs = 0
    failed_pairs = 0
    runtime_blocked_pairs = 0
    for pair in topic_input.pairs:
        paper_a_id = pair.paper_a.metadata.paper_id
        paper_b_id = pair.paper_b.metadata.paper_id
        paper_a = completed_papers.get(paper_a_id)
        paper_b = completed_papers.get(paper_b_id)
        if paper_a is None or paper_b is None:
            runtime_blocked_pairs += 1
            continue
        try:
            relation_outcome = relation_runner(
                paper_a_metadata=paper_a.metadata,
                paper_a_profile=paper_a.profile,
                paper_a_full_text=paper_a.full_text,
                paper_b_metadata=paper_b.metadata,
                paper_b_profile=paper_b.profile,
                paper_b_full_text=paper_b.full_text,
                model=model,
                relation_agent=relation_agent,
                reviewer_agent=reviewer_agent,
                paths=paths,
            )
        except Exception as error:
            failed_pairs += 1
            failures.append(
                TopicRunFailure(
                    scope="pair",
                    business_key=pair.pair_key,
                    code="pair_processing_failed",
                    message=_safe_error_message(error),
                )
            )
            continue
        completed_pairs += 1
        completed_result_metrics = completed_result_metrics.plus(
            relation_outcome.metrics
        )

    expected_papers = len(topic_input.pdf_paths)
    expected_pairs = expected_papers * (expected_papers - 1) // 2
    input_blocked_pairs = expected_pairs - len(topic_input.pairs)
    blocked_pairs = input_blocked_pairs + runtime_blocked_pairs
    paper_counts = PaperRunCounts(
        expected=expected_papers,
        completed=len(completed_papers),
        failed=len(topic_input.matches) - len(completed_papers),
        input_failed=len(topic_input.failures),
    )
    pair_counts = PairRunCounts(
        expected=expected_pairs,
        completed=completed_pairs,
        failed=failed_pairs,
        blocked=blocked_pairs,
    )
    status = (
        "relations_complete"
        if (
            paper_counts.failed == 0
            and paper_counts.input_failed == 0
            and pair_counts.failed == 0
            and pair_counts.blocked == 0
        )
        else "incomplete"
    )
    return TopicRunSummary(
        status=status,
        topic=topic_input.topic,
        papers=paper_counts,
        pairs=pair_counts,
        consumption=TopicConsumptionSummary(
            current_run=StageMetrics(
                elapsed_seconds=max(0.0, clock() - run_started_at),
                token_usage=(
                    current_run_token_usage()
                    if current_run_token_usage is not None
                    else TokenUsage.zero()
                ),
            ),
            completed_result_chain=completed_result_metrics,
        ),
        failures=failures,
    )


def _load_or_extract_full_text(
    *,
    input_pdf: Path,
    output_markdown: Path,
    pdf_writer: PdfWriter,
) -> str:
    if output_markdown.is_file():
        persisted = output_markdown.read_text(encoding="utf-8")
        if persisted.strip():
            return persisted
    pdf_writer(
        input_pdf=input_pdf,
        output_markdown=output_markdown,
    )
    extracted = output_markdown.read_text(encoding="utf-8")
    if not extracted.strip():
        raise ValueError("full_text.md contains no non-whitespace text")
    return extracted


def _ensure_matching_metadata(
    *,
    metadata: PaperMetadata,
    destination: Path,
    temp_root: Path,
) -> None:
    if destination.is_file():
        try:
            persisted = PaperMetadata.model_validate_json(
                destination.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError):
            persisted = None
        if persisted == metadata:
            return

    serialized = json.dumps(
        metadata.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )
    _write_text_atomically(
        serialized + "\n",
        destination=destination,
        temp_root=temp_root,
        business_key=metadata.paper_id,
    )


def _write_text_atomically(
    content: str,
    *,
    destination: Path,
    temp_root: Path,
    business_key: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"{business_key}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_error_message(error: Exception) -> str:
    message = str(error).strip()
    return message or type(error).__name__
