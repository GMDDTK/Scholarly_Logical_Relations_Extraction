import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

from pydantic import TypeAdapter, ValidationError

from .agents.paper_parser import (
    ContentRetriesExhausted,
    NetworkRetriesExhausted,
    PaperParserAgent,
)
from .agents.evidence_reviewer import (
    EvidenceReviewerAgent,
    ProfileReviewContentRetriesExhausted,
    ProfileReviewNetworkRetriesExhausted,
    RelationReviewContentRetriesExhausted,
    RelationReviewNetworkRetriesExhausted,
)
from .agents.relation_inference import (
    RelationInferenceAgent,
    RelationInferenceContentRetriesExhausted,
    RelationInferenceNetworkRetriesExhausted,
)
from .config import EXPERIMENT_UNIT, PROJECT_ROOT, load_deepseek_config
from .contracts.paper_metadata import PaperMetadata
from .contracts.paper_profile import PaperProfileContent
from .contracts.profile_review import ProfileReviewIssue
from .contracts.relation_record import RelationRecord
from .contracts.relation_review import RelationReviewIssue
from .llm_client import DeepSeekJsonClient, TokenTrackingJsonClient
from .metadata_preprocessing import write_paper_metadata
from .pdf_preprocessing import write_pdf_markdown
from .result_paths import build_topic_result_paths
from .topic_input import discover_topic_inputs
from .topic_orchestration import run_topic_orchestration
from .visualization_orchestration import build_topic_visualization
from .workflow import (
    InitialProfileWorkflowResult,
    ProfileReviewWorkflowResult,
    RelationInferenceWorkflowResult,
    RelationReviewWorkflowResult,
    run_initial_profile_generation,
    run_initial_relation_inference,
    run_initial_relation_review,
    run_post_revision_relation_review,
    run_profile_review,
    run_targeted_relation_revision,
    run_targeted_profile_revision,
)


RESULT_UNIT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "generation"
    / "proposed"
    / EXPERIMENT_UNIT
)
TEMP_UNIT_ROOT = PROJECT_ROOT / "tmp" / "resume" / EXPERIMENT_UNIT


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run implemented MA-LRNC generation agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initial = subparsers.add_parser(
        "initial-generation",
        help="Generate a complete initial PaperProfileContent.",
    )
    _add_topic_argument(initial)
    _add_article_arguments(initial)

    revision = subparsers.add_parser(
        "targeted-revision",
        help="Revise a complete profile from structured review issues.",
    )
    _add_topic_argument(revision)
    _add_article_arguments(revision)
    revision.add_argument(
        "--current-profile",
        type=Path,
        required=True,
        help="UTF-8 JSON file containing the current PaperProfileContent.",
    )
    revision.add_argument(
        "--review-issues",
        type=Path,
        required=True,
        help="UTF-8 JSON array containing one or more ProfileReviewIssue objects.",
    )

    review = subparsers.add_parser(
        "profile-review",
        help="Review a complete PaperProfileContent against the article.",
    )
    _add_topic_argument(review)
    _add_article_arguments(review)
    review.add_argument(
        "--current-profile",
        type=Path,
        required=True,
        help="UTF-8 JSON file containing the current PaperProfileContent.",
    )

    relation = subparsers.add_parser(
        "relation-inference",
        help="Infer one academic logical relation for an ordered paper pair.",
    )
    _add_topic_argument(relation)
    _add_relation_paper_arguments(relation, "a")
    _add_relation_paper_arguments(relation, "b")

    relation_review = subparsers.add_parser(
        "relation-review",
        help="Review one current RelationRecord against an ordered paper pair.",
    )
    _add_topic_argument(relation_review)
    _add_relation_paper_arguments(relation_review, "a")
    _add_relation_paper_arguments(relation_review, "b")
    relation_review.add_argument(
        "--current-relation",
        type=Path,
        required=True,
        help="UTF-8 JSON file containing the current RelationRecord.",
    )

    relation_revision = subparsers.add_parser(
        "relation-targeted-revision",
        help="Revise one RelationRecord from structured relation review issues.",
    )
    _add_topic_argument(relation_revision)
    _add_relation_paper_arguments(relation_revision, "a")
    _add_relation_paper_arguments(relation_revision, "b")
    relation_revision.add_argument(
        "--current-relation",
        type=Path,
        required=True,
        help="UTF-8 JSON file containing the current RelationRecord.",
    )
    relation_revision.add_argument(
        "--review-issues",
        type=Path,
        required=True,
        help="UTF-8 JSON array containing RelationReviewIssue objects.",
    )

    post_revision_review = subparsers.add_parser(
        "relation-post-revision-review",
        help="Review a revised RelationRecord and terminate the automatic loop.",
    )
    _add_topic_argument(post_revision_review)
    _add_relation_paper_arguments(post_revision_review, "a")
    _add_relation_paper_arguments(post_revision_review, "b")
    post_revision_review.add_argument(
        "--previous-review-issues",
        type=Path,
        required=True,
        help="UTF-8 JSON array containing the issues that triggered revision.",
    )
    post_revision_review.add_argument(
        "--revised-relation",
        type=Path,
        required=True,
        help="UTF-8 JSON file containing the revised RelationRecord.",
    )

    extraction = subparsers.add_parser(
        "extract-full-text",
        help="Extract one PDF to cleaned single-document Markdown.",
    )
    extraction.add_argument("--input-pdf", type=Path, required=True)
    extraction.add_argument("--output-markdown", type=Path, required=True)

    metadata = subparsers.add_parser(
        "extract-metadata",
        help="Build one PaperMetadata JSON from a catalog workbook row.",
    )
    metadata.add_argument("--workbook", type=Path, required=True)
    metadata.add_argument("--sheet", required=True)
    metadata.add_argument("--row-number", type=int, required=True)
    metadata.add_argument("--output-json", type=Path, required=True)

    topic_run = subparsers.add_parser(
        "run-topic",
        help="Run the serial incremental workflow for one complete topic.",
    )
    _add_topic_argument(topic_run)
    visualization = subparsers.add_parser(
        "build-visualization",
        help="Build reusable offline HTML and publication figures for a complete topic.",
    )
    _add_topic_argument(visualization)
    return parser


def _add_topic_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic name used to isolate formal results and recovery state.",
    )


def _add_article_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--author",
        action="append",
        required=True,
        dest="authors",
        help="Repeat for each author in source order.",
    )
    parser.add_argument(
        "--full-text",
        type=Path,
        required=True,
        help="UTF-8 Markdown file containing the cleaned complete article text.",
    )


def _add_relation_paper_arguments(
    parser: argparse.ArgumentParser,
    label: str,
) -> None:
    parser.add_argument(
        f"--paper-{label}-metadata",
        type=Path,
        required=True,
        help=f"UTF-8 JSON PaperMetadata for paper {label.upper()}.",
    )
    parser.add_argument(
        f"--paper-{label}-profile",
        type=Path,
        required=True,
        help=f"UTF-8 JSON PaperProfileContent for paper {label.upper()}.",
    )
    parser.add_argument(
        f"--paper-{label}-full-text",
        type=Path,
        required=True,
        help=f"UTF-8 Markdown full text for paper {label.upper()}.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    topic_started_at = (
        time.perf_counter()
        if args.command == "run-topic"
        else None
    )
    try:
        if args.command == "extract-full-text":
            markdown = write_pdf_markdown(
                input_pdf=args.input_pdf,
                output_markdown=args.output_markdown,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "character_count": len(markdown),
                        "output_markdown": str(args.output_markdown),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "extract-metadata":
            metadata = write_paper_metadata(
                workbook_path=args.workbook,
                sheet_name=args.sheet,
                row_number=args.row_number,
                output_json=args.output_json,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "paper_id": metadata.paper_id,
                        "output_json": str(args.output_json),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        topic_paths = build_topic_result_paths(
            topic=args.topic,
            result_unit_root=RESULT_UNIT_ROOT,
            temp_unit_root=TEMP_UNIT_ROOT,
        )
        topic_input = None
        if args.command in {"run-topic", "build-visualization"}:
            topic_input = discover_topic_inputs(
                project_root=PROJECT_ROOT,
                topic=args.topic,
            )
        if args.command == "build-visualization":
            assert topic_input is not None
            visualization_result = build_topic_visualization(
                topic_input=topic_input,
                paths=topic_paths,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "reused": visualization_result.reused,
                        "input_fingerprint": (
                            visualization_result.input_fingerprint
                        ),
                        "output_paths": [
                            str(path)
                            for path in visualization_result.output_paths
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        config = load_deepseek_config()
        _clear_proxy_environment()
        base_client = DeepSeekJsonClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
        )
        client = (
            TokenTrackingJsonClient(client=base_client)
            if args.command == "run-topic"
            else base_client
        )
        if args.command == "run-topic":
            assert topic_input is not None
            summary = run_topic_orchestration(
                topic_input=topic_input,
                processed_root=PROJECT_ROOT / "data" / "processed",
                model=config.model,
                parser_agent=PaperParserAgent(client=client),
                reviewer_agent=EvidenceReviewerAgent(client=client),
                relation_agent=RelationInferenceAgent(client=client),
                paths=topic_paths,
                current_run_token_usage=lambda: client.token_usage,
                started_at=topic_started_at,
            )
            if summary.status == "relations_complete":
                build_topic_visualization(
                    topic_input=topic_input,
                    paths=topic_paths,
                )
            print(summary.model_dump_json())
            return 0 if summary.status == "relations_complete" else 1
        if args.command == "initial-generation":
            full_text = args.full_text.read_text(encoding="utf-8")
            agent = PaperParserAgent(client=client)
            workflow_result = run_initial_profile_generation(
                paper_id=args.paper_id,
                title=args.title,
                authors=args.authors,
                article_full_text=full_text,
                model=config.model,
                agent=agent,
                result_root=topic_paths.result_root,
                temp_root=topic_paths.temp_root,
            )
        elif args.command == "targeted-revision":
            full_text = args.full_text.read_text(encoding="utf-8")
            agent = PaperParserAgent(client=client)
            current_profile = PaperProfileContent.model_validate_json(
                args.current_profile.read_text(encoding="utf-8")
            )
            review_issues = TypeAdapter(list[ProfileReviewIssue]).validate_json(
                args.review_issues.read_text(encoding="utf-8")
            )
            workflow_result = run_targeted_profile_revision(
                paper_id=args.paper_id,
                title=args.title,
                authors=args.authors,
                article_full_text=full_text,
                current_profile=current_profile,
                review_issues=review_issues,
                model=config.model,
                agent=agent,
                result_root=topic_paths.result_root,
                temp_root=topic_paths.temp_root,
            )
        elif args.command == "profile-review":
            full_text = args.full_text.read_text(encoding="utf-8")
            current_profile = PaperProfileContent.model_validate_json(
                args.current_profile.read_text(encoding="utf-8")
            )
            workflow_result = run_profile_review(
                paper_id=args.paper_id,
                title=args.title,
                authors=args.authors,
                article_full_text=full_text,
                current_profile=current_profile,
                model=config.model,
                agent=EvidenceReviewerAgent(client=client),
                result_root=topic_paths.result_root,
                temp_root=topic_paths.temp_root,
            )
        elif args.command in {
            "relation-inference",
            "relation-review",
            "relation-targeted-revision",
            "relation-post-revision-review",
        }:
            paper_a_metadata = PaperMetadata.model_validate_json(
                args.paper_a_metadata.read_text(encoding="utf-8")
            )
            paper_a_profile = PaperProfileContent.model_validate_json(
                args.paper_a_profile.read_text(encoding="utf-8")
            )
            paper_b_metadata = PaperMetadata.model_validate_json(
                args.paper_b_metadata.read_text(encoding="utf-8")
            )
            paper_b_profile = PaperProfileContent.model_validate_json(
                args.paper_b_profile.read_text(encoding="utf-8")
            )
            pair_inputs = {
                "paper_a_metadata": paper_a_metadata,
                "paper_a_profile": paper_a_profile,
                "paper_a_full_text": args.paper_a_full_text.read_text(
                    encoding="utf-8"
                ),
                "paper_b_metadata": paper_b_metadata,
                "paper_b_profile": paper_b_profile,
                "paper_b_full_text": args.paper_b_full_text.read_text(
                    encoding="utf-8"
                ),
                "model": config.model,
                "result_root": topic_paths.result_root,
                "temp_root": topic_paths.temp_root,
            }
            if args.command == "relation-inference":
                workflow_result = run_initial_relation_inference(
                    agent=RelationInferenceAgent(client=client),
                    **pair_inputs,
                )
            elif args.command == "relation-review":
                current_relation = TypeAdapter(RelationRecord).validate_json(
                    args.current_relation.read_text(encoding="utf-8")
                )
                workflow_result = run_initial_relation_review(
                    current_relation=current_relation,
                    agent=EvidenceReviewerAgent(client=client),
                    **pair_inputs,
                )
            elif args.command == "relation-targeted-revision":
                current_relation = TypeAdapter(RelationRecord).validate_json(
                    args.current_relation.read_text(encoding="utf-8")
                )
                review_issues = TypeAdapter(
                    list[RelationReviewIssue]
                ).validate_json(args.review_issues.read_text(encoding="utf-8"))
                workflow_result = run_targeted_relation_revision(
                    current_relation=current_relation,
                    review_issues=review_issues,
                    agent=RelationInferenceAgent(client=client),
                    **pair_inputs,
                )
            else:
                previous_review_issues = TypeAdapter(
                    list[RelationReviewIssue]
                ).validate_json(
                    args.previous_review_issues.read_text(encoding="utf-8")
                )
                revised_relation = TypeAdapter(
                    RelationRecord
                ).validate_json(
                    args.revised_relation.read_text(encoding="utf-8")
                )
                workflow_result = run_post_revision_relation_review(
                    previous_review_issues=previous_review_issues,
                    revised_relation=revised_relation,
                    agent=EvidenceReviewerAgent(client=client),
                    **pair_inputs,
                )
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (
        OSError,
        RuntimeError,
        UnicodeError,
        ValidationError,
        ValueError,
    ) as error:
        parser.error(str(error))
    except (
        NetworkRetriesExhausted,
        ContentRetriesExhausted,
        ProfileReviewNetworkRetriesExhausted,
        ProfileReviewContentRetriesExhausted,
        RelationInferenceNetworkRetriesExhausted,
        RelationInferenceContentRetriesExhausted,
        RelationReviewNetworkRetriesExhausted,
        RelationReviewContentRetriesExhausted,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "elapsed_seconds": error.metrics.elapsed_seconds,
                    "token_usage": error.metrics.token_usage.model_dump(),
                },
                ensure_ascii=False,
            )
        )
        return 1

    _print_success(workflow_result)
    return 0


def _clear_proxy_environment() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)


def _print_success(
    result: (
        InitialProfileWorkflowResult
        | ProfileReviewWorkflowResult
        | RelationInferenceWorkflowResult
        | RelationReviewWorkflowResult
    ),
) -> None:
    print(
        json.dumps(
            {
                "status": "completed",
                "reused": result.reused,
                "elapsed_seconds": result.checkpoint.metrics.elapsed_seconds,
                "token_usage": result.checkpoint.metrics.token_usage.model_dump(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
