from __future__ import annotations

import argparse
import json
import sys
import time

from .config import (
    EXPERIMENT_UNIT,
    PROJECT_ROOT,
    ConfigurationError,
    load_deepseek_config,
)
from .contracts import TokenUsage
from .llm_client import DeepSeekJsonClient
from .network import IncompleteTopicError, build_network_bundle, write_network_artifacts
from .result_paths import build_topic_result_paths
from .topic_input import TopicInputError, discover_topic_inputs
from .workflow import completed_result_chain_metrics, process_paper


RESULT_UNIT_ROOT = (
    PROJECT_ROOT / "results" / "generation" / "baselines" / EXPERIMENT_UNIT
)
TEMP_UNIT_ROOT = (
    PROJECT_ROOT / "tmp" / "resume" / "generation" / "baselines" / EXPERIMENT_UNIT
)


def run_topic(*, topic: str) -> dict[str, object]:
    started = time.perf_counter()
    topic_input = discover_topic_inputs(project_root=PROJECT_ROOT, topic=topic)
    config = load_deepseek_config()
    client = DeepSeekJsonClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
    )
    paths = build_topic_result_paths(
        topic=topic_input.topic,
        result_unit_root=RESULT_UNIT_ROOT,
        temp_unit_root=TEMP_UNIT_ROOT,
    )
    results = []
    current_usage = TokenUsage.zero()
    model_stage_elapsed = 0.0
    for document in topic_input.reference_documents:
        try:
            result = process_paper(
                topic_input=topic_input,
                document=document,
                client=client,
                paths=paths,
                model=config.model,
            )
        except Exception as error:
            results.append(
                {
                    "paper_id": document.paper.paper_id,
                    "status": "failed",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            continue
        current_usage = current_usage.plus(
            result.current_run_metrics.token_usage
        )
        model_stage_elapsed += result.current_run_metrics.elapsed_seconds
        results.append(
            {
                "paper_id": result.paper_id,
                "status": result.status,
                "message": result.message,
            }
        )
    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("accepted", "skipped", "needs_retry", "failed")
    }
    complete = counts["accepted"] + counts["skipped"] == 50
    network = None
    if complete:
        bundle = build_network_bundle(
            topic_input=topic_input,
            paths=paths,
            model=config.model,
        )
        write_network_artifacts(bundle=bundle, paths=paths)
        network = {
            "node_count": len(bundle.nodes),
            "edge_count": len(bundle.edges),
            "input_fingerprint": bundle.input_fingerprint,
        }
    completed_metrics = completed_result_chain_metrics(
        topic_input=topic_input,
        paths=paths,
        model=config.model,
    )
    return {
        "topic": topic_input.topic,
        "status": "complete" if complete else "incomplete",
        "counts": counts,
        "papers": results,
        "papers_without_references": [
            {
                "paper_id": document.paper.paper_id,
                "title": document.paper.title,
            }
            for document in topic_input.reference_documents
            if not document.lines
        ],
        "network": network,
        "current_run": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model_stage_elapsed_seconds": round(model_stage_elapsed, 3),
            "token_usage": current_usage.model_dump(mode="json"),
        },
        "completed_result_chain": completed_metrics.model_dump(mode="json"),
    }


def run_paper(*, topic: str, paper_id: str) -> dict[str, object]:
    started = time.perf_counter()
    topic_input = discover_topic_inputs(project_root=PROJECT_ROOT, topic=topic)
    documents = {
        document.paper.paper_id: document
        for document in topic_input.reference_documents
    }
    document = documents.get(paper_id)
    if document is None:
        raise ValueError(
            f"paper ID must be one of P001 through P050; received {paper_id!r}"
        )

    config = load_deepseek_config()
    client = DeepSeekJsonClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
    )
    paths = build_topic_result_paths(
        topic=topic_input.topic,
        result_unit_root=RESULT_UNIT_ROOT,
        temp_unit_root=TEMP_UNIT_ROOT,
    )
    result = process_paper(
        topic_input=topic_input,
        document=document,
        client=client,
        paths=paths,
        model=config.model,
    )
    completed_metrics = completed_result_chain_metrics(
        topic_input=topic_input,
        paths=paths,
        model=config.model,
    )
    return {
        "topic": topic_input.topic,
        "paper_id": document.paper.paper_id,
        "title": document.paper.title,
        "workbook": document.workbook_path.name,
        "reference_count": len(document.lines),
        "status": result.status,
        "message": result.message,
        "has_references": (
            result.outcome.has_references if result.outcome is not None else None
        ),
        "cited_papers": (
            [item.model_dump(mode="json") for item in result.outcome.cited_papers]
            if result.outcome is not None
            else None
        ),
        "outcome_path": (
            str(paths.outcome(document.paper.paper_id))
            if result.outcome is not None
            else None
        ),
        "current_run": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model_stage_elapsed_seconds": round(
                result.current_run_metrics.elapsed_seconds,
                3,
            ),
            "token_usage": result.current_run_metrics.token_usage.model_dump(
                mode="json"
            ),
        },
        "completed_result_chain": completed_metrics.model_dump(mode="json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a within-topic direct citation network")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_topic_parser = subparsers.add_parser("run-topic")
    run_topic_parser.add_argument("--topic", required=True)
    run_paper_parser = subparsers.add_parser("run-paper")
    run_paper_parser.add_argument("--topic", required=True)
    run_paper_parser.add_argument("--paper-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run-topic":
            summary = run_topic(topic=args.topic)
        elif args.command == "run-paper":
            summary = run_paper(topic=args.topic, paper_id=args.paper_id)
        else:
            raise AssertionError("unreachable")
    except (
        ConfigurationError,
        TopicInputError,
        IncompleteTopicError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {"status": "input_error", "message": str(error)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    successful_statuses = (
        {"complete"} if args.command == "run-topic" else {"accepted", "skipped"}
    )
    return 0 if summary["status"] in successful_statuses else 1


if __name__ == "__main__":
    sys.exit(main())
