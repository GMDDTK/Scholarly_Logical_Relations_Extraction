from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import TypeAdapter, ValidationError

from .contracts.network_visualization import (
    LayoutPosition,
    NetworkBundle,
    NetworkEdgeView,
    NetworkLayout,
    NetworkNode,
)
from .contracts.paper_profile import PaperProfileContent
from .contracts.relation_outcome import (
    AcceptedRelationOutcome,
    RelationOutcome,
)
from .contracts.relation_record import (
    RelatedPairRecord,
    RelationDirection,
)
from .result_paths import TopicResultPaths
from .topic_input import TopicInputSet


NETWORK_SCHEMA_VERSION = 1
VISUALIZATION_VERSION = 4
LAYOUT_ALGORITHM_VERSION = 2
LAYOUT_SEED = 20260808
LAYOUT_ITERATIONS = 400

_RELATION_OUTCOME_ADAPTER = TypeAdapter(RelationOutcome)


@dataclass(frozen=True)
class NetworkDataBuildResult:
    bundle: NetworkBundle
    output_path: Path
    reused: bool


def build_network_bundle(
    *,
    topic_input: TopicInputSet,
    paths: TopicResultPaths,
) -> NetworkBundle:
    if topic_input.topic != paths.topic:
        raise ValueError("topic input and result paths must use the same topic")
    if not topic_input.is_complete:
        raise ValueError("formal network requires a complete topic input set")

    nodes = _load_nodes(topic_input=topic_input, paths=paths)
    pairs = _load_pairs(topic_input=topic_input, paths=paths)
    fingerprint = _build_input_fingerprint(
        topic=topic_input.topic,
        nodes=nodes,
        pairs=pairs,
    )
    layout = _build_layout(nodes=nodes, pairs=pairs)
    return NetworkBundle(
        schema_version=NETWORK_SCHEMA_VERSION,
        topic=topic_input.topic,
        input_fingerprint=fingerprint,
        nodes=nodes,
        pairs=pairs,
        layout=layout,
    )


def build_projection_edges(
    bundle: NetworkBundle,
    *,
    projection: Literal["strict", "provisional"],
) -> tuple[NetworkEdgeView, ...]:
    edges: list[NetworkEdgeView] = []
    for outcome in bundle.pairs:
        record = outcome.relation_record
        if not isinstance(record, RelatedPairRecord):
            continue
        if projection == "strict" and outcome.review_status != "accepted":
            continue
        if record.direction is RelationDirection.A_TO_B:
            source = record.paper_a_id
            target = record.paper_b_id
            directed = True
        elif record.direction is RelationDirection.B_TO_A:
            source = record.paper_b_id
            target = record.paper_a_id
            directed = True
        else:
            source = record.paper_a_id
            target = record.paper_b_id
            directed = False
        edges.append(
            NetworkEdgeView(
                pair_key=f"{record.paper_a_id}__{record.paper_b_id}",
                source=source,
                target=target,
                directed=directed,
                relation_type=record.relation_type,
                relation_direction=record.direction,
                review_status=outcome.review_status,
            )
        )
    return tuple(edges)


def write_network_data(
    *,
    topic_input: TopicInputSet,
    paths: TopicResultPaths,
) -> NetworkDataBuildResult:
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)
    output_path = paths.network_root / "network_data.json"
    existing = _load_existing_bundle(output_path)
    if existing == bundle:
        return NetworkDataBuildResult(
            bundle=existing,
            output_path=output_path,
            reused=True,
        )

    payload = bundle.model_dump_json(indent=2) + "\n"
    _write_text_atomically(
        destination=output_path,
        temp_root=paths.temp_root / "visualization",
        text=payload,
    )
    return NetworkDataBuildResult(
        bundle=bundle,
        output_path=output_path,
        reused=False,
    )


def _load_nodes(
    *,
    topic_input: TopicInputSet,
    paths: TopicResultPaths,
) -> list[NetworkNode]:
    ordered_matches = sorted(
        topic_input.matches,
        key=lambda item: item.metadata.paper_id,
    )
    missing_profiles = [
        match.metadata.paper_id
        for match in ordered_matches
        if not (
            paths.final_profile_root / f"{match.metadata.paper_id}.json"
        ).is_file()
    ]
    if missing_profiles:
        raise ValueError(f"missing final profiles: {missing_profiles}")

    nodes: list[NetworkNode] = []
    for match in ordered_matches:
        profile_path = (
            paths.final_profile_root / f"{match.metadata.paper_id}.json"
        )
        try:
            profile = PaperProfileContent.model_validate_json(
                profile_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError) as error:
            raise ValueError(
                f"invalid final profile for {match.metadata.paper_id}: {error}"
            ) from error
        metadata = match.metadata
        nodes.append(
            NetworkNode(
                paper_id=metadata.paper_id,
                title=metadata.title,
                authors=metadata.authors,
                publication_date=metadata.publication_date.isoformat(),
                profile=profile,
            )
        )
    return nodes


def _load_pairs(
    *,
    topic_input: TopicInputSet,
    paths: TopicResultPaths,
) -> list[RelationOutcome]:
    ordered_pairs = sorted(topic_input.pairs, key=lambda item: item.pair_key)
    missing_outcomes = [
        pair.pair_key
        for pair in ordered_pairs
        if not (
            paths.relation_outcome_root / f"{pair.pair_key}.json"
        ).is_file()
    ]
    if missing_outcomes:
        raise ValueError(f"missing relation outcomes: {missing_outcomes}")

    outcomes: list[RelationOutcome] = []
    for pair in ordered_pairs:
        outcome_path = paths.relation_outcome_root / f"{pair.pair_key}.json"
        try:
            outcome = _RELATION_OUTCOME_ADAPTER.validate_json(
                outcome_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError) as error:
            raise ValueError(
                f"invalid relation outcome for {pair.pair_key}: {error}"
            ) from error
        record = outcome.relation_record
        actual_key = f"{record.paper_a_id}__{record.paper_b_id}"
        if actual_key != pair.pair_key:
            raise ValueError(
                f"relation outcome key mismatch for {pair.pair_key}: {actual_key}"
            )
        outcomes.append(outcome)
    return outcomes


def _build_input_fingerprint(
    *,
    topic: str,
    nodes: list[NetworkNode],
    pairs: list[RelationOutcome],
) -> str:
    payload = {
        "schema_version": NETWORK_SCHEMA_VERSION,
        "visualization_version": VISUALIZATION_VERSION,
        "topic": topic,
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "pairs": [
            _RELATION_OUTCOME_ADAPTER.dump_python(outcome, mode="json")
            for outcome in pairs
        ],
        "layout": {
            "algorithm": "spring_layout",
            "algorithm_version": LAYOUT_ALGORITHM_VERSION,
            "networkx_version": nx.__version__,
            "seed": LAYOUT_SEED,
            "iterations": LAYOUT_ITERATIONS,
            "basis_projection": "strict",
        },
    }
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(normalized).hexdigest()


def _build_layout(
    *,
    nodes: list[NetworkNode],
    pairs: list[RelationOutcome],
) -> NetworkLayout:
    graph = nx.Graph()
    graph.add_nodes_from(node.paper_id for node in nodes)
    for outcome in pairs:
        record = outcome.relation_record
        if (
            isinstance(outcome, AcceptedRelationOutcome)
            and isinstance(record, RelatedPairRecord)
        ):
            graph.add_edge(record.paper_a_id, record.paper_b_id)

    positions = _component_aware_spring_layout(graph)
    canonical_positions = {
        paper_id: LayoutPosition(
            x=round(float(positions[paper_id][0]), 8),
            y=round(float(positions[paper_id][1]), 8),
        )
        for paper_id in sorted(graph.nodes)
    }
    return NetworkLayout(
        algorithm="spring_layout",
        algorithm_version=LAYOUT_ALGORITHM_VERSION,
        networkx_version=nx.__version__,
        seed=LAYOUT_SEED,
        basis_projection="strict",
        positions=canonical_positions,
    )


def _component_aware_spring_layout(
    graph: nx.Graph,
) -> dict[str, tuple[float, float]]:
    components = sorted(
        (tuple(sorted(component)) for component in nx.connected_components(graph)),
        key=lambda component: (-len(component), component),
    )
    if len(components) == 1:
        return _spring_component(graph, components[0])

    largest = components[0]
    if len(largest) / graph.number_of_nodes() < 0.6:
        return _spring_component(graph, tuple(sorted(graph.nodes)))

    main_positions = _spring_component(graph, largest)
    positions: dict[str, tuple[float, float]] = {
        paper_id: (
            0.92 * local_x,
            0.70 * local_y + 0.12,
        )
        for paper_id, (local_x, local_y) in main_positions.items()
    }
    satellites = components[1:]
    slot_width = 1.8 / len(satellites)
    for index, component in enumerate(satellites):
        center_x = -0.9 + slot_width * (index + 0.5)
        if len(component) == 1:
            positions[component[0]] = (center_x, -0.92)
            continue
        local_positions = _spring_component(graph, component)
        x_scale = min(0.34, slot_width * 0.38)
        for paper_id, (local_x, local_y) in local_positions.items():
            positions[paper_id] = (
                center_x + x_scale * local_x,
                -0.86 + 0.10 * local_y,
            )
    return positions


def _spring_component(
    graph: nx.Graph,
    component: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    if len(component) == 1:
        return {component[0]: (0.0, 0.0)}
    component_graph = graph.subgraph(component)
    raw_positions = nx.spring_layout(
        component_graph,
        seed=LAYOUT_SEED,
        iterations=LAYOUT_ITERATIONS,
        method="force",
        k=2.0 / math.sqrt(len(component)),
        scale=1.0,
        center=(0.0, 0.0),
    )
    return {
        paper_id: (
            float(raw_positions[paper_id][0]),
            float(raw_positions[paper_id][1]),
        )
        for paper_id in component
    }


def _load_existing_bundle(path: Path) -> NetworkBundle | None:
    try:
        return NetworkBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError):
        return None


def _write_text_atomically(
    *,
    destination: Path,
    temp_root: Path,
    text: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
