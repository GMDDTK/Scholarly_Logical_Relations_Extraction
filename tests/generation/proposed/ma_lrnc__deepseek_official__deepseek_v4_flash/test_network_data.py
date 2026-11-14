import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.network_visualization import (
    NetworkBundle,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_outcome import (
    RelationOutcome,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.network_visualization import (
    build_network_bundle,
    build_projection_edges,
    write_network_data,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.result_paths import (
    build_topic_result_paths,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.topic_input import (
    PaperInputMatch,
    PaperPairInput,
    TopicInputSet,
)


def _profile(label: str) -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[
            {
                "statement": f"{label}研究问题",
                "evidence_quotes": [f"{label}问题证据"],
            }
        ],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


def _metadata(paper_id: str) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}题名",
        authors=[f"{paper_id}作者"],
        publication_date=date(2020 + int(paper_id[-1]), 1, 1),
    )


def _topic_input(tmp_path: Path) -> TopicInputSet:
    matches = tuple(
        PaperInputMatch(
            pdf_path=tmp_path / f"{paper_id}.pdf",
            catalog_row=index + 2,
            metadata=_metadata(paper_id),
        )
        for index, paper_id in enumerate(("P001", "P002", "P003"))
    )
    pairs = tuple(
        PaperPairInput(
            pair_key=(
                f"{matches[left].metadata.paper_id}__"
                f"{matches[right].metadata.paper_id}"
            ),
            paper_a=matches[left],
            paper_b=matches[right],
        )
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    return TopicInputSet(
        topic="测试主题",
        topic_directory=tmp_path,
        workbook_path=tmp_path / "catalog.xlsx",
        worksheet_name="Sheet1",
        pdf_paths=tuple(match.pdf_path for match in matches),
        matches=matches,
        failures=(),
        pairs=pairs,
    )


def _paths(tmp_path: Path):
    return build_topic_result_paths(
        topic="测试主题",
        result_unit_root=tmp_path / "results",
        temp_unit_root=tmp_path / "tmp",
    )


def _write_complete_inputs(tmp_path: Path) -> tuple[TopicInputSet, object]:
    topic_input = _topic_input(tmp_path)
    paths = _paths(tmp_path)
    paths.final_profile_root.mkdir(parents=True)
    paths.relation_outcome_root.mkdir(parents=True)
    for match in topic_input.matches:
        profile_path = paths.final_profile_root / f"{match.metadata.paper_id}.json"
        profile_path.write_text(
            _profile(match.metadata.paper_id).model_dump_json(indent=2),
            encoding="utf-8",
        )

    outcomes = {
        "P001__P002": {
            "review_status": "accepted",
            "relation_record": {
                "record_type": "related",
                "paper_a_id": "P001",
                "paper_b_id": "P002",
                "relation_type": "拓展",
                "direction": "a_to_b",
                "relation_explanation": "P001 对 P002 形成拓展。",
                "paper_a_evidence_quotes": ["P001证据"],
                "paper_b_evidence_quotes": ["P002证据"],
            },
        },
        "P001__P003": {
            "review_status": "manual_review_required",
            "relation_record": {
                "record_type": "related",
                "paper_a_id": "P001",
                "paper_b_id": "P003",
                "relation_type": "挑战",
                "direction": "b_to_a",
                "relation_explanation": "P003 对 P001 形成挑战，仍需人工复核。",
                "paper_a_evidence_quotes": ["P001待复核证据"],
                "paper_b_evidence_quotes": ["P003待复核证据"],
            },
            "outstanding_issues": [
                {
                    "target_fields": ["relation_type"],
                    "issue_description": "挑战与修正边界仍需人工判断。",
                    "paper_a_review_evidence_quotes": [],
                    "paper_b_review_evidence_quotes": [],
                    "revision_guidance": "由人工结合全文确定最终类别。",
                }
            ],
        },
        "P002__P003": {
            "review_status": "accepted",
            "relation_record": {
                "record_type": "unrelated",
                "paper_a_id": "P002",
                "paper_b_id": "P003",
                "no_relation_reason": "只有主题相似，没有实质知识动作。",
            },
        },
    }
    adapter = TypeAdapter(RelationOutcome)
    for pair_key, payload in outcomes.items():
        outcome = adapter.validate_python(payload)
        (paths.relation_outcome_root / f"{pair_key}.json").write_text(
            adapter.dump_json(outcome, indent=2).decode("utf-8"),
            encoding="utf-8",
        )
    return topic_input, paths


def test_network_bundle_aggregates_complete_topic_and_stable_layout(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)

    first = build_network_bundle(topic_input=topic_input, paths=paths)
    second = build_network_bundle(topic_input=topic_input, paths=paths)

    assert first == second
    assert first.schema_version == 1
    assert first.topic == "测试主题"
    assert first.input_fingerprint.startswith("sha256:")
    assert [node.paper_id for node in first.nodes] == ["P001", "P002", "P003"]
    assert len(first.pairs) == 3
    assert first.layout.basis_projection == "strict"
    assert first.layout.seed > 0
    assert set(first.layout.positions) == {"P001", "P002", "P003"}


def test_layout_reserves_a_separate_band_for_isolated_nodes(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)

    bundle = build_network_bundle(topic_input=topic_input, paths=paths)

    isolate = bundle.layout.positions["P003"]
    connected = [
        bundle.layout.positions[paper_id]
        for paper_id in ("P001", "P002")
    ]
    assert bundle.layout.algorithm_version == 2
    assert isolate.y <= -0.85
    assert all(position.y >= -0.7 for position in connected)


def test_projection_edges_preserve_review_status_and_direction(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)

    strict = build_projection_edges(bundle, projection="strict")
    provisional = build_projection_edges(bundle, projection="provisional")

    assert [edge.pair_key for edge in strict] == ["P001__P002"]
    assert strict[0].source == "P001"
    assert strict[0].target == "P002"
    assert strict[0].review_status == "accepted"
    assert [edge.pair_key for edge in provisional] == [
        "P001__P002",
        "P001__P003",
    ]
    manual_edge = provisional[1]
    assert manual_edge.source == "P003"
    assert manual_edge.target == "P001"
    assert manual_edge.review_status == "manual_review_required"


def test_network_bundle_rejects_missing_pair_outcome(tmp_path: Path) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    (paths.relation_outcome_root / "P002__P003.json").unlink()

    with pytest.raises(ValueError, match="P002__P003"):
        build_network_bundle(topic_input=topic_input, paths=paths)


def test_network_bundle_contract_rejects_extra_fields(tmp_path: Path) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)
    payload = bundle.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        NetworkBundle.model_validate(payload)


def test_network_data_write_is_atomic_and_reuses_valid_output(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)

    first = write_network_data(topic_input=topic_input, paths=paths)
    second = write_network_data(topic_input=topic_input, paths=paths)

    assert first.reused is False
    assert second.reused is True
    assert first.output_path == paths.network_root / "network_data.json"
    payload = json.loads(first.output_path.read_text(encoding="utf-8"))
    assert payload["topic"] == "测试主题"
    assert not list(paths.temp_root.rglob("*.tmp"))


def test_network_data_fingerprint_changes_with_profile_content(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    first = write_network_data(topic_input=topic_input, paths=paths)
    profile_path = paths.final_profile_root / "P001.json"
    profile_path.write_text(
        _profile("P001更新").model_dump_json(indent=2),
        encoding="utf-8",
    )

    second = write_network_data(topic_input=topic_input, paths=paths)

    assert second.reused is False
    assert second.bundle.input_fingerprint != first.bundle.input_fingerprint
