import hashlib
import warnings
from pathlib import Path

from pypdf import PdfReader

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.publication_visualization import (
    render_publication_figures,
)

from test_network_data import _write_complete_inputs
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.network_visualization import (
    build_network_bundle,
)


EXPECTED_FIGURE_NAMES = {
    "strict_network.svg",
    "strict_network.pdf",
    "relation_matrix.svg",
    "relation_matrix.pdf",
    "relation_type_distribution.svg",
    "relation_type_distribution.pdf",
}


def test_publication_figures_are_vector_outputs_with_traceable_metadata(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)

    result = render_publication_figures(
        bundle=bundle,
        output_root=paths.network_root / "figures",
        temp_root=paths.temp_root / "visualization" / "figures",
    )

    assert result.reused is False
    assert {path.name for path in result.output_paths} == EXPECTED_FIGURE_NAMES
    for output_path in result.output_paths:
        assert output_path.is_file()
        assert output_path.stat().st_size > 500
        if output_path.suffix == ".svg":
            svg = output_path.read_text(encoding="utf-8")
            assert "<svg" in svg
            assert bundle.input_fingerprint in svg
            assert "测试主题" in svg
        else:
            reader = PdfReader(output_path)
            assert len(reader.pages) == 1
            assert reader.metadata is not None
            assert bundle.input_fingerprint in (reader.metadata.subject or "")


def test_publication_figures_reuse_only_matching_complete_set(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)
    first = render_publication_figures(
        bundle=bundle,
        output_root=paths.network_root / "figures",
        temp_root=paths.temp_root / "visualization" / "figures",
    )
    mtimes = {path.name: path.stat().st_mtime_ns for path in first.output_paths}

    second = render_publication_figures(
        bundle=bundle,
        output_root=paths.network_root / "figures",
        temp_root=paths.temp_root / "visualization" / "figures",
    )

    assert second.reused is True
    assert {
        path.name: path.stat().st_mtime_ns for path in second.output_paths
    } == mtimes


def test_publication_figures_replace_invalid_partial_set(tmp_path: Path) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)
    first = render_publication_figures(
        bundle=bundle,
        output_root=paths.network_root / "figures",
        temp_root=paths.temp_root / "visualization" / "figures",
    )
    damaged = next(path for path in first.output_paths if path.name == "strict_network.pdf")
    damaged.write_bytes(b"not-a-pdf")

    second = render_publication_figures(
        bundle=bundle,
        output_root=paths.network_root / "figures",
        temp_root=paths.temp_root / "visualization" / "figures",
    )

    assert second.reused is False
    assert len(PdfReader(damaged).pages) == 1
    assert not list(paths.temp_root.rglob("*.tmp.*"))


def test_publication_figure_bytes_are_deterministic_after_rebuild(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)
    output_root = paths.network_root / "figures"
    temp_root = paths.temp_root / "visualization" / "figures"

    first = render_publication_figures(
        bundle=bundle,
        output_root=output_root,
        temp_root=temp_root,
    )
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.output_paths
    }
    for path in first.output_paths:
        path.unlink()

    second = render_publication_figures(
        bundle=bundle,
        output_root=output_root,
        temp_root=temp_root,
    )
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.output_paths
    }

    assert second.reused is False
    assert second_hashes == first_hashes


def test_publication_symmetric_edges_emit_no_networkx_warnings(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    relation_path = paths.relation_outcome_root / "P001__P002.json"
    relation_path.write_text(
        relation_path.read_text(encoding="utf-8")
        .replace('"拓展"', '"互补"')
        .replace('"a_to_b"', '"symmetric"'),
        encoding="utf-8",
    )
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render_publication_figures(
            bundle=bundle,
            output_root=paths.network_root / "figures",
            temp_root=paths.temp_root / "visualization" / "figures",
        )
