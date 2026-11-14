from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.visualization_orchestration import (
    build_topic_visualization,
)

from test_network_data import _write_complete_inputs


def test_topic_visualization_builds_one_reusable_complete_output_set(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)

    first = build_topic_visualization(topic_input=topic_input, paths=paths)
    second = build_topic_visualization(topic_input=topic_input, paths=paths)

    expected = {
        paths.network_root / "network_data.json",
        paths.network_root / "index.html",
        paths.network_root / "figures" / "strict_network.svg",
        paths.network_root / "figures" / "strict_network.pdf",
        paths.network_root / "figures" / "relation_matrix.svg",
        paths.network_root / "figures" / "relation_matrix.pdf",
        paths.network_root / "figures" / "relation_type_distribution.svg",
        paths.network_root / "figures" / "relation_type_distribution.pdf",
    }
    assert set(first.output_paths) == expected
    assert all(path.is_file() for path in first.output_paths)
    assert first.reused is False
    assert second.reused is True
    assert second.input_fingerprint == first.input_fingerprint


def test_topic_visualization_rejects_incomplete_relation_coverage(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    (paths.relation_outcome_root / "P002__P003.json").unlink()
    (paths.relation_outcome_root / "P001__P003.json").unlink()

    with pytest.raises(ValueError) as error:
        build_topic_visualization(topic_input=topic_input, paths=paths)

    assert "P001__P003" in str(error.value)
    assert "P002__P003" in str(error.value)
    assert not paths.network_root.exists()
