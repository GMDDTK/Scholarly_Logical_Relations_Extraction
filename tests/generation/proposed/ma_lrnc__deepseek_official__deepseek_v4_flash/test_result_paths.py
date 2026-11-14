from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.result_paths import (
    build_topic_result_paths,
)


def test_topic_result_paths_group_all_outputs_under_one_topic(
    tmp_path: Path,
) -> None:
    paths = build_topic_result_paths(
        topic="信息素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "tmp" / "unit",
    )

    assert paths.result_root == (
        tmp_path / "results" / "unit" / "topics" / "信息素养"
    )
    assert paths.temp_root == (
        tmp_path / "tmp" / "unit" / "topics" / "信息素养"
    )
    assert paths.checkpoint_root == paths.result_root / "checkpoints"
    assert paths.failure_root == paths.result_root / "failures"
    assert paths.final_profile_root == paths.result_root / "final_profiles"
    assert paths.relation_outcome_root == (
        paths.result_root / "relation_outcomes"
    )
    assert paths.network_root == paths.result_root / "network"


def test_same_business_key_is_isolated_between_topics(tmp_path: Path) -> None:
    information_literacy = build_topic_result_paths(
        topic="信息素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "tmp" / "unit",
    )
    data_literacy = build_topic_result_paths(
        topic="数据素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "tmp" / "unit",
    )

    information_checkpoint = (
        information_literacy.checkpoint_root
        / "paper_profile_initial_generation"
        / "P016.json"
    )
    data_checkpoint = (
        data_literacy.checkpoint_root
        / "paper_profile_initial_generation"
        / "P016.json"
    )

    assert information_checkpoint != data_checkpoint
    assert information_literacy.result_root not in data_checkpoint.parents
    assert data_literacy.result_root not in information_checkpoint.parents


@pytest.mark.parametrize(
    "topic",
    [
        "",
        " ",
        ".",
        "..",
        "../信息素养",
        r"..\信息素养",
        "信息/素养",
        r"信息\素养",
        "C:信息素养",
        "信息素养.",
        "信息素养 ",
        "CON",
        "nul.txt",
    ],
)
def test_topic_result_paths_reject_unsafe_topic_names(
    tmp_path: Path,
    topic: str,
) -> None:
    with pytest.raises(ValueError, match="topic"):
        build_topic_result_paths(
            topic=topic,
            result_unit_root=tmp_path / "results" / "unit",
            temp_unit_root=tmp_path / "tmp" / "unit",
        )


def test_topic_result_path_construction_does_not_create_directories(
    tmp_path: Path,
) -> None:
    paths = build_topic_result_paths(
        topic="信息素养",
        result_unit_root=tmp_path / "results" / "unit",
        temp_unit_root=tmp_path / "tmp" / "unit",
    )

    assert not paths.result_root.exists()
    assert not paths.temp_root.exists()
