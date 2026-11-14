from datetime import date
from pathlib import Path

import openpyxl
import pytest

from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.topic_input import (
    TopicInputError,
    discover_topic_inputs,
)


HEADERS = ("Title-题名", "Author-作者", "PubTime-发表时间")


def _write_catalog(
    project_root: Path,
    topic: str,
    count: int = 50,
    *,
    first_publication_time: str = "2020-01-01",
) -> None:
    directory = project_root / "data" / "题录"
    directory.mkdir(parents=True)
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = topic
    worksheet.append(HEADERS)
    for index in range(1, count + 1):
        publication_time = first_publication_time if index == 1 else "2020-01-01"
        worksheet.append((f"测试论文{index:02d}", f"作者{index}", publication_time))
    workbook.save(directory / f"{topic}50高被引.xlsx")
    workbook.close()


def _write_reference_workbooks(
    project_root: Path,
    topic: str,
    count: int = 50,
) -> None:
    directory = project_root / "data" / "参考文献" / topic
    directory.mkdir(parents=True)
    for index in range(1, count + 1):
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        if index == 2:
            worksheet.append(("[1] 作者1. 测试论文01[J]. 测试期刊, 2020.",))
            worksheet.append(("[2] 主题外论文[J].",))
        workbook.save(directory / f"测试论文{index:02d}.xlsx")
        workbook.close()


def test_discovers_exactly_fifty_catalog_papers_and_reference_files(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path, "测试主题")
    _write_reference_workbooks(tmp_path, "测试主题")

    topic_input = discover_topic_inputs(
        project_root=tmp_path,
        topic="测试主题",
    )

    assert len(topic_input.papers) == 50
    assert topic_input.papers[0].paper_id == "P001"
    assert topic_input.papers[0].publication_date == date(2020, 1, 1)
    assert topic_input.reference_documents[0].paper.paper_id == "P001"
    assert topic_input.reference_documents[0].lines == ()
    assert [line.row_number for line in topic_input.reference_documents[1].lines] == [1, 2]
    assert topic_input.reference_documents[1].lines[0].raw_text.startswith("[1]")


def test_filename_normalization_maps_only_exact_unique_catalog_title(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path, "测试主题")
    _write_reference_workbooks(tmp_path, "测试主题")
    reference_directory = tmp_path / "data" / "参考文献" / "测试主题"
    (reference_directory / "测试论文01.xlsx").rename(
        reference_directory / "《测试论文 01》.xlsx"
    )

    topic_input = discover_topic_inputs(project_root=tmp_path, topic="测试主题")

    assert topic_input.reference_documents[0].workbook_path.name == "《测试论文 01》.xlsx"


def test_wrong_reference_file_count_fails_before_model_work(tmp_path: Path) -> None:
    _write_catalog(tmp_path, "测试主题")
    _write_reference_workbooks(tmp_path, "测试主题", count=49)

    with pytest.raises(TopicInputError, match="exactly 50 reference workbooks"):
        discover_topic_inputs(project_root=tmp_path, topic="测试主题")


def test_catalog_datetime_text_is_normalized_to_date(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        "测试主题",
        first_publication_time="2024-02-20 10:47",
    )
    _write_reference_workbooks(tmp_path, "测试主题")

    topic_input = discover_topic_inputs(project_root=tmp_path, topic="测试主题")

    assert topic_input.papers[0].publication_date == date(2024, 2, 20)


def test_duplicate_normalized_source_title_is_rejected(tmp_path: Path) -> None:
    _write_catalog(tmp_path, "测试主题")
    _write_reference_workbooks(tmp_path, "测试主题")
    reference_directory = tmp_path / "data" / "参考文献" / "测试主题"
    (reference_directory / "测试论文02.xlsx").rename(
        reference_directory / "《测试论文01》.xlsx"
    )

    with pytest.raises(TopicInputError, match="unique catalog paper"):
        discover_topic_inputs(project_root=tmp_path, topic="测试主题")
