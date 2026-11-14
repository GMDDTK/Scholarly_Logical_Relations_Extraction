from datetime import datetime
from pathlib import Path

import openpyxl
import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.topic_input import (
    TopicInputError,
    discover_topic_inputs,
)


HEADERS = ("Title-题名", "Author-作者", "PubTime-发表时间")


def _create_topic_directory(project_root: Path, topic: str) -> Path:
    topic_directory = project_root / "data" / "论文" / topic
    topic_directory.mkdir(parents=True)
    return topic_directory


def _create_pdf(topic_directory: Path, filename: str) -> Path:
    path = topic_directory / filename
    path.write_bytes(b"%PDF-1.4 test placeholder")
    return path


def _create_catalog(
    project_root: Path,
    topic: str,
    rows: list[tuple[object, ...]],
    *,
    headers: tuple[str, ...] = HEADERS,
    filename: str | None = None,
    extra_visible_sheet: bool = False,
    extra_hidden_sheet: bool = False,
) -> Path:
    catalog_directory = project_root / "data" / "题录"
    catalog_directory.mkdir(parents=True, exist_ok=True)
    path = catalog_directory / (filename or f"{topic}50高被引.xlsx")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = f"{topic}50高被引"
    worksheet.append(headers)
    for row in rows:
        worksheet.append(row)
    if extra_visible_sheet:
        workbook.create_sheet("其他可见表")
    if extra_hidden_sheet:
        hidden = workbook.create_sheet("隐藏辅助表")
        hidden.sheet_state = "hidden"
    workbook.save(path)
    workbook.close()
    return path


def test_discovers_topic_inputs_by_exact_normalized_title_and_author(
    tmp_path: Path,
) -> None:
    topic = "信息素养"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(
        topic_directory,
        "“翻转课堂”应用于信息素养教育课程的实证研究_隆茜.pdf",
    )
    _create_pdf(
        topic_directory,
        "公众健康信息素养促进中的图书馆参与_驱动因素、国外实践及思考_邓胜利.pdf",
    )
    workbook_path = _create_catalog(
        tmp_path,
        topic,
        [
            (
                "公众健康信息素养促进中的图书馆参与:驱动因素、国外实践及思考",
                "邓胜利; 付少雄",
                datetime(2018, 4, 11, 10, 0),
            ),
            (
                "“翻转课堂”应用于信息素养教育课程的实证研究",
                "隆茜",
                "2014-11-21",
            ),
        ],
        extra_hidden_sheet=True,
    )

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert discovered.topic_directory == topic_directory
    assert discovered.workbook_path == workbook_path
    assert discovered.worksheet_name == "信息素养50高被引"
    assert [item.metadata.paper_id for item in discovered.matches] == [
        "P001",
        "P002",
    ]
    assert discovered.matches[0].metadata.authors == ["邓胜利", "付少雄"]
    assert discovered.failures == ()
    assert [pair.pair_key for pair in discovered.pairs] == ["P001__P002"]
    assert discovered.is_complete is True


def test_discovers_optional_source_publication_date(tmp_path: Path) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "译文_张三.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [("译文", "张三", "2015-11-21", "2015-02-05")],
        headers=HEADERS + ("SourcePubTime-源文献发表时间",),
    )

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert discovered.matches[0].metadata.publication_date.isoformat() == "2015-11-21"
    assert (
        discovered.matches[0].metadata.source_publication_date.isoformat()
        == "2015-02-05"
    )


def test_record_failure_does_not_discard_other_valid_matches(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "有效论文_张三.pdf")
    _create_pdf(topic_directory, "另一篇研究_李四.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [
            ("有效论文", "张三", "2020-01-01"),
            ("另一篇论文", "李四", "2021-01-01"),
        ],
    )

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert [item.metadata.paper_id for item in discovered.matches] == ["P001"]
    assert len(discovered.failures) == 1
    failure = discovered.failures[0]
    assert failure.pdf_path.name == "另一篇研究_李四.pdf"
    assert failure.code == "no_exact_title_match"
    assert any(
        candidate.paper_id == "P002"
        for candidate in failure.diagnostic_candidates
    )
    assert discovered.pairs == ()
    assert discovered.is_complete is False


def test_exact_title_with_mismatched_filename_author_is_record_failure(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "同名论文_李四.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [("同名论文", "张三", "2020-01-01")],
    )

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert discovered.matches == ()
    assert len(discovered.failures) == 1
    assert discovered.failures[0].code == "author_mismatch"


def test_invalid_values_in_one_matching_catalog_row_are_record_failure(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "有效论文_张三.pdf")
    _create_pdf(topic_directory, "日期缺失论文_李四.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [
            ("有效论文", "张三", "2020-01-01"),
            ("日期缺失论文", "李四", None),
        ],
    )

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert [item.metadata.paper_id for item in discovered.matches] == ["P001"]
    assert len(discovered.failures) == 1
    assert discovered.failures[0].code == "invalid_catalog_values"


def test_two_pdfs_cannot_claim_the_same_catalog_row(tmp_path: Path) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "同名论文_张三.pdf")
    _create_pdf(topic_directory, "同名论文_李四.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [("同名论文", "张三; 李四", "2020-01-01")],
    )

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert discovered.matches == ()
    assert len(discovered.failures) == 2
    assert {failure.code for failure in discovered.failures} == {
        "duplicate_catalog_assignment"
    }


def test_missing_topic_directory_is_theme_level_error(tmp_path: Path) -> None:
    with pytest.raises(TopicInputError, match="Topic PDF directory"):
        discover_topic_inputs(project_root=tmp_path, topic="不存在")


def test_multiple_catalog_workbooks_are_theme_level_error(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "测试论文_张三.pdf")
    _create_catalog(tmp_path, topic, [], filename=f"{topic}A.xlsx")
    _create_catalog(tmp_path, topic, [], filename=f"{topic}B.xlsx")

    with pytest.raises(TopicInputError, match="exactly one catalog workbook"):
        discover_topic_inputs(project_root=tmp_path, topic=topic)


def test_excel_lock_file_is_ignored_when_discovering_workbook(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "测试论文_张三.pdf")
    workbook_path = _create_catalog(
        tmp_path,
        topic,
        [("测试论文", "张三", "2020-01-01")],
    )
    lock_file = workbook_path.parent / f"~${topic}编辑中.xlsx"
    lock_file.write_bytes(b"not-a-real-workbook")

    discovered = discover_topic_inputs(project_root=tmp_path, topic=topic)

    assert discovered.workbook_path == workbook_path
    assert [item.metadata.paper_id for item in discovered.matches] == ["P001"]


def test_multiple_visible_worksheets_are_theme_level_error(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "测试论文_张三.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [],
        extra_visible_sheet=True,
    )

    with pytest.raises(TopicInputError, match="exactly one visible worksheet"):
        discover_topic_inputs(project_root=tmp_path, topic=topic)


def test_missing_required_catalog_header_is_theme_level_error(
    tmp_path: Path,
) -> None:
    topic = "测试主题"
    topic_directory = _create_topic_directory(tmp_path, topic)
    _create_pdf(topic_directory, "测试论文_张三.pdf")
    _create_catalog(
        tmp_path,
        topic,
        [],
        headers=("Title-题名", "Author-作者"),
    )

    with pytest.raises(TopicInputError, match="PubTime-发表时间"):
        discover_topic_inputs(project_root=tmp_path, topic=topic)
