from datetime import datetime
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.metadata_preprocessing import (
    load_paper_metadata,
    paper_id_from_catalog_row,
    write_paper_metadata,
)


class FakeWorksheet:
    def __init__(self, rows: dict[int, tuple[object, ...]]) -> None:
        self._rows = rows

    def iter_rows(
        self,
        *,
        min_row: int,
        max_row: int,
        values_only: bool,
    ) -> list[tuple[object, ...]]:
        assert min_row == max_row
        assert values_only is True
        return [self._rows[min_row]]


class FakeWorkbook:
    def __init__(self, worksheet: FakeWorksheet) -> None:
        self._worksheet = worksheet
        self.closed = False

    def __getitem__(self, name: str) -> FakeWorksheet:
        if name != "信息素养50高被引":
            raise KeyError(name)
        return self._worksheet

    def close(self) -> None:
        self.closed = True


def test_paper_id_uses_one_based_catalog_data_order() -> None:
    assert paper_id_from_catalog_row(2) == "P001"
    assert paper_id_from_catalog_row(25) == "P024"
    assert paper_id_from_catalog_row(33) == "P032"


def test_paper_id_rejects_header_row() -> None:
    with pytest.raises(ValueError, match="data row"):
        paper_id_from_catalog_row(1)


def test_load_paper_metadata_reads_required_columns_and_normalizes_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_path = tmp_path / "catalog.xlsx"
    workbook_path.write_bytes(b"xlsx-placeholder")
    worksheet = FakeWorksheet(
        {
            1: (
                "序号",
                "Title-题名",
                "Author-作者",
                "其他字段",
                "PubTime-发表时间",
            ),
            25: (
                24,
                " 数据素养能力指标体系构建及高校师生数据素养能力现状调查与分析 ",
                "隆茜； 王晓光",
                "忽略",
                datetime(2015, 12, 9, 11, 4),
            ),
        }
    )
    workbook = FakeWorkbook(worksheet)
    monkeypatch.setattr(
        "openpyxl.load_workbook",
        lambda *args, **kwargs: workbook,
    )

    metadata = load_paper_metadata(
        workbook_path=workbook_path,
        sheet_name="信息素养50高被引",
        row_number=25,
    )

    assert metadata.model_dump(mode="json", exclude_none=True) == {
        "paper_id": "P024",
        "title": "数据素养能力指标体系构建及高校师生数据素养能力现状调查与分析",
        "authors": ["隆茜", "王晓光"],
        "publication_date": "2015-12-09",
    }
    assert workbook.closed is True


def test_load_paper_metadata_rejects_missing_required_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_path = tmp_path / "catalog.xlsx"
    workbook_path.write_bytes(b"xlsx-placeholder")
    workbook = FakeWorkbook(
        FakeWorksheet(
            {
                1: ("Title-题名", "Author-作者"),
                2: ("测试论文", "张三"),
            }
        )
    )
    monkeypatch.setattr(
        "openpyxl.load_workbook",
        lambda *args, **kwargs: workbook,
    )

    with pytest.raises(ValueError, match="PubTime-发表时间"):
        load_paper_metadata(
            workbook_path=workbook_path,
            sheet_name="信息素养50高被引",
            row_number=2,
        )

    assert workbook.closed is True


def test_write_paper_metadata_preserves_chinese_and_iso_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_path = tmp_path / "catalog.xlsx"
    workbook_path.write_bytes(b"xlsx-placeholder")
    worksheet = FakeWorksheet(
        {
            1: ("Title-题名", "Author-作者", "PubTime-发表时间"),
            33: ("“翻转课堂”应用于信息素养教育课程的实证研究", "隆茜", "2014-11-21"),
        }
    )
    monkeypatch.setattr(
        "openpyxl.load_workbook",
        lambda *args, **kwargs: FakeWorkbook(worksheet),
    )
    output_json = tmp_path / "metadata.json"

    metadata = write_paper_metadata(
        workbook_path=workbook_path,
        sheet_name="信息素养50高被引",
        row_number=33,
        output_json=output_json,
    )

    text = output_json.read_text(encoding="utf-8")
    assert metadata.paper_id == "P032"
    assert '"publication_date": "2014-11-21"' in text
    assert "翻转课堂" in text
    assert "\\u" not in text
    assert "source_publication_date" not in text


def test_load_paper_metadata_reads_optional_source_publication_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_path = tmp_path / "catalog.xlsx"
    workbook_path.write_bytes(b"xlsx-placeholder")
    worksheet = FakeWorksheet(
        {
            1: (
                "Title-题名",
                "Author-作者",
                "PubTime-发表时间",
                "SourcePubTime-源文献发表时间",
            ),
            26: (
                "高等教育信息素养框架",
                "韩丽风; 王茜",
                "2015-11-21",
                datetime(2015, 2, 5, 9, 0),
            ),
        }
    )
    monkeypatch.setattr(
        "openpyxl.load_workbook",
        lambda *args, **kwargs: FakeWorkbook(worksheet),
    )

    metadata = load_paper_metadata(
        workbook_path=workbook_path,
        sheet_name="信息素养50高被引",
        row_number=26,
    )

    assert metadata.publication_date.isoformat() == "2015-11-21"
    assert metadata.source_publication_date.isoformat() == "2015-02-05"
