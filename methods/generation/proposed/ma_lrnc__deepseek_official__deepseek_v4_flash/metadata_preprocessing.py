from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import openpyxl

from .contracts.paper_metadata import PaperMetadata


TITLE_HEADER = "Title-题名"
AUTHORS_HEADER = "Author-作者"
PUBLICATION_TIME_HEADER = "PubTime-发表时间"
SOURCE_PUBLICATION_TIME_HEADER = "SourcePubTime-源文献发表时间"
REQUIRED_CATALOG_HEADERS = (
    TITLE_HEADER,
    AUTHORS_HEADER,
    PUBLICATION_TIME_HEADER,
)
_AUTHOR_SEPARATOR = re.compile(r"[;；]")


def paper_id_from_catalog_row(row_number: int) -> str:
    if row_number < 2:
        raise ValueError("Catalog row_number must identify a data row after the header")
    return f"P{row_number - 1:03d}"


def load_paper_metadata(
    *,
    workbook_path: Path,
    sheet_name: str,
    row_number: int,
) -> PaperMetadata:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Catalog workbook not found: {workbook_path}")

    workbook = openpyxl.load_workbook(
        filename=workbook_path,
        read_only=True,
        data_only=True,
    )
    try:
        try:
            worksheet = workbook[sheet_name]
        except KeyError as error:
            raise ValueError(
                f"Catalog worksheet not found: {sheet_name}"
            ) from error

        header_values = _read_single_row(worksheet, row_number=1)
        header_indices = required_catalog_header_indices(header_values)
        source_publication_time_index = optional_catalog_header_index(
            header_values,
            SOURCE_PUBLICATION_TIME_HEADER,
        )
        row_values = _read_single_row(worksheet, row_number=row_number)
        selected = {
            header: _value_at(row_values, index, header)
            for header, index in header_indices.items()
        }
        return paper_metadata_from_catalog_values(
            row_number=row_number,
            title=selected[TITLE_HEADER],
            authors=selected[AUTHORS_HEADER],
            publication_time=selected[PUBLICATION_TIME_HEADER],
            source_publication_time=_optional_value_at(
                row_values,
                source_publication_time_index,
            ),
        )
    finally:
        workbook.close()


def write_paper_metadata(
    *,
    workbook_path: Path,
    sheet_name: str,
    row_number: int,
    output_json: Path,
) -> PaperMetadata:
    metadata = load_paper_metadata(
        workbook_path=workbook_path,
        sheet_name=sheet_name,
        row_number=row_number,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            metadata.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def _read_single_row(
    worksheet: object,
    *,
    row_number: int,
) -> tuple[object, ...]:
    rows = list(
        worksheet.iter_rows(
            min_row=row_number,
            max_row=row_number,
            values_only=True,
        )
    )
    if not rows:
        raise ValueError(f"Catalog row not found: {row_number}")
    return tuple(rows[0])


def required_catalog_header_indices(
    header_values: Sequence[object],
) -> Mapping[str, int]:
    normalized = {
        str(value).strip(): index
        for index, value in enumerate(header_values)
        if value is not None
    }
    missing = [
        header
        for header in REQUIRED_CATALOG_HEADERS
        if header not in normalized
    ]
    if missing:
        raise ValueError(
            "Catalog is missing required headers: " + ", ".join(missing)
        )
    return {
        header: normalized[header]
        for header in REQUIRED_CATALOG_HEADERS
    }


def optional_catalog_header_index(
    header_values: Sequence[object],
    header: str,
) -> int | None:
    for index, value in enumerate(header_values):
        if value is not None and str(value).strip() == header:
            return index
    return None


def paper_metadata_from_catalog_values(
    *,
    row_number: int,
    title: object,
    authors: object,
    publication_time: object,
    source_publication_time: object = None,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id_from_catalog_row(row_number),
        title=_required_text(title, TITLE_HEADER),
        authors=_parse_authors(authors),
        publication_date=_parse_publication_date(
            publication_time,
            PUBLICATION_TIME_HEADER,
        ),
        source_publication_date=_parse_optional_publication_date(
            source_publication_time,
            SOURCE_PUBLICATION_TIME_HEADER,
        ),
    )


def _value_at(
    values: Sequence[object],
    index: int,
    header: str,
) -> object:
    if index >= len(values):
        raise ValueError(f"Catalog row has no value for required header: {header}")
    return values[index]


def _optional_value_at(
    values: Sequence[object],
    index: int | None,
) -> object:
    if index is None or index >= len(values):
        return None
    return values[index]


def _required_text(value: object, header: str) -> str:
    if value is None:
        raise ValueError(f"Catalog value is empty for required header: {header}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Catalog value is empty for required header: {header}")
    return text


def _parse_authors(value: object) -> list[str]:
    text = _required_text(value, AUTHORS_HEADER)
    authors = [part.strip() for part in _AUTHOR_SEPARATOR.split(text)]
    authors = [author for author in authors if author]
    if not authors:
        raise ValueError("Catalog author list is empty")
    return authors


def _parse_publication_date(value: object, header: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _required_text(value, header)
    try:
        return datetime.fromisoformat(text).date()
    except ValueError as error:
        raise ValueError(
            f"Unsupported {header} value: {text}"
        ) from error


def _parse_optional_publication_date(
    value: object,
    header: str,
) -> date | None:
    if value is None or not str(value).strip():
        return None
    return _parse_publication_date(value, header)
