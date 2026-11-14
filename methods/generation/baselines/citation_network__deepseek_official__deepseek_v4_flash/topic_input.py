from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from zipfile import BadZipFile

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

from .contracts import ReferenceDocument, ReferenceLine, TopicInputSet, TopicPaper


TITLE_HEADER = "Title-题名"
AUTHORS_HEADER = "Author-作者"
PUBLICATION_TIME_HEADER = "PubTime-发表时间"
_INVALID_TOPIC_CHARACTERS = frozenset('\\/:*?"<>|')
_TITLE_IGNORED_CHARACTERS = re.compile(
    r"[\s“”\"'‘’《》〈〉【】\[\]（）()，,。.!！?？:：;；·•—–_＿\-]+",
    flags=re.UNICODE,
)
_AUTHOR_SPLIT = re.compile(r"[;,；，、]+")


class TopicInputError(ValueError):
    pass


def normalize_title(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _TITLE_IGNORED_CHARACTERS.sub("", normalized)


def discover_topic_inputs(*, project_root: Path, topic: str) -> TopicInputSet:
    safe_topic = _validate_topic(topic)
    catalog_path = _discover_catalog(project_root, safe_topic)
    papers = _load_catalog(catalog_path)
    reference_paths = _discover_reference_workbooks(project_root, safe_topic)
    by_title: dict[str, list[TopicPaper]] = {}
    for paper in papers:
        by_title.setdefault(normalize_title(paper.title), []).append(paper)

    mapped: dict[str, ReferenceDocument] = {}
    for path in reference_paths:
        candidates = by_title.get(normalize_title(path.stem), [])
        if len(candidates) != 1:
            raise TopicInputError(
                f"Reference workbook {path.name!r} does not map to a unique catalog paper"
            )
        paper = candidates[0]
        if paper.paper_id in mapped:
            raise TopicInputError(
                "Reference workbook does not map to a unique catalog paper: "
                f"multiple files claim {paper.paper_id}"
            )
        mapped[paper.paper_id] = ReferenceDocument(
            paper=paper,
            workbook_path=path,
            workbook_sha256=_sha256_file(path),
            lines=_load_reference_lines(path),
        )
    missing = [paper.paper_id for paper in papers if paper.paper_id not in mapped]
    if missing:
        raise TopicInputError(
            "Reference workbooks do not cover all catalog papers: " + ", ".join(missing)
        )
    return TopicInputSet(
        topic=safe_topic,
        catalog_workbook_path=catalog_path,
        catalog_sha256=_sha256_file(catalog_path),
        papers=papers,
        reference_documents=tuple(mapped[paper.paper_id] for paper in papers),
    )


def _validate_topic(topic: str) -> str:
    normalized = topic.strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or any(character in _INVALID_TOPIC_CHARACTERS for character in normalized)
    ):
        raise TopicInputError(f"Invalid topic name: {topic!r}")
    return normalized


def _discover_catalog(project_root: Path, topic: str) -> Path:
    directory = project_root / "data" / "题录"
    candidates = sorted(
        (
            path
            for path in directory.glob(f"{topic}*.xlsx")
            if path.is_file() and not path.name.startswith("~$")
        ),
        key=lambda path: path.name.casefold(),
    ) if directory.is_dir() else []
    if len(candidates) != 1:
        raise TopicInputError(
            f"Expected exactly one catalog workbook for {topic}, found {len(candidates)}"
        )
    return candidates[0]


def _discover_reference_workbooks(project_root: Path, topic: str) -> tuple[Path, ...]:
    directory = project_root / "data" / "参考文献" / topic
    candidates = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.casefold() == ".xlsx"
                and not path.name.startswith("~$")
            ),
            key=lambda path: unicodedata.normalize("NFKC", path.name).casefold(),
        )
    ) if directory.is_dir() else ()
    if len(candidates) != 50:
        raise TopicInputError(
            f"Expected exactly 50 reference workbooks, found {len(candidates)}"
        )
    return candidates


def _open_workbook(path: Path):
    try:
        return openpyxl.load_workbook(path, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, OSError, ValueError) as error:
        raise TopicInputError(f"Workbook could not be opened: {path}") from error


def _load_catalog(path: Path) -> tuple[TopicPaper, ...]:
    workbook = _open_workbook(path)
    try:
        visible = [sheet for sheet in workbook.worksheets if sheet.sheet_state == "visible"]
        if len(visible) != 1:
            raise TopicInputError(
                f"Expected exactly one visible catalog worksheet, found {len(visible)}"
            )
        worksheet = visible[0]
        header = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
        if header is None:
            raise TopicInputError("Catalog worksheet has no header row")
        indices = {str(value).strip(): index for index, value in enumerate(header) if value}
        for required in (TITLE_HEADER, AUTHORS_HEADER, PUBLICATION_TIME_HEADER):
            if required not in indices:
                raise TopicInputError(f"Catalog is missing required header {required}")
        doi_index = next(
            (index for name, index in indices.items() if name.casefold().startswith("doi")),
            None,
        )
        rows = []
        for values in worksheet.iter_rows(min_row=2, values_only=True):
            title = _value_at(values, indices[TITLE_HEADER])
            if not str(title or "").strip():
                continue
            rows.append(values)
        if len(rows) != 50:
            raise TopicInputError(
                f"Catalog must contain exactly 50 paper rows, found {len(rows)}"
            )
        papers = []
        for index, values in enumerate(rows, start=1):
            title = str(_value_at(values, indices[TITLE_HEADER])).strip()
            authors = _parse_authors(_value_at(values, indices[AUTHORS_HEADER]))
            publication_date = _parse_date(
                _value_at(values, indices[PUBLICATION_TIME_HEADER])
            )
            raw_doi = _value_at(values, doi_index) if doi_index is not None else None
            doi = str(raw_doi).strip() if raw_doi not in (None, "") else None
            papers.append(
                TopicPaper(
                    paper_id=f"P{index:03d}",
                    title=title,
                    authors=authors,
                    publication_date=publication_date,
                    doi=doi,
                )
            )
        normalized_titles = [normalize_title(paper.title) for paper in papers]
        if len(set(normalized_titles)) != 50:
            raise TopicInputError("Catalog normalized titles must be unique")
        return tuple(papers)
    finally:
        workbook.close()


def _load_reference_lines(path: Path) -> tuple[ReferenceLine, ...]:
    workbook = _open_workbook(path)
    try:
        visible = [sheet for sheet in workbook.worksheets if sheet.sheet_state == "visible"]
        if not visible:
            raise TopicInputError(f"Reference workbook has no visible worksheet: {path}")
        texts: list[str] = []
        for worksheet in visible:
            for row in worksheet.iter_rows(values_only=True):
                for value in row:
                    text = str(value).strip() if value is not None else ""
                    if text:
                        texts.append(text)
        return tuple(
            ReferenceLine(row_number=index, raw_text=text)
            for index, text in enumerate(texts, start=1)
        )
    finally:
        workbook.close()


def _parse_authors(value: object) -> list[str]:
    authors = [part.strip() for part in _AUTHOR_SPLIT.split(str(value or "")) if part.strip()]
    if not authors:
        raise TopicInputError("Catalog paper has no author")
    return authors


def _parse_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    for date_format in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    raise TopicInputError(f"Invalid publication date: {value!r}")


def _value_at(values: tuple[object, ...], index: int) -> object:
    return values[index] if index < len(values) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
