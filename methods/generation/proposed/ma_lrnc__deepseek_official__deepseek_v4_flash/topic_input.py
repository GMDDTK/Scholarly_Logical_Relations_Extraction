from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

from .contracts.paper_metadata import PaperMetadata
from .metadata_preprocessing import (
    AUTHORS_HEADER,
    PUBLICATION_TIME_HEADER,
    SOURCE_PUBLICATION_TIME_HEADER,
    TITLE_HEADER,
    optional_catalog_header_index,
    paper_id_from_catalog_row,
    paper_metadata_from_catalog_values,
    required_catalog_header_indices,
)


_INVALID_TOPIC_CHARACTERS = frozenset('\\/:*?"<>|')
_TITLE_IGNORED_CHARACTERS = re.compile(
    r"""[\s“”"'‘’《》〈〉【】\[\]（）()，,。.!！?？:：;；·•—–_＿\-]+""",
    flags=re.UNICODE,
)
_AUTHOR_IGNORED_CHARACTERS = re.compile(r"[\W_]+", flags=re.UNICODE)
_DIAGNOSTIC_CANDIDATE_LIMIT = 3
PaperInputFailureCode = Literal[
    "no_exact_title_match",
    "ambiguous_exact_title_match",
    "invalid_catalog_values",
    "author_mismatch",
    "duplicate_catalog_assignment",
]


class TopicInputError(ValueError):
    """A theme-level input error that makes the whole topic unsafe to run."""


@dataclass(frozen=True)
class DiagnosticCandidate:
    paper_id: str
    title: str
    score: float


@dataclass(frozen=True)
class PaperInputMatch:
    pdf_path: Path
    catalog_row: int
    metadata: PaperMetadata


@dataclass(frozen=True)
class PaperInputFailure:
    pdf_path: Path
    code: PaperInputFailureCode
    message: str
    diagnostic_candidates: tuple[DiagnosticCandidate, ...] = ()


@dataclass(frozen=True)
class PaperPairInput:
    pair_key: str
    paper_a: PaperInputMatch
    paper_b: PaperInputMatch


@dataclass(frozen=True)
class TopicInputSet:
    topic: str
    topic_directory: Path
    workbook_path: Path
    worksheet_name: str
    pdf_paths: tuple[Path, ...]
    matches: tuple[PaperInputMatch, ...]
    failures: tuple[PaperInputFailure, ...]
    pairs: tuple[PaperPairInput, ...]

    @property
    def is_complete(self) -> bool:
        return not self.failures and len(self.matches) == len(self.pdf_paths)


@dataclass(frozen=True)
class _CatalogCandidate:
    row_number: int
    title: object
    authors: object
    publication_time: object
    source_publication_time: object
    normalized_title: str

    @property
    def paper_id(self) -> str:
        return paper_id_from_catalog_row(self.row_number)

    @property
    def display_title(self) -> str:
        return str(self.title).strip()


def discover_topic_inputs(
    *,
    project_root: Path,
    topic: str,
) -> TopicInputSet:
    normalized_topic = _validate_topic(topic)
    topic_directory = project_root / "data" / "论文" / normalized_topic
    if not topic_directory.is_dir():
        raise TopicInputError(
            f"Topic PDF directory not found: {topic_directory}"
        )

    pdf_paths = tuple(
        sorted(
            (
                path
                for path in topic_directory.iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: unicodedata.normalize(
                "NFKC", path.name
            ).casefold(),
        )
    )
    if not pdf_paths:
        raise TopicInputError(
            f"Topic PDF directory contains no PDF files: {topic_directory}"
        )

    workbook_path = _discover_catalog_workbook(
        project_root=project_root,
        topic=normalized_topic,
    )
    worksheet_name, catalog_candidates = _load_catalog_candidates(
        workbook_path
    )
    matches, failures = _match_pdfs(
        pdf_paths=pdf_paths,
        catalog_candidates=catalog_candidates,
    )
    pairs = _build_pairs(matches)
    return TopicInputSet(
        topic=normalized_topic,
        topic_directory=topic_directory,
        workbook_path=workbook_path,
        worksheet_name=worksheet_name,
        pdf_paths=pdf_paths,
        matches=matches,
        failures=failures,
        pairs=pairs,
    )


def normalize_title(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _TITLE_IGNORED_CHARACTERS.sub("", normalized)


def normalize_author(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _AUTHOR_IGNORED_CHARACTERS.sub("", normalized)


def _validate_topic(topic: str) -> str:
    normalized = topic.strip()
    if not normalized:
        raise TopicInputError("Topic name must not be empty")
    if normalized in {".", ".."} or any(
        character in _INVALID_TOPIC_CHARACTERS
        for character in normalized
    ):
        raise TopicInputError(f"Topic name contains invalid characters: {topic}")
    return normalized


def _discover_catalog_workbook(
    *,
    project_root: Path,
    topic: str,
) -> Path:
    catalog_directory = project_root / "data" / "题录"
    candidates: list[Path] = []
    if catalog_directory.is_dir():
        topic_key = topic.casefold()
        candidates = sorted(
            (
                path
                for path in catalog_directory.iterdir()
                if path.is_file()
                and path.suffix.casefold() == ".xlsx"
                and not path.name.startswith("~$")
                and path.stem.casefold().startswith(topic_key)
            ),
            key=lambda path: path.name.casefold(),
        )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise TopicInputError(
            "Expected exactly one catalog workbook matching "
            f"{topic}*.xlsx, found {len(candidates)}: {names}"
        )
    return candidates[0]


def _load_catalog_candidates(
    workbook_path: Path,
) -> tuple[str, tuple[_CatalogCandidate, ...]]:
    try:
        workbook = openpyxl.load_workbook(
            filename=workbook_path,
            read_only=True,
            data_only=True,
        )
    except (
        BadZipFile,
        InvalidFileException,
        OSError,
        ValueError,
    ) as error:
        raise TopicInputError(
            f"Catalog workbook could not be opened: {workbook_path}"
        ) from error
    try:
        visible = [
            worksheet
            for worksheet in workbook.worksheets
            if worksheet.sheet_state == "visible"
        ]
        if len(visible) != 1:
            raise TopicInputError(
                "Expected exactly one visible worksheet in catalog workbook, "
                f"found {len(visible)}"
            )
        worksheet = visible[0]
        header_rows = list(
            worksheet.iter_rows(
                min_row=1,
                max_row=1,
                values_only=True,
            )
        )
        if not header_rows:
            raise TopicInputError("Catalog worksheet has no header row")
        try:
            header_indices = required_catalog_header_indices(header_rows[0])
        except ValueError as error:
            raise TopicInputError(str(error)) from error
        source_publication_time_index = optional_catalog_header_index(
            header_rows[0],
            SOURCE_PUBLICATION_TIME_HEADER,
        )

        candidates: list[_CatalogCandidate] = []
        for row_number, values in enumerate(
            worksheet.iter_rows(min_row=2, values_only=True),
            start=2,
        ):
            title = _value_at(
                values,
                header_indices[TITLE_HEADER],
            )
            normalized_title = normalize_title(title)
            if not normalized_title:
                continue
            candidates.append(
                _CatalogCandidate(
                    row_number=row_number,
                    title=title,
                    authors=_value_at(
                        values,
                        header_indices[AUTHORS_HEADER],
                    ),
                    publication_time=_value_at(
                        values,
                        header_indices[PUBLICATION_TIME_HEADER],
                    ),
                    source_publication_time=(
                        _value_at(values, source_publication_time_index)
                        if source_publication_time_index is not None
                        else None
                    ),
                    normalized_title=normalized_title,
                )
            )
        return worksheet.title, tuple(candidates)
    finally:
        workbook.close()


def _value_at(values: tuple[object, ...], index: int) -> object:
    if index >= len(values):
        return None
    return values[index]


def _match_pdfs(
    *,
    pdf_paths: tuple[Path, ...],
    catalog_candidates: tuple[_CatalogCandidate, ...],
) -> tuple[tuple[PaperInputMatch, ...], tuple[PaperInputFailure, ...]]:
    provisional_matches: list[PaperInputMatch] = []
    failures: list[PaperInputFailure] = []
    for pdf_path in pdf_paths:
        filename_title, filename_author = _split_pdf_filename(pdf_path)
        normalized_filename_title = normalize_title(filename_title)
        exact_candidates = [
            candidate
            for candidate in catalog_candidates
            if candidate.normalized_title == normalized_filename_title
        ]
        diagnostics = _diagnostic_candidates(
            normalized_filename_title,
            catalog_candidates,
        )
        if len(exact_candidates) == 0:
            failures.append(
                PaperInputFailure(
                    pdf_path=pdf_path,
                    code="no_exact_title_match",
                    message=(
                        "No catalog title exactly matches the normalized "
                        f"PDF filename title: {filename_title}"
                    ),
                    diagnostic_candidates=diagnostics,
                )
            )
            continue
        if len(exact_candidates) > 1:
            failures.append(
                PaperInputFailure(
                    pdf_path=pdf_path,
                    code="ambiguous_exact_title_match",
                    message=(
                        "Multiple catalog rows exactly match the normalized "
                        f"PDF filename title: {filename_title}"
                    ),
                    diagnostic_candidates=tuple(
                        DiagnosticCandidate(
                            paper_id=candidate.paper_id,
                            title=candidate.display_title,
                            score=1.0,
                        )
                        for candidate in exact_candidates
                    ),
                )
            )
            continue

        candidate = exact_candidates[0]
        try:
            metadata = paper_metadata_from_catalog_values(
                row_number=candidate.row_number,
                title=candidate.title,
                authors=candidate.authors,
                publication_time=candidate.publication_time,
                source_publication_time=candidate.source_publication_time,
            )
        except ValueError as error:
            failures.append(
                PaperInputFailure(
                    pdf_path=pdf_path,
                    code="invalid_catalog_values",
                    message=str(error),
                    diagnostic_candidates=diagnostics,
                )
            )
            continue

        if filename_author is not None and normalize_author(
            filename_author
        ) not in {
            normalize_author(author)
            for author in metadata.authors
        }:
            failures.append(
                PaperInputFailure(
                    pdf_path=pdf_path,
                    code="author_mismatch",
                    message=(
                        f"PDF filename author {filename_author!r} does not "
                        f"match catalog authors for {metadata.paper_id}"
                    ),
                    diagnostic_candidates=diagnostics,
                )
            )
            continue

        provisional_matches.append(
            PaperInputMatch(
                pdf_path=pdf_path,
                catalog_row=candidate.row_number,
                metadata=metadata,
            )
        )

    matches, collision_failures = _remove_catalog_collisions(
        provisional_matches
    )
    failures.extend(collision_failures)
    return (
        tuple(sorted(matches, key=lambda item: item.metadata.paper_id)),
        tuple(
            sorted(
                failures,
                key=lambda failure: failure.pdf_path.name.casefold(),
            )
        ),
    )


def _split_pdf_filename(pdf_path: Path) -> tuple[str, str | None]:
    title, separator, author = pdf_path.stem.rpartition("_")
    if separator and title.strip() and author.strip():
        return title.strip(), author.strip()
    return pdf_path.stem.strip(), None


def _remove_catalog_collisions(
    provisional_matches: list[PaperInputMatch],
) -> tuple[list[PaperInputMatch], list[PaperInputFailure]]:
    by_row: dict[int, list[PaperInputMatch]] = {}
    for match in provisional_matches:
        by_row.setdefault(match.catalog_row, []).append(match)

    accepted: list[PaperInputMatch] = []
    failures: list[PaperInputFailure] = []
    for matches in by_row.values():
        if len(matches) == 1:
            accepted.append(matches[0])
            continue
        paper_id = matches[0].metadata.paper_id
        for match in matches:
            failures.append(
                PaperInputFailure(
                    pdf_path=match.pdf_path,
                    code="duplicate_catalog_assignment",
                    message=(
                        "Multiple PDFs map to the same catalog record: "
                        f"{paper_id}"
                    ),
                )
            )
    return accepted, failures


def _diagnostic_candidates(
    normalized_filename_title: str,
    catalog_candidates: tuple[_CatalogCandidate, ...],
) -> tuple[DiagnosticCandidate, ...]:
    ranked = sorted(
        (
            DiagnosticCandidate(
                paper_id=candidate.paper_id,
                title=candidate.display_title,
                score=round(
                    _dice_similarity(
                        normalized_filename_title,
                        candidate.normalized_title,
                    ),
                    4,
                ),
            )
            for candidate in catalog_candidates
        ),
        key=lambda candidate: (-candidate.score, candidate.paper_id),
    )
    return tuple(ranked[:_DIAGNOSTIC_CANDIDATE_LIMIT])


def _dice_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_bigrams = _bigrams(left)
    right_bigrams = _bigrams(right)
    if not left_bigrams or not right_bigrams:
        return 0.0
    overlap = len(left_bigrams & right_bigrams)
    return (2 * overlap) / (len(left_bigrams) + len(right_bigrams))


def _bigrams(value: str) -> set[str]:
    return {
        value[index : index + 2]
        for index in range(max(0, len(value) - 1))
    }


def _build_pairs(
    matches: tuple[PaperInputMatch, ...],
) -> tuple[PaperPairInput, ...]:
    return tuple(
        PaperPairInput(
            pair_key=(
                f"{paper_a.metadata.paper_id}__"
                f"{paper_b.metadata.paper_id}"
            ),
            paper_a=paper_a,
            paper_b=paper_b,
        )
        for paper_a, paper_b in combinations(matches, 2)
    )
