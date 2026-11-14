from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PAPER_ID_PATTERN = r"^P\d{3}$"
FINGERPRINT_PATTERN = r"^sha256:[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopicPaper(StrictModel):
    paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    publication_date: date
    doi: str | None = None

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title must not be blank")
        return stripped

    @field_validator("doi")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("authors")
    @classmethod
    def clean_authors(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if not cleaned:
            raise ValueError("authors must contain at least one non-empty name")
        return cleaned


class ReferenceLine(StrictModel):
    row_number: int = Field(ge=1)
    raw_text: str = Field(min_length=1)

    @field_validator("raw_text")
    @classmethod
    def strip_raw_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("raw_text must not be blank")
        return stripped


class ReferenceDocument(StrictModel):
    paper: TopicPaper
    workbook_path: Path
    workbook_sha256: str = Field(pattern=FINGERPRINT_PATTERN)
    lines: tuple[ReferenceLine, ...]


class TopicInputSet(StrictModel):
    topic: str = Field(min_length=1)
    catalog_workbook_path: Path
    catalog_sha256: str = Field(pattern=FINGERPRINT_PATTERN)
    papers: tuple[TopicPaper, ...]
    reference_documents: tuple[ReferenceDocument, ...]

    @model_validator(mode="after")
    def validate_fixed_topic(self) -> "TopicInputSet":
        if len(self.papers) != 50:
            raise ValueError("topic catalog must contain exactly 50 papers")
        if len(self.reference_documents) != 50:
            raise ValueError("topic must contain exactly 50 reference documents")
        paper_ids = [paper.paper_id for paper in self.papers]
        document_ids = [item.paper.paper_id for item in self.reference_documents]
        if paper_ids != [f"P{index:03d}" for index in range(1, 51)]:
            raise ValueError("paper IDs must be the ordered range P001 through P050")
        if document_ids != paper_ids:
            raise ValueError("reference documents must align with ordered papers")
        return self


class CitedPaperIdentification(StrictModel):
    cited_paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    reference_row_numbers: list[int] = Field(min_length=1)

    @field_validator("reference_row_numbers")
    @classmethod
    def sort_unique_rows(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("reference row numbers must be positive")
        return sorted(set(values))


class CitationIdentification(StrictModel):
    cited_papers: list[CitedPaperIdentification]

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "CitationIdentification":
        target_ids = [item.cited_paper_id for item in self.cited_papers]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("each cited paper must appear exactly once")
        self.cited_papers.sort(key=lambda item: item.cited_paper_id)
        return self


class CitationReviewIssue(StrictModel):
    description: str = Field(min_length=1)
    reference_row_numbers: list[int] = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("description must not be blank")
        return stripped

    @field_validator("reference_row_numbers")
    @classmethod
    def sort_unique_rows(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("review issue row numbers must be positive")
        return sorted(set(values))


class CitationReviewResult(StrictModel):
    status: Literal["pass", "needs_retry"]
    issues: list[CitationReviewIssue]

    @model_validator(mode="after")
    def validate_status(self) -> "CitationReviewResult":
        if self.status == "pass" and self.issues:
            raise ValueError("pass review must not contain issues")
        if self.status == "needs_retry" and not self.issues:
            raise ValueError("needs_retry review must contain issues")
        return self


class CitationEvidence(StrictModel):
    row_number: int = Field(ge=1)
    raw_text: str = Field(min_length=1)


class CitedPaperEvidence(StrictModel):
    cited_paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    evidence: list[CitationEvidence] = Field(min_length=1)


class CitationPaperOutcome(StrictModel):
    outcome_version: Literal[1] = 1
    topic: str = Field(min_length=1)
    paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    input_fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
    has_references: bool
    cited_papers: list[CitedPaperEvidence]

    @model_validator(mode="after")
    def validate_empty_document(self) -> "CitationPaperOutcome":
        if not self.has_references and self.cited_papers:
            raise ValueError("a paper without references cannot contain citation edges")
        targets = [item.cited_paper_id for item in self.cited_papers]
        if len(targets) != len(set(targets)):
            raise ValueError("outcome target papers must be unique")
        self.cited_papers.sort(key=lambda item: item.cited_paper_id)
        return self


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @classmethod
    def zero(cls) -> "TokenUsage":
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def plus(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class StageMetrics(StrictModel):
    elapsed_seconds: float = Field(ge=0)
    token_usage: TokenUsage


class IdentificationCheckpoint(StrictModel):
    checkpoint_version: Literal[1] = 1
    stage: Literal["citation_identification"] = "citation_identification"
    business_key: str = Field(pattern=PAPER_ID_PATTERN)
    task_fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
    result: CitationIdentification
    metrics: StageMetrics


class ReviewCheckpoint(StrictModel):
    checkpoint_version: Literal[1] = 1
    stage: Literal["citation_review"] = "citation_review"
    business_key: str = Field(pattern=PAPER_ID_PATTERN)
    task_fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
    result: CitationReviewResult
    metrics: StageMetrics


class CitationFailureRecord(StrictModel):
    failure_record_version: Literal[1] = 1
    stage: Literal["citation_identification", "citation_review"]
    business_key: str = Field(pattern=PAPER_ID_PATTERN)
    task_fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
    failure_kind: Literal[
        "content_exhausted",
        "network_exhausted",
        "review_needs_retry",
        "unexpected_error",
    ]
    message: str = Field(min_length=1)
    metrics: StageMetrics
    review_result: CitationReviewResult | None = None


class CitationNetworkNode(StrictModel):
    paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    title: str
    authors: list[str]
    publication_date: date
    x: float
    y: float


class CitationNetworkEdge(StrictModel):
    source_paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    target_paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    evidence: list[CitationEvidence] = Field(min_length=1)


class CitationNetworkBundle(StrictModel):
    bundle_version: Literal[1] = 1
    topic: str
    input_fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
    nodes: list[CitationNetworkNode]
    edges: list[CitationNetworkEdge]
    papers_without_references: list[str]
