from datetime import date
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator, model_validator

from .base import StrictModel


PaperId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^P\d{3}$"),
]
NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class PaperMetadata(StrictModel):
    paper_id: PaperId
    title: NonEmptyText
    authors: list[NonEmptyText] = Field(min_length=1)
    publication_date: date
    source_publication_date: date | None = None

    @field_validator("authors")
    @classmethod
    def authors_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("authors must not contain duplicates")
        return value

    @model_validator(mode="after")
    def source_date_must_not_follow_current_publication(self) -> "PaperMetadata":
        if (
            self.source_publication_date is not None
            and self.source_publication_date > self.publication_date
        ):
            raise ValueError(
                "source_publication_date must not be after publication_date"
            )
        return self

    @property
    def relation_reference_date(self) -> date:
        return self.source_publication_date or self.publication_date
