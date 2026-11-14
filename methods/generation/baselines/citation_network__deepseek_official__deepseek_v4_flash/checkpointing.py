from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import (
    CitationPaperOutcome,
    IdentificationCheckpoint,
    ReviewCheckpoint,
)


T = TypeVar("T", bound=BaseModel)


def compute_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def write_model_atomically(
    *,
    model: BaseModel,
    destination: Path,
    temp_root: Path,
    business_key: str,
) -> None:
    write_text_atomically(
        text=model.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=business_key,
    )


def write_text_atomically(
    *,
    text: str,
    destination: Path,
    temp_root: Path,
    business_key: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"{business_key}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path: Path, model_type: type[T]) -> T | None:
    if not path.is_file():
        return None
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError):
        return None


def load_valid_outcome(
    *,
    path: Path,
    topic: str,
    paper_id: str,
    input_fingerprint: str,
) -> CitationPaperOutcome | None:
    outcome = load_model(path, CitationPaperOutcome)
    if outcome is None:
        return None
    if (
        outcome.topic != topic
        or outcome.paper_id != paper_id
        or outcome.input_fingerprint != input_fingerprint
    ):
        return None
    return outcome


def load_valid_identification_checkpoint(
    *,
    path: Path,
    business_key: str,
    task_fingerprint: str,
) -> IdentificationCheckpoint | None:
    checkpoint = load_model(path, IdentificationCheckpoint)
    if checkpoint is None:
        return None
    if (
        checkpoint.business_key != business_key
        or checkpoint.task_fingerprint != task_fingerprint
    ):
        return None
    return checkpoint


def load_valid_review_checkpoint(
    *,
    path: Path,
    business_key: str,
    task_fingerprint: str,
) -> ReviewCheckpoint | None:
    checkpoint = load_model(path, ReviewCheckpoint)
    if checkpoint is None:
        return None
    if (
        checkpoint.business_key != business_key
        or checkpoint.task_fingerprint != task_fingerprint
    ):
        return None
    return checkpoint
