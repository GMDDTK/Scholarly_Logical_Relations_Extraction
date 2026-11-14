import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .contracts.paper_metadata import PaperMetadata
from .contracts.paper_profile import PaperProfileContent
from .contracts.relation_outcome import RelationOutcome
from .contracts.relation_record import RelationRecord
from .contracts.runtime import (
    PaperParserFailureRecord,
    PaperProfileCheckpoint,
    ProfileReviewCheckpoint,
    ProfileReviewerFailureRecord,
    RelationInferenceCheckpoint,
    RelationInferenceFailureRecord,
    RelationReviewCheckpoint,
    RelationReviewerFailureRecord,
)
from .relation_validation import (
    RelationHandoffError,
    validate_relation_handoff,
    validate_relation_revision_handoff,
)
from .relation_review_validation import (
    RelationReviewFieldError,
    validate_relation_review_fields,
)
from .review_validation import (
    ProfileReviewLocationError,
    validate_profile_review_locations,
)


_RELATION_OUTCOME_ADAPTER = TypeAdapter(RelationOutcome)


def compute_task_fingerprint(
    *,
    stage: str,
    business_key: str,
    dynamic_input: dict[str, Any],
    system_prompt: str,
    output_schema: dict[str, Any],
    experiment_unit: str,
    model: str,
) -> str:
    payload = {
        "stage": stage,
        "business_key": business_key,
        "dynamic_input": dynamic_input,
        "system_prompt": system_prompt,
        "output_schema": output_schema,
        "experiment_unit": experiment_unit,
        "model": model,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def write_paper_profile_checkpoint(
    *,
    checkpoint: PaperProfileCheckpoint,
    destination: Path,
    temp_root: Path,
) -> None:
    _write_json_atomically(
        serialized=checkpoint.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=checkpoint.business_key,
    )


def write_final_paper_profile(
    *,
    profile: PaperProfileContent,
    destination: Path,
    temp_root: Path,
    paper_id: str,
) -> None:
    _write_json_atomically(
        serialized=profile.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=paper_id,
    )


def load_matching_final_paper_profile(
    *,
    path: Path,
    expected_profile: PaperProfileContent,
) -> PaperProfileContent | None:
    if not path.is_file():
        return None
    try:
        profile = PaperProfileContent.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError):
        return None
    if profile != expected_profile:
        return None
    return profile


def write_relation_outcome(
    *,
    outcome: RelationOutcome,
    destination: Path,
    temp_root: Path,
    pair_key: str,
) -> None:
    serialized = _RELATION_OUTCOME_ADAPTER.dump_json(
        outcome,
        indent=2,
    ).decode("utf-8")
    _write_json_atomically(
        serialized=serialized,
        destination=destination,
        temp_root=temp_root,
        business_key=pair_key,
    )


def load_matching_relation_outcome(
    *,
    path: Path,
    expected_outcome: RelationOutcome,
) -> RelationOutcome | None:
    if not path.is_file():
        return None
    try:
        outcome = _RELATION_OUTCOME_ADAPTER.validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError):
        return None
    if outcome != expected_outcome:
        return None
    return outcome


def write_failure_record(
    *,
    failure: (
        PaperParserFailureRecord
        | ProfileReviewerFailureRecord
        | RelationInferenceFailureRecord
        | RelationReviewerFailureRecord
    ),
    destination: Path,
    temp_root: Path,
) -> None:
    _write_json_atomically(
        serialized=failure.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=failure.business_key,
    )


def _write_json_atomically(
    *,
    serialized: str,
    destination: Path,
    temp_root: Path,
    business_key: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"{business_key}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_valid_paper_profile_checkpoint(
    *,
    path: Path,
    expected_stage: str,
    expected_business_key: str,
    expected_fingerprint: str,
) -> PaperProfileCheckpoint | None:
    if not path.is_file():
        return None
    try:
        checkpoint = PaperProfileCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if checkpoint.stage != expected_stage:
            return None
        if checkpoint.business_key != expected_business_key:
            return None
        if checkpoint.task_fingerprint != expected_fingerprint:
            return None
    except (OSError, UnicodeError, ValidationError):
        return None
    return checkpoint


def write_profile_review_checkpoint(
    *,
    checkpoint: ProfileReviewCheckpoint,
    destination: Path,
    temp_root: Path,
) -> None:
    _write_json_atomically(
        serialized=checkpoint.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=checkpoint.business_key,
    )


def load_valid_profile_review_checkpoint(
    *,
    path: Path,
    expected_business_key: str,
    expected_fingerprint: str,
    current_profile: PaperProfileContent,
) -> ProfileReviewCheckpoint | None:
    if not path.is_file():
        return None
    try:
        checkpoint = ProfileReviewCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if checkpoint.business_key != expected_business_key:
            return None
        if checkpoint.task_fingerprint != expected_fingerprint:
            return None
        validate_profile_review_locations(
            checkpoint.result,
            current_profile,
        )
    except (
        OSError,
        UnicodeError,
        ValidationError,
        ProfileReviewLocationError,
    ):
        return None
    return checkpoint


def write_relation_inference_checkpoint(
    *,
    checkpoint: RelationInferenceCheckpoint,
    destination: Path,
    temp_root: Path,
) -> None:
    _write_json_atomically(
        serialized=checkpoint.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=checkpoint.business_key,
    )


def load_valid_relation_inference_checkpoint(
    *,
    path: Path,
    expected_stage: str,
    expected_business_key: str,
    expected_fingerprint: str,
    paper_a_metadata: PaperMetadata,
    paper_b_metadata: PaperMetadata,
    current_relation: RelationRecord | None = None,
) -> RelationInferenceCheckpoint | None:
    if not path.is_file():
        return None
    try:
        checkpoint = RelationInferenceCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if checkpoint.stage != expected_stage:
            return None
        if checkpoint.business_key != expected_business_key:
            return None
        if checkpoint.task_fingerprint != expected_fingerprint:
            return None
        if current_relation is None:
            validate_relation_handoff(
                checkpoint.result,
                paper_a_metadata,
                paper_b_metadata,
            )
        else:
            validate_relation_revision_handoff(
                checkpoint.result,
                current_relation,
                paper_a_metadata,
                paper_b_metadata,
            )
    except (
        OSError,
        UnicodeError,
        ValidationError,
        RelationHandoffError,
    ):
        return None
    return checkpoint


def write_relation_review_checkpoint(
    *,
    checkpoint: RelationReviewCheckpoint,
    destination: Path,
    temp_root: Path,
) -> None:
    _write_json_atomically(
        serialized=checkpoint.model_dump_json(indent=2),
        destination=destination,
        temp_root=temp_root,
        business_key=checkpoint.business_key,
    )


def load_valid_relation_review_checkpoint(
    *,
    path: Path,
    expected_stage: str,
    expected_business_key: str,
    expected_fingerprint: str,
    current_relation: RelationRecord,
) -> RelationReviewCheckpoint | None:
    if not path.is_file():
        return None
    try:
        checkpoint = RelationReviewCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if checkpoint.stage != expected_stage:
            return None
        if checkpoint.business_key != expected_business_key:
            return None
        if checkpoint.task_fingerprint != expected_fingerprint:
            return None
        validate_relation_review_fields(
            checkpoint.result,
            current_relation,
        )
    except (
        OSError,
        UnicodeError,
        ValidationError,
        RelationReviewFieldError,
    ):
        return None
    return checkpoint
