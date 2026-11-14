from itertools import combinations
from typing import Literal

from pydantic import Field, StringConstraints, model_validator
from typing_extensions import Annotated

from .base import StrictModel
from .paper_metadata import NonEmptyText, PaperId
from .paper_profile import PaperProfileContent
from .relation_outcome import RelationOutcome
from .relation_record import RelationDirection, RelationType


Sha256Fingerprint = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]


class NetworkNode(StrictModel):
    paper_id: PaperId
    title: NonEmptyText
    authors: list[NonEmptyText] = Field(min_length=1)
    publication_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    profile: PaperProfileContent


class LayoutPosition(StrictModel):
    x: float = Field(ge=-1.1, le=1.1, allow_inf_nan=False)
    y: float = Field(ge=-1.1, le=1.1, allow_inf_nan=False)


class NetworkLayout(StrictModel):
    algorithm: Literal["spring_layout"]
    algorithm_version: Literal[2]
    networkx_version: NonEmptyText
    seed: int = Field(ge=0)
    basis_projection: Literal["strict"]
    positions: dict[PaperId, LayoutPosition]


class NetworkEdgeView(StrictModel):
    pair_key: NonEmptyText
    source: PaperId
    target: PaperId
    directed: bool
    relation_type: RelationType
    relation_direction: RelationDirection
    review_status: Literal["accepted", "manual_review_required"]


class NetworkBundle(StrictModel):
    schema_version: Literal[1]
    topic: NonEmptyText
    input_fingerprint: Sha256Fingerprint
    nodes: list[NetworkNode] = Field(min_length=1)
    pairs: list[RelationOutcome]
    layout: NetworkLayout

    @model_validator(mode="after")
    def validate_complete_canonical_topic(self) -> "NetworkBundle":
        node_ids = [node.paper_id for node in self.nodes]
        if node_ids != sorted(node_ids):
            raise ValueError("network nodes must be sorted by paper_id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("network nodes must have unique paper_id values")

        pair_keys = [
            _pair_key(
                outcome.relation_record.paper_a_id,
                outcome.relation_record.paper_b_id,
            )
            for outcome in self.pairs
        ]
        if pair_keys != sorted(pair_keys):
            raise ValueError("network pairs must be sorted by canonical pair key")
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("network pairs must be unique")

        expected_pairs = {
            _pair_key(paper_a_id, paper_b_id)
            for paper_a_id, paper_b_id in combinations(node_ids, 2)
        }
        if set(pair_keys) != expected_pairs:
            missing = sorted(expected_pairs - set(pair_keys))
            extra = sorted(set(pair_keys) - expected_pairs)
            raise ValueError(
                f"network pair coverage mismatch; missing={missing}, extra={extra}"
            )

        position_ids = set(self.layout.positions)
        if position_ids != set(node_ids):
            missing = sorted(set(node_ids) - position_ids)
            extra = sorted(position_ids - set(node_ids))
            raise ValueError(
                f"layout position coverage mismatch; missing={missing}, extra={extra}"
            )
        return self


def _pair_key(paper_a_id: str, paper_b_id: str) -> str:
    return f"{paper_a_id}__{paper_b_id}"
