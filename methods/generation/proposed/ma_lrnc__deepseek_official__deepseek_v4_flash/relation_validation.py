from .contracts.paper_metadata import PaperMetadata
from .contracts.relation_record import (
    RelatedPairRecord,
    RelationDirection,
    RelationRecord,
)


class RelationHandoffError(ValueError):
    pass


def validate_relation_pair_inputs(
    paper_a_metadata: PaperMetadata,
    paper_b_metadata: PaperMetadata,
) -> None:
    if paper_a_metadata.paper_id >= paper_b_metadata.paper_id:
        raise RelationHandoffError(
            "paper A and B must be ordered by ascending paper_id"
        )


def validate_relation_handoff(
    record: RelationRecord,
    paper_a_metadata: PaperMetadata,
    paper_b_metadata: PaperMetadata,
) -> None:
    validate_relation_pair_inputs(paper_a_metadata, paper_b_metadata)
    if record.paper_a_id != paper_a_metadata.paper_id:
        raise RelationHandoffError("output paper_a_id does not match input paper A")
    if record.paper_b_id != paper_b_metadata.paper_id:
        raise RelationHandoffError("output paper_b_id does not match input paper B")
    if not isinstance(record, RelatedPairRecord):
        return
    if (
        record.direction is RelationDirection.A_TO_B
        and paper_a_metadata.relation_reference_date
        < paper_b_metadata.relation_reference_date
    ):
        raise RelationHandoffError(
            "a_to_b is impossible because paper A predates paper B"
        )
    if (
        record.direction is RelationDirection.B_TO_A
        and paper_b_metadata.relation_reference_date
        < paper_a_metadata.relation_reference_date
    ):
        raise RelationHandoffError(
            "b_to_a is impossible because paper B predates paper A"
        )


def validate_relation_revision_handoff(
    revised_record: RelationRecord,
    current_relation: RelationRecord,
    paper_a_metadata: PaperMetadata,
    paper_b_metadata: PaperMetadata,
) -> None:
    validate_relation_handoff(
        revised_record,
        paper_a_metadata,
        paper_b_metadata,
    )
    if revised_record == current_relation:
        raise RelationHandoffError(
            "revised relation record must differ from current relation"
        )
