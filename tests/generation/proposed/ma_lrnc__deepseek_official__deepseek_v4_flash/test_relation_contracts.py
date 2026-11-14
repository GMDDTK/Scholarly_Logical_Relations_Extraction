from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    RelationRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.relation_validation import (
    RelationHandoffError,
    validate_relation_handoff,
)


RELATION_ADAPTER = TypeAdapter(RelationRecord)


def metadata(paper_id: str, published: date) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}论文",
        authors=["张三"],
        publication_date=published,
    )


def related_payload(**overrides):
    payload = {
        "record_type": "related",
        "paper_a_id": "P001",
        "paper_b_id": "P002",
        "relation_type": "拓展",
        "direction": "b_to_a",
        "relation_explanation": "论文B在论文A模型中增加了新的作用变量。",
        "paper_a_evidence_quotes": ["论文A提出基础模型。"],
        "paper_b_evidence_quotes": ["论文B在基础模型中增加变量。"],
    }
    payload.update(overrides)
    return payload


def test_relation_record_discriminates_related_and_unrelated() -> None:
    related = RELATION_ADAPTER.validate_python(related_payload())
    unrelated = RELATION_ADAPTER.validate_python(
        {
            "record_type": "unrelated",
            "paper_a_id": "P001",
            "paper_b_id": "P002",
            "no_relation_reason": "两篇论文只有宽泛主题相似。",
        }
    )

    assert related.record_type == "related"
    assert unrelated.record_type == "unrelated"

    with pytest.raises(ValidationError):
        RELATION_ADAPTER.validate_python(
            {
                "record_type": "unrelated",
                "paper_a_id": "P001",
                "paper_b_id": "P002",
                "no_relation_reason": "主题相似但不存在实质关系。",
                "confidence": 0.8,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        related_payload(paper_a_id="P002", paper_b_id="P001"),
        related_payload(paper_b_id="P001"),
        related_payload(
            paper_a_evidence_quotes=["重复证据", "重复证据"],
        ),
        related_payload(relation_type="互补", direction="a_to_b"),
        related_payload(relation_type="拓展", direction="symmetric"),
    ],
)
def test_relation_record_rejects_invalid_pair_and_cross_field_values(
    payload,
) -> None:
    with pytest.raises(ValidationError):
        RELATION_ADAPTER.validate_python(payload)


def test_relation_handoff_checks_ids_and_temporal_direction() -> None:
    paper_a = metadata("P001", date(2020, 1, 1))
    paper_b = metadata("P002", date(2021, 1, 1))
    valid = RELATION_ADAPTER.validate_python(related_payload())

    validate_relation_handoff(valid, paper_a, paper_b)

    wrong_ids = RELATION_ADAPTER.validate_python(
        related_payload(paper_a_id="P003", paper_b_id="P004")
    )
    with pytest.raises(RelationHandoffError):
        validate_relation_handoff(wrong_ids, paper_a, paper_b)

    impossible = RELATION_ADAPTER.validate_python(
        related_payload(direction="a_to_b")
    )
    with pytest.raises(RelationHandoffError):
        validate_relation_handoff(impossible, paper_a, paper_b)


def test_relation_handoff_allows_same_day_direction() -> None:
    same_day = date(2021, 1, 1)
    paper_a = metadata("P001", same_day)
    paper_b = metadata("P002", same_day)
    record = RELATION_ADAPTER.validate_python(
        related_payload(direction="a_to_b")
    )

    validate_relation_handoff(record, paper_a, paper_b)


def test_relation_handoff_prefers_source_publication_date_for_translation() -> None:
    paper_a = PaperMetadata(
        paper_id="P025",
        title="高等教育信息素养框架",
        authors=["韩丽风"],
        publication_date=date(2015, 11, 21),
        source_publication_date=date(2015, 2, 5),
    )
    paper_b = PaperMetadata(
        paper_id="P027",
        title="ACRL《高等教育信息素养框架》解读",
        authors=["秦小燕"],
        publication_date=date(2015, 10, 5),
    )
    record = RELATION_ADAPTER.validate_python(
        related_payload(
            paper_a_id="P025",
            paper_b_id="P027",
            direction="b_to_a",
        )
    )

    validate_relation_handoff(record, paper_a, paper_b)


def test_paper_metadata_rejects_source_date_after_current_publication() -> None:
    with pytest.raises(ValidationError, match="source_publication_date"):
        PaperMetadata(
            paper_id="P025",
            title="译文",
            authors=["张三"],
            publication_date=date(2015, 2, 5),
            source_publication_date=date(2015, 11, 21),
        )
