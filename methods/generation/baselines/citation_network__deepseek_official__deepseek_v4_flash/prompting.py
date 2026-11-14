import json

from .contracts import CitationIdentification, ReferenceDocument, TopicInputSet


def build_identification_message(
    *,
    topic_input: TopicInputSet,
    document: ReferenceDocument,
) -> str:
    payload = {
        "topic": topic_input.topic,
        "current_paper": _paper_payload(document.paper),
        "topic_paper_catalog": [_paper_payload(paper) for paper in topic_input.papers],
        "reference_lines": [line.model_dump(mode="json") for line in document.lines],
    }
    return "请识别当前论文对主题内其他论文的直接引用。\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )


def build_review_message(
    *,
    topic_input: TopicInputSet,
    document: ReferenceDocument,
    identification: CitationIdentification,
) -> str:
    payload = {
        "topic": topic_input.topic,
        "current_paper": _paper_payload(document.paper),
        "topic_paper_catalog": [_paper_payload(paper) for paper in topic_input.papers],
        "reference_lines": [line.model_dump(mode="json") for line in document.lines],
        "identification_to_review": identification.model_dump(mode="json"),
    }
    return "请独立审核当前引用识别结果。\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )


def _paper_payload(paper) -> dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "publication_date": paper.publication_date.isoformat(),
        "doi": paper.doi,
    }
