import json
import math
import re
from collections import deque
from datetime import date
from pathlib import Path

from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.contracts import (
    ReferenceDocument,
    ReferenceLine,
    TokenUsage,
    TopicInputSet,
    TopicPaper,
)
from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
)
from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.network import (
    build_network_bundle,
    write_network_artifacts,
)
from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.result_paths import (
    CitationResultPaths,
)
from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.workflow import (
    process_paper,
)


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = deque(contents)
        self.call_count = 0

    def create_json_completion(self, messages: list[dict[str, str]]) -> CompletionResult:
        self.call_count += 1
        return CompletionResult(
            content=self.contents.popleft(),
            finish_reason="stop",
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


def _paper(index: int) -> TopicPaper:
    return TopicPaper(
        paper_id=f"P{index:03d}",
        title=f"论文{index}",
        authors=[f"作者{index}"],
        publication_date=date(2020, 1, 1),
    )


def _topic_input(tmp_path: Path, *, nonempty: bool) -> TopicInputSet:
    papers = tuple(_paper(index) for index in range(1, 51))
    workbook = tmp_path / "P001.xlsx"
    workbook.write_bytes(b"fixture")
    lines = (
        ReferenceLine(row_number=1, raw_text="[1] 作者2. 论文2[J]."),
    ) if nonempty else ()
    documents = tuple(
        ReferenceDocument(
            paper=paper,
            workbook_path=workbook,
            workbook_sha256="sha256:" + str(index).zfill(64),
            lines=lines if index == 1 else (),
        )
        for index, paper in enumerate(papers, start=1)
    )
    return TopicInputSet(
        topic="测试主题",
        catalog_workbook_path=tmp_path / "catalog.xlsx",
        catalog_sha256="sha256:" + "a" * 64,
        papers=papers,
        reference_documents=documents,
    )


def test_empty_workbook_creates_zero_edge_outcome_without_llm(tmp_path: Path) -> None:
    topic_input = _topic_input(tmp_path, nonempty=False)
    client = FakeClient([])
    paths = CitationResultPaths(result_root=tmp_path / "results", temp_root=tmp_path / "tmp")

    result = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=client,
        paths=paths,
        model="deepseek-v4-flash",
    )

    assert result.status == "accepted"
    assert result.outcome is not None
    assert result.outcome.has_references is False
    assert result.outcome.cited_papers == []
    assert client.call_count == 0


def test_nonempty_workbook_requires_identifier_then_reviewer(tmp_path: Path) -> None:
    topic_input = _topic_input(tmp_path, nonempty=True)
    client = FakeClient(
        [
            json.dumps({"cited_papers": [{"cited_paper_id": "P002", "reference_row_numbers": [1]}]}),
            json.dumps({"status": "pass", "issues": []}),
        ]
    )
    paths = CitationResultPaths(result_root=tmp_path / "results", temp_root=tmp_path / "tmp")

    result = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=client,
        paths=paths,
        model="deepseek-v4-flash",
    )

    assert result.status == "accepted"
    assert client.call_count == 2
    assert result.outcome.cited_papers[0].cited_paper_id == "P002"
    assert result.outcome.cited_papers[0].evidence[0].raw_text.startswith("[1]")

    reused = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=client,
        paths=paths,
        model="deepseek-v4-flash",
    )
    assert reused.status == "skipped"
    assert client.call_count == 2


def test_review_failure_has_no_outcome_and_next_run_restarts_identification(tmp_path: Path) -> None:
    topic_input = _topic_input(tmp_path, nonempty=True)
    client = FakeClient(
        [
            json.dumps({"cited_papers": []}),
            json.dumps({"status": "needs_retry", "issues": [{"description": "漏报", "reference_row_numbers": [1]}]}),
            json.dumps({"cited_papers": [{"cited_paper_id": "P002", "reference_row_numbers": [1]}]}),
            json.dumps({"status": "pass", "issues": []}),
        ]
    )
    paths = CitationResultPaths(result_root=tmp_path / "results", temp_root=tmp_path / "tmp")

    first = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=client,
        paths=paths,
        model="deepseek-v4-flash",
    )
    assert first.status == "needs_retry"
    assert first.outcome is None
    assert not paths.identification_checkpoint("P001").exists()
    assert not paths.review_checkpoint("P001").exists()

    second = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=client,
        paths=paths,
        model="deepseek-v4-flash",
    )
    assert second.status == "accepted"
    assert client.call_count == 4


def test_review_technical_failure_reuses_valid_identification_checkpoint(
    tmp_path: Path,
) -> None:
    topic_input = _topic_input(tmp_path, nonempty=True)
    paths = CitationResultPaths(result_root=tmp_path / "results", temp_root=tmp_path / "tmp")
    first_client = FakeClient(
        [
            json.dumps({"cited_papers": [{"cited_paper_id": "P002", "reference_row_numbers": [1]}]}),
            "bad",
            "bad",
            "bad",
        ]
    )

    first = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=first_client,
        paths=paths,
        model="deepseek-v4-flash",
    )
    second_client = FakeClient([json.dumps({"status": "pass", "issues": []})])
    second = process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=second_client,
        paths=paths,
        model="deepseek-v4-flash",
    )

    assert first.status == "failed"
    assert first_client.call_count == 4
    assert second.status == "accepted"
    assert second_client.call_count == 1


def test_network_bundle_has_fixed_nodes_directed_edges_and_empty_record(tmp_path: Path) -> None:
    topic_input = _topic_input(tmp_path, nonempty=True)
    paths = CitationResultPaths(result_root=tmp_path / "results", temp_root=tmp_path / "tmp")
    passing = FakeClient(
        [
            json.dumps({"cited_papers": [{"cited_paper_id": "P002", "reference_row_numbers": [1]}]}),
            json.dumps({"status": "pass", "issues": []}),
        ]
    )
    process_paper(
        topic_input=topic_input,
        document=topic_input.reference_documents[0],
        client=passing,
        paths=paths,
        model="deepseek-v4-flash",
    )
    empty_client = FakeClient([])
    for document in topic_input.reference_documents[1:]:
        process_paper(
            topic_input=topic_input,
            document=document,
            client=empty_client,
            paths=paths,
            model="deepseek-v4-flash",
        )

    bundle = build_network_bundle(topic_input=topic_input, paths=paths, model="deepseek-v4-flash")

    assert len(bundle.nodes) == 50
    assert [(edge.source_paper_id, edge.target_paper_id) for edge in bundle.edges] == [("P001", "P002")]
    assert "P002" in bundle.papers_without_references

    write_network_artifacts(bundle=bundle, paths=paths)
    assert (paths.network_root / "network_data.json").stat().st_size > 0
    html = (paths.network_root / "index.html").read_text(encoding="utf-8")
    assert "P001" in html
    assert "P002" in html
    assert "作者2. 论文2" in html
    assert "引用论文 → 被引论文" in html
    assert 'data-source="P001" data-target="P002"' in html
    assert "P001 引用 P002" in html
    edge = re.search(
        r'data-source="P001" data-target="P002".*?'
        r'<line class="edge-line" x1="([0-9.]+)" y1="([0-9.]+)" '
        r'x2="([0-9.]+)" y2="([0-9.]+)"',
        html,
        flags=re.DOTALL,
    )
    assert edge is not None
    edge_x2, edge_y2 = float(edge.group(3)), float(edge.group(4))
    target = bundle.nodes[1]
    target_x = 500 + 310 * target.x
    target_y = 380 + 310 * target.y
    assert math.dist((edge_x2, edge_y2), (target_x, target_y)) >= 14
    assert (paths.network_root / "figures" / "citation_network.svg").stat().st_size > 0
    assert (paths.network_root / "figures" / "citation_network.pdf").stat().st_size > 0
