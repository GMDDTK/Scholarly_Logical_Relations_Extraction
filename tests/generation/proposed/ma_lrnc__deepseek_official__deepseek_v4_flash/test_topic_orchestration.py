import json
from datetime import date
from pathlib import Path

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.result_paths import (
    build_topic_result_paths,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.topic_input import (
    PaperInputFailure,
    PaperInputMatch,
    PaperPairInput,
    TopicInputSet,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.topic_orchestration import (
    run_topic_orchestration,
)


def _metadata(paper_id: str) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}题名",
        authors=[f"{paper_id}作者"],
        publication_date=date(2020, 1, int(paper_id[-1])),
    )


def _empty_profile() -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


def _stage_metrics(
    *,
    elapsed_seconds: float = 1.0,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> StageMetrics:
    return StageMetrics(
        elapsed_seconds=elapsed_seconds,
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _profile_outcome() -> object:
    return type(
        "ProfileOutcome",
        (),
        {
            "profile": _empty_profile(),
            "metrics": _stage_metrics(),
        },
    )()


def _topic_input(
    tmp_path: Path,
    *,
    paper_ids: tuple[str, ...],
    failures: tuple[PaperInputFailure, ...] = (),
) -> TopicInputSet:
    topic_directory = tmp_path / "data" / "论文" / "测试主题"
    topic_directory.mkdir(parents=True)
    matches = []
    for row_number, paper_id in enumerate(paper_ids, start=2):
        pdf_path = topic_directory / f"{paper_id}题名.pdf"
        pdf_path.write_bytes(b"%PDF-placeholder")
        matches.append(
            PaperInputMatch(
                pdf_path=pdf_path,
                catalog_row=row_number,
                metadata=_metadata(paper_id),
            )
        )
    pairs = tuple(
        PaperPairInput(
            pair_key=(
                f"{matches[left].metadata.paper_id}__"
                f"{matches[right].metadata.paper_id}"
            ),
            paper_a=matches[left],
            paper_b=matches[right],
        )
        for left in range(len(matches))
        for right in range(left + 1, len(matches))
    )
    return TopicInputSet(
        topic="测试主题",
        topic_directory=topic_directory,
        workbook_path=tmp_path / "catalog.xlsx",
        worksheet_name="Sheet1",
        pdf_paths=tuple(match.pdf_path for match in matches)
        + tuple(failure.pdf_path for failure in failures),
        matches=tuple(matches),
        failures=failures,
        pairs=pairs,
    )


def _paths(tmp_path: Path):
    return build_topic_result_paths(
        topic="测试主题",
        result_unit_root=tmp_path / "results",
        temp_unit_root=tmp_path / "tmp",
    )


def test_topic_orchestration_runs_serial_papers_then_pairs_and_reuses_full_text(
    tmp_path: Path,
) -> None:
    topic_input = _topic_input(
        tmp_path,
        paper_ids=("P001", "P002"),
    )
    events: list[str] = []

    def pdf_writer(*, input_pdf: Path, output_markdown: Path) -> str:
        paper_id = input_pdf.stem[:4]
        events.append(f"extract:{paper_id}")
        text = f"{paper_id}全文"
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(text, encoding="utf-8")
        return text

    def profile_runner(**kwargs: object) -> object:
        metadata = kwargs["paper_metadata"]
        assert isinstance(metadata, PaperMetadata)
        assert kwargs["article_full_text"] == f"{metadata.paper_id}全文"
        events.append(f"profile:{metadata.paper_id}")
        return _profile_outcome()

    def relation_runner(**kwargs: object) -> object:
        paper_a = kwargs["paper_a_metadata"]
        paper_b = kwargs["paper_b_metadata"]
        assert isinstance(paper_a, PaperMetadata)
        assert isinstance(paper_b, PaperMetadata)
        events.append(f"relation:{paper_a.paper_id}__{paper_b.paper_id}")
        return type(
            "RelationOutcome",
            (),
            {"metrics": _stage_metrics(elapsed_seconds=2.0)},
        )()

    first = run_topic_orchestration(
        topic_input=topic_input,
        processed_root=tmp_path / "processed",
        model="deepseek-v4-flash",
        parser_agent=object(),
        reviewer_agent=object(),
        relation_agent=object(),
        paths=_paths(tmp_path),
        pdf_writer=pdf_writer,
        profile_runner=profile_runner,
        relation_runner=relation_runner,
        current_run_token_usage=lambda: TokenUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
        started_at=10.0,
        clock=lambda: 12.5,
    )

    assert first.status == "relations_complete"
    assert first.papers.model_dump() == {
        "expected": 2,
        "completed": 2,
        "failed": 0,
        "input_failed": 0,
    }
    assert first.pairs.model_dump() == {
        "expected": 1,
        "completed": 1,
        "failed": 0,
        "blocked": 0,
    }
    assert first.consumption.current_run == StageMetrics(
        elapsed_seconds=2.5,
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )
    assert first.consumption.completed_result_chain == StageMetrics(
        elapsed_seconds=4.0,
        token_usage=TokenUsage(
            prompt_tokens=30,
            completion_tokens=15,
            total_tokens=45,
        ),
    )
    assert events == [
        "extract:P001",
        "profile:P001",
        "extract:P002",
        "profile:P002",
        "relation:P001__P002",
    ]
    metadata_path = (
        tmp_path / "processed" / "测试主题" / "P001" / "metadata.json"
    )
    assert json.loads(metadata_path.read_text(encoding="utf-8"))[
        "paper_id"
    ] == "P001"

    events.clear()

    def must_not_extract(**_: object) -> str:
        raise AssertionError("existing full_text.md must be reused")

    second = run_topic_orchestration(
        topic_input=topic_input,
        processed_root=tmp_path / "processed",
        model="deepseek-v4-flash",
        parser_agent=object(),
        reviewer_agent=object(),
        relation_agent=object(),
        paths=_paths(tmp_path),
        pdf_writer=must_not_extract,
        profile_runner=profile_runner,
        relation_runner=relation_runner,
    )

    assert second.status == "relations_complete"
    assert events == [
        "profile:P001",
        "profile:P002",
        "relation:P001__P002",
    ]


def test_paper_failure_blocks_only_dependent_pairs(
    tmp_path: Path,
) -> None:
    topic_input = _topic_input(
        tmp_path,
        paper_ids=("P001", "P002", "P003"),
    )
    attempted_pairs: list[str] = []

    def pdf_writer(*, input_pdf: Path, output_markdown: Path) -> str:
        if input_pdf.stem.startswith("P002"):
            raise RuntimeError("simulated PDF failure")
        text = f"{input_pdf.stem[:4]}全文"
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(text, encoding="utf-8")
        return text

    def profile_runner(**_: object) -> object:
        return _profile_outcome()

    def relation_runner(**kwargs: object) -> object:
        paper_a = kwargs["paper_a_metadata"]
        paper_b = kwargs["paper_b_metadata"]
        key = f"{paper_a.paper_id}__{paper_b.paper_id}"
        attempted_pairs.append(key)
        return type(
            "RelationOutcome",
            (),
            {"metrics": _stage_metrics()},
        )()

    result = run_topic_orchestration(
        topic_input=topic_input,
        processed_root=tmp_path / "processed",
        model="deepseek-v4-flash",
        parser_agent=object(),
        reviewer_agent=object(),
        relation_agent=object(),
        paths=_paths(tmp_path),
        pdf_writer=pdf_writer,
        profile_runner=profile_runner,
        relation_runner=relation_runner,
    )

    assert result.status == "incomplete"
    assert result.papers.model_dump() == {
        "expected": 3,
        "completed": 2,
        "failed": 1,
        "input_failed": 0,
    }
    assert result.pairs.model_dump() == {
        "expected": 3,
        "completed": 1,
        "failed": 0,
        "blocked": 2,
    }
    assert attempted_pairs == ["P001__P003"]
    assert not (
        tmp_path
        / "processed"
        / "测试主题"
        / "P002"
        / "full_text.md"
    ).exists()


def test_empty_existing_full_text_is_reextracted(tmp_path: Path) -> None:
    topic_input = _topic_input(
        tmp_path,
        paper_ids=("P001",),
    )
    full_text_path = (
        tmp_path
        / "processed"
        / "测试主题"
        / "P001"
        / "full_text.md"
    )
    full_text_path.parent.mkdir(parents=True)
    full_text_path.write_text(" \n", encoding="utf-8")
    extraction_count = 0

    def pdf_writer(*, output_markdown: Path, **_: object) -> str:
        nonlocal extraction_count
        extraction_count += 1
        output_markdown.write_text("恢复后的全文", encoding="utf-8")
        return "恢复后的全文"

    def profile_runner(**kwargs: object) -> object:
        assert kwargs["article_full_text"] == "恢复后的全文"
        return _profile_outcome()

    result = run_topic_orchestration(
        topic_input=topic_input,
        processed_root=tmp_path / "processed",
        model="deepseek-v4-flash",
        parser_agent=object(),
        reviewer_agent=object(),
        relation_agent=object(),
        paths=_paths(tmp_path),
        pdf_writer=pdf_writer,
        profile_runner=profile_runner,
        relation_runner=lambda **_: object(),
    )

    assert result.status == "relations_complete"
    assert extraction_count == 1
    assert full_text_path.read_text(encoding="utf-8") == "恢复后的全文"


def test_pair_failure_does_not_stop_other_pairs(tmp_path: Path) -> None:
    topic_input = _topic_input(
        tmp_path,
        paper_ids=("P001", "P002", "P003"),
    )
    attempted_pairs: list[str] = []

    def pdf_writer(*, input_pdf: Path, output_markdown: Path) -> str:
        text = f"{input_pdf.stem[:4]}全文"
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(text, encoding="utf-8")
        return text

    def profile_runner(**_: object) -> object:
        return _profile_outcome()

    def relation_runner(**kwargs: object) -> object:
        paper_a = kwargs["paper_a_metadata"]
        paper_b = kwargs["paper_b_metadata"]
        key = f"{paper_a.paper_id}__{paper_b.paper_id}"
        attempted_pairs.append(key)
        if key == "P001__P002":
            raise RuntimeError("simulated relation failure")
        return type(
            "RelationOutcome",
            (),
            {"metrics": _stage_metrics()},
        )()

    result = run_topic_orchestration(
        topic_input=topic_input,
        processed_root=tmp_path / "processed",
        model="deepseek-v4-flash",
        parser_agent=object(),
        reviewer_agent=object(),
        relation_agent=object(),
        paths=_paths(tmp_path),
        pdf_writer=pdf_writer,
        profile_runner=profile_runner,
        relation_runner=relation_runner,
    )

    assert result.status == "incomplete"
    assert result.pairs.model_dump() == {
        "expected": 3,
        "completed": 2,
        "failed": 1,
        "blocked": 0,
    }
    assert attempted_pairs == [
        "P001__P002",
        "P001__P003",
        "P002__P003",
    ]


def test_input_failure_is_counted_and_its_pairs_are_blocked(
    tmp_path: Path,
) -> None:
    failed_pdf = (
        tmp_path / "data" / "论文" / "测试主题" / "未匹配论文.pdf"
    )
    failure = PaperInputFailure(
        pdf_path=failed_pdf,
        code="no_exact_title_match",
        message="no exact match",
    )
    topic_input = _topic_input(
        tmp_path,
        paper_ids=("P001",),
        failures=(failure,),
    )

    def pdf_writer(*, output_markdown: Path, **_: object) -> str:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text("全文", encoding="utf-8")
        return "全文"

    result = run_topic_orchestration(
        topic_input=topic_input,
        processed_root=tmp_path / "processed",
        model="deepseek-v4-flash",
        parser_agent=object(),
        reviewer_agent=object(),
        relation_agent=object(),
        paths=_paths(tmp_path),
        pdf_writer=pdf_writer,
        profile_runner=lambda **_: _profile_outcome(),
        relation_runner=lambda **_: object(),
    )

    assert result.status == "incomplete"
    assert result.papers.model_dump() == {
        "expected": 2,
        "completed": 1,
        "failed": 0,
        "input_failed": 1,
    }
    assert result.pairs.model_dump() == {
        "expected": 1,
        "completed": 0,
        "failed": 0,
        "blocked": 1,
    }
