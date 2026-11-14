from pathlib import Path

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.pdf_preprocessing import (
    clean_pymupdf4llm_markup,
    write_pdf_markdown,
)


def test_clean_pymupdf4llm_markup_keeps_text_and_removes_selected_tags() -> None:
    markdown = (
        "正文<sup>［1］</sup><u>重点</u>"
        "information<br>literacy<br/>education<br />course"
        "<SUP>2</SUP><U>结尾</U>"
    )

    cleaned = clean_pymupdf4llm_markup(markdown)

    assert cleaned == (
        "正文［1］重点information literacy education course2结尾"
    )


def test_write_pdf_markdown_outputs_one_full_document_without_page_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "paper.pdf"
    input_pdf.write_bytes(b"%PDF-placeholder")
    output_markdown = tmp_path / "paper.md"
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_to_markdown(path: str, **kwargs: object) -> str:
        calls.append((path, kwargs))
        return "# 论文标题\n\n正文内容。<br>\n"

    monkeypatch.setattr("pymupdf4llm.to_markdown", fake_to_markdown)

    result = write_pdf_markdown(
        input_pdf=input_pdf,
        output_markdown=output_markdown,
    )

    assert result == "# 论文标题\n\n正文内容。 \n"
    assert output_markdown.read_text(encoding="utf-8") == result
    assert calls == [
        (
            str(input_pdf),
            {
                "page_chunks": False,
                "page_separators": False,
                "header": False,
                "footer": False,
                "use_ocr": False,
            },
        )
    ]


def test_write_pdf_markdown_does_not_leave_output_when_extraction_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "paper.pdf"
    input_pdf.write_bytes(b"%PDF-placeholder")
    output_markdown = tmp_path / "paper.md"

    def failing_to_markdown(*_: object, **__: object) -> str:
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr("pymupdf4llm.to_markdown", failing_to_markdown)

    try:
        write_pdf_markdown(
            input_pdf=input_pdf,
            output_markdown=output_markdown,
        )
    except RuntimeError as error:
        assert str(error) == "simulated extraction failure"
    else:
        raise AssertionError("expected extraction failure")

    assert not output_markdown.exists()


def test_write_pdf_markdown_rejects_empty_cleaned_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "paper.pdf"
    input_pdf.write_bytes(b"%PDF-placeholder")
    output_markdown = tmp_path / "paper.md"
    monkeypatch.setattr(
        "pymupdf4llm.to_markdown",
        lambda *_args, **_kwargs: "<br>",
    )

    try:
        write_pdf_markdown(
            input_pdf=input_pdf,
            output_markdown=output_markdown,
        )
    except ValueError as error:
        assert str(error) == "PyMuPDF4LLM produced no non-whitespace text"
    else:
        raise AssertionError("empty cleaned document must be rejected")

    assert not output_markdown.exists()
