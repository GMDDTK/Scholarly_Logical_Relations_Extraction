import html
import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import TypeAdapter

from .contracts.paper_metadata import PaperMetadata
from .contracts.paper_profile import PaperProfileContent
from .contracts.profile_review import ProfileReviewIssue, ProfileReviewResult
from .contracts.relation_record import RelationRecord
from .contracts.relation_review import RelationReviewIssue, RelationReviewResult
from .relation_review_validation import validate_relation_review_issue_fields
from .relation_validation import validate_relation_pair_inputs
from .relation_validation import validate_relation_handoff


PaperParserMode = Literal["initial_generation", "targeted_revision"]
PROMPT_ROOT = Path(__file__).resolve().parent / "prompts" / "paper_parser"
EVIDENCE_REVIEWER_PROMPT_ROOT = (
    Path(__file__).resolve().parent / "prompts" / "evidence_reviewer"
)
SHARED_PROMPT_ROOT = Path(__file__).resolve().parent / "prompts" / "shared"
RELATION_INFERENCE_PROMPT_ROOT = (
    Path(__file__).resolve().parent / "prompts" / "relation_inference"
)


def _read_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8").strip()


def build_system_prompt(mode: PaperParserMode) -> str:
    mode_filename = {
        "initial_generation": "initial_generation.md",
        "targeted_revision": "targeted_revision.md",
    }[mode]
    schema_json = json.dumps(
        PaperProfileContent.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return (
        f"{_read_prompt('common_rules.md')}\n\n"
        f"{_read_prompt(mode_filename)}\n\n"
        "<output_json_schema>\n"
        f"{schema_json}\n"
        "</output_json_schema>"
    )


def _validate_article_inputs(
    title: str,
    authors: Sequence[str],
    article_full_text: str,
) -> tuple[str, list[str], str]:
    clean_title = title.strip()
    clean_authors = [author.strip() for author in authors if author.strip()]
    if not clean_title:
        raise ValueError("title must not be blank")
    if not clean_authors:
        raise ValueError("authors must contain at least one nonblank author")
    if not article_full_text.strip():
        raise ValueError("article_full_text must not be blank")
    return clean_title, clean_authors, article_full_text


def _article_block(
    title: str,
    authors: Sequence[str],
    article_full_text: str,
) -> str:
    clean_title, clean_authors, full_text = _validate_article_inputs(
        title,
        authors,
        article_full_text,
    )
    escaped_title = html.escape(clean_title, quote=False)
    escaped_authors = html.escape("；".join(clean_authors), quote=False)
    return (
        "<article_metadata>\n"
        f"<title>{escaped_title}</title>\n"
        f"<authors>{escaped_authors}</authors>\n"
        "</article_metadata>\n\n"
        "<article_full_text>\n"
        f"{full_text}\n"
        "</article_full_text>"
    )


def build_initial_user_message(
    *,
    title: str,
    authors: Sequence[str],
    article_full_text: str,
) -> str:
    return _article_block(title, authors, article_full_text)


def build_revision_user_message(
    *,
    title: str,
    authors: Sequence[str],
    article_full_text: str,
    current_profile: PaperProfileContent,
    review_issues: Sequence[ProfileReviewIssue],
) -> str:
    if not review_issues:
        raise ValueError("review_issues must contain at least one issue")
    current_profile_json = current_profile.model_dump_json(indent=2)
    review_issues_json = json.dumps(
        [issue.model_dump(mode="json") for issue in review_issues],
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"{_article_block(title, authors, article_full_text)}\n\n"
        "<current_profile>\n"
        f"{current_profile_json}\n"
        "</current_profile>\n\n"
        "<review_issues>\n"
        f"{review_issues_json}\n"
        "</review_issues>"
    )


def build_profile_review_system_prompt() -> str:
    fixed_prompt = (
        EVIDENCE_REVIEWER_PROMPT_ROOT / "profile_review.md"
    ).read_text(encoding="utf-8").strip()
    schema_json = json.dumps(
        TypeAdapter(ProfileReviewResult).json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return (
        f"{fixed_prompt}\n\n"
        "<output_json_schema>\n"
        f"{schema_json}\n"
        "</output_json_schema>"
    )


def build_profile_review_user_message(
    *,
    title: str,
    authors: Sequence[str],
    article_full_text: str,
    current_profile: PaperProfileContent,
) -> str:
    return (
        f"{_article_block(title, authors, article_full_text)}\n\n"
        "<current_profile>\n"
        f"{current_profile.model_dump_json(indent=2)}\n"
        "</current_profile>"
    )


def build_relation_inference_system_prompt() -> str:
    return _build_relation_inference_system_prompt("initial_inference.md")


def build_relation_revision_system_prompt() -> str:
    return _build_relation_inference_system_prompt("targeted_revision.md")


def _build_relation_inference_system_prompt(mode_filename: str) -> str:
    fixed_parts = [
        (SHARED_PROMPT_ROOT / "relation_rules.md")
        .read_text(encoding="utf-8")
        .strip(),
        (RELATION_INFERENCE_PROMPT_ROOT / "common_rules.md")
        .read_text(encoding="utf-8")
        .strip(),
        (RELATION_INFERENCE_PROMPT_ROOT / mode_filename)
        .read_text(encoding="utf-8")
        .strip(),
    ]
    schema_json = json.dumps(
        TypeAdapter(RelationRecord).json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return (
        "\n\n".join(fixed_parts)
        + "\n\n<output_json_schema>\n"
        + schema_json
        + "\n</output_json_schema>"
    )


def build_relation_inference_user_message(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
) -> str:
    validate_relation_pair_inputs(paper_a_metadata, paper_b_metadata)
    if not paper_a_full_text.strip():
        raise ValueError("paper_a_full_text must not be blank")
    if not paper_b_full_text.strip():
        raise ValueError("paper_b_full_text must not be blank")

    pair_message = (
        "<relation_pair>\n"
        f"{_relation_paper_block('paper_a', paper_a_metadata, paper_a_profile, paper_a_full_text)}\n"
        f"{_relation_paper_block('paper_b', paper_b_metadata, paper_b_profile, paper_b_full_text)}\n"
        "</relation_pair>"
    )
    source_date_rule = _source_publication_date_rule(
        paper_a_metadata,
        paper_b_metadata,
    )
    if source_date_rule is None:
        return pair_message
    return f"{pair_message}\n\n{source_date_rule}"


def build_relation_review_system_prompt() -> str:
    fixed_parts = [
        (SHARED_PROMPT_ROOT / "relation_rules.md")
        .read_text(encoding="utf-8")
        .strip(),
        (EVIDENCE_REVIEWER_PROMPT_ROOT / "relation_review.md")
        .read_text(encoding="utf-8")
        .strip(),
    ]
    schema_json = json.dumps(
        TypeAdapter(RelationReviewResult).json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return (
        "\n\n".join(fixed_parts)
        + "\n\n<output_json_schema>\n"
        + schema_json
        + "\n</output_json_schema>"
    )


def build_relation_post_revision_review_system_prompt() -> str:
    fixed_parts = [
        (SHARED_PROMPT_ROOT / "relation_rules.md")
        .read_text(encoding="utf-8")
        .strip(),
        (EVIDENCE_REVIEWER_PROMPT_ROOT / "relation_review.md")
        .read_text(encoding="utf-8")
        .strip(),
        (EVIDENCE_REVIEWER_PROMPT_ROOT / "post_revision_review.md")
        .read_text(encoding="utf-8")
        .strip(),
    ]
    schema_json = json.dumps(
        TypeAdapter(RelationReviewResult).json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return (
        "\n\n".join(fixed_parts)
        + "\n\n<output_json_schema>\n"
        + schema_json
        + "\n</output_json_schema>"
    )


def build_relation_review_user_message(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    current_relation: RelationRecord,
) -> str:
    validate_relation_handoff(
        current_relation,
        paper_a_metadata,
        paper_b_metadata,
    )
    pair_message = build_relation_inference_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
    )
    return (
        f"{pair_message}\n\n"
        "<current_relation_record>\n"
        f"{TypeAdapter(RelationRecord).dump_json(current_relation, indent=2).decode('utf-8')}\n"
        "</current_relation_record>"
    )


def build_relation_post_revision_review_user_message(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    previous_review_issues: Sequence[RelationReviewIssue],
    revised_relation: RelationRecord,
) -> str:
    if not previous_review_issues:
        raise ValueError("previous_review_issues must contain at least one issue")
    validate_relation_handoff(
        revised_relation,
        paper_a_metadata,
        paper_b_metadata,
    )
    pair_message = build_relation_inference_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
    )
    review_issues_json = json.dumps(
        [issue.model_dump(mode="json") for issue in previous_review_issues],
        ensure_ascii=False,
        indent=2,
    )
    revised_relation_json = (
        TypeAdapter(RelationRecord)
        .dump_json(revised_relation, indent=2)
        .decode("utf-8")
    )
    return (
        f"{pair_message}\n\n"
        "<previous_review_issues>\n"
        f"{review_issues_json}\n"
        "</previous_review_issues>\n\n"
        "<revised_relation_record>\n"
        f"{revised_relation_json}\n"
        "</revised_relation_record>"
    )


def build_relation_revision_user_message(
    *,
    paper_a_metadata: PaperMetadata,
    paper_a_profile: PaperProfileContent,
    paper_a_full_text: str,
    paper_b_metadata: PaperMetadata,
    paper_b_profile: PaperProfileContent,
    paper_b_full_text: str,
    current_relation: RelationRecord,
    review_issues: Sequence[RelationReviewIssue],
) -> str:
    validate_relation_handoff(
        current_relation,
        paper_a_metadata,
        paper_b_metadata,
    )
    validate_relation_review_issue_fields(review_issues, current_relation)
    pair_message = build_relation_inference_user_message(
        paper_a_metadata=paper_a_metadata,
        paper_a_profile=paper_a_profile,
        paper_a_full_text=paper_a_full_text,
        paper_b_metadata=paper_b_metadata,
        paper_b_profile=paper_b_profile,
        paper_b_full_text=paper_b_full_text,
    )
    current_relation_json = (
        TypeAdapter(RelationRecord)
        .dump_json(current_relation, indent=2)
        .decode("utf-8")
    )
    review_issues_json = json.dumps(
        [issue.model_dump(mode="json") for issue in review_issues],
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"{pair_message}\n\n"
        "<current_relation_record>\n"
        f"{current_relation_json}\n"
        "</current_relation_record>\n\n"
        "<review_issues>\n"
        f"{review_issues_json}\n"
        "</review_issues>"
    )


def _relation_paper_block(
    tag: str,
    metadata: PaperMetadata,
    profile: PaperProfileContent,
    full_text: str,
) -> str:
    metadata_json = metadata.model_dump_json(indent=2, exclude_none=True)
    profile_json = profile.model_dump_json(indent=2)
    return (
        f"<{tag}>\n"
        "<metadata>\n"
        f"{metadata_json}\n"
        "</metadata>\n"
        "<profile>\n"
        f"{profile_json}\n"
        "</profile>\n"
        "<full_text>\n"
        f"{full_text}\n"
        "</full_text>\n"
        f"</{tag}>"
    )


def _source_publication_date_rule(
    paper_a_metadata: PaperMetadata,
    paper_b_metadata: PaperMetadata,
) -> str | None:
    if (
        paper_a_metadata.source_publication_date is None
        and paper_b_metadata.source_publication_date is None
    ):
        return None

    def reference_line(label: str, metadata: PaperMetadata) -> str:
        field_name = (
            "source_publication_date"
            if metadata.source_publication_date is not None
            else "publication_date"
        )
        return (
            f"- 论文 {label}：{metadata.relation_reference_date.isoformat()}"
            f"（使用 `{field_name}`）"
        )

    return (
        "<source_publication_date_rule>\n"
        "`publication_date` 表示当前输入论文自身的发表日期；\n"
        "`source_publication_date` 表示该论文所承载源文献的原始发表日期。\n"
        "本论文对的时间方向基准：\n"
        f"{reference_line('A', paper_a_metadata)}\n"
        f"{reference_line('B', paper_b_metadata)}\n"
        "只使用输入中明确提供的日期，不得推测、补写或修改缺失的源文献日期。\n"
        "该日期规则只用于排除时间上不可能的方向，不能单独证明关系存在、"
        "关系类别、引用、阅读、思想来源或历史影响。\n"
        "译文发表较晚不能单独否决另一篇较早论文对其源文献内容实施知识动作；"
        "关系仍须由双方全文证据支持。\n"
        "</source_publication_date_rule>"
    )
