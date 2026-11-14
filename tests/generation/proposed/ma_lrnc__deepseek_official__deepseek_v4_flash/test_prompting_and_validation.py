import json

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.profile_review import (
    ProfileReviewIssue,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.prompting import (
    build_initial_user_message,
    build_revision_user_message,
    build_system_prompt,
)
def empty_profile() -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


def test_system_prompt_combines_common_mode_and_generated_schema() -> None:
    prompt = build_system_prompt("initial_generation")

    assert "# 文献解析智能体：共同规则" in prompt
    assert "# 文献解析智能体：首次生成模式" in prompt
    assert "<output_json_schema>" in prompt
    assert '"research_questions"' in prompt


def test_initial_user_message_contains_only_confirmed_dynamic_inputs() -> None:
    message = build_initial_user_message(
        title="测试<title>",
        authors=["张三", "李四"],
        article_full_text="# 正文\n本文采用某方法。",
    )

    assert "<title>测试&lt;title&gt;</title>" in message
    assert "<authors>张三；李四</authors>" in message
    assert "<article_full_text>\n# 正文\n本文采用某方法。\n</article_full_text>" in message
    assert "paper_id" not in message
    assert "publication_date" not in message


def test_revision_message_places_review_issues_last_and_requires_nonempty_issues() -> None:
    issue = ProfileReviewIssue(
        target_locations=[{"field": "main_findings", "item_index": None}],
        issue_description="缺少主要发现。",
        review_evidence_quotes=[],
        revision_guidance="核对全文并补充主要发现。",
    )
    message = build_revision_user_message(
        title="测试论文",
        authors=["张三"],
        article_full_text="正文。",
        current_profile=empty_profile(),
        review_issues=[issue],
    )

    assert message.index("<current_profile>") < message.index("<review_issues>")
    assert message.rstrip().endswith("</review_issues>")
    assert json.loads(
        message.split("<review_issues>\n", 1)[1].split("\n</review_issues>", 1)[0]
    )[0]["issue_description"] == "缺少主要发现。"

    with pytest.raises(ValueError):
        build_revision_user_message(
            title="测试论文",
            authors=["张三"],
            article_full_text="正文。",
            current_profile=empty_profile(),
            review_issues=[],
        )
