from datetime import date

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.prompting import (
    build_relation_inference_system_prompt,
    build_relation_inference_user_message,
    build_relation_review_system_prompt,
    build_relation_review_user_message,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    UnrelatedPairRecord,
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


def metadata(
    paper_id: str,
    title: str,
    *,
    source_publication_date: date | None = None,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["张三"],
        publication_date=date(2021, 1, 1),
        source_publication_date=source_publication_date,
    )


def test_relation_system_prompt_combines_shared_common_mode_and_schema_once() -> None:
    prompt = build_relation_inference_system_prompt()

    assert prompt.count("# 学术逻辑关系共同规则") == 1
    assert prompt.count("# 关系推断智能体：共同规则") == 1
    assert prompt.count("# 关系推断智能体：首次推断模式") == 1
    assert prompt.index("# 学术逻辑关系共同规则") < prompt.index(
        "# 关系推断智能体：共同规则"
    )
    assert prompt.index("# 关系推断智能体：共同规则") < prompt.index(
        "# 关系推断智能体：首次推断模式"
    )
    assert '"record_type"' in prompt
    assert "<output_json_schema>" in prompt
    assert "source_publication_date" not in prompt
    assert "发表日期 只用于排除时间上不可能的方向" in " ".join(prompt.split())


def test_relation_user_message_groups_each_paper_and_preserves_markdown() -> None:
    message = build_relation_inference_user_message(
        paper_a_metadata=metadata("P001", "论文A"),
        paper_a_profile=empty_profile(),
        paper_a_full_text="# A正文\nA内容。",
        paper_b_metadata=metadata("P002", "论文B"),
        paper_b_profile=empty_profile(),
        paper_b_full_text="# B正文\nB内容。",
    )

    assert message.startswith("<relation_pair>")
    assert message.rstrip().endswith("</relation_pair>")
    assert message.index("<paper_a>") < message.index("<paper_b>")
    assert '"paper_id": "P001"' in message
    assert '"paper_id": "P002"' in message
    assert "<full_text>\n# A正文\nA内容。\n</full_text>" in message
    assert "<full_text>\n# B正文\nB内容。\n</full_text>" in message
    assert "pair_id" not in message
    assert "<source_publication_date_rule>" not in message


def test_relation_user_message_includes_source_date_only_when_present() -> None:
    message = build_relation_inference_user_message(
        paper_a_metadata=metadata(
            "P001",
            "译文",
            source_publication_date=date(2020, 1, 1),
        ),
        paper_a_profile=empty_profile(),
        paper_a_full_text="A正文。",
        paper_b_metadata=metadata("P002", "非译文"),
        paper_b_profile=empty_profile(),
        paper_b_full_text="B正文。",
    )

    assert message.count('"source_publication_date"') == 1
    assert '"source_publication_date": "2020-01-01"' in message
    assert message.count("<source_publication_date_rule>") == 1
    assert "论文 A：2020-01-01（使用 `source_publication_date`）" in message
    assert "论文 B：2021-01-01（使用 `publication_date`）" in message
    assert "不得推测、补写或修改缺失的源文献日期" in message
    assert "不能单独证明关系存在、关系类别、引用、阅读、思想来源或历史影响" in message
    assert "译文发表较晚不能单独否决" in message


def test_relation_review_prompt_reuses_shared_rules_and_appends_current_record() -> None:
    system_prompt = build_relation_review_system_prompt()
    user_message = build_relation_review_user_message(
        paper_a_metadata=metadata("P001", "论文A"),
        paper_a_profile=empty_profile(),
        paper_a_full_text="# A正文\nA内容。",
        paper_b_metadata=metadata("P002", "论文B"),
        paper_b_profile=empty_profile(),
        paper_b_full_text="# B正文\nB内容。",
        current_relation=UnrelatedPairRecord(
            record_type="unrelated",
            paper_a_id="P001",
            paper_b_id="P002",
            no_relation_reason="两篇论文只有主题相似。",
        ),
    )

    assert system_prompt.count("# 学术逻辑关系共同规则") == 1
    assert system_prompt.count("# 证据核验智能体：关系核验模式") == 1
    assert '"target_fields"' in system_prompt
    assert user_message.index("<relation_pair>") < user_message.index(
        "<current_relation_record>"
    )
    assert '"record_type": "unrelated"' in user_message


def test_relation_review_user_message_inherits_conditional_source_date_rule() -> None:
    user_message = build_relation_review_user_message(
        paper_a_metadata=metadata(
            "P001",
            "译文",
            source_publication_date=date(2020, 1, 1),
        ),
        paper_a_profile=empty_profile(),
        paper_a_full_text="A正文。",
        paper_b_metadata=metadata("P002", "非译文"),
        paper_b_profile=empty_profile(),
        paper_b_full_text="B正文。",
        current_relation=UnrelatedPairRecord(
            record_type="unrelated",
            paper_a_id="P001",
            paper_b_id="P002",
            no_relation_reason="两篇论文只有主题相似。",
        ),
    )

    assert user_message.count("<source_publication_date_rule>") == 1
    assert user_message.index("<source_publication_date_rule>") < user_message.index(
        "<current_relation_record>"
    )
