import json
from collections import deque
from datetime import date
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.relation_inference import (
    RelationInferenceAgent,
    RelationInferenceContentRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    RelatedPairRecord,
    UnrelatedPairRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_review import (
    RelationRecordField,
    RelationReviewIssue,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    RelationInferenceRunResult,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.prompting import (
    build_relation_revision_system_prompt,
    build_relation_revision_user_message,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.workflow import (
    run_targeted_relation_revision,
)


def metadata(paper_id: str, published: date) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}论文",
        authors=["张三"],
        publication_date=published,
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


def current_unrelated() -> UnrelatedPairRecord:
    return UnrelatedPairRecord(
        record_type="unrelated",
        paper_a_id="P001",
        paper_b_id="P002",
        no_relation_reason="两篇论文只有主题相似。",
    )


def revision_issues() -> list[RelationReviewIssue]:
    return [
        RelationReviewIssue(
            target_fields=[
                RelationRecordField.RECORD_TYPE,
                RelationRecordField.NO_RELATION_REASON,
            ],
            issue_description="遗漏了论文B对论文A模型的实质拓展。",
            paper_a_review_evidence_quotes=["A提出基础模型。"],
            paper_b_review_evidence_quotes=["B增加新的变量。"],
            revision_guidance="回查全文并重新判断有无关系。",
        )
    ]


def revised_related() -> RelatedPairRecord:
    return RelatedPairRecord(
        record_type="related",
        paper_a_id="P001",
        paper_b_id="P002",
        relation_type="拓展",
        direction="b_to_a",
        relation_explanation="B在A模型上增加变量。",
        paper_a_evidence_quotes=["A提出基础模型。"],
        paper_b_evidence_quotes=["B增加新的变量。"],
    )


def revised_unrelated() -> UnrelatedPairRecord:
    return UnrelatedPairRecord(
        record_type="unrelated",
        paper_a_id="P001",
        paper_b_id="P002",
        no_relation_reason="两篇论文只有方法名称相近，缺少可对齐的具体知识动作。",
    )


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = deque(contents)
        self.calls = 0

    def create_json_completion(self, messages):
        self.calls += 1
        return CompletionResult(
            content=self.contents.popleft(),
            finish_reason="stop",
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


def test_relation_revision_prompt_combines_layers_and_orders_dynamic_inputs() -> None:
    system_prompt = build_relation_revision_system_prompt()
    user_message = build_relation_revision_user_message(
        paper_a_metadata=metadata("P001", date(2020, 1, 1)),
        paper_a_profile=empty_profile(),
        paper_a_full_text="A全文。",
        paper_b_metadata=metadata("P002", date(2021, 1, 1)),
        paper_b_profile=empty_profile(),
        paper_b_full_text="B全文。",
        current_relation=current_unrelated(),
        review_issues=revision_issues(),
    )

    assert system_prompt.count("# 学术逻辑关系共同规则") == 1
    assert system_prompt.count("# 关系推断智能体：共同规则") == 1
    assert system_prompt.count("# 关系推断智能体：定向修订模式") == 1
    assert system_prompt.index("# 学术逻辑关系共同规则") < system_prompt.index(
        "# 关系推断智能体：共同规则"
    )
    assert system_prompt.index("# 关系推断智能体：共同规则") < system_prompt.index(
        "# 关系推断智能体：定向修订模式"
    )
    assert user_message.index("<relation_pair>") < user_message.index(
        "<current_relation_record>"
    )
    assert user_message.index("<current_relation_record>") < user_message.index(
        "<review_issues>"
    )
    assert '"target_fields"' in user_message


def test_relation_revision_retries_unchanged_record_and_accumulates_usage() -> None:
    current = current_unrelated()
    client = FakeClient(
        [
            current.model_dump_json(),
            revised_related().model_dump_json(),
        ]
    )
    agent = RelationInferenceAgent(client=client, sleep=lambda _: None)

    result = agent.targeted_revision(
        paper_a_metadata=metadata("P001", date(2020, 1, 1)),
        paper_a_profile=empty_profile(),
        paper_a_full_text="A全文。",
        paper_b_metadata=metadata("P002", date(2021, 1, 1)),
        paper_b_profile=empty_profile(),
        paper_b_full_text="B全文。",
        current_relation=current,
        review_issues=revision_issues(),
    )

    assert client.calls == 2
    assert result.record == revised_related()
    assert result.metrics.token_usage.total_tokens == 30


def test_relation_revision_allows_related_to_unrelated_branch_switch() -> None:
    current = revised_related()
    issue = RelationReviewIssue(
        target_fields=[
            RelationRecordField.RECORD_TYPE,
            RelationRecordField.RELATION_TYPE,
            RelationRecordField.RELATION_EXPLANATION,
        ],
        issue_description="当前记录只有方法名称相似，不能成立拓展。",
        paper_a_review_evidence_quotes=["A提出基础模型。"],
        paper_b_review_evidence_quotes=["B使用另一种模型。"],
        revision_guidance="重新判断关系是否成立。",
    )
    client = FakeClient([revised_unrelated().model_dump_json()])
    agent = RelationInferenceAgent(client=client, sleep=lambda _: None)

    result = agent.targeted_revision(
        paper_a_metadata=metadata("P001", date(2020, 1, 1)),
        paper_a_profile=empty_profile(),
        paper_a_full_text="A全文。",
        paper_b_metadata=metadata("P002", date(2021, 1, 1)),
        paper_b_profile=empty_profile(),
        paper_b_full_text="B全文。",
        current_relation=current,
        review_issues=[issue],
    )

    assert result.record == revised_unrelated()
    assert client.calls == 1


def test_relation_revision_rejects_issues_for_fields_absent_from_current_branch() -> None:
    invalid_issue = RelationReviewIssue(
        target_fields=[RelationRecordField.RELATION_TYPE],
        issue_description="关系类别不正确。",
        paper_a_review_evidence_quotes=[],
        paper_b_review_evidence_quotes=[],
        revision_guidance="修改关系类别。",
    )
    client = FakeClient([revised_related().model_dump_json()])
    agent = RelationInferenceAgent(client=client, sleep=lambda _: None)

    with pytest.raises(ValueError):
        agent.targeted_revision(
            paper_a_metadata=metadata("P001", date(2020, 1, 1)),
            paper_a_profile=empty_profile(),
            paper_a_full_text="A全文。",
            paper_b_metadata=metadata("P002", date(2021, 1, 1)),
            paper_b_profile=empty_profile(),
            paper_b_full_text="B全文。",
            current_relation=current_unrelated(),
            review_issues=[invalid_issue],
        )

    assert client.calls == 0


def test_relation_revision_exhausts_content_retries_when_output_never_changes() -> None:
    current = current_unrelated()
    client = FakeClient([current.model_dump_json()] * 3)
    agent = RelationInferenceAgent(client=client, sleep=lambda _: None)

    with pytest.raises(RelationInferenceContentRetriesExhausted) as captured:
        agent.targeted_revision(
            paper_a_metadata=metadata("P001", date(2020, 1, 1)),
            paper_a_profile=empty_profile(),
            paper_a_full_text="A全文。",
            paper_b_metadata=metadata("P002", date(2021, 1, 1)),
            paper_b_profile=empty_profile(),
            paper_b_full_text="B全文。",
            current_relation=current,
            review_issues=revision_issues(),
        )

    assert client.calls == 3
    assert captured.value.metrics.token_usage.total_tokens == 45


class SuccessfulRevisionAgent:
    def __init__(self) -> None:
        self.calls = 0

    def targeted_revision(self, **kwargs) -> RelationInferenceRunResult:
        self.calls += 1
        client = FakeClient([revised_related().model_dump_json()])
        return RelationInferenceAgent(
            client=client,
            sleep=lambda _: None,
        ).targeted_revision(**kwargs)


def test_relation_revision_workflow_reuses_matching_checkpoint(
    tmp_path: Path,
) -> None:
    kwargs = {
        "paper_a_metadata": metadata("P001", date(2020, 1, 1)),
        "paper_a_profile": empty_profile(),
        "paper_a_full_text": "A全文。",
        "paper_b_metadata": metadata("P002", date(2021, 1, 1)),
        "paper_b_profile": empty_profile(),
        "paper_b_full_text": "B全文。",
        "current_relation": current_unrelated(),
        "review_issues": revision_issues(),
        "model": "deepseek-v4-flash",
        "result_root": tmp_path / "results",
        "temp_root": tmp_path / "resume",
    }
    first_agent = SuccessfulRevisionAgent()
    first = run_targeted_relation_revision(agent=first_agent, **kwargs)

    assert first.reused is False
    assert first_agent.calls == 1
    assert first.checkpoint.stage == "relation_targeted_revision"

    second_agent = SuccessfulRevisionAgent()
    second = run_targeted_relation_revision(agent=second_agent, **kwargs)

    assert second.reused is True
    assert second_agent.calls == 0
