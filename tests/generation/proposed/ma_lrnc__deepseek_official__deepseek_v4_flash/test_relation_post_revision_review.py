import json
from collections import deque
from datetime import date
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.evidence_reviewer import (
    EvidenceReviewerAgent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_record import (
    RelatedPairRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.relation_review import (
    RelationRecordField,
    RelationReviewIssue,
    RelationReviewPassed,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    RelationReviewerRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.prompting import (
    build_relation_post_revision_review_system_prompt,
    build_relation_post_revision_review_user_message,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.workflow import (
    run_post_revision_relation_review,
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


def revised_relation() -> RelatedPairRecord:
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


def previous_issues() -> list[RelationReviewIssue]:
    return [
        RelationReviewIssue(
            target_fields=[
                RelationRecordField.RECORD_TYPE,
                RelationRecordField.NO_RELATION_REASON,
            ],
            issue_description="原无关系记录遗漏实质拓展。",
            paper_a_review_evidence_quotes=["A提出基础模型。"],
            paper_b_review_evidence_quotes=["B增加新的变量。"],
            revision_guidance="重新判断有无关系。",
        )
    ]


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.outcomes = deque(
            CompletionResult(
                content=json.dumps(payload, ensure_ascii=False),
                finish_reason="stop",
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                ),
            )
            for payload in payloads
        )
        self.calls = 0
        self.messages_seen: list[list[dict[str, str]]] = []

    def create_json_completion(self, messages):
        self.calls += 1
        self.messages_seen.append(messages)
        return self.outcomes.popleft()


def pair_kwargs() -> dict:
    return {
        "paper_a_metadata": metadata("P001", date(2020, 1, 1)),
        "paper_a_profile": empty_profile(),
        "paper_a_full_text": "A全文。",
        "paper_b_metadata": metadata("P002", date(2021, 1, 1)),
        "paper_b_profile": empty_profile(),
        "paper_b_full_text": "B全文。",
        "previous_review_issues": previous_issues(),
        "revised_relation": revised_relation(),
    }


def test_post_revision_review_prompt_adds_terminal_mode_and_orders_inputs() -> None:
    system_prompt = build_relation_post_revision_review_system_prompt()
    user_message = build_relation_post_revision_review_user_message(
        **pair_kwargs()
    )

    assert system_prompt.count("# 学术逻辑关系共同规则") == 1
    assert system_prompt.count("# 证据核验智能体：关系核验模式") == 1
    assert system_prompt.count("# 证据核验智能体：修订后关系复核模式") == 1
    assert system_prompt.index("# 证据核验智能体：关系核验模式") < (
        system_prompt.index("# 证据核验智能体：修订后关系复核模式")
    )
    assert user_message.index("<relation_pair>") < user_message.index(
        "<previous_review_issues>"
    )
    assert user_message.index("<previous_review_issues>") < user_message.index(
        "<revised_relation_record>"
    )


def test_post_revision_review_requires_previous_issues_before_model_call() -> None:
    client = FakeClient([{"status": "pass"}])
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)
    kwargs = pair_kwargs()
    kwargs["previous_review_issues"] = []

    with pytest.raises(ValueError):
        agent.review_relation_after_revision(**kwargs)

    assert client.calls == 0


def test_post_revision_review_reports_only_current_outstanding_fields() -> None:
    outstanding = {
        "status": "needs_revision",
        "issues": [
            {
                "target_fields": ["paper_b_evidence_quotes"],
                "issue_description": "修订后的论文B引句仍不能支持新增变量。",
                "paper_a_review_evidence_quotes": [],
                "paper_b_review_evidence_quotes": ["B只复用了原模型。"],
                "revision_guidance": "人工核对论文B证据与拓展判断。",
            }
        ],
    }
    client = FakeClient([outstanding])
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    result = agent.review_relation_after_revision(**pair_kwargs())

    assert result.review.status == "needs_revision"
    assert result.review.issues[0].target_fields == [
        RelationRecordField.PAPER_B_EVIDENCE_QUOTES
    ]
    assert result.metrics.token_usage.total_tokens == 120


class SuccessfulPostReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review_relation_after_revision(self, **kwargs) -> RelationReviewerRunResult:
        self.calls += 1
        return RelationReviewerRunResult(
            review=RelationReviewPassed(status="pass"),
            metrics=StageMetrics(
                elapsed_seconds=2.0,
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            ),
        )


def test_post_revision_review_workflow_reuses_pair_checkpoint(
    tmp_path: Path,
) -> None:
    kwargs = {
        **pair_kwargs(),
        "model": "deepseek-v4-flash",
        "result_root": tmp_path / "results",
        "temp_root": tmp_path / "resume",
    }
    first_agent = SuccessfulPostReviewer()
    first = run_post_revision_relation_review(agent=first_agent, **kwargs)

    assert first.reused is False
    assert first_agent.calls == 1
    assert first.checkpoint.stage == "relation_post_revision_review"

    second_agent = SuccessfulPostReviewer()
    second = run_post_revision_relation_review(agent=second_agent, **kwargs)

    assert second.reused is True
    assert second_agent.calls == 0

