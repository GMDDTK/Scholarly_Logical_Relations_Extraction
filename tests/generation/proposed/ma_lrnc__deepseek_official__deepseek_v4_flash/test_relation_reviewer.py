import json
from collections import deque
from datetime import date

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
    UnrelatedPairRecord,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
)


def metadata(paper_id: str) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=f"{paper_id}论文",
        authors=["张三"],
        publication_date=date(2021, 1, 1),
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


def current_relation() -> UnrelatedPairRecord:
    return UnrelatedPairRecord(
        record_type="unrelated",
        paper_a_id="P001",
        paper_b_id="P002",
        no_relation_reason="两篇论文只有主题相似。",
    )


def completion(payload: dict) -> CompletionResult:
    return CompletionResult(
        content=json.dumps(payload, ensure_ascii=False),
        finish_reason="stop",
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.outcomes = deque(completion(payload) for payload in payloads)
        self.calls = 0
        self.messages_seen: list[list[dict[str, str]]] = []

    def create_json_completion(self, messages):
        self.calls += 1
        self.messages_seen.append(messages)
        return self.outcomes.popleft()


def review_kwargs() -> dict:
    return {
        "paper_a_metadata": metadata("P001"),
        "paper_a_profile": empty_profile(),
        "paper_a_full_text": "A全文。",
        "paper_b_metadata": metadata("P002"),
        "paper_b_profile": empty_profile(),
        "paper_b_full_text": "B全文。",
        "current_relation": current_relation(),
    }


def test_relation_reviewer_accepts_pass_and_builds_pair_context() -> None:
    client = FakeClient([{"status": "pass"}])
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    result = agent.review_relation(**review_kwargs())

    assert result.review.status == "pass"
    assert result.metrics.token_usage.total_tokens == 120
    user_message = client.messages_seen[0][1]["content"]
    assert "<relation_pair>" in user_message
    assert "<current_relation_record>" in user_message


def test_relation_reviewer_retries_invalid_target_field_then_accepts() -> None:
    invalid = {
        "status": "needs_revision",
        "issues": [
            {
                "target_fields": ["relation_type"],
                "issue_description": "目标字段不属于当前无关系分支。",
                "paper_a_review_evidence_quotes": [],
                "paper_b_review_evidence_quotes": [],
                "revision_guidance": "修正字段定位。",
            }
        ],
    }
    client = FakeClient([invalid, {"status": "pass"}])
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    result = agent.review_relation(**review_kwargs())

    assert result.review.status == "pass"
    assert client.calls == 2
    assert result.metrics.token_usage.total_tokens == 240
