import json
from collections import deque

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.evidence_reviewer import (
    EvidenceReviewerAgent,
    ProfileReviewContentRetriesExhausted,
    ProfileReviewNetworkRetriesExhausted,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
    TransientNetworkError,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.prompting import (
    build_profile_review_system_prompt,
    build_profile_review_user_message,
)


def profile() -> PaperProfileContent:
    return PaperProfileContent(
        research_questions=[
            {
                "statement": "本文研究信息素养。",
                "evidence_quotes": ["本文研究信息素养。"],
            }
        ],
        theories_and_viewpoints=[],
        methods_and_models=[],
        data_and_contexts=[],
        main_findings=[],
        limitations_and_boundary_conditions=[],
    )


def completion(payload: dict) -> CompletionResult:
    return CompletionResult(
        content=json.dumps(payload, ensure_ascii=False),
        finish_reason="stop",
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )


class FakeClient:
    def __init__(
        self,
        outcomes: list[CompletionResult | Exception],
    ) -> None:
        self.outcomes = deque(outcomes)
        self.messages_seen: list[list[dict[str, str]]] = []

    def create_json_completion(
        self,
        messages: list[dict[str, str]],
    ) -> CompletionResult:
        self.messages_seen.append(messages)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_profile_review_prompt_contains_checklist_schema_and_inputs() -> None:
    system_prompt = build_profile_review_system_prompt()
    user_message = build_profile_review_user_message(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=profile(),
    )

    assert "完整性" in system_prompt
    assert "原文忠实性" in system_prompt
    assert "主体归属" in system_prompt
    assert "<output_json_schema>" in system_prompt
    assert '"needs_revision"' in system_prompt
    assert "<article_full_text>" in user_message
    assert "<current_profile>" in user_message
    assert user_message.index("<article_full_text>") < user_message.index(
        "<current_profile>"
    )


def test_profile_reviewer_accepts_pass_without_content_retry() -> None:
    client = FakeClient([completion({"status": "pass"})])
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    result = agent.review_profile(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=profile(),
    )

    assert result.review.status == "pass"
    assert len(client.messages_seen) == 1
    assert result.metrics.token_usage.total_tokens == 150


def test_profile_reviewer_accepts_non_verbatim_review_quote() -> None:
    client = FakeClient(
        [
            completion(
                {
                    "status": "needs_revision",
                    "issues": [
                        {
                            "target_locations": [
                                {
                                    "field": "research_questions",
                                    "item_index": 0,
                                }
                            ],
                            "issue_description": "证据没有充分支持陈述。",
                            "review_evidence_quotes": ["经过格式调整的核验引句。"],
                            "revision_guidance": "核对全文并替换证据。",
                        }
                    ],
                }
            )
        ]
    )
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    result = agent.review_profile(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=profile(),
    )

    assert result.review.status == "needs_revision"
    assert len(client.messages_seen) == 1


def test_profile_reviewer_retries_invalid_location_then_accepts() -> None:
    invalid_location = {
        "status": "needs_revision",
        "issues": [
            {
                "target_locations": [
                    {"field": "research_questions", "item_index": 3}
                ],
                "issue_description": "位置不存在。",
                "review_evidence_quotes": [],
                "revision_guidance": "修正问题位置。",
            }
        ],
    }
    client = FakeClient(
        [completion(invalid_location), completion({"status": "pass"})]
    )
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    result = agent.review_profile(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=profile(),
    )

    assert result.review.status == "pass"
    assert len(client.messages_seen) == 2
    assert result.metrics.token_usage.total_tokens == 300


def test_profile_reviewer_stops_after_three_invalid_outputs() -> None:
    client = FakeClient(
        [
            completion({"status": "needs_revision", "issues": []}),
            completion({"status": "needs_revision", "issues": []}),
            completion({"status": "needs_revision", "issues": []}),
        ]
    )
    agent = EvidenceReviewerAgent(client=client, sleep=lambda _: None)

    with pytest.raises(ProfileReviewContentRetriesExhausted):
        agent.review_profile(
            title="测试论文",
            authors=["张三"],
            article_full_text="本文研究信息素养。",
            current_profile=profile(),
        )

    assert len(client.messages_seen) == 3


def test_profile_reviewer_network_retry_and_exhaustion_are_separate() -> None:
    sleeps: list[float] = []
    recovering_client = FakeClient(
        [
            TransientNetworkError("timeout"),
            TransientNetworkError("limited", retry_after=7.0),
            completion({"status": "pass"}),
        ]
    )
    recovering_agent = EvidenceReviewerAgent(
        client=recovering_client,
        sleep=sleeps.append,
    )

    result = recovering_agent.review_profile(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
        current_profile=profile(),
    )

    assert result.review.status == "pass"
    assert sleeps == [2.0, 7.0]

    exhausted_client = FakeClient(
        [
            TransientNetworkError("1"),
            TransientNetworkError("2"),
            TransientNetworkError("3"),
        ]
    )
    exhausted_agent = EvidenceReviewerAgent(
        client=exhausted_client,
        sleep=lambda _: None,
    )

    with pytest.raises(ProfileReviewNetworkRetriesExhausted):
        exhausted_agent.review_profile(
            title="测试论文",
            authors=["张三"],
            article_full_text="本文研究信息素养。",
            current_profile=profile(),
        )

    assert len(exhausted_client.messages_seen) == 3
