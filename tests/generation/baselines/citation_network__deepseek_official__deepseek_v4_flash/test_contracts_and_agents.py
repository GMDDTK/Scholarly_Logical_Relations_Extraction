import json
from collections import deque

import pytest
from pydantic import ValidationError

from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.agents import (
    CitationIdentificationAgent,
    CitationReviewAgent,
    ContentRetriesExhausted,
)
from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.contracts import (
    CitationIdentification,
    CitationReviewResult,
    TokenUsage,
)
from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
    TransientNetworkError,
)


class FakeClient:
    def __init__(self, contents: list[str | Exception]) -> None:
        self.contents = deque(contents)
        self.messages_seen: list[list[dict[str, str]]] = []

    def create_json_completion(
        self,
        messages: list[dict[str, str]],
    ) -> CompletionResult:
        self.messages_seen.append(messages)
        outcome = self.contents.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return CompletionResult(
            content=outcome,
            finish_reason="stop",
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


def test_identification_contract_deduplicates_and_sorts_evidence_rows() -> None:
    result = CitationIdentification.model_validate(
        {
            "cited_papers": [
                {
                    "cited_paper_id": "P002",
                    "reference_row_numbers": [3, 1, 3],
                }
            ]
        }
    )

    assert result.cited_papers[0].reference_row_numbers == [1, 3]


def test_identification_contract_rejects_duplicate_target_records() -> None:
    with pytest.raises(ValidationError):
        CitationIdentification.model_validate(
            {
                "cited_papers": [
                    {"cited_paper_id": "P002", "reference_row_numbers": [1]},
                    {"cited_paper_id": "P002", "reference_row_numbers": [2]},
                ]
            }
        )


def test_review_contract_enforces_status_issue_invariant() -> None:
    assert CitationReviewResult(status="pass", issues=[]).status == "pass"
    with pytest.raises(ValidationError):
        CitationReviewResult(status="pass", issues=[{"description": "漏报", "reference_row_numbers": [1]}])
    with pytest.raises(ValidationError):
        CitationReviewResult(status="needs_retry", issues=[])


def test_identifier_retries_invalid_content_but_not_semantic_reasoning() -> None:
    client = FakeClient(
        [
            "not json",
            json.dumps(
                {
                    "cited_papers": [
                        {
                            "cited_paper_id": "P002",
                            "reference_row_numbers": [1],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        ]
    )
    agent = CitationIdentificationAgent(client=client, sleep=lambda _: None)

    execution = agent.identify(user_message="识别输入")

    assert execution.result.cited_papers[0].cited_paper_id == "P002"
    assert len(client.messages_seen) == 2
    assert execution.metrics.token_usage.total_tokens == 30


def test_reviewer_returns_needs_retry_without_revision_loop() -> None:
    client = FakeClient(
        [
            json.dumps(
                {
                    "status": "needs_retry",
                    "issues": [
                        {
                            "description": "第2行遗漏了P003",
                            "reference_row_numbers": [2],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = CitationReviewAgent(client=client, sleep=lambda _: None)

    execution = agent.review(user_message="审核输入")

    assert execution.result.status == "needs_retry"
    assert len(client.messages_seen) == 1


def test_invalid_content_stops_after_three_attempts() -> None:
    agent = CitationIdentificationAgent(
        client=FakeClient(["bad", "bad", "bad"]),
        sleep=lambda _: None,
    )

    with pytest.raises(ContentRetriesExhausted):
        agent.identify(user_message="识别输入")


def test_each_content_attempt_has_its_own_three_network_attempts() -> None:
    sleeps: list[float] = []
    client = FakeClient([TransientNetworkError("temporary")] * 9)
    agent = CitationIdentificationAgent(client=client, sleep=sleeps.append)

    with pytest.raises(Exception, match="network retries exhausted"):
        agent.identify(user_message="识别输入")

    assert len(client.messages_seen) == 9
    assert sleeps == [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]
