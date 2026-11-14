import json
from collections import deque
from pathlib import Path

import pytest

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.paper_parser import (
    ContentRetriesExhausted,
    NetworkRetriesExhausted,
    PaperParserAgent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.checkpointing import (
    compute_task_fingerprint,
    load_valid_paper_profile_checkpoint,
    write_failure_record,
    write_paper_profile_checkpoint,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    PaperParserFailureRecord,
    PaperProfileCheckpoint,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
    TransientNetworkError,
)


def profile_json(quote: str = "本文研究信息素养。") -> str:
    return json.dumps(
        {
            "research_questions": [
                {
                    "statement": "本文研究信息素养。",
                    "evidence_quotes": [quote],
                }
            ],
            "theories_and_viewpoints": [],
            "methods_and_models": [],
            "data_and_contexts": [],
            "main_findings": [],
            "limitations_and_boundary_conditions": [],
        },
        ensure_ascii=False,
    )


class FakeClient:
    def __init__(self, outcomes: list[CompletionResult | Exception]) -> None:
        self.outcomes = deque(outcomes)
        self.messages_seen: list[list[dict[str, str]]] = []

    def create_json_completion(
        self, messages: list[dict[str, str]]
    ) -> CompletionResult:
        self.messages_seen.append(messages)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def completion(content: str, *, finish_reason: str = "stop") -> CompletionResult:
    return CompletionResult(
        content=content,
        finish_reason=finish_reason,
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )


def test_network_retry_uses_two_and_four_second_backoff() -> None:
    sleeps: list[float] = []
    client = FakeClient(
        [
            TransientNetworkError("timeout"),
            TransientNetworkError("server"),
            completion(profile_json()),
        ]
    )
    agent = PaperParserAgent(client=client, sleep=sleeps.append)

    result = agent.initial_generation(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
    )

    assert sleeps == [2.0, 4.0]
    assert len(client.messages_seen) == 3
    assert result.metrics.token_usage.total_tokens == 150


def test_retry_after_overrides_default_backoff() -> None:
    sleeps: list[float] = []
    client = FakeClient(
        [
            TransientNetworkError("limited", retry_after=7.5),
            completion(profile_json()),
        ]
    )
    agent = PaperParserAgent(client=client, sleep=sleeps.append)

    agent.initial_generation(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
    )

    assert sleeps == [7.5]


def test_network_exhaustion_does_not_turn_into_content_retry() -> None:
    client = FakeClient(
        [
            TransientNetworkError("1"),
            TransientNetworkError("2"),
            TransientNetworkError("3"),
        ]
    )
    agent = PaperParserAgent(client=client, sleep=lambda _: None)

    with pytest.raises(NetworkRetriesExhausted) as error:
        agent.initial_generation(
            title="测试论文",
            authors=["张三"],
            article_full_text="本文研究信息素养。",
        )

    assert len(client.messages_seen) == 3
    assert error.value.metrics.token_usage.total_tokens == 0


def test_content_retry_accumulates_returned_token_usage() -> None:
    client = FakeClient(
        [
            completion("not json"),
            completion(json.dumps({"research_questions": []})),
            completion(profile_json()),
        ]
    )
    agent = PaperParserAgent(client=client, sleep=lambda _: None)

    result = agent.initial_generation(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
    )

    assert len(client.messages_seen) == 3
    assert result.metrics.token_usage.prompt_tokens == 300
    assert result.metrics.token_usage.completion_tokens == 150
    assert result.metrics.token_usage.total_tokens == 450


def test_non_verbatim_evidence_does_not_trigger_content_retry() -> None:
    client = FakeClient([completion(profile_json("经过轻微改写的证据。"))])
    agent = PaperParserAgent(client=client, sleep=lambda _: None)

    result = agent.initial_generation(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
    )

    assert len(client.messages_seen) == 1
    assert (
        result.profile.research_questions[0].evidence_quotes[0]
        == "经过轻微改写的证据。"
    )


def test_content_retry_stops_after_three_content_attempts() -> None:
    client = FakeClient([completion("bad")] * 3)
    agent = PaperParserAgent(client=client, sleep=lambda _: None)

    with pytest.raises(ContentRetriesExhausted) as error:
        agent.initial_generation(
            title="测试论文",
            authors=["张三"],
            article_full_text="本文研究信息素养。",
        )

    assert len(client.messages_seen) == 3
    assert error.value.metrics.token_usage.total_tokens == 450


def test_checkpoint_roundtrip_and_fingerprint_invalidation(tmp_path: Path) -> None:
    fingerprint = compute_task_fingerprint(
        stage="paper_profile_initial_generation",
        business_key="P001",
        dynamic_input={"title": "测试论文", "full_text": "正文"},
        system_prompt="prompt",
        output_schema={"type": "object"},
        experiment_unit="ma_lrnc__deepseek_official__deepseek_v4_flash",
        model="deepseek-v4-flash",
    )
    result = PaperParserAgent(
        client=FakeClient([completion(profile_json())]),
        sleep=lambda _: None,
    ).initial_generation(
        title="测试论文",
        authors=["张三"],
        article_full_text="本文研究信息素养。",
    )
    checkpoint = PaperProfileCheckpoint(
        checkpoint_version=1,
        stage="paper_profile_initial_generation",
        business_key="P001",
        task_fingerprint=fingerprint,
        result=result.profile,
        metrics=StageMetrics(
            elapsed_seconds=result.metrics.elapsed_seconds,
            token_usage=result.metrics.token_usage,
        ),
    )
    destination = tmp_path / "checkpoints" / "P001.json"
    temp_root = tmp_path / "resume"

    write_paper_profile_checkpoint(
        checkpoint=checkpoint,
        destination=destination,
        temp_root=temp_root,
    )

    loaded = load_valid_paper_profile_checkpoint(
        path=destination,
        expected_stage="paper_profile_initial_generation",
        expected_business_key="P001",
        expected_fingerprint=fingerprint,
    )
    assert loaded is not None
    assert list(temp_root.glob("*")) == []

    stale = load_valid_paper_profile_checkpoint(
        path=destination,
        expected_stage="paper_profile_initial_generation",
        expected_business_key="P001",
        expected_fingerprint="sha256:" + "b" * 64,
    )
    assert stale is None


def test_checkpoint_accepts_schema_valid_non_verbatim_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = PaperProfileCheckpoint(
        checkpoint_version=1,
        stage="paper_profile_initial_generation",
        business_key="P001",
        task_fingerprint="sha256:" + "d" * 64,
        result=json.loads(profile_json("经过轻微改写的证据。")),
        metrics=StageMetrics(
            elapsed_seconds=1.0,
            token_usage=TokenUsage.zero(),
        ),
    )
    destination = tmp_path / "checkpoints" / "P001.json"
    write_paper_profile_checkpoint(
        checkpoint=checkpoint,
        destination=destination,
        temp_root=tmp_path / "resume",
    )

    loaded = load_valid_paper_profile_checkpoint(
        path=destination,
        expected_stage="paper_profile_initial_generation",
        expected_business_key="P001",
        expected_fingerprint="sha256:" + "d" * 64,
    )

    assert loaded is not None


def test_failure_record_is_separate_from_success_checkpoint(tmp_path: Path) -> None:
    failure = PaperParserFailureRecord(
        failure_record_version=1,
        stage="paper_profile_initial_generation",
        business_key="P001",
        task_fingerprint="sha256:" + "c" * 64,
        failure_kind="network_exhausted",
        metrics=StageMetrics(
            elapsed_seconds=6.0,
            token_usage=TokenUsage.zero(),
        ),
    )
    destination = tmp_path / "failures" / "P001.json"

    write_failure_record(
        failure=failure,
        destination=destination,
        temp_root=tmp_path / "resume",
    )

    persisted = PaperParserFailureRecord.model_validate_json(
        destination.read_text(encoding="utf-8")
    )
    assert persisted.failure_kind == "network_exhausted"
