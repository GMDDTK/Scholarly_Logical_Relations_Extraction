import json
from collections import deque
from datetime import date
from pathlib import Path

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.agents.relation_inference import (
    RelationInferenceAgent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_metadata import (
    PaperMetadata,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.paper_profile import (
    PaperProfileContent,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.contracts.runtime import (
    RelationInferenceRunResult,
    StageMetrics,
    TokenUsage,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.llm_client import (
    CompletionResult,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.workflow import (
    run_initial_relation_inference,
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


def relation_json(*, direction: str = "b_to_a") -> str:
    return json.dumps(
        {
            "record_type": "related",
            "paper_a_id": "P001",
            "paper_b_id": "P002",
            "relation_type": "拓展",
            "direction": direction,
            "relation_explanation": "B在A模型上增加变量。",
            "paper_a_evidence_quotes": ["A提出基础模型。"],
            "paper_b_evidence_quotes": ["B增加新的变量。"],
        },
        ensure_ascii=False,
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


def test_relation_agent_retries_invalid_handoff_and_accumulates_usage() -> None:
    client = FakeClient([relation_json(direction="a_to_b"), relation_json()])
    agent = RelationInferenceAgent(client=client, sleep=lambda _: None)

    result = agent.initial_inference(
        paper_a_metadata=metadata("P001", date(2020, 1, 1)),
        paper_a_profile=empty_profile(),
        paper_a_full_text="A全文。",
        paper_b_metadata=metadata("P002", date(2021, 1, 1)),
        paper_b_profile=empty_profile(),
        paper_b_full_text="B全文。",
    )

    assert client.calls == 2
    assert result.record.direction == "b_to_a"
    assert result.metrics.token_usage.total_tokens == 30


class SuccessfulRelationAgent:
    def __init__(self) -> None:
        self.calls = 0

    def initial_inference(self, **kwargs) -> RelationInferenceRunResult:
        self.calls += 1
        client = FakeClient([relation_json()])
        return RelationInferenceAgent(
            client=client,
            sleep=lambda _: None,
        ).initial_inference(**kwargs)


def test_relation_workflow_reuses_matching_pair_checkpoint(tmp_path: Path) -> None:
    kwargs = {
        "paper_a_metadata": metadata("P001", date(2020, 1, 1)),
        "paper_a_profile": empty_profile(),
        "paper_a_full_text": "A全文。",
        "paper_b_metadata": metadata("P002", date(2021, 1, 1)),
        "paper_b_profile": empty_profile(),
        "paper_b_full_text": "B全文。",
        "model": "deepseek-v4-flash",
        "result_root": tmp_path / "results",
        "temp_root": tmp_path / "resume",
    }
    first_agent = SuccessfulRelationAgent()
    first = run_initial_relation_inference(agent=first_agent, **kwargs)

    assert first.reused is False
    assert first_agent.calls == 1
    assert first.checkpoint.business_key == "P001__P002"

    second_agent = SuccessfulRelationAgent()
    second = run_initial_relation_inference(agent=second_agent, **kwargs)

    assert second.reused is True
    assert second_agent.calls == 0
