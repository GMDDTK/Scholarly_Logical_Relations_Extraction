from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CitationResultPaths:
    result_root: Path
    temp_root: Path

    def identification_checkpoint(self, paper_id: str) -> Path:
        return self.result_root / "checkpoints" / "citation_identification" / f"{paper_id}.json"

    def review_checkpoint(self, paper_id: str) -> Path:
        return self.result_root / "checkpoints" / "citation_review" / f"{paper_id}.json"

    def failure(self, stage: str, paper_id: str) -> Path:
        return self.result_root / "failures" / stage / f"{paper_id}.json"

    def outcome(self, paper_id: str) -> Path:
        return self.result_root / "citation_outcomes" / f"{paper_id}.json"

    @property
    def network_root(self) -> Path:
        return self.result_root / "network"


def build_topic_result_paths(
    *,
    topic: str,
    result_unit_root: Path,
    temp_unit_root: Path,
) -> CitationResultPaths:
    safe = validate_topic_name(topic)
    return CitationResultPaths(
        result_root=result_unit_root / "topics" / safe,
        temp_root=temp_unit_root / "topics" / safe,
    )


def validate_topic_name(topic: str) -> str:
    forbidden = frozenset('<>:"/\\|?*')
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{number}" for number in range(1, 10))
    reserved.update(f"LPT{number}" for number in range(1, 10))
    if not topic or topic != topic.strip() or topic in {".", ".."}:
        raise ValueError("topic must be one non-empty safe directory name")
    if topic.endswith(".") or any(character in forbidden for character in topic):
        raise ValueError("topic contains a Windows-forbidden path character")
    if any(ord(character) < 32 for character in topic):
        raise ValueError("topic contains a control character")
    if topic.split(".", maxsplit=1)[0].upper() in reserved:
        raise ValueError("topic uses a Windows-reserved device name")
    return topic
