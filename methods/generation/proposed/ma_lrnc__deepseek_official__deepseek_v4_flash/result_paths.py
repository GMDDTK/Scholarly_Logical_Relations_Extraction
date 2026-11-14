from dataclasses import dataclass
from pathlib import Path


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"/\\|?*')


@dataclass(frozen=True)
class TopicResultPaths:
    topic: str
    result_root: Path
    temp_root: Path

    @property
    def checkpoint_root(self) -> Path:
        return self.result_root / "checkpoints"

    @property
    def failure_root(self) -> Path:
        return self.result_root / "failures"

    @property
    def final_profile_root(self) -> Path:
        return self.result_root / "final_profiles"

    @property
    def relation_outcome_root(self) -> Path:
        return self.result_root / "relation_outcomes"

    @property
    def network_root(self) -> Path:
        return self.result_root / "network"


def build_topic_result_paths(
    *,
    topic: str,
    result_unit_root: Path,
    temp_unit_root: Path,
) -> TopicResultPaths:
    safe_topic = validate_topic_name(topic)
    return TopicResultPaths(
        topic=safe_topic,
        result_root=result_unit_root / "topics" / safe_topic,
        temp_root=temp_unit_root / "topics" / safe_topic,
    )


def validate_topic_name(topic: str) -> str:
    if not isinstance(topic, str):
        raise ValueError("topic must be a string")
    if not topic or topic != topic.strip():
        raise ValueError("topic must be non-empty without surrounding whitespace")
    if topic in {".", ".."}:
        raise ValueError("topic must be a single safe directory name")
    if topic.endswith("."):
        raise ValueError("topic must not end with a dot")
    if any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in topic):
        raise ValueError("topic contains a Windows-forbidden path character")
    if any(ord(character) < 32 for character in topic):
        raise ValueError("topic contains a control character")

    windows_stem = topic.split(".", maxsplit=1)[0].upper()
    if windows_stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError("topic uses a Windows-reserved device name")
    return topic
