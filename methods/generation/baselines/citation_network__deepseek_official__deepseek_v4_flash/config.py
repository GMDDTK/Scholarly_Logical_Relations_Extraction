import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


EXPERIMENT_UNIT = "citation_network__deepseek_official__deepseek_v4_flash"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_URL_ENV = "LLM_DEEPSEEK_OFFICIAL_BASE_URL"
API_KEY_ENV = "LLM_DEEPSEEK_OFFICIAL_API_KEY"
API_STYLE_ENV = "LLM_DEEPSEEK_OFFICIAL_API_STYLE"
MODEL_ENV = "LLM_DEEPSEEK_OFFICIAL_DEEPSEEK_V4_FLASH_MODEL"
REQUIRED_ENV_NAMES = (BASE_URL_ENV, API_KEY_ENV, API_STYLE_ENV, MODEL_ENV)


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    api_key: str
    api_style: str
    model: str


def load_deepseek_config(
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> DeepSeekConfig:
    source_file = env_file if env_file is not None else PROJECT_ROOT / ".env"
    file_values = dotenv_values(source_file) if source_file.is_file() else {}
    process_values = os.environ if environ is None else environ
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_ENV_NAMES:
        raw = process_values.get(name, file_values.get(name))
        value = str(raw).strip() if raw is not None else ""
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise ConfigurationError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    if values[API_STYLE_ENV] != "openai":
        raise ConfigurationError(f"{API_STYLE_ENV} must be configured as openai")
    return DeepSeekConfig(
        base_url=values[BASE_URL_ENV],
        api_key=values[API_KEY_ENV],
        api_style=values[API_STYLE_ENV],
        model=values[MODEL_ENV],
    )
