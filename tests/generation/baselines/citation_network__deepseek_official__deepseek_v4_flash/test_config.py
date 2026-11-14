from pathlib import Path

import pytest

from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.config import (
    ConfigurationError,
    load_deepseek_config,
)


def test_loads_only_the_experiment_specific_deepseek_configuration(
    tmp_path: Path,
) -> None:
    config = load_deepseek_config(
        env_file=tmp_path / "missing.env",
        environ={
            "LLM_DEEPSEEK_OFFICIAL_BASE_URL": "https://api.deepseek.com",
            "LLM_DEEPSEEK_OFFICIAL_API_KEY": "test-key",
            "LLM_DEEPSEEK_OFFICIAL_API_STYLE": "openai",
            "LLM_DEEPSEEK_OFFICIAL_DEEPSEEK_V4_FLASH_MODEL": "deepseek-v4-flash",
        },
    )

    assert config.model == "deepseek-v4-flash"
    assert config.api_style == "openai"


def test_missing_configuration_reports_names_without_values(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as error:
        load_deepseek_config(env_file=tmp_path / "missing.env", environ={})

    message = str(error.value)
    assert "LLM_DEEPSEEK_OFFICIAL_API_KEY" in message
    assert "test-key" not in message
