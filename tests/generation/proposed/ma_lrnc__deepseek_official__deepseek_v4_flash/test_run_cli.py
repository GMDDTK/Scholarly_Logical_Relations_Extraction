from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash import (
    run as run_module,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.run import (
    build_argument_parser,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.topic_input import (
    TopicInputError,
)


def test_cli_exposes_both_paper_parser_modes() -> None:
    parser = build_argument_parser()

    initial = parser.parse_args(
        [
            "initial-generation",
            "--topic",
            "信息素养",
            "--paper-id",
            "P001",
            "--title",
            "测试论文",
            "--author",
            "张三",
            "--full-text",
            "paper.md",
        ]
    )
    assert initial.command == "initial-generation"
    assert initial.topic == "信息素养"

    revision = parser.parse_args(
        [
            "targeted-revision",
            "--topic",
            "信息素养",
            "--paper-id",
            "P001",
            "--title",
            "测试论文",
            "--author",
            "张三",
            "--full-text",
            "paper.md",
            "--current-profile",
            "profile.json",
            "--review-issues",
            "issues.json",
        ]
    )
    assert revision.command == "targeted-revision"


def test_cli_exposes_profile_review_mode() -> None:
    parser = build_argument_parser()

    review = parser.parse_args(
        [
            "profile-review",
            "--topic",
            "信息素养",
            "--paper-id",
            "P001",
            "--title",
            "测试论文",
            "--author",
            "张三",
            "--full-text",
            "paper.md",
            "--current-profile",
            "profile.json",
        ]
    )

    assert review.command == "profile-review"


def test_cli_exposes_relation_inference_mode() -> None:
    parser = build_argument_parser()

    relation = parser.parse_args(
        [
            "relation-inference",
            "--topic",
            "信息素养",
            "--paper-a-metadata",
            "a_metadata.json",
            "--paper-a-profile",
            "a_profile.json",
            "--paper-a-full-text",
            "a.md",
            "--paper-b-metadata",
            "b_metadata.json",
            "--paper-b-profile",
            "b_profile.json",
            "--paper-b-full-text",
            "b.md",
        ]
    )

    assert relation.command == "relation-inference"


def test_cli_exposes_relation_review_mode() -> None:
    parser = build_argument_parser()

    review = parser.parse_args(
        [
            "relation-review",
            "--topic",
            "信息素养",
            "--paper-a-metadata",
            "a_metadata.json",
            "--paper-a-profile",
            "a_profile.json",
            "--paper-a-full-text",
            "a.md",
            "--paper-b-metadata",
            "b_metadata.json",
            "--paper-b-profile",
            "b_profile.json",
            "--paper-b-full-text",
            "b.md",
            "--current-relation",
            "relation.json",
        ]
    )

    assert review.command == "relation-review"


def test_cli_exposes_relation_targeted_revision_mode() -> None:
    parser = build_argument_parser()

    revision = parser.parse_args(
        [
            "relation-targeted-revision",
            "--topic",
            "信息素养",
            "--paper-a-metadata",
            "a_metadata.json",
            "--paper-a-profile",
            "a_profile.json",
            "--paper-a-full-text",
            "a.md",
            "--paper-b-metadata",
            "b_metadata.json",
            "--paper-b-profile",
            "b_profile.json",
            "--paper-b-full-text",
            "b.md",
            "--current-relation",
            "relation.json",
            "--review-issues",
            "issues.json",
        ]
    )

    assert revision.command == "relation-targeted-revision"


def test_cli_exposes_relation_post_revision_review_mode() -> None:
    parser = build_argument_parser()

    review = parser.parse_args(
        [
            "relation-post-revision-review",
            "--topic",
            "信息素养",
            "--paper-a-metadata",
            "a_metadata.json",
            "--paper-a-profile",
            "a_profile.json",
            "--paper-a-full-text",
            "a.md",
            "--paper-b-metadata",
            "b_metadata.json",
            "--paper-b-profile",
            "b_profile.json",
            "--paper-b-full-text",
            "b.md",
            "--previous-review-issues",
            "issues.json",
            "--revised-relation",
            "relation.json",
        ]
    )

    assert review.command == "relation-post-revision-review"


def test_cli_exposes_extract_full_text_mode() -> None:
    parser = build_argument_parser()

    extraction = parser.parse_args(
        [
            "extract-full-text",
            "--input-pdf",
            "paper.pdf",
            "--output-markdown",
            "paper.md",
        ]
    )

    assert extraction.command == "extract-full-text"


def test_cli_exposes_extract_metadata_mode() -> None:
    parser = build_argument_parser()

    extraction = parser.parse_args(
        [
            "extract-metadata",
            "--workbook",
            "catalog.xlsx",
            "--sheet",
            "信息素养50高被引",
            "--row-number",
            "25",
            "--output-json",
            "metadata.json",
        ]
    )

    assert extraction.command == "extract-metadata"
    assert extraction.row_number == 25


def test_model_stage_requires_topic() -> None:
    parser = build_argument_parser()

    try:
        parser.parse_args(
            [
                "initial-generation",
                "--paper-id",
                "P001",
                "--title",
                "测试论文",
                "--author",
                "张三",
                "--full-text",
                "paper.md",
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("model stage must require --topic")


def test_cli_exposes_unified_topic_mode_with_only_topic_argument() -> None:
    parser = build_argument_parser()

    topic_run = parser.parse_args(
        [
            "run-topic",
            "--topic",
            "信息素养",
        ]
    )

    assert topic_run.command == "run-topic"
    assert topic_run.topic == "信息素养"


def test_cli_exposes_deterministic_visualization_mode() -> None:
    parser = build_argument_parser()

    visualization = parser.parse_args(
        ["build-visualization", "--topic", "信息素养"]
    )

    assert visualization.command == "build-visualization"
    assert visualization.topic == "信息素养"


def test_build_visualization_does_not_load_model_config(
    monkeypatch,
    capsys,
) -> None:
    events: list[str] = []

    def discover(**_: object) -> object:
        events.append("discover")
        return object()

    class VisualizationResult:
        reused = False
        input_fingerprint = "sha256:" + "0" * 64
        output_paths = ("network_data.json", "index.html")

    def build(**_: object) -> object:
        events.append("build")
        return VisualizationResult()

    def must_not_load_config() -> object:
        events.append("config")
        raise AssertionError("visualization must not load LLM configuration")

    monkeypatch.setattr(run_module, "discover_topic_inputs", discover)
    monkeypatch.setattr(run_module, "build_topic_visualization", build)
    monkeypatch.setattr(run_module, "load_deepseek_config", must_not_load_config)

    exit_code = run_module.main(
        ["build-visualization", "--topic", "测试主题"]
    )

    assert exit_code == 0
    assert events == ["discover", "build"]
    output = capsys.readouterr().out
    assert '"status": "completed"' in output
    assert '"input_fingerprint"' in output


def test_run_topic_builds_visualization_only_after_relations_complete(
    monkeypatch,
    capsys,
) -> None:
    events: list[str] = []

    class Config:
        base_url = "https://example.invalid"
        api_key = "test"
        model = "test-model"

    class TrackingClient:
        token_usage = object()

    class Summary:
        status = "relations_complete"

        @staticmethod
        def model_dump_json() -> str:
            return '{"status":"relations_complete"}'

    monkeypatch.setattr(
        run_module,
        "discover_topic_inputs",
        lambda **_: object(),
    )
    monkeypatch.setattr(run_module, "load_deepseek_config", Config)
    monkeypatch.setattr(
        run_module,
        "DeepSeekJsonClient",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        run_module,
        "TokenTrackingJsonClient",
        lambda **_: TrackingClient(),
    )

    def orchestrate(**_: object) -> Summary:
        events.append("orchestrate")
        return Summary()

    def visualize(**_: object) -> object:
        events.append("visualize")
        return object()

    monkeypatch.setattr(run_module, "run_topic_orchestration", orchestrate)
    monkeypatch.setattr(run_module, "build_topic_visualization", visualize)

    exit_code = run_module.main(["run-topic", "--topic", "测试主题"])

    assert exit_code == 0
    assert events == ["orchestrate", "visualize"]
    assert '"relations_complete"' in capsys.readouterr().out


def test_run_topic_discovers_inputs_before_loading_model_config(
    monkeypatch,
) -> None:
    events: list[str] = []

    def fail_discovery(**_: object) -> object:
        events.append("discover")
        raise TopicInputError("simulated topic input failure")

    def must_not_load_config() -> object:
        events.append("config")
        raise AssertionError("model config must not load before input preflight")

    monkeypatch.setattr(
        run_module,
        "discover_topic_inputs",
        fail_discovery,
    )
    monkeypatch.setattr(
        run_module,
        "load_deepseek_config",
        must_not_load_config,
    )

    try:
        run_module.main(["run-topic", "--topic", "测试主题"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("topic-level input failure must stop the batch")

    assert events == ["discover"]
