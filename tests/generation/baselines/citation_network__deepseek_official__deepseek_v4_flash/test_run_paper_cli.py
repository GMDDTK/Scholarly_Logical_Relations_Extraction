import json

from methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash import run


def test_parser_accepts_single_paper_command() -> None:
    args = run.build_parser().parse_args(
        ["run-paper", "--topic", "信息素养", "--paper-id", "P006"]
    )

    assert args.command == "run-paper"
    assert args.topic == "信息素养"
    assert args.paper_id == "P006"


def test_main_returns_zero_only_when_single_paper_is_accepted_or_skipped(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        run,
        "run_paper",
        lambda **_: {"topic": "信息素养", "paper_id": "P006", "status": "accepted"},
        raising=False,
    )

    accepted_code = run.main(
        ["run-paper", "--topic", "信息素养", "--paper-id", "P006"]
    )
    accepted_output = json.loads(capsys.readouterr().out)

    monkeypatch.setattr(
        run,
        "run_paper",
        lambda **_: {"topic": "信息素养", "paper_id": "P006", "status": "needs_retry"},
    )
    retry_code = run.main(
        ["run-paper", "--topic", "信息素养", "--paper-id", "P006"]
    )
    retry_output = json.loads(capsys.readouterr().out)

    assert accepted_code == 0
    assert accepted_output["status"] == "accepted"
    assert retry_code == 1
    assert retry_output["status"] == "needs_retry"


def test_single_paper_command_does_not_accept_topic_complete_as_its_gate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run,
        "run_paper",
        lambda **_: {"topic": "信息素养", "paper_id": "P006", "status": "complete"},
        raising=False,
    )

    exit_code = run.main(
        ["run-paper", "--topic", "信息素养", "--paper-id", "P006"]
    )

    assert exit_code == 1
