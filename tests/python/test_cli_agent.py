from __future__ import annotations

import json
from pathlib import Path

from oaa.cli import build_parser, main


def plan_payload():
    return {
        "goal": "Calculate a result",
        "steps": [
            {
                "tool": "calculator",
                "arguments": {"expression": "(25 * 4) / 2"},
                "reason": "Calculate the requested value",
            }
        ],
    }


def test_agent_plan_parser_and_preview_are_non_executing(capsys):
    payload = plan_payload()
    parsed = build_parser().parse_args([
        "agent", "plan", "--plan-json", json.dumps(payload),
    ])
    assert parsed.command == "agent"
    assert parsed.agent_action == "plan"

    assert main(["agent", "plan", "--plan-json", json.dumps(payload)]) == 0
    output = capsys.readouterr().out
    assert "Goal: Calculate a result" in output
    assert "calculator [compute]" in output
    assert "Preview only: no tool handlers were executed." in output


def test_agent_run_requires_allow_list_and_per_step_approval(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert main([
        "agent", "run", "--plan-json", json.dumps(plan_payload()),
        "--allow-tool", "calculator",
    ]) == 0
    output = capsys.readouterr().out
    assert "Approval required: calculator [compute]" in output
    assert '"status": "executed"' in output
    assert '"result": 50' in output


def test_agent_rejection_does_not_run_tool_or_continue(monkeypatch, capsys):
    payload = plan_payload()
    payload["steps"].append({
        "tool": "calculator",
        "arguments": {"expression": "7 * 8"},
        "reason": "Second step must not run after rejection",
    })
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert main([
        "agent", "run", "--plan-json", json.dumps(payload),
        "--allow-tool", "calculator",
    ]) == 1
    output = capsys.readouterr().out
    assert output.count("Approval required:") == 1
    assert '"status": "rejected"' in output
    assert '"status": "executed"' not in output


def test_agent_preflights_every_tool_before_prompting(tmp_path: Path, monkeypatch, capsys):
    workspace = tmp_path / "knowledge"
    workspace.mkdir()
    (workspace / "manual.md").write_text("OAA local notes", encoding="utf-8")
    payload = plan_payload()
    payload["steps"].append({
        "tool": "read_text_file",
        "arguments": {"path": "manual.md"},
        "reason": "Read local notes",
    })

    def unexpected_prompt(_):
        raise AssertionError("approval prompt must not run for a denied plan")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    code = main([
        "agent", "run", "--plan-json", json.dumps(payload),
        "--workspace", str(workspace),
        "--allow-tool", "calculator",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "policy denies tool 'read_text_file'" in captured.err


def test_agent_accepts_json_plan_file_and_rejects_oversized_file(tmp_path: Path, capsys):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan_payload()), encoding="utf-8")
    assert main(["agent", "plan", "--plan-file", str(plan_file)]) == 0
    capsys.readouterr()

    plan_file.write_text(" " * (64 * 1024 + 1), encoding="utf-8")
    assert main(["agent", "plan", "--plan-file", str(plan_file)]) == 2
    assert "65536-byte limit" in capsys.readouterr().err
