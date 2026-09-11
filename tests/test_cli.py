import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from aaa_v0.contracts import AssayConfig, SequenceBalanceRule
from aaa_v0.prereg import SPEC_COMMIT


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    return env


def _run(*args, check=True):
    return subprocess.run(
        [sys.executable, "-m", "aaa_v0.cli", *args],
        cwd=Path.cwd(),
        env=_env(),
        text=True,
        capture_output=True,
        check=check,
    )


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixtures(tmp_path):
    cfg = AssayConfig.exploratory_default()
    seq = SequenceBalanceRule(2.0, 2.0, 2.0, 3, 1000)
    cfg_path = tmp_path / "config.json"
    seq_path = tmp_path / "sequence.json"
    rules_path = tmp_path / "rules.json"
    _write(cfg_path, cfg.to_dict())
    _write(seq_path, seq.__dict__)
    _write(rules_path, {
        "sequence_balance_rule": seq.__dict__,
        "d1_rule": {"min_positive_budget_points": 1, "min_mean_delta": 0.01, "max_zero_budget_abs_delta": 0.0},
        "d2_rule": {"min_positive_budget_points": 1, "min_mean_delta": 0.01, "max_zero_budget_abs_delta": 0.0},
        "d3_rule": {"max_post_change_old_rule_adherence": 0.50, "min_post_change_competence": 0.99, "evaluation_start_offset": 4},
        "spec_commit": SPEC_COMMIT,
    })
    return cfg_path, seq_path, rules_path


def test_cli_surface_has_only_approved_workflow_and_rejects_frontier_command():
    result = _run("--help")
    for name in ("audit-history", "calibrate-controls", "freeze-prereg", "validate-controls"):
        assert name in result.stdout
    bad = _run("run-frontier", check=False)
    assert bad.returncode != 0


def test_audit_history_writes_manifested_balanced_output(tmp_path):
    cfg_path, seq_path, _ = _fixtures(tmp_path)
    out = tmp_path / "audit"
    _run(
        "audit-history",
        "--config", str(cfg_path),
        "--sequence-rule", str(seq_path),
        "--history-seed", "17",
        "--output-dir", str(out),
    )
    payload = json.loads((out / "history_audit.json").read_text())
    assert payload["status"] == "BALANCED"
    assert payload["treatment_q_theta_mi"] == 0.0
    assert payload["control_q_theta_mi"] == 0.0
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert "history_audit.json" in manifest


def test_full_offline_control_workflow_is_manifested_and_unscored(tmp_path):
    cfg_path, seq_path, rules_path = _fixtures(tmp_path)
    cal = tmp_path / "cal"
    _run(
        "calibrate-controls",
        "--config", str(cfg_path),
        "--sequence-rule", str(seq_path),
        "--history-seeds", "17",
        "--future-seeds", "101",
        "--output-dir", str(cal),
    )
    exploratory = json.loads((cal / "exploratory_calibration.json").read_text())
    assert exploratory["status"] == "EXPLORATORY_UNSCORED"
    assert exploratory["aaa_evidence"] == "NONE"
    raw_line = json.loads((cal / "raw_control_runs.jsonl").read_text().splitlines()[0])
    assert raw_line["raw_trial_records"]
    assert raw_line["bundle"]["raw_trial_records_hash"]

    frozen = tmp_path / "frozen"
    _run(
        "freeze-prereg",
        "--exploratory-report", str(cal / "exploratory_calibration.json"),
        "--decision-rules", str(rules_path),
        "--confirmatory-history-seeds", "18",
        "--confirmatory-future-seeds", "102",
        "--output-dir", str(frozen),
   )
    prereg = json.loads((frozen / "preregistration.json").read_text())
    assert prereg["spec_commit"] == SPEC_COMMIT
    expected_hash = hashlib.sha256((frozen / "preregistration.json").read_bytes()).hexdigest()
    assert (frozen / "preregistration.sha256").read_text().strip() == expected_hash

    validated = tmp_path / "validated"
    _run(
        "validate-controls",
        "--preregistration", str(frozen / "preregistration.json"),
        "--output-dir", str(validated),
    )
    result = json.loads((validated / "confirmatory_controls.json").read_text())
    assert result["status"] == "CONTROL_VALIDATED"
    assert result["aaa_evidence"] == "NONE"
    assert result["preregistration_hash"]
    confirmatory_raw = json.loads((validated / "confirmatory_control_runs.jsonl").read_text().splitlines()[0])
    assert confirmatory_raw["raw_trial_records"]
    manifest = json.loads((validated / "MANIFEST.json").read_text())
    assert "confirmatory_controls.json" in manifest
    assert "confirmatory_control_runs.jsonl" in manifest
