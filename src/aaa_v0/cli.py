from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from aaa_v0.agents import FixedMetaAgent, LocalAgent, RevisingMetaAgent
from aaa_v0.balance import audit_history_pair
from aaa_v0.contracts import AssayConfig, SequenceBalanceRule
from aaa_v0.histories import BalanceConstructionError, make_history_pair
from aaa_v0.prereg import (
    Preregistration,
    RecoveryRule,
    TransferRule,
    freeze_preregistration,
    seed_identity,
    validate_confirmatory_controls,
)
from aaa_v0.runner import ProtocolStop, run_reference_pair
from aaa_v0.serialization import canonical_json_bytes, sha256_json


def _read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(directory: Path) -> None:
    entries: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            entries[path.relative_to(directory).as_posix()] = _sha256_file(path)
    _write_canonical(directory / "MANIFEST.json", entries)


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _load_config(path: str) -> AssayConfig:
    return AssayConfig(**_read_json(path))  # type: ignore[arg-type]


def _load_sequence_rule(path: str) -> SequenceBalanceRule:
    return SequenceBalanceRule(**_read_json(path))  # type: ignore[arg-type]


def _cmd_audit_history(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    rule = _load_sequence_rule(args.sequence_rule)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        pair = make_history_pair(cfg, args.history_seed, rule)
        report = audit_history_pair(pair, rule)
        payload = {
            "status": "BALANCED" if report.ok else "BALANCE_FAILURE",
            "config_hash": sha256_json(cfg.to_dict()),
            "history_pair_hash": sha256_json({
                "treatment": [e.to_dict() for e in pair.treatment],
                "control": [e.to_dict() for e in pair.control],
                "treatment_phi": list(pair.treatment_phi),
            }),
            "treatment_mi": report.treatment_mi,
            "control_mi": report.control_mi,
            "sequence": asdict(report.sequence),
            "mismatched_fields": list(report.mismatched_fields),
        }
        code = 0 if report.ok else 1
    except BalanceConstructionError as exc:
        payload = {
            "status": "BALANCE_FAILURE",
            "error": str(exc),
            "config_hash": sha256_json(cfg.to_dict()),
        }
        code = 1
    _write_canonical(out / "history_audit.json", payload)
    _write_manifest(out)
    return code


def _reference_classes():
    return (
        ("LOCAL", LocalAgent),
        ("FIXED_META", FixedMetaAgent),
        ("REVISING_META", RevisingMetaAgent),
    )


def _cmd_calibrate_controls(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    rule = _load_sequence_rule(args.sequence_rule)
    history_seeds = _parse_int_list(args.history_seeds)
    future_seeds = _parse_int_list(args.future_seeds)
    if not history_seeds or not future_seeds:
        raise SystemExit("exploratory seed lists must be non-empty")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines: list[bytes] = []
    summaries: dict[str, list[dict[str, object]]] = {name: [] for name, _ in _reference_classes()}
    for history_seed in history_seeds:
        for future_seed in future_seeds:
            for name, cls in _reference_classes():
                bundle = run_reference_pair(cfg, rule, history_seed, future_seed, cls, 0.0)
                entry = {
                    "agent": name,
                    "history_seed": history_seed,
                    "future_seed": future_seed,
                    "bundle": bundle.to_dict(),
                }
                lines.append(canonical_json_bytes(entry))
                summaries[name].append(bundle.to_dict())
    (out / "raw_control_runs.jsonl").write_bytes(b"\n".join(lines) + b"\n")

    exploratory_ids = tuple(
        sorted(
            [seed_identity("history", s) for s in history_seeds]
            + [seed_identity("future", s) for s in future_seeds]
        )
    )
    report = {
        "status": "EXPLORATORY_UNSCORED",
        "next_phase_gate_passed": False,
        "aaa_evidence": "NONE",
        "config": cfg.to_dict(),
        "sequence_balance_rule": asdict(rule),
        "history_seeds": list(history_seeds),
        "future_seeds": list(future_seeds),
        "exploratory_seed_hashes": list(exploratory_ids),
        "control_summaries": summaries,
    }
    _write_canonical(out / "exploratory_calibration.json", report)
    _write_manifest(out)
    return 0


def _cmd_freeze_prereg(args: argparse.Namespace) -> int:
    exploratory = _read_json(args.exploratory_report)
    rules = _read_json(args.decision_rules)
    cfg = AssayConfig(**dict(exploratory["config"]))  # type: ignore[arg-type]
    report_seq = SequenceBalanceRule(**dict(exploratory["sequence_balance_rule"]))  # type: ignore[arg-type]
    frozen_seq = SequenceBalanceRule(**dict(rules["sequence_balance_rule"]))  # type: ignore[arg-type]
    if report_seq != frozen_seq:
        raise SystemExit("decision-rule sequence balance contract differs from exploratory report")
    d1 = TransferRule(**dict(rules["d1_rule"]))  # type: ignore[arg-type]
    d2 = TransferRule(**dict(rules["d2_rule"]))  # type: ignore[arg-type]
    d3 = RecoveryRule(**dict(rules["d3_rule"]))  # type: ignore[arg-type]
    history_seeds = _parse_int_list(args.confirmatory_history_seeds)
    future_seeds = _parse_int_list(args.confirmatory_future_seeds)
    prereg = freeze_preregistration(
        cfg,
        frozen_seq,
        d1,
        d2,
        d3,
        tuple(str(x) for x in exploratory["exploratory_seed_hashes"]),  # type: ignore[arg-type]
        history_seeds,
        future_seeds,
        str(rules["spec_commit"]),
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = out / "preregistration.json"
    _write_canonical(prereg_path, prereg.to_dict())
    (out / "preregistration.sha256").write_text(_sha256_file(prereg_path) + "\n", encoding="utf-8")
    _write_manifest(out)
    return 0


def _cmd_validate_controls(args: argparse.Namespace) -> int:
    prereg = Preregistration.from_dict(_read_json(args.preregistration))
    if len(prereg.confirmatory_history_seeds) != 1 or len(prereg.confirmatory_future_seeds) != 1:
        raise SystemExit("multi-seed confirmatory aggregation is not frozen in AAA-v0")
    history_seed = prereg.confirmatory_history_seeds[0]
    future_seed = prereg.confirmatory_future_seeds[0]
    bundles = {}
    try:
        for name, cls in _reference_classes():
            bundles[name] = run_reference_pair(
                prereg.config,
                prereg.sequence_balance_rule,
                history_seed,
                future_seed,
                cls,
                prereg.d1_rule.max_zero_budget_abs_delta,
            )
    except ProtocolStop as exc:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": exc.code,
            "next_phase_gate_passed": False,
            "aaa_evidence": "NONE",
            "preregistration_hash": sha256_json(prereg.to_dict()),
        }
        _write_canonical(out / "confirmatory_controls.json", payload)
        _write_manifest(out)
        return 1

    report = validate_confirmatory_controls(
        prereg,
        bundles,
        sequence_balance_rule=prereg.sequence_balance_rule,
        history_seeds=prereg.confirmatory_history_seeds,
        future_seeds=prereg.confirmatory_future_seeds,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_canonical(out / "confirmatory_controls.json", report.to_dict())
    _write_manifest(out)
    return 0 if report.next_phase_gate_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aaa-v0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit-history")
    p.add_argument("--config", required=True)
    p.add_argument("--sequence-rule", required=True)
    p.add_argument("--history-seed", required=True, type=int)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=_cmd_audit_history)

    p = sub.add_parser("calibrate-controls")
    p.add_argument("--config", required=True)
    p.add_argument("--sequence-rule", required=True)
    p.add_argument("--history-seeds", required=True)
    p.add_argument("--future-seeds", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=_cmd_calibrate_controls)

    p = sub.add_parser("freeze-prereg")
    p.add_argument("--exploratory-report", required=True)
    p.add_argument("--decision-rules", required=True)
    p.add_argument("--confirmatory-history-seeds", required=True)
    p.add_argument("--confirmatory-future-seeds", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=_cmd_freeze_prereg)

    p = sub.add_parser("validate-controls")
    p.add_argument("--preregistration", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=_cmd_validate_controls)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
