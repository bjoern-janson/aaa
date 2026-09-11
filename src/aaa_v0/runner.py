from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Type

from aaa_v0.agents import Agent
from aaa_v0.balance import BalanceReport, audit_history_pair
from aaa_v0.contracts import AssayConfig, ProtocolStatus, SequenceBalanceRule, TrialRecord
from aaa_v0.environment import MetaProbeEnvironment
from aaa_v0.histories import BalanceConstructionError, make_history_pair
from aaa_v0.measurement import (
    LearningCurve,
    PairedCurve,
    RecoveryTrajectory,
    compute_learning_curve,
    compute_paired_curve,
    compute_recovery_trajectory,
)
from aaa_v0.provenance import SeedCustody
from aaa_v0.serialization import sha256_json
from aaa_v0.tasks import context_signature, make_future_tasks, render_history_episode, render_task


class ProtocolStop(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class MatchReport:
    treatment: tuple[tuple[str, float], ...]
    control: tuple[tuple[str, float], ...]
    absolute_delta: tuple[tuple[str, float], ...]
    tolerance: float
    ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "treatment": dict(self.treatment),
            "control": dict(self.control),
            "absolute_delta": dict(self.absolute_delta),
            "tolerance": self.tolerance,
            "ok": self.ok,
        }


def assert_match_gate(
    family_deltas: dict[str, float],
    tolerance: float,
    *,
    treatment: dict[str, float] | None = None,
    control: dict[str, float] | None = None,
) -> MatchReport:
    if tolerance < 0:
        raise ValueError("match tolerance must be non-negative")
    abs_delta = {family: abs(float(value)) for family, value in family_deltas.items()}
    ok = all(value <= tolerance for value in abs_delta.values())
    report = MatchReport(
        treatment=tuple(sorted((treatment or {}).items())),
        control=tuple(sorted((control or {}).items())),
        absolute_delta=tuple(sorted(abs_delta.items())),
        tolerance=tolerance,
        ok=ok,
    )
    if not ok:
        raise ProtocolStop("MATCH_FAILURE")
    return report


@dataclass(frozen=True)
class RunBundle:
    config_hash: str
    history_pair_hash: str
    balance_report: BalanceReport
    terminal_snapshot_hash: str
    future_seed_commitment: str
    future_seed_reveal_hash: str
    raw_trial_records_hash: str
    match_report: MatchReport
    d1_paired_curve: PairedCurve
    d2_paired_curve: PairedCurve
    d3_treatment_recovery: RecoveryTrajectory
    d3_control_curve: LearningCurve
    protocol_status: ProtocolStatus
    raw_trial_records: tuple[TrialRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config_hash": self.config_hash,
            "history_pair_hash": self.history_pair_hash,
            "balance_report": asdict(self.balance_report),
            "terminal_snapshot_hash": self.terminal_snapshot_hash,
            "future_seed_commitment": self.future_seed_commitment,
            "future_seed_reveal_hash": self.future_seed_reveal_hash,
            "raw_trial_records_hash": self.raw_trial_records_hash,
            "match_report": self.match_report.to_dict(),
            "d1_paired_curve": asdict(self.d1_paired_curve),
            "d2_paired_curve": asdict(self.d2_paired_curve),
            "d3_treatment_recovery": asdict(self.d3_treatment_recovery),
            "d3_control_curve": asdict(self.d3_control_curve),
            "protocol_status": asdict(self.protocol_status),
        }


def _history_pair_dict(pair) -> dict[str, object]:
    return {
        "treatment": [e.to_dict() for e in pair.treatment],
        "control": [e.to_dict() for e in pair.control],
        "treatment_phi": list(pair.treatment_phi),
    }


def _new_agent_pair(agent_cls: Type[Agent], theta_count: int, treatment_history, control_history):
    treatment_agent = agent_cls(theta_count=theta_count)
    control_agent = agent_cls(theta_count=theta_count)
    treatment_agent.observe_history(treatment_history)
    control_agent.observe_history(control_history)
    return treatment_agent, control_agent


def _run_family(
    tasks,
    config: AssayConfig,
    treatment_agent: Agent,
    control_agent: Agent,
    terminal_treatment_preferred: dict[tuple[int, ...], tuple[int, ...] | None],
) -> tuple[TrialRecord, ...]:
    records: list[TrialRecord] = []
    for task in tasks:
        view = render_task(task, config)
        treatment_agent.start_task(view)
        control_agent.start_task(view)
        t_env = MetaProbeEnvironment(task, config.q_count, config.max_budget)
        c_env = MetaProbeEnvironment(task, config.q_count, config.max_budget)

        for condition, agent in (("treatment", treatment_agent), ("control", control_agent)):
            guess = agent.guess_theta()
            records.append(
                TrialRecord(
                    condition=condition,
                    agent_name=agent.name,
                    task_id=task.task_id,
                    family=task.family,
                    budget=0,
                    guess=guess,
                    theta=task.theta,
                    correct=guess == task.theta,
                    chosen_probe_signature=None,
                    old_rule_adherence=None,
                )
            )

        first_treatment_probe = None
        for budget in range(1, config.max_budget + 1):
            t_probe = treatment_agent.choose_probe()
            c_probe = control_agent.choose_probe()
            if first_treatment_probe is None:
                first_treatment_probe = t_probe
            t_obs = t_env.probe(t_probe)
            c_obs = c_env.probe(c_probe)
            treatment_agent.observe_probe(t_obs)
            control_agent.observe_probe(c_obs)
            t_guess = treatment_agent.guess_theta()
            c_guess = control_agent.guess_theta()
            old_adherence = None
            if task.family == "D3" and budget == 1:
                old = terminal_treatment_preferred.get(view.context_signature)
                old_adherence = old is not None and first_treatment_probe == old
            records.append(
                TrialRecord(
                    condition="treatment",
                    agent_name=treatment_agent.name,
                    task_id=task.task_id,
                    family=task.family,
                    budget=budget,
                    guess=t_guess,
                    theta=task.theta,
                    correct=t_guess == task.theta,
                    chosen_probe_signature=t_probe,
                    old_rule_adherence=old_adherence,
                )
            )
            records.append(
                TrialRecord(
                    condition="control",
                    agent_name=control_agent.name,
                    task_id=task.task_id,
                    family=task.family,
                    budget=budget,
                    guess=c_guess,
                    theta=task.theta,
                    correct=c_guess == task.theta,
                    chosen_probe_signature=c_probe,
                    old_rule_adherence=None,
                )
            )
    return tuple(records)


def run_reference_pair(
    config: AssayConfig,
    sequence_rule: SequenceBalanceRule,
    history_seed: int,
    future_seed: int,
    agent_cls: Type[Agent],
    match_tolerance: float,
) -> RunBundle:
    try:
        pair = make_history_pair(config, history_seed, sequence_rule)
    except BalanceConstructionError as exc:
        raise ProtocolStop("BALANCE_FAILURE", str(exc)) from exc
    balance = audit_history_pair(pair, sequence_rule)
    if not balance.ok:
        raise ProtocolStop("BALANCE_FAILURE")

    treatment_history = tuple(render_history_episode(e, config) for e in pair.treatment)
    control_history = tuple(render_history_episode(e, config) for e in pair.control)

    terminal_treatment, terminal_control = _new_agent_pair(
        agent_cls, config.theta_count, treatment_history, control_history
    )
    treatment_state_hash = sha256_json(terminal_treatment.state_dict())
    control_state_hash = sha256_json(terminal_control.state_dict())
    history_pair_hash = sha256_json(_history_pair_dict(pair))
    config_hash = sha256_json(config.to_dict())

    terminal_preferred = {
        context_signature(z, config.z_count): terminal_treatment.preferred_probe(context_signature(z, config.z_count))
        for z in range(config.z_count)
    }

    custody = SeedCustody(history_seed=history_seed, sealed_future_seed=future_seed)
    snapshot = custody.freeze_terminal(
        treatment_state_hash,
        control_state_hash,
        history_pair_hash,
        config_hash,
    )
    reveal = custody.reveal_future_seed(snapshot)
    families = make_future_tasks(config, reveal, pair.treatment_phi)

    all_records: list[TrialRecord] = []
    for family in ("D1", "D2", "D3"):
        treatment_agent, control_agent = _new_agent_pair(
            agent_cls, config.theta_count, treatment_history, control_history
        )
        family_records = _run_family(
            families[family],
            config,
            treatment_agent,
            control_agent,
            terminal_preferred,
        )
        all_records.extend(family_records)
    raw_records = tuple(all_records)

    treatment_zero: dict[str, float] = {}
    control_zero: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for family in ("D1", "D2", "D3"):
        t0 = [float(r.correct) for r in raw_records if r.family == family and r.condition == "treatment" and r.budget == 0]
        c0 = [float(r.correct) for r in raw_records if r.family == family and r.condition == "control" and r.budget == 0]
        treatment_zero[family] = sum(t0) / len(t0)
        control_zero[family] = sum(c0) / len(c0)
        deltas[family] = treatment_zero[family] - control_zero[family]
    match_report = assert_match_gate(
        deltas,
        match_tolerance,
        treatment=treatment_zero,
        control=control_zero,
    )

    d1 = compute_paired_curve(raw_records, family="D1")
    d2 = compute_paired_curve(raw_records, family="D2")
    d3_recovery = compute_recovery_trajectory(raw_records, change_after=config.d3_change_after)
    d3_control = compute_learning_curve(raw_records, condition="control", family="D3")

    return RunBundle(
        config_hash=config_hash,
        history_pair_hash=history_pair_hash,
        balance_report=balance,
        terminal_snapshot_hash=snapshot.snapshot_hash,
        future_seed_commitment=custody.future_seed_commitment,
        future_seed_reveal_hash=sha256_json({"seed": reveal.seed, "commitment": reveal.commitment}),
        raw_trial_records_hash=sha256_json([asdict(r) for r in raw_records]),
        match_report=match_report,
        d1_paired_curve=d1,
        d2_paired_curve=d2,
        d3_treatment_recovery=d3_recovery,
        d3_control_curve=d3_control,
        protocol_status=ProtocolStatus(
            balance_ok=True,
            custody_ok=True,
            match_ok=True,
            controls_ok=False,
            stop_code=None,
        ),
        raw_trial_records=raw_records,
    )
