"""Minimum complete-call package in the fixed, jointly calibrated 20-call library."""
from __future__ import annotations

import json
import time
import warnings
import copy
import csv
from dataclasses import fields

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from normal_case_parameters import NormalCaseParameters
from normal_customer import build_history, QUARTER
from historical_duration import HistoricalDuration
from common import ROOT, CORES

OUT = ROOT / "checks/min_records/certificate.json"
SOURCE = ROOT / "checks/exp2_check.json"
CAPACITY_SOURCE = ROOT / "results/exp2/plot/evidence_capacity.csv"
TARGET_H = 7.17785615395177  # Source: the saved 18-record duration in the main sequential-submission CSV.
MIP_SECONDS = 45.0  # Existing author-chosen computation limit for each mixed-integer solve.
TOTAL_SECONDS = 180.0  # Existing author-chosen three-minute limit for the complete package search.


def main():
    started = time.monotonic()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(NormalCaseParameters)}
    c = NormalCaseParameters(**{k: v for k, v in saved["parameters"].items() if k in allowed})
    tolerance = c.solver_tolerance  # Source: the original historical LP feasibility tolerance.
    public, meter, power, maximum, truth, records, check = build_history(c)
    calibration = saved["history_calibration"]["calibration"]
    critical = np.array([
        calibration["global" if family == "global" else "other"]["joint_multiplier"]
        for family in public["error_families"]
    ])
    public["bounds"] = public["error_stat_SD_MW"] * critical
    history = HistoricalDuration(
        public, meter, public["J"], power,
        bias_bound=c.bias_bound_rate * c.load_mw,
        solver_tolerance=tolerance,
        duration_check_tolerance=c.duration_check_tolerance_hours,
    )
    n, s, calls = history.n, public["W"].shape[0], public["J"]
    # Sources: the original HistoricalDuration.A,b and its original d/mu bounds.
    upper_response = history.cap(TARGET_H)
    identity = sparse.eye(n + 1, format="csr")
    L = sparse.vstack([history.A, identity, -identity], format="csr")
    upper = np.r_[upper_response, history.bias_bound]
    lower = np.r_[np.zeros(n), -history.bias_bound]
    rhs = np.r_[history.b, upper, -lower]
    rows = L.shape[0]
    support = [
        sorted(set(int(i) for i in public["call_ids"][public["W"].getrow(r).indices]))
        for r in range(s)
    ]
    dates = [str(item["task"]["date"]) for item in public["designs"]]
    labels = []
    for r in range(rows):
        if r < 2 * s:
            index = r % s
            labels.append(dict(kind="statistic", statistic=index,
                               sign="minus" if r < s else "plus",
                               family=str(public["error_families"][index]),
                               required_dates=[dates[k] for k in support[index]]))
        else:
            index = (r - 2 * s) % (n + 1)
            labels.append(dict(kind="upper_bound" if r < 2 * s + n + 1 else "lower_bound",
                               variable="mu" if index == n else f"d[{index}]"))

    # A normalized multiplier is at most one; this is an exact consequence of sum(lambda)=1.
    gate_i, gate_j, gate_v = [], [], []
    gate_number = 0
    for r in range(2 * s):
        for k in support[r % s]:
            gate_i.extend([gate_number, gate_number])
            gate_j.extend([r, rows + k])
            gate_v.extend([1.0, -1.0])
            gate_number += 1
    gate = sparse.csr_matrix((gate_v, (gate_i, gate_j)),
                             shape=(gate_number, rows + calls))
    equal = sparse.hstack([L.T, sparse.csr_matrix((n + 1, calls))], format="csr")
    normalization = sparse.csr_matrix(
        (np.ones(rows), (np.zeros(rows, dtype=int), np.arange(rows))),
        shape=(1, rows + calls),
    )
    budget_row = sparse.csr_matrix(
        (np.ones(calls), (np.zeros(calls, dtype=int), rows + np.arange(calls))),
        shape=(1, rows + calls),
    )
    objective = np.r_[rhs, np.zeros(calls)]
    integrality = np.r_[np.zeros(rows, dtype=int), np.ones(calls, dtype=int)]
    bounds = Bounds(np.zeros(rows + calls), np.ones(rows + calls))
    base_constraints = [
        LinearConstraint(equal, np.zeros(n + 1), np.zeros(n + 1)),
        LinearConstraint(normalization, np.ones(1), np.ones(1)),
        LinearConstraint(gate, -np.inf, np.zeros(gate_number)),
    ]
    output = dict(
        analysis="ex_post selection from the already generated complete 20-call history; not an ex-ante optimum",
        source=SOURCE.relative_to(ROOT).as_posix(), parameters=c.values(), target_duration_hours=TARGET_H,
        call_dates=dates, readings=n, statistics=s, power_upper_MW=power,
        calibration=calibration, original_tolerance=tolerance,
        full_history_status=int(history.solve(TARGET_H).status), mip_runs=[],
    )

    def save():
        output["elapsed_seconds"] = time.monotonic() - started
        OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    def verify_package(selected, multipliers):
        active = np.array([set(r).issubset(selected) for r in support])
        stat_active = np.r_[active, active]
        A = history.A[stat_active]
        b = history.b[stat_active]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options")
            options = dict(threads=CORES, primal_feasibility_tolerance=tolerance,
                           dual_feasibility_tolerance=tolerance)
            primal = linprog(np.zeros(n + 1), A_ub=A, b_ub=b,
                             bounds=np.c_[lower, upper], method="highs", options=options)
            # Diagnostic phase-I: relax only submitted statistic bounds by t MW;
            # all original latent-response and shared-mu bounds remain exact.
            phase = linprog(
                np.r_[np.zeros(n + 1), 1.0],
                A_ub=sparse.hstack([A, -np.ones((A.shape[0], 1))], format="csr"),
                b_ub=b, bounds=np.c_[np.r_[lower, 0.0], np.r_[upper, np.inf]],
                method="highs", options=options,
            )
        result = dict(
            original_LP_status=int(primal.status), original_LP_message=primal.message,
            retained_statistics=int(active.sum()),
            dual_equation_max_abs=float(np.max(np.abs(L.T @ multipliers))),
            dual_normalization=float(multipliers.sum()),
            dual_rhs=float(rhs @ multipliers),
            positive_constraints=[
                dict(row=int(i), multiplier=float(multipliers[i]), rhs=float(rhs[i]), **labels[i])
                for i in np.flatnonzero(multipliers > tolerance)
            ],
            phase_I_status=int(phase.status),
        )
        if phase.success:
            result["phase_I_uniform_statistic_relaxation_MW"] = float(phase.fun)
            result["phase_I_mu_MW"] = float(phase.x[n])
            result["phase_I_latent_response_MW"] = phase.x[:n].tolist()
            result["phase_I_statistic_residuals_MW"] = (A @ phase.x[:n + 1] - b).tolist()
        if primal.success:
            result["original_feasible_mu_MW"] = float(primal.x[n])
            result["original_feasible_latent_response_MW"] = primal.x[:n].tolist()
        return result

    def run(budget):
        remaining = TOTAL_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            return None
        limit = min(MIP_SECONDS, remaining)
        constraints = base_constraints + [LinearConstraint(budget_row, -np.inf, [budget])]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options")
            solved = milp(
                objective, integrality=integrality, bounds=bounds, constraints=constraints,
                # Zero requested MIP gap is a computation choice to certify the minimum count.
                options=dict(time_limit=limit, mip_rel_gap=0.0, threads=CORES,
                             mip_feasibility_tolerance=tolerance),
            )
        result = dict(
            budget=int(budget), status=int(solved.status), message=solved.message,
            objective=None if solved.fun is None else float(solved.fun),
            lower_bound=None if getattr(solved, "mip_dual_bound", None) is None
            else float(solved.mip_dual_bound),
            mip_gap=None if getattr(solved, "mip_gap", None) is None else float(solved.mip_gap),
            computation_limit_seconds=limit,
        )
        if solved.x is not None:
            # One half only decodes binary selectors; it is not a scientific threshold.
            selected = set(int(k) for k in np.flatnonzero(solved.x[rows:] > 0.5))
            result["selected_call_ids_zero_based"] = sorted(selected)
            result["selected_dates"] = [dates[k] for k in sorted(selected)]
            if solved.fun < -tolerance:
                result["independent_verification"] = verify_package(selected, solved.x[:rows])
        result["excludes"] = bool(
            result["objective"] is not None and result["objective"] < -tolerance
            and result.get("independent_verification", {}).get("original_LP_status") == 2
            and result["independent_verification"]["dual_equation_max_abs"] <= tolerance
        )
        result["proved_no_excluding_package"] = bool(
            result["lower_bound"] is not None and result["lower_bound"] >= -tolerance
        )
        output["mip_runs"].append(result)
        save()
        print(json.dumps({k: result[k] for k in (
            "budget", "status", "objective", "lower_bound", "excludes",
            "proved_no_excluding_package")}, ensure_ascii=False), flush=True)
        return result

    # Reproduce the original feasible bracket before refining the minimum count.
    for budget in (6, 7):
        run(budget)
    tested = {r["budget"] for r in output["mip_runs"]}
    while time.monotonic() - started < TOTAL_SECONDS:
        positives = [r["budget"] for r in output["mip_runs"] if r["excludes"]]
        negatives = [r["budget"] for r in output["mip_runs"] if r["proved_no_excluding_package"]]
        high = min(positives) if positives else calls
        low = max(negatives) if negatives else 0  # Empty package cannot exclude: its physical box contains zero.
        if high <= low + 1:
            output["minimum_record_count"] = high
            output["minimum_proven"] = True
            break
        budget = (low + high) // 2
        if budget in tested:
            break
        run(budget)
        tested.add(budget)
    output.setdefault("minimum_proven", False)
    if output["mip_runs"]:
        positives = [r["budget"] for r in output["mip_runs"] if r["excludes"]]
        negatives = [r["budget"] for r in output["mip_runs"] if r["proved_no_excluding_package"]]
        output["minimum_record_upper_bound"] = min(positives) if positives else None
        output["minimum_record_lower_bound"] = max(negatives) + 1 if negatives else 1
        if positives:
            best = min((r for r in output["mip_runs"] if r["excludes"]), key=lambda r: r["budget"])
            selected = set(best["selected_call_ids_zero_based"])
            active = np.array([set(r).issubset(selected) for r in support])
            selected_history = copy.copy(history)
            selected_history.A = history.A[np.r_[active, active]]
            selected_history.b = history.b[np.r_[active, active]]
            selected_minimum = selected_history.minimum(maximum)
            full_minimum = history.minimum(maximum)
            best["independent_verification"]["minimum_duration_hours"] = selected_minimum
            witness_duration = selected_minimum
            minimum_witness = selected_history.solve(witness_duration)
            if not minimum_witness.success:
                # Source: the original duration solution check adds its saved 1e-6-hour tolerance.
                witness_duration += c.duration_check_tolerance_hours
                minimum_witness = selected_history.solve(witness_duration)
            best["independent_verification"]["minimum_duration_primal"] = dict(
                duration_hours=witness_duration, status=int(minimum_witness.status),
                message=minimum_witness.message,
            )
            if minimum_witness.success:
                best["independent_verification"]["minimum_duration_primal"].update(
                    shared_mean_MW=float(minimum_witness.x[n]),
                    latent_response_MW=minimum_witness.x[:n].tolist(),
                    maximum_statistic_residual_MW=float(np.max(
                        selected_history.A @ minimum_witness.x - selected_history.b)),
                )
            output["full_history_minimum_duration_hours"] = full_minimum
            output["minimum_package_vs_full_duration_difference_hours"] = full_minimum - selected_minimum
            details = []
            seen = set()
            for constraint in best["independent_verification"]["positive_constraints"]:
                if constraint["kind"] != "statistic" or constraint["statistic"] in seen:
                    continue
                r = constraint["statistic"]
                seen.add(r)
                w = public["W"].getrow(r)
                details.append(dict(
                    statistic=r, family=str(public["error_families"][r]),
                    required_recent_count=int(public["required"][r]), weights=w.data.tolist(),
                    measured_mean_MW=float((w @ (public["baseline"] - meter))[0]),
                    calibrated_error_bound_MW=float(public["bounds"][r]),
                    readings=[dict(
                        index=int(i), date=dates[int(public["call_ids"][i])],
                        elapsed_left_hours=float(public["left"][i] - public["starts"][i]),
                        elapsed_right_hours=float(public["left"][i] + QUARTER - public["starts"][i]),
                        measured_reduction_MW=float(public["baseline"][i] - meter[i]),
                    ) for i in w.indices],
                ))
            output["binding_statistics_details"] = details
            # Source: the existing main continuation comparison; no annual model is run here.
            with CAPACITY_SOURCE.open(encoding="utf-8-sig", newline="") as handle:
                capacity_rows = list(csv.DictReader(handle))
            full_row = next(r for r in capacity_rows if
                            abs(float(r["确认时长_小时"]) - full_minimum) <= c.duration_check_tolerance_hours)
            old_row = next(r for r in capacity_rows if
                           abs(float(r["确认时长_小时"]) - TARGET_H) <= c.duration_check_tolerance_hours)
            output["capacity_mapping"] = dict(
                source=str(CAPACITY_SOURCE),
                annual_maximum_extension_hours=None,
                archived_history_parameter_maximum_extension_hours=c.maximum_extension_hours,
                note="H is historical-only. Capacity mapping uses the existing main comparison with "
                     "continuation until shortage clearance, not the archived four-hour annual setting.",
                full_net_lower_MW=float(full_row["认证下界_兆瓦"]),
                full_net_upper_MW=float(full_row["认证上界_兆瓦"]),
                three_record_upper_MW=float(old_row["认证上界_兆瓦"]),
                minimum_four_vs_three_increment_MW=float(full_row["认证下界_兆瓦"])
                - float(old_row["认证上界_兆瓦"]),
            )
    save()
    if output.get("minimum_proven") and output.get("capacity_mapping"):
        best = min((r for r in output["mip_runs"] if r["excludes"]), key=lambda r: r["budget"])
        ruled_out = max(r["budget"] for r in output["mip_runs"] if r["proved_no_excluding_package"])
        capacity = output["capacity_mapping"]
        # All table values come from the solver output and the existing annual map.
        table_rows = [
            dict(记录包=f"任意不超过{ruled_out}条完整记录", 记录数=f"≤{ruled_out}",
                 确认时长_小时=output["target_duration_hours"],
                 时长结果类型="所有此类记录包所能确认时长的统一上界", 净认证下界_MW="",
                 净认证上界_MW=capacity["three_record_upper_MW"],
                 容量结果类型="统一上界；未求精确最优三条额度"),
            dict(记录包=f"所选{best['budget']}条完整记录", 记录数=best["budget"],
                 确认时长_小时=best["independent_verification"]["minimum_duration_hours"],
                 时长结果类型="该包支持的最短可行持续时间",
                 净认证下界_MW=capacity["full_net_lower_MW"],
                 净认证上界_MW=capacity["full_net_upper_MW"], 容量结果类型="达到全部二十条的计算区间"),
            dict(记录包="全部二十条完整记录", 记录数=len(dates),
                 确认时长_小时=output["full_history_minimum_duration_hours"],
                 时长结果类型="全集支持的最短可行持续时间",
                 净认证下界_MW=capacity["full_net_lower_MW"],
                 净认证上界_MW=capacity["full_net_upper_MW"], 容量结果类型="原主实验计算区间"),
        ]
        with OUT.with_name("record_package_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
    print(f"Result: {OUT}", flush=True)


if __name__ == "__main__":
    main()
