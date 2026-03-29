#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


EPS = 1e-9
STAGES = ("A", "B", "C", "E")
ABC = ("A", "B", "C")
RESULT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RESULT_DIR.parent
FILE_A_PATH = PROJECT_DIR / "simulation.py"
FILE_C_PATH = PROJECT_DIR / "optimized_robot_simulation_v3.py"


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def load_file_a_module() -> Any:
    return _load_module("codex_original_simulation_a", FILE_A_PATH)


@lru_cache(maxsize=None)
def load_file_c_module() -> Any:
    return _load_module("codex_original_simulation_c", FILE_C_PATH)


@dataclass(frozen=True)
class AnalyticalParams:
    calibration_h: Dict[str, float] = field(
        default_factory=lambda: {"A": 20.0 / 60.0, "B": 15.0 / 60.0, "C": 15.0 / 60.0, "E": 30.0 / 60.0}
    )
    test_h: Dict[str, float] = field(default_factory=lambda: {"A": 2.0, "B": 1.5, "C": 2.0, "E": 2.5})
    defect_rate: Dict[str, float] = field(default_factory=lambda: {"A": 0.03, "B": 0.025, "C": 0.02, "D": 0.002})
    op_error_rate: Dict[str, float] = field(default_factory=lambda: {"A": 0.025, "B": 0.03, "C": 0.015, "E": 0.02})
    false_positive_ratio: float = 0.60
    false_negative_ratio: float = 0.40
    load_h: float = 0.5
    unload_h: float = 0.5


def part1_closed_form(params: Optional[AnalyticalParams] = None) -> Dict[str, float]:
    params = params or AnalyticalParams()
    alpha = params.false_positive_ratio
    beta = params.false_negative_ratio

    s: Dict[str, float] = {}
    f: Dict[str, float] = {}
    p_stage: Dict[str, float] = {}
    missed: Dict[str, float] = {}

    for stage in ABC:
        p = params.defect_rate[stage]
        q = params.op_error_rate[stage]
        s[stage] = (1.0 - p) * (1.0 - alpha * q) + p * (beta * q)
        f[stage] = 1.0 - s[stage]
        p_stage[stage] = 1.0 - f[stage] ** 2
        missed[stage] = p * beta * q

    missed["D"] = params.defect_rate["D"]
    p_detect_pool = 1.0 - (1.0 - missed["A"]) * (1.0 - missed["B"]) * (1.0 - missed["C"]) * (1.0 - missed["D"])

    denom_l = missed["A"] + missed["B"] + missed["C"] + missed["D"]
    lam = {
        "A": missed["A"] / max(denom_l, EPS),
        "B": missed["B"] / max(denom_l, EPS),
        "C": missed["C"] / max(denom_l, EPS),
        "D": missed["D"] / max(denom_l, EPS),
    }

    q_e = params.op_error_rate["E"]
    s_e = (1.0 - p_detect_pool) * (1.0 - alpha * q_e) + p_detect_pool * (beta * q_e)
    f_e = 1.0 - s_e
    p_e = 1.0 - f_e**2
    p_overall = p_stage["A"] * p_stage["B"] * p_stage["C"] * p_e

    d = {stage: params.calibration_h[stage] + params.test_h[stage] for stage in STAGES}
    expected_total_h = (
        params.load_h
        + d["A"] * (1.0 + f["A"])
        + p_stage["A"] * d["B"] * (1.0 + f["B"])
        + p_stage["A"] * p_stage["B"] * d["C"] * (1.0 + f["C"])
        + p_stage["A"] * p_stage["B"] * p_stage["C"] * d["E"] * (1.0 + f_e)
        + params.unload_h
    )

    return {
        "s_A": s["A"],
        "s_B": s["B"],
        "s_C": s["C"],
        "f_A": f["A"],
        "f_B": f["B"],
        "f_C": f["C"],
        "P_A": p_stage["A"],
        "P_B": p_stage["B"],
        "P_C": p_stage["C"],
        "P_G": p_detect_pool,
        "lambda_1": lam["A"],
        "lambda_2": lam["B"],
        "lambda_3": lam["C"],
        "lambda_4": lam["D"],
        "s_E": s_e,
        "f_E": f_e,
        "P_E": p_e,
        "P_overall": p_overall,
        "E_total_h": expected_total_h,
        "missed_A": missed["A"],
        "missed_B": missed["B"],
        "missed_C": missed["C"],
        "missed_D": missed["D"],
    }


def part1_monte_carlo(
    params: Optional[AnalyticalParams] = None,
    *,
    n: int = 10_000,
    seed: int = 20260329,
) -> Dict[str, float]:
    params = params or AnalyticalParams()
    rng = random.Random(seed)

    qualified = 0
    stage_pass_counts = {stage: 0 for stage in STAGES}
    stage_enter_counts = {stage: 0 for stage in STAGES}
    total_time = 0.0
    hidden_source_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    e_detect_pool_count = 0

    alpha = params.false_positive_ratio
    beta = params.false_negative_ratio

    for _ in range(n):
        hidden = {"A": False, "B": False, "C": False, "D": rng.random() < params.defect_rate["D"]}
        attempts = {stage: 0 for stage in STAGES}
        passed = {stage: False for stage in STAGES}
        rejected = False
        e_pool_counted = False
        total_robot_h = params.load_h + params.unload_h

        for stage in STAGES:
            stage_enter_counts[stage] += 1
            while attempts[stage] < 2 and not passed[stage]:
                attempts[stage] += 1
                total_robot_h += params.calibration_h[stage] + params.test_h[stage]

                if stage in ABC:
                    has_defect = rng.random() < params.defect_rate[stage]
                    if has_defect:
                        if rng.random() < beta * params.op_error_rate[stage]:
                            passed[stage] = True
                            hidden[stage] = True
                    else:
                        if rng.random() >= alpha * params.op_error_rate[stage]:
                            passed[stage] = True
                else:
                    has_hidden = hidden["A"] or hidden["B"] or hidden["C"] or hidden["D"]
                    if has_hidden:
                        if not e_pool_counted:
                            e_pool_counted = True
                            e_detect_pool_count += 1
                            for key in ("A", "B", "C", "D"):
                                if hidden[key]:
                                    hidden_source_counts[key] += 1
                        if rng.random() < beta * params.op_error_rate["E"]:
                            passed["E"] = True
                    else:
                        if rng.random() >= alpha * params.op_error_rate["E"]:
                            passed["E"] = True

            if not passed[stage]:
                rejected = True
                break
            stage_pass_counts[stage] += 1

        if not rejected and all(passed.values()):
            qualified += 1
        total_time += total_robot_h

    denom_source = sum(hidden_source_counts.values())
    lam = {key: hidden_source_counts[key] / max(denom_source, 1) for key in hidden_source_counts}

    return {
        "sim_P_A": stage_pass_counts["A"] / max(1, stage_enter_counts["A"]),
        "sim_P_B": stage_pass_counts["B"] / max(1, stage_enter_counts["B"]),
        "sim_P_C": stage_pass_counts["C"] / max(1, stage_enter_counts["C"]),
        "sim_P_E": stage_pass_counts["E"] / max(1, stage_enter_counts["E"]),
        "sim_P_overall": qualified / max(n, 1),
        "sim_E_total_h": total_time / max(n, 1),
        "sim_P_G": e_detect_pool_count / max(n, 1),
        "sim_lambda_1": lam["A"],
        "sim_lambda_2": lam["B"],
        "sim_lambda_3": lam["C"],
        "sim_lambda_4": lam["D"],
    }


def build_integrated_config(
    *,
    failure_multiplier: float = 1.0,
    processing_time_multiplier: float = 1.0,
    queue_capacity_multiplier: float = 1.0,
    stage_capacity_multiplier: float = 1.0,
    defect_multiplier: float = 1.0,
    operator_error_multiplier: float = 1.0,
    benches: Optional[int] = None,
) -> Any:
    mod = load_file_c_module()
    base = mod.build_default_config()

    stages = {}
    for name, cfg in base.stages.items():
        queue_capacity = None
        if cfg.queue_capacity is not None:
            queue_capacity = max(1, int(round(cfg.queue_capacity * queue_capacity_multiplier)))
        worker_count = max(1, int(round(cfg.worker_count * stage_capacity_multiplier)))
        station_count = max(1, int(round(cfg.station_count * stage_capacity_multiplier)))
        failure_bands = tuple(
            mod.FailureBand(band.max_hour, max(0.0, band.rate_per_hour * failure_multiplier))
            for band in cfg.failure_bands
        )
        stages[name] = mod.StageConfig(
            name=name,
            calibration_h=cfg.calibration_h * processing_time_multiplier,
            test_h=cfg.test_h * processing_time_multiplier,
            defect_rate=min(1.0, cfg.defect_rate * defect_multiplier),
            operator_error_rate=min(1.0, cfg.operator_error_rate * operator_error_multiplier),
            worker_count=worker_count,
            station_count=station_count,
            queue_capacity=queue_capacity,
            queue_policy=cfg.queue_policy,
            failure_bands=failure_bands,
        )

    return mod.SimulationConfig(
        stages=stages,
        alpha=base.alpha,
        beta=base.beta,
        load_h=base.load_h * processing_time_multiplier,
        unload_h=base.unload_h * processing_time_multiplier,
        predebug_h=base.predebug_h * processing_time_multiplier,
        predebug_reduce_e=base.predebug_reduce_e,
        benches=base.benches if benches is None else max(1, benches),
        repair_h=base.repair_h,
        max_attempts_per_stage=base.max_attempts_per_stage,
        latent_defect_rate_d=min(1.0, base.latent_defect_rate_d * defect_multiplier),
    )


def build_file_a_params(
    *,
    failure_multiplier: float = 1.0,
    processing_time_multiplier: float = 1.0,
    defect_multiplier: float = 1.0,
    operator_error_multiplier: float = 1.0,
    benches: Optional[int] = None,
) -> Any:
    mod = load_file_a_module()
    base = mod.SimulationParams()
    failure_rate_table = {
        stage: [mod.FailureBand(b.max_hour, max(0.0, b.rate_per_hour * failure_multiplier)) for b in bands]
        for stage, bands in base.failure_rate_table.items()
    }
    return mod.SimulationParams(
        p={key: min(1.0, value * defect_multiplier) for key, value in base.p.items()},
        q={key: min(1.0, value * operator_error_multiplier) for key, value in base.q.items()},
        alpha=base.alpha,
        beta=base.beta,
        delta={key: value * processing_time_multiplier for key, value in base.delta.items()},
        theta={key: value * processing_time_multiplier for key, value in base.theta.items()},
        load_time=base.load_time * processing_time_multiplier,
        unload_time=base.unload_time * processing_time_multiplier,
        predebug_time=base.predebug_time * processing_time_multiplier,
        predebug_theta_reduction=base.predebug_theta_reduction,
        benches=base.benches if benches is None else max(1, benches),
        stations=base.stations,
        failure_rate_table=failure_rate_table,
        repair_time_after_failure=base.repair_time_after_failure,
    )


def normalize_integrated_summary(summary: Dict[str, float], *, n_robots: int) -> Dict[str, float]:
    out = dict(summary)
    out["model"] = "integrated"
    out["n_robots"] = float(n_robots)
    out["makespan_hours"] = summary["makespan_h"]
    out["qualified_count"] = summary["qualified_count"]
    out["throughput_per_day"] = summary["qualified_count"] / max(summary["makespan_days"], EPS)
    return out


def normalize_file_a_summary(summary: Dict[str, float], *, n_robots: int) -> Dict[str, float]:
    out = dict(summary)
    out["model"] = "file_a"
    out["n_robots"] = float(n_robots)
    out["completed_count"] = float(n_robots)
    out["hidden_qualified_count"] = summary["qualified_count"] * summary["omission_probability"]
    out["rejected_count"] = max(0.0, float(n_robots) - summary["qualified_count"])
    out["false_positive_events"] = summary["total_misjudge_events"]
    out["false_negative_events"] = summary["total_omission_events"]
    out["makespan_h"] = summary["makespan_hours"]
    out["throughput_per_day"] = summary["qualified_count"] / max(summary["makespan_days"], EPS)
    out["stage_A_station_utilization"] = summary.get("station_A", 0.0)
    out["stage_B_station_utilization"] = summary.get("station_B", 0.0)
    out["stage_C_station_utilization"] = summary.get("station_C", 0.0)
    out["stage_E_station_utilization"] = summary.get("station_E", 0.0)
    return out


def run_integrated_model(
    *,
    part: int,
    n_robots: int,
    shift_hours: float,
    replications: int = 1,
    seed: int = 20260329,
    handover_h: float = 0.5,
    maintenance_every_days: int = 7,
    maintenance_h: float = 4.0,
    enable_predebug: bool = False,
    failure_multiplier: float = 1.0,
    processing_time_multiplier: float = 1.0,
    queue_capacity_multiplier: float = 1.0,
    stage_capacity_multiplier: float = 1.0,
    defect_multiplier: float = 1.0,
    operator_error_multiplier: float = 1.0,
    benches: Optional[int] = None,
) -> Dict[str, float]:
    mod = load_file_c_module()
    config = build_integrated_config(
        failure_multiplier=failure_multiplier,
        processing_time_multiplier=processing_time_multiplier,
        queue_capacity_multiplier=queue_capacity_multiplier,
        stage_capacity_multiplier=stage_capacity_multiplier,
        defect_multiplier=defect_multiplier,
        operator_error_multiplier=operator_error_multiplier,
        benches=benches,
    )
    scenario = mod.Scenario(
        n_robots=n_robots,
        part=part,
        shift_hours=shift_hours,
        handover_h=handover_h,
        maintenance_every_days=maintenance_every_days,
        maintenance_h=maintenance_h,
        enable_predebug=enable_predebug,
    )
    if replications > 1:
        raw = mod.run_replications(config=config, scenario=scenario, replications=replications, seed0=seed)
    else:
        raw = mod.SimulationController(config=config, scenario=scenario, seed=seed).run()
    return normalize_integrated_summary(raw, n_robots=n_robots)


def run_file_a_model(
    *,
    part: int,
    n_robots: int,
    shift_hours: float,
    replications: int = 1,
    seed: int = 20260329,
    maintenance_every_days: int = 7,
    maintenance_h: float = 4.0,
    enable_predebug: bool = False,
    failure_multiplier: float = 1.0,
    processing_time_multiplier: float = 1.0,
    defect_multiplier: float = 1.0,
    operator_error_multiplier: float = 1.0,
    benches: Optional[int] = None,
) -> Dict[str, float]:
    mod = load_file_a_module()
    params = build_file_a_params(
        failure_multiplier=failure_multiplier,
        processing_time_multiplier=processing_time_multiplier,
        defect_multiplier=defect_multiplier,
        operator_error_multiplier=operator_error_multiplier,
        benches=benches,
    )
    if part == 2:
        scenario = mod.Scenario(
            part=2,
            n_robots=n_robots,
            shift_hours=shift_hours,
            enable_predebug=enable_predebug,
        )
    elif part == 3:
        scenario = mod.Scenario(
            part=3,
            n_robots=n_robots,
            shift_hours=0.0,
            k_hours=shift_hours,
            gap_between_shifts=0.5,
            maintenance_every_days=maintenance_every_days,
            maintenance_duration=maintenance_h,
            enable_predebug=enable_predebug,
        )
    else:
        raise ValueError("part must be 2 or 3")

    if replications > 1:
        raw = mod.run_replications(params=params, scenario=scenario, replications=replications, seed0=seed)
    else:
        raw = mod.FactorySimulation(params=params, scenario=scenario, seed=seed).run()
    return normalize_file_a_summary(raw, n_robots=n_robots)


class IntegratedSimulationModel:
    def __init__(self, analytical_params: Optional[AnalyticalParams] = None) -> None:
        self.analytical_params = analytical_params or AnalyticalParams()

    def solve_problem1(self, *, monte_carlo_runs: int = 10_000, seed: int = 20260329) -> Dict[str, Dict[str, float]]:
        return {
            "closed_form": part1_closed_form(self.analytical_params),
            "monte_carlo": part1_monte_carlo(self.analytical_params, n=monte_carlo_runs, seed=seed),
        }

    def solve_problem2(self, **kwargs: Any) -> Dict[str, float]:
        return run_integrated_model(part=2, **kwargs)

    def solve_problem3(self, **kwargs: Any) -> Dict[str, float]:
        return run_integrated_model(part=3, **kwargs)

    def optimize_problem3(
        self,
        *,
        n_robots: int = 240,
        replications: int = 20,
        k_min: float = 9.0,
        k_max: float = 12.0,
        k_step: float = 0.5,
        seed: int = 20260329,
    ) -> Dict[str, Any]:
        mod = load_file_c_module()
        config = build_integrated_config()
        best_k, best_result, all_results = mod.optimize_shift_length(
            config=config,
            n_robots=n_robots,
            replications=replications,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            seed0=seed,
        )
        normalized = {
            str(k): normalize_integrated_summary(value, n_robots=n_robots)
            for k, value in all_results.items()
        }
        return {
            "best_k": best_k,
            "best_result": normalize_integrated_summary(best_result, n_robots=n_robots),
            "all_results": normalized,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated simulation model for Problems 1-3.")
    parser.add_argument("--problem", choices=("1", "2", "3"), default="2")
    parser.add_argument("--robots", type=int, default=120)
    parser.add_argument("--shift-hours", type=float, default=12.0)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--optimize-k", action="store_true")
    parser.add_argument("--k-min", type=float, default=9.0)
    parser.add_argument("--k-max", type=float, default=12.0)
    parser.add_argument("--k-step", type=float, default=0.5)
    parser.add_argument("--enable-predebug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = IntegratedSimulationModel()

    if args.problem == "1":
        payload = model.solve_problem1(seed=args.seed)
    elif args.problem == "2":
        payload = model.solve_problem2(
            n_robots=args.robots,
            shift_hours=args.shift_hours,
            replications=args.replications,
            seed=args.seed,
            enable_predebug=args.enable_predebug,
        )
    else:
        if args.optimize_k:
            payload = model.optimize_problem3(
                n_robots=args.robots,
                replications=args.replications,
                k_min=args.k_min,
                k_max=args.k_max,
                k_step=args.k_step,
                seed=args.seed,
            )
        else:
            payload = model.solve_problem3(
                n_robots=args.robots,
                shift_hours=args.shift_hours,
                replications=args.replications,
                seed=args.seed,
                enable_predebug=True if not args.enable_predebug else args.enable_predebug,
            )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
