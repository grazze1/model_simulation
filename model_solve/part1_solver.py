#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


STAGES_ABC = ("A", "B", "C")
STAGES_ALL = ("A", "B", "C", "E")


@dataclass
class Part1Params:
    p: Dict[str, float] = field(
        default_factory=lambda: {"A": 0.03, "B": 0.025, "C": 0.02, "D": 0.002}
    )
    q: Dict[str, float] = field(
        default_factory=lambda: {"A": 0.025, "B": 0.03, "C": 0.015, "E": 0.02}
    )
    alpha: float = 0.6
    beta: float = 0.4
    delta: Dict[str, float] = field(
        default_factory=lambda: {"A": 20 / 60, "B": 15 / 60, "C": 15 / 60, "E": 30 / 60}
    )
    theta: Dict[str, float] = field(
        default_factory=lambda: {"A": 2.0, "B": 1.5, "C": 2.0, "E": 2.5}
    )
    load_time: float = 0.5
    unload_time: float = 0.5


def part1_calculation(params: Part1Params) -> Dict[str, float]:
    out: Dict[str, float] = {}

    s: Dict[str, float] = {}
    f: Dict[str, float] = {}
    pass_probability: Dict[str, float] = {}
    leakage_probability: Dict[str, float] = {}

    for stage in STAGES_ABC:
        defect_rate = params.p[stage]
        error_rate = params.q[stage]
        success = (1 - defect_rate) * (1 - params.alpha * error_rate) + defect_rate * params.beta * error_rate
        failure = 1 - success
        pass_once_or_twice = 1 - failure**2
        leakage = defect_rate * params.beta * error_rate

        s[stage] = success
        f[stage] = failure
        pass_probability[stage] = pass_once_or_twice
        leakage_probability[stage] = leakage

        out[f"s_{stage}"] = success
        out[f"f_{stage}"] = failure
        out[f"P_{stage}"] = pass_once_or_twice
        out[f"P_L_{stage}"] = leakage

    p_g = 1 - (
        (1 - leakage_probability["A"])
        * (1 - leakage_probability["B"])
        * (1 - leakage_probability["C"])
        * (1 - params.p["D"])
    )
    out["P_G"] = p_g

    lambdas = {
        "A": leakage_probability["A"] / p_g,
        "B": leakage_probability["B"] / p_g,
        "C": leakage_probability["C"] / p_g,
        "D": params.p["D"] / p_g,
    }
    for key, value in lambdas.items():
        out[f"lambda_{key}"] = value

    s_e = (1 - p_g) * (1 - params.alpha * params.q["E"]) + p_g * params.beta * params.q["E"]
    f_e = 1 - s_e
    p_e = 1 - (1 - s_e) ** 2
    out["s_E"] = s_e
    out["f_E"] = f_e
    out["P_E"] = p_e

    p_out = pass_probability["A"] * pass_probability["B"] * pass_probability["C"] * p_e
    out["P_out"] = p_out

    expected_stage_time: Dict[str, float] = {}
    for stage in STAGES_ALL:
        failure = f[stage] if stage in STAGES_ABC else f_e
        expected_time = (1 + failure) * (params.delta[stage] + params.theta[stage])
        expected_stage_time[stage] = expected_time
        out[f"E_T_{stage}"] = expected_time

    out["E_T_all"] = (
        params.load_time
        + expected_stage_time["A"]
        + pass_probability["A"] * expected_stage_time["B"]
        + pass_probability["A"] * pass_probability["B"] * expected_stage_time["C"]
        + pass_probability["A"] * pass_probability["B"] * pass_probability["C"] * expected_stage_time["E"]
        + params.unload_time
    )
    return out


def print_part1_results(results: Dict[str, float]) -> None:
    print("=" * 72)
    print("Part 1: Single Robot Closed-form Results")
    print("=" * 72)
    keys_order = [
        "s_A", "f_A", "P_A", "P_L_A",
        "s_B", "f_B", "P_B", "P_L_B",
        "s_C", "f_C", "P_C", "P_L_C",
        "P_G",
        "lambda_A", "lambda_B", "lambda_C", "lambda_D",
        "s_E", "f_E", "P_E",
        "P_out",
        "E_T_A", "E_T_B", "E_T_C", "E_T_E",
        "E_T_all",
    ]
    for key in keys_order:
        print(f"{key:>12s} = {results[key]:.10f}")


def run_part1_simulation() -> Dict[str, float]:
    params = Part1Params()
    results = part1_calculation(params)
    print_part1_results(results)
    return results


if __name__ == "__main__":
    run_part1_simulation()
