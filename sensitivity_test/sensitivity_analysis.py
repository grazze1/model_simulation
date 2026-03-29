#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont
try:
    with contextlib.redirect_stderr(io.StringIO()):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from integrated_simulation_model import run_integrated_model


RESULT_DIR = Path(__file__).resolve().parent
BASE_ROBOTS = 120
BASE_SHIFT_HOURS = 12.0
BASE_BENCHES = 2
COLOR = (31, 119, 180)


def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def average_utilization(result: Dict[str, float]) -> float:
    values = [
        float(result.get("stage_A_station_utilization", 0.0)),
        float(result.get("stage_B_station_utilization", 0.0)),
        float(result.get("stage_C_station_utilization", 0.0)),
        float(result.get("stage_E_station_utilization", 0.0)),
    ]
    return fmean(values)


def run_sweep(
    *,
    parameter_name: str,
    values: Iterable[float],
    replications: int,
    seed0: int,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []

    for index, value in enumerate(values):
        n_robots = BASE_ROBOTS
        shift_hours = BASE_SHIFT_HOURS
        benches = BASE_BENCHES
        failure_multiplier = 1.0
        processing_time_multiplier = 1.0
        queue_capacity_multiplier = 1.0
        stage_capacity_multiplier = 1.0

        if parameter_name == "failure_multiplier":
            failure_multiplier = value
        elif parameter_name == "service_time_multiplier":
            processing_time_multiplier = value
        elif parameter_name == "arrival_rate":
            n_robots = int(round(value))
        elif parameter_name == "resource_capacity":
            stage_capacity_multiplier = value
            queue_capacity_multiplier = value
            benches = max(1, int(round(BASE_BENCHES * value)))
        else:
            raise ValueError(parameter_name)

        result = run_integrated_model(
            part=2,
            n_robots=n_robots,
            shift_hours=shift_hours,
            replications=replications,
            seed=seed0 + index * 1000,
            failure_multiplier=failure_multiplier,
            processing_time_multiplier=processing_time_multiplier,
            queue_capacity_multiplier=queue_capacity_multiplier,
            stage_capacity_multiplier=stage_capacity_multiplier,
            benches=benches,
        )

        rows.append(
            {
                "parameter_name": parameter_name,
                "parameter_value": float(value),
                "replications": float(replications),
                "n_robots": float(n_robots),
                "benches": float(benches),
                "throughput_per_day": float(result["throughput_per_day"]),
                "completion_time_days": float(result["makespan_days"]),
                "system_utilization": average_utilization(result),
            }
        )

    return rows


def _draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = box
    tw, th = draw.textsize(text, font=font)
    draw.text((left + (right - left - tw) / 2, top + (bottom - top - th) / 2), text, fill="black", font=font)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    points: List[tuple[float, float]],
) -> None:
    left, top, right, bottom = box
    font = ImageFont.load_default()
    plot_left = left + 52
    plot_top = top + 24
    plot_right = right - 18
    plot_bottom = bottom - 42
    draw.rectangle((left, top, right, bottom), outline=(180, 180, 180), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=1)
    _draw_centered(draw, (left, top, right, top + 18), title, font)
    _draw_centered(draw, (left, bottom - 18, right, bottom), x_label, font)
    draw.text((left + 4, plot_top - 2), y_label, fill="black", font=font)

    if not points:
        return
    x_min = min(point[0] for point in points)
    x_max = max(point[0] for point in points)
    y_min = min(point[1] for point in points)
    y_max = max(point[1] for point in points)
    if abs(x_max - x_min) < 1e-12:
        x_max += 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max += 1.0
        y_min -= 1.0

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    scaled = [(sx(x), sy(y)) for x, y in points]
    if len(scaled) > 1:
        draw.line(scaled, fill=COLOR, width=2)
    for px, py in scaled:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=COLOR, outline=COLOR)


def save_plot(rows: List[Dict[str, float | str]], parameter_name: str) -> None:
    subset = [row for row in rows if row["parameter_name"] == parameter_name]
    subset.sort(key=lambda item: float(item["parameter_value"]))

    metrics = [
        ("throughput_per_day", "Throughput"),
        ("completion_time_days", "Completion time (days)"),
        ("system_utilization", "System utilization"),
    ]
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        x_values = [float(item["parameter_value"]) for item in subset]
        for ax, (metric_key, title) in zip(axes, metrics):
            y_values = [float(item[metric_key]) for item in subset]
            ax.plot(x_values, y_values, marker="o", linewidth=2, color="#1f77b4")
            ax.set_title(title)
            ax.set_xlabel(parameter_name.replace("_", " "))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Metric value")
        fig.suptitle(f"Sensitivity Analysis: {parameter_name}")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / f"sensitivity_{parameter_name}.png", dpi=180)
        plt.close(fig)
        return

    image = Image.new("RGB", (1500, 430), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_centered(draw, (0, 6, 1500, 26), f"Sensitivity Analysis: {parameter_name}", font)

    x_values = [float(item["parameter_value"]) for item in subset]
    for index, (metric_key, title) in enumerate(metrics):
        y_values = [float(item[metric_key]) for item in subset]
        left = 20 + index * 490
        _draw_panel(
            draw,
            (left, 36, left + 460, 410),
            title,
            parameter_name.replace("_", " "),
            "value",
            list(zip(x_values, y_values)),
        )

    image.save(RESULT_DIR / f"sensitivity_{parameter_name}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for the integrated simulation model.")
    parser.add_argument("--replications", type=int, default=8, help="Average each design point over N replications.")
    parser.add_argument("--seed", type=int, default=20260329, help="Base random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sweeps = {
        "failure_multiplier": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "service_time_multiplier": [0.75, 0.9, 1.0, 1.1, 1.25, 1.5],
        "arrival_rate": [60, 90, 120, 150, 180, 240],
        "resource_capacity": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    }

    rows: List[Dict[str, float | str]] = []
    for offset, (name, values) in enumerate(sweeps.items()):
        rows.extend(run_sweep(parameter_name=name, values=values, replications=args.replications, seed0=args.seed + offset * 10_000))

    write_csv(RESULT_DIR / "sensitivity_results.csv", rows)
    for name in sweeps:
        save_plot(rows, name)

    summary = {
        "result_count": len(rows),
        "output_csv": str(RESULT_DIR / "sensitivity_results.csv"),
        "plots": [str(RESULT_DIR / f"sensitivity_{name}.png") for name in sweeps],
    }
    (RESULT_DIR / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
