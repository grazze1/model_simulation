#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
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

from integrated_simulation_model import run_file_a_model, run_integrated_model


RESULT_DIR = Path(__file__).resolve().parent
BASE_ROBOTS = 120
BASE_SHIFT_HOURS = 12.0
BASE_BENCHES = 2
COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14)]


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


def _draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    tw, th = draw.textsize(text, font=font)
    draw.text((left + (width - tw) / 2, top + (height - th) / 2), text, fill="black", font=font)


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    series_map: Dict[str, List[tuple[float, float]]],
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

    all_points = [point for points in series_map.values() for point in points]
    if not all_points:
        return
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if abs(x_max - x_min) < 1e-12:
        x_max += 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max += 1.0
        y_min -= 1.0

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for tick in range(5):
        x = plot_left + tick * (plot_right - plot_left) / 4
        y = plot_top + tick * (plot_bottom - plot_top) / 4
        draw.line((x, plot_bottom - 3, x, plot_bottom + 3), fill="black", width=1)
        draw.line((plot_left - 3, y, plot_left + 3, y), fill="black", width=1)

    for index, (name, points) in enumerate(series_map.items()):
        color = COLORS[index % len(COLORS)]
        pixel_points = [(sx(x), sy(y)) for x, y in points]
        if len(pixel_points) > 1:
            draw.line(pixel_points, fill=color, width=2)
        for px, py in pixel_points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)
        legend_y = top + 4 + index * 12
        draw.line((right - 84, legend_y + 5, right - 68, legend_y + 5), fill=color, width=2)
        draw.text((right - 64, legend_y), name, fill="black", font=font)


def save_plot(rows: List[Dict[str, float | str]], experiment: str) -> None:
    experiment_rows = [row for row in rows if row["experiment"] == experiment]
    metrics = [
        ("makespan_days", "Makespan (days)"),
        ("throughput_per_day", "Throughput (qualified/day)"),
        ("omission_probability", "Omission probability"),
    ]
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (metric_key, title) in zip(axes, metrics):
            for model_name in ("file_a", "integrated"):
                series = [row for row in experiment_rows if row["model"] == model_name]
                series.sort(key=lambda item: float(item["parameter_value"]))
                ax.plot(
                    [float(item["parameter_value"]) for item in series],
                    [float(item[metric_key]) for item in series],
                    marker="o",
                    linewidth=2,
                    label=model_name,
                )
            ax.set_title(title)
            ax.set_xlabel(experiment.replace("_", " "))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Metric value")
        axes[-1].legend()
        fig.suptitle(f"Robustness Sweep: {experiment}")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / f"robustness_{experiment}.png", dpi=180)
        plt.close(fig)
        return
    image = Image.new("RGB", (1500, 430), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_centered(draw, (0, 6, 1500, 26), f"Robustness Sweep: {experiment}", font)

    for index, (metric_key, title) in enumerate(metrics):
        series_map: Dict[str, List[tuple[float, float]]] = {}
        for model_name in ("file_a", "integrated"):
            series = [row for row in experiment_rows if row["model"] == model_name]
            series.sort(key=lambda item: float(item["parameter_value"]))
            series_map[model_name] = [
                (float(item["parameter_value"]), float(item[metric_key]))
                for item in series
            ]
        left = 20 + index * 490
        _draw_line_panel(
            draw,
            (left, 36, left + 460, 410),
            title,
            experiment.replace("_", " "),
            "value",
            series_map,
        )

    image.save(RESULT_DIR / f"robustness_{experiment}.png")


def run_experiment(
    *,
    experiment: str,
    values: Iterable[float],
    replications: int,
    seed0: int,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []

    for index, raw_value in enumerate(values):
        seed = seed0 + index * 1000
        n_robots = BASE_ROBOTS
        shift_hours = BASE_SHIFT_HOURS
        benches = BASE_BENCHES
        failure_multiplier = 1.0
        processing_time_multiplier = 1.0

        parameter_value = float(raw_value)
        if experiment == "robot_quantity":
            n_robots = int(round(parameter_value))
        elif experiment == "failure_multiplier":
            failure_multiplier = parameter_value
        elif experiment == "queue_load":
            benches = max(1, int(round(parameter_value)))
            parameter_value = BASE_ROBOTS / benches
        elif experiment == "processing_time_multiplier":
            processing_time_multiplier = parameter_value
        elif experiment == "shift_duration":
            shift_hours = parameter_value
        else:
            raise ValueError(experiment)

        common_kwargs = {
            "part": 2,
            "n_robots": n_robots,
            "shift_hours": shift_hours,
            "replications": replications,
            "seed": seed,
            "failure_multiplier": failure_multiplier,
            "processing_time_multiplier": processing_time_multiplier,
            "benches": benches,
        }

        outputs = [
            run_file_a_model(**common_kwargs),
            run_integrated_model(**common_kwargs),
        ]

        for output in outputs:
            row: Dict[str, float | str] = {
                "experiment": experiment,
                "parameter_value": parameter_value,
                "model": str(output["model"]),
                "seed": float(seed),
                "replications": float(replications),
                "n_robots": float(n_robots),
                "shift_hours": float(shift_hours),
                "benches": float(benches),
                "failure_multiplier": float(failure_multiplier),
                "processing_time_multiplier": float(processing_time_multiplier),
            }
            for key in (
                "makespan_days",
                "makespan_hours",
                "qualified_count",
                "omission_probability",
                "misjudgment_probability",
                "throughput_per_day",
                "bench_effective_work_ratio",
                "bench_occupancy_ratio",
                "stage_A_station_utilization",
                "stage_B_station_utilization",
                "stage_C_station_utilization",
                "stage_E_station_utilization",
            ):
                row[key] = float(output.get(key, 0.0))
            rows.append(row)

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robustness testing for File A and the integrated model.")
    parser.add_argument("--replications", type=int, default=6, help="Average each design point over N replications.")
    parser.add_argument("--seed", type=int, default=20260329, help="Base random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiments = {
        "robot_quantity": [60, 120, 180, 240],
        "failure_multiplier": [0.5, 1.0, 1.5, 2.0],
        "queue_load": [4, 3, 2, 1],
        "processing_time_multiplier": [0.75, 1.0, 1.25, 1.5],
        "shift_duration": [8.0, 10.0, 12.0, 14.0],
    }

    all_rows: List[Dict[str, float | str]] = []
    for offset, (name, values) in enumerate(experiments.items()):
        all_rows.extend(run_experiment(experiment=name, values=values, replications=args.replications, seed0=args.seed + offset * 10_000))

    write_csv(RESULT_DIR / "robustness_results.csv", all_rows)
    for name in experiments:
        save_plot(all_rows, name)

    summary = {
        "result_count": len(all_rows),
        "experiments": list(experiments.keys()),
        "output_csv": str(RESULT_DIR / "robustness_results.csv"),
        "plots": [str(RESULT_DIR / f"robustness_{name}.png") for name in experiments],
    }
    (RESULT_DIR / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
