#!/usr/bin/env python3
"""
Plot KDE-style trends from OVERALL.json against model metadata (dates or sizes).

Examples
--------
From the project root:

  # MHS vs release date
  python db/plot_overall_kde.py --x date --y MHS \
      --output plots/mhs_vs_date.png

  # Creativity vs model size (log10 of active params)
  python db/plot_overall_kde.py --x size --y Creativity \
      --output plots/creativity_vs_size.png

Notes
-----
- X-axis:
    * "date": days since the earliest model date in db/model_dates.json
    * "size": log10 of active parameters from db/model_size.json
- Y-axis:
    * "MHS" (overall score from OVERALL.json)
    * or any personality dimension name present in OVERALL.json, e.g.
      "Creativity", "Resilience", "Work-related Stress", ...
- Only a KDE-style smoothed curve is plotted (no raw points).
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OVERALL = BASE_DIR.parent / "OVERALL.json"
DEFAULT_MODEL_DATES = BASE_DIR / "model_dates.json"
DEFAULT_MODEL_SIZES = BASE_DIR / "model_size.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_json(path: Path):
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: could not parse JSON from {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def get_available_metrics(overall_data: Sequence[dict]) -> List[str]:
    if not overall_data:
        print("Error: OVERALL.json appears to be empty.", file=sys.stderr)
        raise SystemExit(1)
    first = overall_data[0]
    score_block = first.get("scores", {})
    trait_keys = list(score_block.keys())
    return ["MHS"] + trait_keys


def resolve_metric_name(requested: str, available: Sequence[str]) -> str:
    """
    Resolve a metric name in a case-insensitive and partially matching way.

    - Exact case-insensitive match wins.
    - Otherwise, substring match in available names.
    """
    req = requested.strip().lower()
    mapping = {name.lower(): name for name in available}

    if req in mapping:
        return mapping[req]

    # Try substring match
    matches = [name for name in available if req in name.lower()]
    if len(matches) == 1:
        return matches[0]

    msg = [f"Error: metric '{requested}' not found."]
    msg.append("Available metrics:")
    for name in available:
        msg.append(f"  - {name}")
    print("\n".join(msg), file=sys.stderr)
    raise SystemExit(1)


def extract_metric(entry: dict, metric_name: str) -> float:
    if metric_name == "MHS":
        return float(entry["MHS"])
    scores = entry.get("scores", {})
    stats = scores.get(metric_name)
    if not isinstance(stats, dict) or "mean" not in stats:
        raise KeyError(f"Metric '{metric_name}' not found in entry for LLM={entry.get('LLM')!r}")
    return float(stats["mean"])


def estimate_bandwidth(x: np.ndarray) -> float:
    """
    Simple rule-of-thumb bandwidth (Silverman's rule of thumb on x).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return 1.0

    std = float(np.std(x, ddof=1))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_range = x_max - x_min if x_max > x_min else 1.0

    if std <= 0:
        return max(x_range * 0.1, 1e-3)

    h = 1.06 * std * (n ** (-1.0 / 5.0))
    if h <= 0 or not math.isfinite(h):
        h = max(x_range * 0.1, 1e-3)
    return h


def kernel_smoother(x: np.ndarray,
                    y: np.ndarray,
                    grid: np.ndarray,
                    bandwidth: float) -> np.ndarray:
    """
    Nadaraya-Watson kernel regression with a Gaussian kernel.

    This isn't a pure KDE of (x, y) but a smooth estimate of E[y | x],
    which is typically what you want to see "trend lines" for.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive for kernel smoothing.")

    diffs = (grid[:, None] - x[None, :]) / bandwidth
    weights = np.exp(-0.5 * diffs * diffs)  # Gaussian, no normalization (cancels out)
    weight_sums = weights.sum(axis=1)

    # Prevent division by 0 (should not really happen with Gaussian, but be safe)
    weight_sums[weight_sums == 0] = np.nan

    y_hat = (weights @ y) / weight_sums
    return y_hat


def build_xy_date(overall_data: Sequence[dict],
                  metric_name: str,
                  model_dates_path: Path) -> Tuple[np.ndarray, np.ndarray, datetime]:
    dates_json = load_json(model_dates_path)
    if not isinstance(dates_json, dict):
        print(f"Error: {model_dates_path} must contain an object mapping model -> date string.",
              file=sys.stderr)
        raise SystemExit(1)

    points = []
    missing = []
    invalid = []

    for entry in overall_data:
        model = entry["LLM"]
        value = extract_metric(entry, metric_name)
        date_str = dates_json.get(model, "")
        if not isinstance(date_str, str) or not date_str.strip():
            missing.append(model)
            continue
        try:
            dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
        except ValueError:
            invalid.append((model, date_str))
            continue
        points.append((model, dt, value))

    if missing:
        print(f"Warning: skipped {len(missing)} models without a date in {model_dates_path.name}.",
              file=sys.stderr)
    if invalid:
        print(f"Warning: skipped {len(invalid)} models with invalid dates:", file=sys.stderr)
        for model, dstr in invalid:
            print(f"  - {model}: {dstr!r}", file=sys.stderr)

    if not points:
        print("Error: no usable (date, metric) points found. "
              "Check db/model_dates.json.", file=sys.stderr)
        raise SystemExit(1)

    min_date = min(dt for _, dt, _ in points)

    x_vals = np.array([(dt - min_date).days for _, dt, _ in points], dtype=float)
    y_vals = np.array([float(v) for _, _, v in points], dtype=float)

    return x_vals, y_vals, min_date


def build_xy_size(overall_data: Sequence[dict],
                  metric_name: str,
                  model_sizes_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    sizes_json = load_json(model_sizes_path)
    if not isinstance(sizes_json, dict):
        print(f"Error: {model_sizes_path} must contain an object mapping model -> [active_params, ...].",
              file=sys.stderr)
        raise SystemExit(1)

    points = []
    missing = []
    invalid = []

    for entry in overall_data:
        model = entry["LLM"]
        value = extract_metric(entry, metric_name)
        size_list = sizes_json.get(model)

        if not isinstance(size_list, list) or not size_list:
            missing.append(model)
            continue

        active = size_list[0]
        try:
            active_f = float(active)
        except (TypeError, ValueError):
            invalid.append((model, active))
            continue

        if active_f <= 0:
            invalid.append((model, active))
            continue

        log_size = math.log10(active_f)
        points.append((model, log_size, value))

    if missing:
        print(f"Warning: skipped {len(missing)} models without a size in {model_sizes_path.name}.",
              file=sys.stderr)
    if invalid:
        print(f"Warning: skipped {len(invalid)} models with invalid or non-positive sizes:",
              file=sys.stderr)
        for model, s in invalid:
            print(f"  - {model}: {s!r}", file=sys.stderr)

    if not points:
        print("Error: no usable (size, metric) points found. "
              "Check db/model_size.json.", file=sys.stderr)
        raise SystemExit(1)

    x_vals = np.array([log_size for _, log_size, _ in points], dtype=float)
    y_vals = np.array([float(v) for _, _, v in points], dtype=float)

    return x_vals, y_vals


# --------------------------------------------------------------------------- #
# CLI + main logic
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot KDE-style trends from OVERALL.json vs model dates or sizes."
    )

    parser.add_argument(
        "--x", "--x-axis",
        dest="x_axis",
        choices=["date", "size"],
        default="date",
        help="X-axis perspective: 'date' (days since earliest model) or "
             "'size' (log10 of active params). Default: date.",
    )
    parser.add_argument(
        "--y", "--metric",
        dest="metric",
        default="MHS",
        help="Metric on the Y-axis: 'MHS' or one of the personality traits "
             "present in OVERALL.json (e.g. 'Creativity').",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        required=True,
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--overall-path",
        dest="overall_path",
        type=Path,
        default=DEFAULT_OVERALL,
        help=f"Path to OVERALL.json (default: {DEFAULT_OVERALL}).",
    )
    parser.add_argument(
        "--model-dates",
        dest="model_dates_path",
        type=Path,
        default=DEFAULT_MODEL_DATES,
        help=f"Path to model_dates.json (default: {DEFAULT_MODEL_DATES}). "
             "Used when --x date.",
    )
    parser.add_argument(
        "--model-sizes",
        dest="model_sizes_path",
        type=Path,
        default=DEFAULT_MODEL_SIZES,
        help=f"Path to model_size.json (default: {DEFAULT_MODEL_SIZES}). "
             "Used when --x size.",
    )
    parser.add_argument(
        "-b", "--bandwidth",
        dest="bandwidth",
        type=float,
        default=None,
        help="Optional bandwidth for the kernel smoother. If omitted, a "
             "rule-of-thumb value is estimated from the X data.",
    )
    parser.add_argument(
        "--grid-points",
        dest="grid_points",
        type=int,
        default=200,
        help="Number of X positions used to evaluate the smoothed curve. "
             "Default: 200.",
    )
    parser.add_argument(
        "--dpi",
        dest="dpi",
        type=int,
        default=150,
        help="DPI for the output PNG. Default: 150.",
    )
    parser.add_argument(
        "--figsize",
        dest="figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(8.0, 5.0),
        help="Figure size in inches, e.g. --figsize 8 5. Default: 8 5.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    overall_data = load_json(args.overall_path)
    if not isinstance(overall_data, list):
        print(f"Error: OVERALL.json should contain a list of objects, got {type(overall_data).__name__}.",
              file=sys.stderr)
        raise SystemExit(1)

    available_metrics = get_available_metrics(overall_data)
    metric_name = resolve_metric_name(args.metric, available_metrics)

    # Build X, Y from the chosen perspective
    if args.x_axis == "date":
        x, y, min_date = build_xy_date(overall_data, metric_name, args.model_dates_path)
        x_label = f"Model release (days since {min_date.isoformat()})"
        x_descriptor = "release date"
    else:
        x, y = build_xy_size(overall_data, metric_name, args.model_sizes_path)
        x_label = "Active parameters (log10 of #params)"
        x_descriptor = "model size (log10 params)"

    if x.size == 0:
        print("Error: no data points available after filtering.", file=sys.stderr)
        raise SystemExit(1)

    x_min = float(x.min())
    x_max = float(x.max())

    if x_max == x_min:
        # Degenerate case: all Xs are identical; plot a flat line
        grid_x = np.linspace(x_min - 0.5, x_max + 0.5, args.grid_points)
        y_hat = np.full_like(grid_x, float(y.mean()))
        used_bandwidth = float("nan")
    else:
        pad = 0.05 * (x_max - x_min)
        left = x_min - pad
        right = x_max + pad
        grid_x = np.linspace(left, right, args.grid_points)

        if args.bandwidth is not None:
            used_bandwidth = args.bandwidth
        else:
            used_bandwidth = estimate_bandwidth(x)

        y_hat = kernel_smoother(x, y, grid_x, used_bandwidth)

    # Plot (only the smoothed KDE-style curve, no raw points)
    width, height = args.figsize
    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(grid_x, y_hat)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{metric_name} (smoothed)")
    title = f"{metric_name} vs {x_descriptor} (KDE-style trend)"
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    # Log what we did
    info = [
        f"Wrote {output_path}",
        f"  points used: {x.size}",
        f"  x-axis: {args.x_axis}",
        f"  metric: {metric_name}",
    ]
    if math.isfinite(used_bandwidth):
        info.append(f"  bandwidth: {used_bandwidth:.4g}")
    print("\n".join(info), file=sys.stderr)


if __name__ == "__main__":
    main()
