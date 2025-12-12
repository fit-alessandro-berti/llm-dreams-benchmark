"""Train random forest regressors to predict model sizes from psychological scores.

The script fits two regressors:
- total parameter count (billions, as stored in db/model_size.json)
- active parameter count when available

By default targets are log1p-transformed before fitting and transformed back for
reporting. After training, a Markdown table of all models in OVERALL.json is
written to PREDICTIONS.md with predicted total/active parameter counts alongside
known values when available.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "scikit-learn is required for this script. Install with `pip install scikit-learn`."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OVERALL_PATH = BASE_DIR.parent / "OVERALL.json"
DEFAULT_MODEL_SIZE_PATH = BASE_DIR / "model_size.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trait_names(overall: Sequence[dict]) -> List[str]:
    if not overall:
        raise ValueError("OVERALL.json is empty; no samples to train on.")
    # Use the ordering from the first entry to keep feature vectors stable.
    return list(overall[0]["scores"].keys())


def vector_from_scores(scores: Dict[str, dict], names: Iterable[str]) -> List[float]:
    vector: List[float] = []
    for name in names:
        trait = scores.get(name)
        if trait is None:
            raise KeyError(f"Trait '{name}' missing from scores; cannot build feature vector.")
        vector.extend([float(trait["mean"]), float(trait["std"])])
    return vector


def transform_target(value: float, use_log: bool) -> float:
    if not use_log:
        return value
    if value <= -1:
        raise ValueError(f"Cannot log-transform non-positive value {value}.")
    return math.log1p(value)


def inverse_transform(value: float, use_log: bool) -> float:
    return math.expm1(value) if use_log else value


def prepare_training_sets(
    overall: Sequence[dict],
    size_map: Dict[str, List],
    names: Sequence[str],
    use_log: bool,
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    total_X: List[List[float]] = []
    total_y: List[float] = []
    active_X: List[List[float]] = []
    active_y: List[float] = []

    for entry in overall:
        model = entry["LLM"]
        size_entry = size_map.get(model)
        if not isinstance(size_entry, list) or len(size_entry) == 0:
            continue

        features = vector_from_scores(entry["scores"], names)
        total_val = size_entry[0]
        if isinstance(total_val, (int, float)):
            total_X.append(features)
            total_y.append(transform_target(float(total_val), use_log))

        if len(size_entry) > 1 and isinstance(size_entry[1], (int, float)):
            active_X.append(features)
            active_y.append(transform_target(float(size_entry[1]), use_log))

    return total_X, total_y, active_X, active_y


def train_regressor(
    X: List[List[float]],
    y: List[float],
    n_estimators: int,
    max_depth: Optional[int],
    random_state: int,
) -> RandomForestRegressor:
    if len(X) < 2:
        raise ValueError("Need at least two samples to train a random forest regressor.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def format_size(value: float) -> str:
    return f"{value:.3f} B params"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train regressors to predict model sizes from psychological score vectors."
    )
    parser.add_argument(
        "--overall",
        type=Path,
        default=DEFAULT_OVERALL_PATH,
        help=f"Path to OVERALL.json (default: {DEFAULT_OVERALL_PATH})",
    )
    parser.add_argument(
        "--model-size",
        type=Path,
        default=DEFAULT_MODEL_SIZE_PATH,
        help=f"Path to model_size.json (default: {DEFAULT_MODEL_SIZE_PATH})",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in each random forest (default: 200).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional tree depth cap; defaults to None (unbounded).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log1p transform of parameter counts before fitting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="PREDICTIONS.md",
        help="Path to write Markdown table of predictions (default: PREDICTIONS.md in repo root).",
    )
    return parser.parse_args()


def predict_all_models(
    overall: Sequence[dict],
    size_map: Dict[str, List],
    names: Sequence[str],
    total_reg: RandomForestRegressor,
    active_reg: Optional[RandomForestRegressor],
    use_log: bool,
) -> List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]]:
    predictions: List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]] = []
    for entry in overall:
        features = vector_from_scores(entry["scores"], names)
        predicted_total = inverse_transform(float(total_reg.predict([features])[0]), use_log)

        predicted_active: Optional[float] = None
        if active_reg is not None:
            predicted_active = inverse_transform(float(active_reg.predict([features])[0]), use_log)

        size_entry = size_map.get(entry["LLM"], [])
        actual_total = (
            float(size_entry[0]) if len(size_entry) > 0 and isinstance(size_entry[0], (int, float)) else None
        )
        actual_active = (
            float(size_entry[1]) if len(size_entry) > 1 and isinstance(size_entry[1], (int, float)) else None
        )

        predictions.append((entry["LLM"], predicted_total, predicted_active, actual_total, actual_active))
    return predictions


def build_markdown_table(
    predictions: Sequence[Tuple[str, float, Optional[float], Optional[float], Optional[float]]]
) -> str:
    sorted_rows = sorted(predictions, key=lambda row: row[1], reverse=True)
    lines = [
        "| Model | Predicted Total Params | Known Total Params | Predicted Active Params | Known Active Params |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model, total, active, actual_total, actual_active in sorted_rows:
        active_str = format_size(active) if active is not None else "n/a"
        total_actual_str = format_size(actual_total) if actual_total is not None else "n/a"
        active_actual_str = format_size(actual_active) if actual_active is not None else "n/a"
        lines.append(
            f"| {model} | {format_size(total)} | {total_actual_str} | {active_str} | {active_actual_str} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    use_log = not args.no_log

    if not args.overall.exists():
        raise FileNotFoundError(f"Could not find OVERALL data at {args.overall}")
    if not args.model_size.exists():
        raise FileNotFoundError(f"Could not find size data at {args.model_size}")

    overall = load_json(args.overall)
    size_map = load_json(args.model_size)
    names = trait_names(overall)

    total_X, total_y, active_X, active_y = prepare_training_sets(
        overall, size_map, names, use_log
    )

    print(
        f"Fitting regressors (log1p targets: {use_log}) "
        f"with {len(names)*2} features per sample."
    )
    print(f"- Total size samples: {len(total_X)}")
    print(f"- Active size samples: {len(active_X)}")

    total_reg = train_regressor(
        total_X, total_y, args.n_estimators, args.max_depth, args.random_state
    )
    active_reg = None
    if len(active_X) >= 2:
        active_reg = train_regressor(
            active_X, active_y, args.n_estimators, args.max_depth, args.random_state
        )
    else:
        print("Not enough active-parameter labels to train an active regressor; skipping.")

    predictions = predict_all_models(overall, size_map, names, total_reg, active_reg, use_log)
    table_md = build_markdown_table(predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table_md, encoding="utf-8")
    print(f"\nPredictions table written to {args.output}\n")
    print(table_md)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
