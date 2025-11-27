import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
MODEL_DATES_PATH = BASE_DIR / "model_dates.json"
MODEL_SIZE_PATH = BASE_DIR / "model_size.json"


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def find_missing_info(
    model_dates: Dict[str, str], model_sizes: Dict[str, List]
) -> List[Tuple[str, List[str]]]:
    models = set(model_dates) | set(model_sizes)
    missing = []

    for model in sorted(models):
        missing_fields = []

        date_str = model_dates.get(model, "")
        if not isinstance(date_str, str) or not date_str.strip():
            missing_fields.append("date")

        size = model_sizes.get(model)
        if not isinstance(size, list) or len(size) == 0:
            missing_fields.append("size")

        if missing_fields:
            missing.append((model, missing_fields))

    return missing


def sorted_by_date(model_dates: Dict[str, str]) -> List[Tuple[str, date]]:
    parsed = []
    today = date.today()

    for model, date_str in model_dates.items():
        if not isinstance(date_str, str) or not date_str.strip():
            continue

        dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        if dt > today:
            raise ValueError(f"Date in the future for model '{model}': {date_str}")

        parsed.append((model, dt))

    return sorted(parsed, key=lambda x: x[1])


def sorted_by_active_params(model_sizes: Dict[str, List]) -> List[Tuple[str, float]]:
    parsed = []

    for model, size_list in model_sizes.items():
        if not isinstance(size_list, list) or len(size_list) == 0:
            continue

        active_params = size_list[0]
        if not isinstance(active_params, (int, float)):
            raise ValueError(
                f"Active parameters must be numeric for model '{model}', got {active_params!r}"
            )

        parsed.append((model, float(active_params)))

    return sorted(parsed, key=lambda x: x[1])


def main() -> None:
    if not MODEL_DATES_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_DATES_PATH}")
    if not MODEL_SIZE_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_SIZE_PATH}")

    model_dates = load_json(MODEL_DATES_PATH)
    model_sizes = load_json(MODEL_SIZE_PATH)

    missing = find_missing_info(model_dates, model_sizes)
    if missing:
        print("Models missing information:")
        for model, fields in missing:
            print(f"- {model}: missing {', '.join(fields)}")
    else:
        print("All models have both date and size entries.")

    print("\nModels sorted by date:")
    for model, dt in sorted_by_date(model_dates):
        print(f"- {model}: {dt.isoformat()}")

    print("\nModels sorted by active parameters (ascending):")
    for model, active in sorted_by_active_params(model_sizes):
        print(f"- {model}: {active}")


if __name__ == "__main__":
    main()
