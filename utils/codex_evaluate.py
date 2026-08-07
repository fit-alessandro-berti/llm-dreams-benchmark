"""Evaluate dream answers with the OpenAI Codex CLI.

Each answering model receives four evaluations, matching ``evaluation.py``:
two answer executions (``__0`` and ``__1``), each evaluated twice. Relative
paths are resolved from the repository root.

Example:

    python utils/codex_evaluate.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file_utils import read_file_with_fallback


NUMBER_EVALUATIONS = 2
ANSWER_EXECUTION_INDEXES = ("0.txt", "1.txt")
DEFAULT_RETRY_DELAY_SECONDS = 17.0
TARGET_MODEL = "gpt-5.6-terra"
TARGET_REASONING_EFFORT = "low"
MAX_WORKERS = 90
EVALUATION_FOLDER = Path("evaluations-gpt56terra")

TRAIT_KEYS = (
    "Anxiety and Stress Levels",
    "Emotional Stability",
    "Problem-solving Skills",
    "Creativity",
    "Interpersonal Relationships",
    "Confidence and Self-efficacy",
    "Conflict Resolution",
    "Work-related Stress",
    "Adaptability",
    "Achievement Motivation",
    "Fear of Failure",
    "Need for Control",
    "Cognitive Load",
    "Social Support",
    "Resilience",
)

EVALUATION_INSTRUCTIONS = (
    "A person did the following dreams. I ask you to estimate the personality "
    "trait of this person. The final output should be a JSON containing the "
    "following keys: 'Anxiety and Stress Levels', 'Emotional Stability', "
    "'Problem-solving Skills', 'Creativity', 'Interpersonal Relationships', "
    "'Confidence and Self-efficacy', 'Conflict Resolution', 'Work-related "
    "Stress', 'Adaptability', 'Achievement Motivation', 'Fear of Failure', "
    "'Need for Control', 'Cognitive Load', 'Social Support', 'Resilience'. "
    "Each key should be associated to a number from 1.0 (minimum score) to "
    "10.0 (maximum score). Please follow strictly the provided JSON schema in "
    "the evaluation!"
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "number", "minimum": 1.0, "maximum": 10.0}
        for key in TRAIT_KEYS
    },
    "required": list(TRAIT_KEYS),
    "additionalProperties": False,
}

_print_lock = threading.Lock()


def log(message: str) -> None:
    """Print one complete line without interleaving worker output."""
    with _print_lock:
        print(message, flush=True)


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def resolve_from_repo(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every model in answers/ with Codex, preserving the "
            "benchmark's existing evaluation layout."
        )
    )
    parser.add_argument(
        "--answers-folder",
        type=Path,
        default=Path("answers"),
        help="Answer directory relative to the repository root (default: answers).",
    )
    parser.add_argument(
        "--incipits-folder",
        type=Path,
        default=Path("incipits"),
        help="Incipit directory relative to the repository root (default: incipits).",
    )
    parser.add_argument(
        "--codex-command",
        default="codex",
        help="Codex CLI executable (default: codex).",
    )
    parser.add_argument(
        "--retry-delay",
        type=non_negative_float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help="Seconds to wait after a failed or invalid attempt (default: 17).",
    )
    return parser.parse_args(argv)


def collect_answers(answers_folder: Path) -> dict[str, list[Path]]:
    """Group well-formed answer files by answering model."""
    answers_by_model: dict[str, list[Path]] = {}
    for answer_path in sorted(answers_folder.iterdir()):
        if not answer_path.is_file() or answer_path.suffix != ".txt":
            continue

        parts = answer_path.name.split("__")
        if len(parts) != 3 or not parts[0] or not parts[1]:
            log(f"Ignoring malformed answer filename: {answer_path.name}")
            continue

        answers_by_model.setdefault(parts[0], []).append(answer_path)

    return answers_by_model


def build_evaluation_prompt(
    model_answers: list[Path],
    answer_execution_index: str,
    incipits_folder: Path,
) -> str:
    """Build the same combined-dream prompt used by ``evaluation.py``."""
    selected_answers = [
        path
        for path in model_answers
        if path.name.split("__")[-1] == answer_execution_index
    ]
    contents = [EVALUATION_INSTRUCTIONS]

    for answer_path in selected_answers:
        dream_name = answer_path.name.split("__")[1]
        incipit_path = incipits_folder / f"{dream_name}.txt"
        incipit = read_file_with_fallback(incipit_path)
        answer = read_file_with_fallback(answer_path)
        answer = answer.replace("\n", " ").replace("\r", " ")
        contents.append(f"{incipit} {answer}")

    return "\n\n".join(contents)


def validate_evaluation(evaluation_path: Path) -> dict[str, Any]:
    """Load and strictly validate one benchmark evaluation."""
    value = json.loads(read_file_with_fallback(evaluation_path))
    if not isinstance(value, dict):
        raise ValueError("evaluation must be a JSON object")

    actual_keys = set(value)
    expected_keys = set(TRAIT_KEYS)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"incorrect trait keys (missing={missing}, extra={extra})")

    for key in TRAIT_KEYS:
        score = value[key]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{key!r} must be numeric")
        if not math.isfinite(float(score)) or not 1.0 <= float(score) <= 10.0:
            raise ValueError(f"{key!r} must be between 1.0 and 10.0")

    return {key: value[key] for key in TRAIT_KEYS}


def remove_invalid_evaluation(evaluation_path: Path) -> bool:
    """Return True for a valid existing file; remove it otherwise."""
    if not evaluation_path.exists():
        return False

    try:
        validate_evaluation(evaluation_path)
    except Exception as exc:
        log(f"Removing invalid evaluation {evaluation_path.name}: {exc}")
        evaluation_path.unlink(missing_ok=True)
        return False

    return True


def codex_instruction(prompt_path: Path, evaluation_path: Path) -> str:
    return (
        f"Read the evaluation request in {prompt_path}. Perform exactly the "
        "evaluation requested there and return only the resulting JSON object. "
        f"Your final response will be saved automatically to {evaluation_path}; "
        "do not read that file and do not try to write it with a tool. Do not "
        "list or search directories and do not inspect any other files, including "
        "any answer or evaluation files. Do not use web search, network access, "
        "MCP, connectors, skills, plugins, subagents, or any tools other than the "
        "single file-read operation needed to open the specified prompt file."
    )


def build_codex_command(
    codex_command: str,
    workspace: Path,
    prompt_path: Path,
    schema_path: Path,
    evaluation_path: Path,
) -> list[str]:
    """Build an isolated, read-only non-interactive Codex invocation."""
    return [
        codex_command,
        "exec",
        "--model",
        TARGET_MODEL,
        "--config",
        f'model_reasoning_effort="{TARGET_REASONING_EFFORT}"',
        "--config",
        'web_search="disabled"',
        "--sandbox",
        "read-only",
        "--cd",
        str(workspace),
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(evaluation_path),
        "--color",
        "never",
        codex_instruction(prompt_path, evaluation_path),
    ]


def run_one_attempt(
    prompt: str,
    evaluation_path: Path,
    codex_command: str,
) -> bool:
    """Run Codex once, then validate and canonicalize its output."""
    evaluation_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="llm-dreams-codex-eval-") as temp_dir:
        workspace = Path(temp_dir).resolve()
        prompt_path = workspace / "evaluation_prompt.txt"
        schema_path = workspace / "evaluation_schema.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        schema_path.write_text(json.dumps(OUTPUT_SCHEMA), encoding="utf-8")

        command = build_codex_command(
            codex_command=codex_command,
            workspace=workspace,
            prompt_path=prompt_path,
            schema_path=schema_path,
            evaluation_path=evaluation_path,
        )
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        evaluation_path.unlink(missing_ok=True)
        output = result.stdout.strip()
        if output:
            log(f"Codex exited with status {result.returncode}: {output[-2000:]}")
        else:
            log(f"Codex exited with status {result.returncode}")
        return False

    try:
        normalized = validate_evaluation(evaluation_path)
    except Exception as exc:
        log(f"Invalid Codex evaluation for {evaluation_path.name}: {exc}")
        evaluation_path.unlink(missing_ok=True)
        return False

    evaluation_path.write_text(json.dumps(normalized), encoding="utf-8")
    return True


def evaluate_until_valid(
    label: str,
    prompt: str,
    evaluation_path: Path,
    codex_command: str,
    retry_delay: float,
) -> bool:
    """Retry one evaluation until Codex creates a valid output file."""
    if remove_invalid_evaluation(evaluation_path):
        log(f"{label} Skipping valid evaluation: {evaluation_path.name}")
        return True

    attempt = 0
    while True:
        attempt += 1
        log(f"{label} Evaluating {evaluation_path.name} (attempt {attempt})")
        try:
            if run_one_attempt(
                prompt=prompt,
                evaluation_path=evaluation_path,
                codex_command=codex_command,
            ):
                log(f"{label} Wrote valid evaluation: {evaluation_path.name}")
                return True
        except Exception as exc:
            evaluation_path.unlink(missing_ok=True)
            log(f"{label} Evaluation attempt failed: {exc!r}")

        if retry_delay:
            log(f"{label} Retrying in {retry_delay:g} seconds")
            time.sleep(retry_delay)


def build_tasks(
    answers_by_model: dict[str, list[Path]],
    incipits_folder: Path,
    evaluation_folder: Path,
) -> list[tuple[str, str, Path]]:
    """Return (answering model, prompt, output path) evaluation tasks."""
    tasks: list[tuple[str, str, Path]] = []
    for answering_model, model_answers in sorted(answers_by_model.items()):
        sanitized_model = answering_model.replace("/", "").replace(":", "")
        for answer_index_number, answer_index in enumerate(ANSWER_EXECUTION_INDEXES):
            prompt = build_evaluation_prompt(
                model_answers=model_answers,
                answer_execution_index=answer_index,
                incipits_folder=incipits_folder,
            )
            for evaluation_number in range(NUMBER_EVALUATIONS):
                evaluation_name = (
                    f"{sanitized_model}__{answer_index_number}__"
                    f"{evaluation_number}.txt"
                )
                tasks.append(
                    (answering_model, prompt, evaluation_folder / evaluation_name)
                )
    return tasks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    answers_folder = resolve_from_repo(args.answers_folder)
    incipits_folder = resolve_from_repo(args.incipits_folder)
    evaluation_folder = resolve_from_repo(EVALUATION_FOLDER)

    if not answers_folder.is_dir():
        print(f"Answers folder does not exist: {answers_folder}", file=sys.stderr)
        return 2
    if not incipits_folder.is_dir():
        print(f"Incipits folder does not exist: {incipits_folder}", file=sys.stderr)
        return 2
    if shutil.which(args.codex_command) is None:
        print(f"Codex CLI executable was not found: {args.codex_command}", file=sys.stderr)
        return 2

    evaluation_folder.mkdir(parents=True, exist_ok=True)
    answers_by_model = collect_answers(answers_folder)
    tasks = build_tasks(answers_by_model, incipits_folder, evaluation_folder)

    log(
        f"Found {len(answers_by_model)} answering model(s); "
        f"{len(tasks)} evaluation file(s) expected"
    )
    log(
        f"Codex model={TARGET_MODEL!r}, reasoning_effort="
        f"{TARGET_REASONING_EFFORT!r}, max_workers={MAX_WORKERS}"
    )
    if not tasks:
        log("Nothing to evaluate.")
        return 0

    pending: list[tuple[str, str, Path]] = []
    for answering_model, prompt, evaluation_path in tasks:
        if remove_invalid_evaluation(evaluation_path):
            log(f"Skipping valid evaluation: {evaluation_path.name}")
        else:
            pending.append((answering_model, prompt, evaluation_path))

    if not pending:
        log("All evaluations are already valid.")
        return 0

    log(
        f"Running {len(pending)} pending evaluation(s) with up to "
        f"{MAX_WORKERS} concurrent Codex process(es)"
    )
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                evaluate_until_valid,
                f"[{index}/{len(pending)}] {answering_model}",
                prompt,
                evaluation_path,
                args.codex_command,
                args.retry_delay,
            ): evaluation_path
            for index, (answering_model, prompt, evaluation_path) in enumerate(
                pending, start=1
            )
        }
        for future in as_completed(futures):
            future.result()

    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
