"""Answer dream incipits using the OpenAI Codex CLI.

Hard-code TARGET_MODEL_NAME (answer filename prefix), TARGET_MODEL (codex --model),
TARGET_REASONING_EFFORT, MAX_WORKERS, MAX_ANSWERS, and CODEX_COMMAND below,
then run:

    python -m utils.codex_answer
    # or: python utils/codex_answer.py

Each incipit is answered NUMBER_EXECUTIONS times, matching ``answer.py``.
Files are written as ``answers/<model>__<dream>__<execution>.txt``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Hard-coded run configuration — edit these before launching.
# ---------------------------------------------------------------------------
TARGET_MODEL_NAME = "gpt-5.6-sol-XHIGH"  # prefix used in answers/
TARGET_MODEL = "gpt-5.6-sol"  # value passed to codex --model
TARGET_REASONING_EFFORT = "xhigh"  # none | low | medium | high | xhigh

# Max concurrent Codex CLI invocations. Each worker handles one answer
# end-to-end (including its own retries) independently of the others.
MAX_WORKERS = 100

# Max unanswered (incipit, execution) pairs to process in this run.
# Set to None for no limit.
MAX_ANSWERS = None

# Two completions per incipit, matching answer.py.
NUMBER_EXECUTIONS = 2

# Command template. {prompt}, {model}, and {effort} are filled per answer.
# The prompt itself instructs Codex to read incipits/<dream>.txt and write
# answers/<model>__<dream>__<execution>.txt.
# Equivalent shell form:
#   codex exec "..." --model gpt-5.6-luna \
#     -c model_reasoning_effort='"xhigh"' \
#     --dangerously-bypass-approvals-and-sandbox
CODEX_COMMAND = (
    'codex exec "{prompt}" --model {model} '
    '-c model_reasoning_effort=\'"{effort}"\' '
    "--dangerously-bypass-approvals-and-sandbox"
)

# Used only when MAX_WORKERS == 1 (sequential mode).
SLEEP_BETWEEN_ANSWERS_SEC = 60
# Backoff after consecutive failures of the *same* answer. After the last
# entry, further retries keep using that delay (max every 10 minutes).
RETRY_BACKOFF_SEC = (60, 300, 600)
INCIPITS_DIR = "incipits"
ANSWERS_DIR = "answers"

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def clean_model_name(m_name: str) -> str:
    return m_name.replace("/", "").replace(":", "")


def is_completed_output(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def list_incipits() -> list[str]:
    return sorted(
        name for name in os.listdir(INCIPITS_DIR) if name.endswith(".txt")
    )


def answer_path_for(incipit_name: str, model_name: str, execution_index: int) -> str:
    stem = os.path.splitext(incipit_name)[0]
    return os.path.join(
        ANSWERS_DIR,
        clean_model_name(model_name) + "__" + stem + "__" + str(execution_index) + ".txt",
    )


def build_prompt(incipit_name: str, answer_path: str) -> str:
    incipit_path = os.path.join(INCIPITS_DIR, incipit_name)
    return (
        f"Respond to the dream incipit contained in {incipit_path}, "
        f"writing the output to {answer_path}. "
        "You are dreaming. Can you complete the following dream? "
        "Read the incipit file carefully and produce a complete dream continuation. "
        "Write only the final dream text into the output file "
        "(no meta-commentary about this instruction). "
        "STRICTLY FORBIDDEN: do not look at, open, list, search, copy, or otherwise "
        f"consult any existing files under the {ANSWERS_DIR}/ folder (including other "
        "models' answers or any prior answer to this or other dreams). "
        "It is also strictly forbidden to connect to GitHub, the web, remote repos, "
        "or any external source to spy on, fetch, or recover benchmark answers. "
        "Complete the dream solely from the incipit file and your own knowledge."
    )


def run_codex(incipit_name: str, answer_path: str) -> bool:
    """Invoke codex exec for one dream. Returns True on success.

    Does not re-run when the answer file already exists and is non-empty.
    """
    if is_completed_output(answer_path):
        _log(f"Skipping (already answered): {answer_path}")
        return True

    os.makedirs(ANSWERS_DIR, exist_ok=True)
    prompt = build_prompt(incipit_name, answer_path)

    cmd_str = CODEX_COMMAND.format(
        prompt=prompt,
        model=TARGET_MODEL,
        effort=TARGET_REASONING_EFFORT,
    )
    cmd = shlex.split(cmd_str)

    _log(f"Running: {cmd_str}")
    result = subprocess.run(cmd, cwd=str(Path.cwd()))
    if result.returncode != 0:
        _log(f"codex exited with status {result.returncode} for {incipit_name}")
        return False

    if not is_completed_output(answer_path):
        _log(f"No completed answer written to {answer_path}")
        return False

    _log(f"Wrote {answer_path}")
    return True


def process_answer(incipit_name: str, execution_index: int, label: str) -> bool:
    """Run one (incipit, execution) pair to completion with independent retries."""
    path = answer_path_for(incipit_name, TARGET_MODEL_NAME, execution_index)
    if is_completed_output(path):
        _log(f"{label} Skipping (already answered): {path}")
        return True

    _log(f"{label} {incipit_name} execution {execution_index} -> {path}")
    fail_count = 0
    while True:
        # Re-check in case another worker (or a previous attempt) finished it.
        if is_completed_output(path):
            _log(f"{label} Skipping (already answered): {path}")
            return True

        ok = run_codex(incipit_name, path)
        if ok:
            return True
        delay = RETRY_BACKOFF_SEC[min(fail_count, len(RETRY_BACKOFF_SEC) - 1)]
        fail_count += 1
        _log(
            f"{label} Failed on {incipit_name} execution {execution_index} "
            f"(attempt {fail_count}); retrying same answer after {delay} seconds..."
        )
        time.sleep(delay)


def main() -> None:
    os.chdir(REPO_ROOT)

    if not os.path.isdir(INCIPITS_DIR):
        print(f"Missing {INCIPITS_DIR}/ directory", file=sys.stderr)
        sys.exit(1)

    if shutil.which("codex") is None:
        print("Codex CLI executable was not found: codex", file=sys.stderr)
        sys.exit(1)

    if MAX_WORKERS < 1:
        print("MAX_WORKERS must be >= 1", file=sys.stderr)
        sys.exit(1)
    if MAX_ANSWERS is not None and MAX_ANSWERS < 1:
        print("MAX_ANSWERS must be >= 1 or None", file=sys.stderr)
        sys.exit(1)

    incipits = list_incipits()
    print(
        f"Codex answering with model_name={TARGET_MODEL_NAME!r}, "
        f"model={TARGET_MODEL!r}, reasoning_effort={TARGET_REASONING_EFFORT!r}, "
        f"max_workers={MAX_WORKERS}, max_answers={MAX_ANSWERS}, "
        f"number_executions={NUMBER_EXECUTIONS}"
    )
    print(f"{len(incipits)} incipit file(s) under {INCIPITS_DIR}/")

    # Only (incipit, execution) pairs without an existing non-empty answer run.
    pending: list[tuple[str, int]] = []
    for incipit in incipits:
        for execution_index in range(NUMBER_EXECUTIONS):
            path = answer_path_for(incipit, TARGET_MODEL_NAME, execution_index)
            if is_completed_output(path):
                print(f"Skipping (already answered): {path}")
            else:
                pending.append((incipit, execution_index))

    pending_count = len(pending)
    if MAX_ANSWERS is not None:
        pending = pending[:MAX_ANSWERS]

    total = len(pending)
    print(f"{pending_count} answer(s) remaining; {total} selected for this run")
    if not pending:
        print("\nDone.")
        return

    if MAX_WORKERS == 1:
        # Sequential path preserves the original inter-answer sleep.
        for index, (incipit, execution_index) in enumerate(pending):
            process_answer(
                incipit, execution_index, label=f"[{index + 1}/{total}]"
            )
            if index + 1 < total:
                print(
                    f"Sleeping {SLEEP_BETWEEN_ANSWERS_SEC} seconds "
                    f"before next answer..."
                )
                time.sleep(SLEEP_BETWEEN_ANSWERS_SEC)
    else:
        # Concurrent: each answer runs independently on its own worker
        # (own subprocess + own retry/backoff loop).
        print(f"Running up to {MAX_WORKERS} concurrent Codex process(es)")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_answer,
                    incipit,
                    execution_index,
                    f"[{index + 1}/{total}]",
                ): (incipit, execution_index)
                for index, (incipit, execution_index) in enumerate(pending)
            }
            for future in as_completed(futures):
                incipit, execution_index = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    _log(
                        f"Unexpected error on {incipit} execution "
                        f"{execution_index}: {exc!r}"
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
