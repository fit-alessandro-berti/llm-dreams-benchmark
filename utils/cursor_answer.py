"""Answer dream incipits using the Cursor CLI (`agent`).

Hard-code TARGET_MODEL_NAME (answer filename prefix), TARGET_MODEL (agent --model),
TARGET_REASONING_EFFORT, MAX_WORKERS, and MAX_ANSWERS below, then run:

    python -m utils.cursor_answer
    # or: python utils/cursor_answer.py

Each incipit is answered NUMBER_EXECUTIONS times, matching ``answer.py``.
Files are written as ``answers/<model>__<dream>__<execution>.txt``.
"""

from __future__ import annotations

import os
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
TARGET_MODEL_NAME = "claude-fable-5-1-high"  # prefix used in answers/
# Base id passed to `agent --model`. Effort is appended as a hyphen suffix
# (e.g. claude-opus-5-thinking-high). Cursor does not accept [effort=...]
# overrides. You may also paste a full effort-suffixed slug here and leave
# TARGET_REASONING_EFFORT empty.
TARGET_MODEL = "claude-fable-5-1"
# none | low | medium | high | xhigh | extra-high | max | minimal; "" to omit
TARGET_REASONING_EFFORT = "high"

# Max concurrent Cursor CLI invocations. Each worker handles one answer
# end-to-end (including its own retries) independently of the others.
MAX_WORKERS = 57

# Max unanswered (incipit, execution) pairs to process in this run.
# Set to None for no limit.
MAX_ANSWERS = None

# Two completions per incipit, matching answer.py.
NUMBER_EXECUTIONS = 2

# Cursor CLI executable name or path.
AGENT_COMMAND = "agent"

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


def cursor_model_spec() -> str:
    """Compose the value passed to `agent --model`."""
    if TARGET_REASONING_EFFORT:
        return f"{TARGET_MODEL}-{TARGET_REASONING_EFFORT}"
    return TARGET_MODEL


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


def build_agent_command(prompt: str) -> list[str]:
    return [
        AGENT_COMMAND,
        "-p",
        prompt,
        "--model",
        cursor_model_spec(),
        "--force",
        "--trust",
        "--sandbox",
        "disabled",
    ]


def run_cursor(incipit_name: str, answer_path: str) -> bool:
    """Invoke `agent -p` for one dream. Returns True on success.

    Does not re-run when the answer file already exists and is non-empty.
    """
    if is_completed_output(answer_path):
        _log(f"Skipping (already answered): {answer_path}")
        return True

    os.makedirs(ANSWERS_DIR, exist_ok=True)
    prompt = build_prompt(incipit_name, answer_path)
    cmd = build_agent_command(prompt)

    _log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path.cwd()))
    if result.returncode != 0:
        _log(f"agent exited with status {result.returncode} for {incipit_name}")
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

        ok = run_cursor(incipit_name, path)
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

    if shutil.which(AGENT_COMMAND) is None:
        print(f"Cursor CLI executable was not found: {AGENT_COMMAND}", file=sys.stderr)
        sys.exit(1)

    if MAX_WORKERS < 1:
        print("MAX_WORKERS must be >= 1", file=sys.stderr)
        sys.exit(1)
    if MAX_ANSWERS is not None and MAX_ANSWERS < 1:
        print("MAX_ANSWERS must be >= 1 or None", file=sys.stderr)
        sys.exit(1)

    incipits = list_incipits()
    print(
        f"Cursor answering with model_name={TARGET_MODEL_NAME!r}, "
        f"model={cursor_model_spec()!r}, reasoning_effort={TARGET_REASONING_EFFORT!r}, "
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
        print(f"Running up to {MAX_WORKERS} concurrent Cursor process(es)")
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
