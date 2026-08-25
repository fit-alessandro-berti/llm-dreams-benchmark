#!/usr/bin/env python3
"""Embed each answer in answers/ with OpenAI and write vectors to embeddings/output/."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file_utils import read_file_with_fallback

ANSWERS_DIR = REPO_ROOT / "answers"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

EMBEDDING_MODEL = "text-embedding-3-small"
API_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MAX_WORKERS = 32
WAITING_TIME_RETRY = 15
REQUEST_TIMEOUT_SECONDS = 120


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Embed answers/ files with OpenAI and write them to embeddings/output/."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("EMBED_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
        help="Maximum number of concurrent embedding threads (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument(
        "--api-key-file",
        default=str(REPO_ROOT.parent / "api_openai.txt"),
        help="Fallback file containing the OpenAI API key.",
    )
    return parser.parse_args(argv)


def read_api_key(api_key_env: str, api_key_file: str) -> str:
    if os.environ.get(api_key_env):
        return os.environ[api_key_env]
    candidate = Path(api_key_file)
    if candidate.exists():
        return read_file_with_fallback(candidate).strip()
    return ""


def is_embedded(output_path: Path) -> bool:
    if not output_path.is_file():
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    embedding = payload.get("embedding") if isinstance(payload, dict) else payload
    return isinstance(embedding, list) and len(embedding) > 0 and all(
        isinstance(value, (int, float)) for value in embedding
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def fetch_embedding(text: str, api_key: str) -> list[float]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text if text else " ",
    }
    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI embeddings HTTP {response.status_code}: {response.text[:500]}"
        )
    data = response.json()["data"]
    if not data:
        raise RuntimeError("OpenAI embeddings response contained no vectors.")
    return data[0]["embedding"]


def embed_answer(answer_path: Path, output_path: Path, api_key: str) -> str:
    if is_embedded(output_path):
        return "skipped"

    text = read_file_with_fallback(answer_path)
    while True:
        try:
            embedding = fetch_embedding(text, api_key)
            write_json_atomic(
                output_path,
                {
                    "source": answer_path.name,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding": embedding,
                },
            )
            return "embedded"
        except Exception:
            traceback.print_exc()
            print(f"sleeping {WAITING_TIME_RETRY} seconds ...")
            time.sleep(WAITING_TIME_RETRY)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")

    api_key = read_api_key(args.api_key_env, args.api_key_file)
    if not api_key:
        raise SystemExit(
            f"OpenAI API key not found in {args.api_key_env} or {args.api_key_file}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    answer_paths = sorted(ANSWERS_DIR.glob("*.txt"))
    pending = [
        path for path in answer_paths if not is_embedded(OUTPUT_DIR / f"{path.stem}.json")
    ]
    skipped = len(answer_paths) - len(pending)

    print(f"Found {len(answer_paths)} answers; {skipped} already embedded.")
    if not pending:
        print("Nothing to embed.")
        return 0

    max_workers = min(args.max_workers, len(pending))
    print(f"Embedding {len(pending)} answers with {max_workers} concurrent threads.")

    progress_lock = Lock()
    completed = 0
    embedded = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                embed_answer,
                path,
                OUTPUT_DIR / f"{path.stem}.json",
                api_key,
            ): path
            for path in pending
        }
        for future in as_completed(futures):
            path = futures[future]
            status = future.result()
            with progress_lock:
                completed += 1
                if status == "embedded":
                    embedded += 1
                current = completed
            print(f"[{current}/{len(pending)}] {status} {path.name}")

    print(f"Done. Newly embedded: {embedded}; previously skipped: {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
