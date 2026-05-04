#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent

PROVIDER_API_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/",
    "openai": "https://api.openai.com/v1/",
    "google": "https://generativelanguage.googleapis.com/v1beta/",
    "claude": "https://api.anthropic.com/v1/",
    "anthropic": "https://api.anthropic.com/v1/",
    "grok": "https://api.x.ai/v1/",
    "x-ai": "https://api.x.ai/v1/",
    "mistral": "https://api.mistral.ai/v1/",
    "deepinfra": "https://api.deepinfra.com/v1/openai/",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
    "nvidia": "https://integrate.api.nvidia.com/v1/",
    "perplexity": "https://api.perplexity.ai/",
    "groq": "https://api.groq.com/openai/v1/",
}

PROVIDER_API_KEY_FILES = {
    "openrouter": "../api_openrouter.txt",
    "openai": "../api_openai.txt",
    "google": "../api_google.txt",
    "claude": "../api_anthropic.txt",
    "anthropic": "../api_anthropic.txt",
    "grok": "../api_grok.txt",
    "x-ai": "../api_grok.txt",
    "mistral": "../api_mistral.txt",
    "deepinfra": "../api_deepinfra.txt",
    "qwen": "../api_qwen.txt",
    "nvidia": "../api_nvidia.txt",
    "perplexity": "../api_perplexity.txt",
    "groq": "../api_groq.txt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute llm-dreams-benchmark for one target model.")
    parser.add_argument("model_name", help="Model alias to benchmark.")
    parser.add_argument("--provider", default="openrouter", help="Model provider. Defaults to openrouter.")
    parser.add_argument("--base-model", help="Underlying API model. Defaults to model_name.")
    parser.add_argument("--alias", help="Alias used inside the benchmark. Defaults to model_name.")
    parser.add_argument("--api-url", help="Override API URL.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key.")
    parser.add_argument("--api-key-file", help="Path to a file containing the API key.")
    parser.add_argument("--reasoning-effort", help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--reasoning-enabled", action="store_true", help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--thinking-tokens", type=int, help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--temperature", type=float, help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--max-tokens", type=int, help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--system-prompt", help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--add-prompt", help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--payload-json", help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--tools-json", help="Accepted for CLI compatibility; unused here.")
    parser.add_argument("--config-json", help="Extra JSON object merged into the config.")
    parser.add_argument("--config-file", help="Path to a JSON file merged into the config.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess phases.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    return parser


def merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_json_object(raw: str | None, label: str) -> Dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a JSON object.")
    return parsed


def load_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as handler:
            file_config = json.load(handler)
        if not isinstance(file_config, dict):
            raise ValueError("config-file must contain a JSON object.")
        config = merge_dicts(config, file_config)
    config = merge_dicts(config, parse_json_object(args.config_json, "config-json"))

    config.setdefault("provider", args.provider)
    config.setdefault("alias", args.alias or args.model_name)
    config.setdefault("base_model", args.base_model or args.model_name)
    config.setdefault("api_url", args.api_url or PROVIDER_API_URLS.get(config["provider"]))
    config.setdefault("api_key_file", args.api_key_file or PROVIDER_API_KEY_FILES.get(config["provider"]))
    if args.api_key_env:
        config["api_key_env"] = args.api_key_env
    return config


def run_subprocess(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def sync_repository(dry_run: bool) -> None:
    git_commands = [
        ["git", "reset", "--hard", "HEAD"],
        ["git", "clean", "-x", "-f"],
        ["git", "pull"],
    ]
    for command in git_commands:
        run_subprocess(command, cwd=REPO_ROOT, dry_run=dry_run)


def run_answers(config: Dict[str, Any], answer_module: Any) -> None:
    answer_module.configure_model(config["base_model"])
    answer_module.m_name = config["alias"].replace("/", "").replace(":", "")
    answer_module.configure_api(config["api_url"], str((REPO_ROOT / config["api_key_file"]).resolve()))

    max_workers = min(answer_module.MAX_WORKERS, max(1, len(answer_module.incipits) * answer_module.NUMBER_EXECUTIONS))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for execution_index in range(answer_module.NUMBER_EXECUTIONS):
            for incipit in answer_module.incipits:
                futures.append(executor.submit(answer_module.generate_answer_for_incipit, incipit, execution_index))
        for future in as_completed(futures):
            future.result()


def execute_pipeline(config: Dict[str, Any], python_executable: str, dry_run: bool) -> None:
    if dry_run:
        print(f"Would execute llm-dreams-benchmark for alias={config['alias']} base_model={config['base_model']}")
        return

    common_module = importlib.import_module("common")
    answer_module = importlib.import_module("answer")
    evaluation_module = importlib.import_module("evaluation")

    temp_api_key_path: str | None = None
    if config.get("api_key_env") and config.get("api_key_file") is None:
        api_key_value = os.environ.get(config["api_key_env"])
        if api_key_value:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handler:
                handler.write(api_key_value)
                temp_api_key_path = handler.name
            config = dict(config)
            config["api_key_file"] = temp_api_key_path

    try:
        run_answers(config, answer_module)

        original_answering_model_name = common_module.ANSWERING_MODEL_NAME
        try:
            common_module.ANSWERING_MODEL_NAME = config["alias"]
            evaluation_module.dispatch_all_evaluations([common_module.EVALUATING_MODEL_NAME], massive=False)
        finally:
            common_module.ANSWERING_MODEL_NAME = original_answering_model_name

        run_subprocess([python_executable, "git_table_results.py"], cwd=REPO_ROOT, dry_run=False)
        run_subprocess([python_executable, "utils/rank_comparison.py"], cwd=REPO_ROOT, dry_run=False)
        run_subprocess([python_executable, "utils/single_voices_report.py"], cwd=REPO_ROOT, dry_run=False)
        run_subprocess([python_executable, "utils/parse_compute_metrics.py"], cwd=REPO_ROOT, dry_run=False)
    finally:
        if temp_api_key_path:
            Path(temp_api_key_path).unlink(missing_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args)
    sync_repository(args.dry_run)
    execute_pipeline(config, python_executable=args.python, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
