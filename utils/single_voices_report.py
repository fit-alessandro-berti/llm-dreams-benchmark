from typing import Dict, List
import os
import sys
import re

# Voices/metrics we expect in the markdown tables
HEADERS = [
    "Anxiety and Stress Levels", "Emotional Stability", "Problem-solving Skills",
    "Creativity", "Interpersonal Relationships", "Confidence and Self-efficacy",
    "Conflict Resolution", "Work-related Stress", "Adaptability",
    "Achievement Motivation", "Fear of Failure", "Need for Control",
    "Cognitive Load", "Social Support", "Resilience"
]


def parse_markdown_table(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse a markdown table like in your previous script and return:
    {
        "model_name": {
            "Voice/Metric": mean_value (float),
            ...
        },
        ...
    }
    Only the mean part of "AVG ± STDEV" is captured for each metric.
    """
    result: Dict[str, Dict[str, float]] = {}

    def _parse_cell_mean(cell: str) -> float:
        # Primary path: values like "3.5$\\pm$0.2"
        if "$" in cell:
            try:
                return float(cell.split("$")[0].strip())
            except Exception:
                pass
        # Common variant: "3.5 ± 0.2"
        if "±" in cell:
            try:
                return float(cell.split("±")[0].strip())
            except Exception:
                pass
        # Fallback: first float in the cell
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell)
        if not m:
            raise ValueError(f"Could not parse numeric mean from cell: {cell!r}")
        return float(m.group(0))

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.splitlines()
        in_table = False
        header_found = False

        for line in lines:
            s = line.strip()
            if s.startswith("| LLM"):
                in_table = True
                header_found = True
                continue
            if in_table and s.startswith("|:--"):
                # separator row
                continue
            if in_table and s.startswith("|"):
                # data row
                columns = [col.strip() for col in line.split("|")[1:-1]]
                if len(columns) < 2:
                    continue
                llm_name = columns[0].strip()
                metrics: Dict[str, float] = {}
                # Skip first two columns, assume metrics start from columns[2:]
                for header, cell in zip(HEADERS, columns[2:]):
                    try:
                        metrics[header] = _parse_cell_mean(cell)
                    except Exception:
                        # If parsing fails for a cell, just skip that metric for this model
                        continue
                if metrics:
                    result[llm_name] = metrics
            if in_table and not s.startswith("|") and header_found:
                break

        return result

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found")
    except Exception as e:
        raise Exception(f"Error processing file '{file_path}': {e}")


def aggregate_averages(models_metrics_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate averages across multiple files.
    Returns:
    {
      "Voice": {"model": avg_across_files, ...},
      ...
    }
    """
    # voice -> model -> [sum, count]
    agg: Dict[str, Dict[str, List[float]]] = {}

    for data in models_metrics_list:
        for model, metrics in data.items():
            for voice, mean_val in metrics.items():
                if voice not in agg:
                    agg[voice] = {}
                if model not in agg[voice]:
                    agg[voice][model] = [0.0, 0.0]
                agg[voice][model][0] += float(mean_val)
                agg[voice][model][1] += 1.0

    # Convert to averages
    out: Dict[str, Dict[str, float]] = {}
    for voice, model_map in agg.items():
        out[voice] = {}
        for model, (s, c) in model_map.items():
            if c > 0:
                out[voice][model] = s / c
    return out


def voice_to_filename(voice: str) -> str:
    """
    Convert a voice name to a clean Markdown filename.
    - Keep letters, digits, spaces, underscores, hyphens.
    - Replace spaces and hyphens with underscores.
    - Collapse multiple underscores.
    """
    cleaned = re.sub(r"[^\w\s-]", "", voice)         # drop anything weird
    cleaned = cleaned.replace("-", " ")              # treat hyphens like spaces
    cleaned = "_".join(cleaned.split())              # spaces -> underscores (collapses repeats)
    return f"{cleaned}.md"


def write_per_voice_tables(output_dir: str, voice_model_avgs: Dict[str, Dict[str, float]]):
    """
    For each voice, write stats/single_voices/<Voice>.md containing a Markdown table:
    | Model | Average |
    |:--|--:|
    | model_a | 3.500 |
    | model_b | 3.250 |
    """
    os.makedirs(output_dir, exist_ok=True)

    # Order voices: keep HEADERS order first, then any unexpected voices alphabetically
    known = [v for v in HEADERS if v in voice_model_avgs]
    unknown = sorted([v for v in voice_model_avgs.keys() if v not in HEADERS])
    voices = known + unknown

    for voice in voices:
        items = list(voice_model_avgs[voice].items())
        # Sort descending by average; tie-break by model name
        items.sort(key=lambda kv: (-kv[1], kv[0].lower()))

        filename = voice_to_filename(voice)
        path = os.path.join(output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {voice}\n\n")
            f.write("| Model | Average |\n")
            f.write("|:--|--:|\n")
            for model, avg in items:
                f.write(f"| {model} | {avg:.3f} |\n")

        print(f"Wrote: {path}")


def main():
    """
    Parse all judge files (same source as your previous script),
    compute per-voice per-model averages across files,
    and write one Markdown table per voice into stats/single_voices/.
    """
    try:
        # Move to project root (like your other script)
        os.chdir("..")
    except Exception:
        pass

    try:
        from common import ALL_JUDGES
    except Exception as e:
        print(f"Error importing ALL_JUDGES from common.py: {e}", file=sys.stderr)
        sys.exit(1)

    parsed_list: List[Dict[str, Dict[str, float]]] = []
    for idx, judge in enumerate(ALL_JUDGES):
        print(idx, judge)
        input_path = ALL_JUDGES[judge]["git_table_result"]
        try:
            parsed = parse_markdown_table(input_path)
            if parsed:
                parsed_list.append(parsed)
            else:
                print(f"Warning: No data parsed from {input_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Skipping '{input_path}' due to error: {e}", file=sys.stderr)
            continue

    if not parsed_list:
        print("Error: No input files could be parsed.", file=sys.stderr)
        sys.exit(1)

    avgs = aggregate_averages(parsed_list)
    output_dir = os.path.join("stats", "single_voices")
    write_per_voice_tables(output_dir, avgs)


if __name__ == "__main__":
    main()
