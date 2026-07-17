from pathlib import Path
import sys
from scipy.stats import pearsonr
import pandas as pd


def is_markdown_separator_row(line):
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False

    columns = [col.strip() for col in stripped.split("|")[1:-1]]
    if not columns:
        return False

    for column in columns:
        if not column:
            return False
        if any(char not in "-:" for char in column):
            return False

    return True



REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file_utils import read_file_with_fallback


def repo_file(path):
    return REPO_ROOT / path


def interpret(content):
    llm_scores = {}
    in_overall_table = False

    for row in content.splitlines():
        stripped = row.strip()
        if stripped.startswith("| LLM "):
            in_overall_table = True
            continue
        if not in_overall_table:
            continue
        if "## Individual" in stripped:
            break
        if is_markdown_separator_row(stripped):
            continue
        if not stripped.startswith("|"):
            if llm_scores:
                break
            continue

        columns = [col.strip() for col in stripped.split("|")[1:-1]]
        if len(columns) < 2:
            continue

        model = columns[0].lower()
        mhs = float(columns[1].replace("*", ""))
        llm_scores[model] = mhs

    return llm_scores


JUDGES = {
    "grok-4.3": interpret(read_file_with_fallback(repo_file("alt_results_grok43.md"))),
    "gpt-5.6-sol": interpret(read_file_with_fallback(repo_file("results_gpt56sol.md"))),
    "gpt-5.4": interpret(read_file_with_fallback(repo_file("alt_results_gpt54.md"))),
    "gpt-5.5": interpret(read_file_with_fallback(repo_file("alt_results_gpt55.md"))),
    "qwen36-plus": interpret(read_file_with_fallback(repo_file("alt_results_qwen36-plus.md"))),
    "mistral-small-2603": interpret(read_file_with_fallback(repo_file("alt_results_mistral2603.md"))),
}

model_keys = sorted(set.intersection(*(set(scores.keys()) for scores in JUDGES.values())))

for judge in JUDGES:
    lst = []
    for k in model_keys:
        lst.append((k, JUDGES[judge][k]))
    JUDGES[judge] = [x[1] for x in lst]

dataframe = []

for judge in JUDGES:
    row = {"Model": judge}
    summ = 0
    for judge2 in JUDGES:
        if len(JUDGES[judge]) < 2 or len(JUDGES[judge2]) < 2:
            corr = 0.0
        else:
            corr, pv = pearsonr(JUDGES[judge], JUDGES[judge2])
        row[judge2] = corr
        summ += corr
    row["SUM"] = summ
    dataframe.append(row)

dataframe = pd.DataFrame(dataframe)
dataframe.sort_values(["SUM", "Model"], ascending=False, inplace=True)

repo_file("stats/JUDGES_RANK.md").write_text(dataframe.to_markdown(index=False), encoding="utf-8")
