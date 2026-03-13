import os
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
    "grok-3": interpret(open(os.path.join("..", "alt_results_grok3.md"), "r").read()),
    "grok-4-1-fast-non-reasoning": interpret(open(os.path.join("..", "alt_results_grok41fast.md"), "r").read()),
    "grok-4.2": interpret(open(os.path.join("..", "alt_results_grok42.md"), "r").read()),
    "gemini-2.5-flash": interpret(open(os.path.join("..", "alt_results_gemini25_flash.md"), "r").read()),
    "gpt-5.1": interpret(open(os.path.join("..", "alt_results_gpt51.md"), "r").read()),
    "gpt-5.2": interpret(open(os.path.join("..", "alt_results_gpt52.md"), "r").read()),
    "gpt-5.4": interpret(open(os.path.join("..", "results_gpt54.md"), "r").read()),
    "qwen3-max": interpret(open(os.path.join("..", "alt_results_qwen3-max.md"), "r").read()),
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

dataframe.to_markdown("../stats/JUDGES_RANK.md", index=False)
