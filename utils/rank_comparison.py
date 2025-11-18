import os
from scipy.stats import pearsonr
import pandas as pd


def interpret(content):
    llm_scores = {}

    content = "".join(content.split("|:--")[1:])
    content = content.split("\n")[1:]
    #print(content)
    for row in content:
        if "## Individual" in row:
            break

        row = row.split("|")
        model = row[1].strip().lower()
        mhs = float(row[2].replace("*", ""))
        llm_scores[model] = mhs

    return llm_scores


JUDGES = {
    "grok-3": interpret(open(os.path.join("..", "alt_results_grok3.md"), "r").read()),
    "grok-4-fast-non-reasoning": interpret(open(os.path.join("..", "alt_results_grok4fast.md"), "r").read()),
    "mistral-small-2503": interpret(open(os.path.join("..", "alt_results_mistral-small-2503.md"), "r").read()),
    "gemini-2.5-flash": interpret(open(os.path.join("..", "alt_results_gemini25_flash.md"), "r").read()),
    "gpt-5-mini": interpret(open(os.path.join("..", "alt_results_gpt5-mini.md"), "r").read()),
    "gpt-5.1": interpret(open(os.path.join("..", "results_gpt51.md"), "r").read()),
    "qwen3-max": interpret(open(os.path.join("..", "alt_results_qwen3-max.md"), "r").read()),
}

REFERENCE_JUDGE = "grok-4-fast-non-reasoning"
model_keys = list(JUDGES[REFERENCE_JUDGE].keys())

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
        corr, pv = pearsonr(JUDGES[judge], JUDGES[judge2])
        row[judge2] = corr
        summ += corr
    row["SUM"] = summ
    dataframe.append(row)

dataframe = pd.DataFrame(dataframe)
dataframe.sort_values(["SUM", "Model"], ascending=False, inplace=True)

dataframe.to_markdown("../stats/JUDGES_RANK.md", index=False)
