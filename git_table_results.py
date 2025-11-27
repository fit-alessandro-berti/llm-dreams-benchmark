import json
import os
import numpy as np
import pandas as pd
import common

PERSONALITY_KEYS = [
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
]

NEGATED_KEYS = {
    "Anxiety and Stress Levels",
    "Fear of Failure",
    "Need for Control",
    "Cognitive Load",
}


def format_mean_std(mean, std):
    return f"{mean} $\\pm$ {std}"


def aggregate_llm_scores(file_paths, normalization_divisor=1):
    scores = {k: [] for k in PERSONALITY_KEYS}
    total_score = 0.0
    divisor = normalization_divisor if normalization_divisor else 1

    for full_path in file_paths:
        dictio = json.load(open(full_path, "r"))

        for key in PERSONALITY_KEYS:
            value = dictio[key]
            scores[key].append(value)
            total_score += (10.0 - value) if key in NEGATED_KEYS else value

    score_stats = {}
    formatted_scores = {}
    for key in PERSONALITY_KEYS:
        mean = round(np.mean(scores[key]), 1)
        std = round(np.std(scores[key]), 1)
        score_stats[key] = {"mean": mean, "std": std}
        formatted_scores[key] = format_mean_std(mean, std)

    return total_score / divisor, formatted_scores, score_stats


def build_overall_dataframe(llms, mhs, all_llms_scores):
    overall_columns = {"LLM": llms, "MHS": [mhs[llm] for llm in llms]}

    for k in PERSONALITY_KEYS:
        overall_columns[k] = [all_llms_scores[llm][k] for llm in llms]

    dataframe = pd.DataFrame(overall_columns)
    dataframe.sort_values(["MHS", "LLM"], ascending=False, inplace=True)
    dataframe["MHS"] = dataframe["MHS"].apply(lambda x: "**%.1f**" % x)

    return dataframe


def render_individual_results(llms, all_llms_scores):
    individual_results = ["## Individual Results"]

    for llm in llms:
        values = [all_llms_scores[llm][k] for k in PERSONALITY_KEYS]

        dataframe = pd.DataFrame(
            {"Personality Trait": PERSONALITY_KEYS, "Score (1.0-10.0)": values}
        )
        individual_results.append("\n")
        individual_results.append("### " + llm)
        individual_results.append("\n")
        individual_results.append(dataframe.to_markdown(index=False))
        individual_results.append("\n\n\n")

    return "\n".join(individual_results)


def group_files_by_llm(evaluation_folder):
    evaluations = [x for x in os.listdir(evaluation_folder) if x.endswith(".txt")]
    evaluations = sorted(
        evaluations,
        key=lambda x: (
            x.split("__")[0].lower(),
            os.path.getctime(os.path.join(evaluation_folder, x)),
        ),
    )

    llm_files = {}
    for ev in evaluations:
        llm = ev.split("__")[0]
        llm_files.setdefault(llm, []).append(os.path.join(evaluation_folder, ev))

    return llm_files


def write_table(evaluation_folder, target_git_table_result):
    if not os.path.exists(evaluation_folder):
        os.mkdir(evaluation_folder)

    llm_files = group_files_by_llm(evaluation_folder)
    llms = sorted(llm_files.keys(), key=lambda x: x.lower())

    mhs = {}
    all_llms_scores = {}

    for llm in llms:
        total_s, scores, _ = aggregate_llm_scores(llm_files[llm])
        mhs[llm] = total_s
        all_llms_scores[llm] = scores

    dataframe = build_overall_dataframe(llms, mhs, all_llms_scores)

    overall_results = ["## Overall Results\n", dataframe.to_markdown(index=False)]
    individual_results = render_individual_results(llms, all_llms_scores)

    combined_stru = "\n".join(overall_results) + "\n" + individual_results

    with open(target_git_table_result, "w") as F:
        F.write(combined_stru)


def collect_all_llm_files(evaluation_folders):
    llm_files = {}
    llm_folder_counts = {}

    for evaluation_folder in evaluation_folders:
        if not os.path.exists(evaluation_folder):
            continue

        folder_llm_files = group_files_by_llm(evaluation_folder)
        for llm, files in folder_llm_files.items():
            llm_files.setdefault(llm, []).extend(files)
            llm_folder_counts[llm] = llm_folder_counts.get(llm, 0) + 1

    for llm in llm_files:
        llm_files[llm] = sorted(llm_files[llm])

    return llm_files, llm_folder_counts


def write_overall_results(
    target_overall_path="OVERALL.md", target_overall_json="OVERALL.json"
):
    evaluation_folders = [
        config["evaluation_folder"] for config in common.ALL_JUDGES.values()
    ]
    llm_files, llm_folder_counts = collect_all_llm_files(evaluation_folders)

    if not llm_files:
        print("No evaluations found to build OVERALL.md")
        return

    llms = sorted(llm_files.keys(), key=lambda x: x.lower())
    mhs = {}
    all_llms_scores = {}
    all_llms_score_stats = {}

    for llm in llms:
        normalization_divisor = llm_folder_counts.get(llm, 1)
        total_s, scores, score_stats = aggregate_llm_scores(
            llm_files[llm], normalization_divisor=normalization_divisor
        )
        mhs[llm] = total_s
        all_llms_scores[llm] = scores
        all_llms_score_stats[llm] = score_stats

    dataframe = build_overall_dataframe(llms, mhs, all_llms_scores)
    overall_results = ["## Overall Results\n", dataframe.to_markdown(index=False)]
    individual_results = render_individual_results(llms, all_llms_scores)

    combined_stru = "\n".join(overall_results) + "\n" + individual_results

    with open(target_overall_path, "w") as F:
        F.write(combined_stru)

    overall_json = []
    for llm in llms:
        overall_json.append(
            {
                "LLM": llm,
                "MHS": round(mhs[llm], 1),
                "scores": {
                    key: all_llms_score_stats[llm][key] for key in PERSONALITY_KEYS
                },
            }
        )

    overall_json_path = target_overall_json
    if not os.path.isabs(overall_json_path):
        output_dir = os.path.dirname(target_overall_path)
        if output_dir:
            overall_json_path = os.path.join(output_dir, overall_json_path)

    with open(overall_json_path, "w") as F:
        json.dump(overall_json, F, indent=2)


if __name__ == "__main__":
    model_list = list(common.ALL_JUDGES)

    for index, m in enumerate(model_list):
        print(index, m)
        evaluation_folder = common.get_evaluation_folder(m)
        target_git_table_result = common.get_git_table_result(m)

        write_table(evaluation_folder, target_git_table_result)

    write_overall_results()
