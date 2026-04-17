ANSWERING_MODEL_NAME = "qwen3.6:35b-a3b"
EVALUATING_MODEL_NAME = "gpt-5.2"


ALL_JUDGES = {
    "grok-3": {
        "evaluation_folder": "evaluations-grok3",
        "git_table_result": "alt_results_grok3.md",
        "evaluation_api_url": "https://api.x.ai/v1/",
        "api_key": open("../api_grok.txt", "r").read().strip(),
    },
    "grok-4.20-0309-non-reasoning": {
        "evaluation_folder": "evaluations-grok42",
        "git_table_result": "alt_results_grok42.md",
        "evaluation_api_url": "https://api.x.ai/v1/",
        "api_key": open("../api_grok.txt", "r").read().strip(),
    },
    "gemini-3-flash-preview": {
        "evaluation_folder": "evaluations-gemini3-flash",
        "git_table_result": "alt_results_gemini3_flash.md",
        "evaluation_api_url": "https://generativelanguage.googleapis.com/v1beta/",
        "api_key": open("../api_google.txt", "r").read().strip(),
    },
    "gpt-5.1": {
        "evaluation_folder": "evaluations-gpt51",
        "git_table_result": "alt_results_gpt51.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "gpt-5.2": {
        "evaluation_folder": "evaluations-gpt52",
        "git_table_result": "alt_results_gpt52.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "gpt-5.4": {
        "evaluation_folder": "evaluations-gpt54",
        "git_table_result": "results_gpt54.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "qwen/qwen3.6-plus": {
        "evaluation_folder": "evaluations-qwen36-plus",
        "git_table_result": "alt_results_qwen36-plus.md",
        "evaluation_api_url": "https://openrouter.ai/api/v1/",
        "api_key": open("../api_openrouter.txt", "r").read().strip()
    },
    "mistral-small-2603": {
        "evaluation_folder": "evaluations-mistral2603",
        "git_table_result": "alt_results_mistral2603.md",
        "evaluation_api_url": "https://api.mistral.ai/v1/",
        "api_key": open("../api_mistral.txt", "r").read().strip()
    },
}

def get_evaluation_folder(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_folder"]
    elif "grok-4.2" in evaluating_model_name:
        return ALL_JUDGES["grok-4.20-0309-non-reasoning"]["evaluation_folder"]
    elif "gemini-3-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-3-flash-preview"]["evaluation_folder"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["evaluation_folder"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["evaluation_folder"]
    elif "gpt-5.4" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.4"]["evaluation_folder"]
    elif "qwen3.6-plus" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3.6-plus"]["evaluation_folder"]
    elif "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2603"]["evaluation_folder"]

def get_git_table_result(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["git_table_result"]
    elif "grok-4.2" in evaluating_model_name:
        return ALL_JUDGES["grok-4.20-0309-non-reasoning"]["git_table_result"]
    elif "gemini-3-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-3-flash-preview"]["git_table_result"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["git_table_result"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["git_table_result"]
    elif "gpt-5.4" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.4"]["git_table_result"]
    elif "qwen3.6-plus" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3.6-plus"]["git_table_result"]
    elif "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2603"]["git_table_result"]

def get_evaluation_api_url(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_api_url"]
    elif "grok-4.2" in evaluating_model_name:
        return ALL_JUDGES["grok-4.20-0309-non-reasoning"]["evaluation_api_url"]
    elif "gemini-3-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-3-flash-preview"]["evaluation_api_url"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["evaluation_api_url"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["evaluation_api_url"]
    elif "gpt-5.4" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.4"]["evaluation_api_url"]
    elif "qwen3.6-plus" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3.6-plus"]["evaluation_api_url"]
    elif "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2603"]["evaluation_api_url"]

def get_manual(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    return False


def get_api_key(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["api_key"]
    elif "grok-4.2" in evaluating_model_name:
        return ALL_JUDGES["grok-4.20-0309-non-reasoning"]["api_key"]
    elif "gemini-3-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-3-flash-preview"]["api_key"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["api_key"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["api_key"]
    elif "gpt-5.4" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.4"]["api_key"]
    elif "qwen3.6-plus" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3.6-plus"]["api_key"]
    elif "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2603"]["api_key"]

#EVALUATION_FOLDER = get_evaluation_folder()
#TARGET_GIT_TABLE_RESULT = get_git_table_result()
#EVALUATION_API_URL = get_evaluation_api_url()
#MANUAL = get_manual()
