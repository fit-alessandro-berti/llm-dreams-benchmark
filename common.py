ANSWERING_MODEL_NAME = "anthropic/claude-opus-4.6"
EVALUATING_MODEL_NAME = "gpt-4.1"


ALL_JUDGES = {
    "mistral-small-2503": {
        "evaluation_folder": "evaluations-mistral-small",
        "git_table_result": "alt_results_mistral-small-2503.md",
        "evaluation_api_url": "https://api.mistral.ai/v1/",
        "api_key": open("../api_mistral.txt", "r").read().strip(),
    },
    "grok-3": {
        "evaluation_folder": "evaluations-grok3",
        "git_table_result": "alt_results_grok3.md",
        "evaluation_api_url": "https://api.x.ai/v1/",
        "api_key": open("../api_grok.txt", "r").read().strip(),
    },
    "grok-4-1-fast-non-reasoning": {
        "evaluation_folder": "evaluations-grok41-fast",
        "git_table_result": "alt_results_grok41fast.md",
        "evaluation_api_url": "https://api.x.ai/v1/",
        "api_key": open("../api_grok.txt", "r").read().strip(),
    },
    "gemini-2.5-flash": {
        "evaluation_folder": "evaluations-gemini25-flash",
        "git_table_result": "alt_results_gemini25_flash.md",
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
        "git_table_result": "results_gpt52.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "qwen/qwen3-max": {
        "evaluation_folder": "evaluations-qwen3-max",
        "git_table_result": "alt_results_qwen3-max.md",
        "evaluation_api_url": "https://openrouter.ai/api/v1/",
        "api_key": open("../api_openrouter.txt", "r").read().strip()
    }
}

#del ALL_JUDGES["qwen/qwen3-max"]

def get_evaluation_folder(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_folder"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_folder"]
    elif "grok-4-1-fast-non-reasoning" in evaluating_model_name:
        return ALL_JUDGES["grok-4-1-fast-non-reasoning"]["evaluation_folder"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["evaluation_folder"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["evaluation_folder"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["evaluation_folder"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["evaluation_folder"]


def get_git_table_result(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["git_table_result"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["git_table_result"]
    elif "grok-4-1-fast-non-reasoning" in evaluating_model_name:
        return ALL_JUDGES["grok-4-1-fast-non-reasoning"]["git_table_result"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["git_table_result"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["git_table_result"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["git_table_result"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["git_table_result"]


def get_evaluation_api_url(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_api_url"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_api_url"]
    elif "grok-4-1-fast-non-reasoning" in evaluating_model_name:
        return ALL_JUDGES["grok-4-1-fast-non-reasoning"]["evaluation_api_url"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["evaluation_api_url"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["evaluation_api_url"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["evaluation_api_url"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["evaluation_api_url"]


def get_manual(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    return False


def get_api_key(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    elif "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["api_key"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["api_key"]
    elif "grok-4-1-fast-non-reasoning" in evaluating_model_name:
        return ALL_JUDGES["grok-4-1-fast-non-reasoning"]["api_key"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["api_key"]
    elif "gpt-5.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.1"]["api_key"]
    elif "gpt-5.2" in evaluating_model_name:
        return ALL_JUDGES["gpt-5.2"]["api_key"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["api_key"]


#EVALUATION_FOLDER = get_evaluation_folder()
#TARGET_GIT_TABLE_RESULT = get_git_table_result()
#EVALUATION_API_URL = get_evaluation_api_url()
#MANUAL = get_manual()
