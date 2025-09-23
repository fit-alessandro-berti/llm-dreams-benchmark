ANSWERING_MODEL_NAME = "gpt-5-chat-latest"
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
    "gemini-2.5-flash": {
        "evaluation_folder": "evaluations-gemini25-flash",
        "git_table_result": "alt_results_gemini25_flash.md",
        "evaluation_api_url": "https://generativelanguage.googleapis.com/v1beta/",
        "api_key": open("../api_google.txt", "r").read().strip(),
    },
    "gpt-5": {
        "evaluation_folder": "evaluations-gpt50",
        "git_table_result": "results_gpt5.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "gpt-5-mini": {
        "evaluation_folder": "evaluations-gpt50-mini",
        "git_table_result": "alt_results_gpt5-mini.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "gpt-4.5-preview": {
        "evaluation_folder": "OLD/evaluations-gpt45",
        "git_table_result": "results_gpt_45.md",
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


def get_evaluation_folder(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_folder"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_folder"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["evaluation_folder"]
    elif "gpt-5-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-5-mini"]["evaluation_folder"]
    elif "gpt-5" in evaluating_model_name:
        return ALL_JUDGES["gpt-5"]["evaluation_folder"]
    elif "gpt-4.5" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["evaluation_folder"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["evaluation_folder"]


def get_git_table_result(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["git_table_result"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["git_table_result"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["git_table_result"]
    elif "gpt-5-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-5-mini"]["git_table_result"]
    elif "gpt-5" in evaluating_model_name:
        return ALL_JUDGES["gpt-5"]["git_table_result"]
    elif "gpt-4.5" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["git_table_result"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["git_table_result"]


def get_evaluation_api_url(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_api_url"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_api_url"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["evaluation_api_url"]
    elif "gpt-5-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-5-mini"]["evaluation_api_url"]
    elif "gpt-5" in evaluating_model_name:
        return ALL_JUDGES["gpt-5"]["evaluation_api_url"]
    elif "gpt-4.5" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["evaluation_api_url"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["evaluation_api_url"]


def get_manual(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name:
        return True

    return False


def get_api_key(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    elif "mistral-small" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["api_key"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["api_key"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["api_key"]
    elif "gpt-5-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-5-mini"]["api_key"]
    elif "gpt-5" in evaluating_model_name:
        return ALL_JUDGES["gpt-5"]["api_key"]
    elif "gpt-4.5" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["api_key"]
    elif "qwen3-max" in evaluating_model_name:
        return ALL_JUDGES["qwen/qwen3-max"]["api_key"]


#EVALUATION_FOLDER = get_evaluation_folder()
#TARGET_GIT_TABLE_RESULT = get_git_table_result()
#EVALUATION_API_URL = get_evaluation_api_url()
#MANUAL = get_manual()
