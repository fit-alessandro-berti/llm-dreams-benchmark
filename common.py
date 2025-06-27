ANSWERING_MODEL_NAME = "gemini-2.5-pro-preview-06-05"
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
    "claude-sonnet-4-20250514": {
        "evaluation_folder": "evaluations-claude-40-sonnet",
        "git_table_result": "alt_results_claude-40-sonnet.md",
        "evaluation_api_url": "https://api.anthropic.com/v1/",
        "api_key": open("../api_anthropic.txt", "r").read().strip(),
    },
    "gpt-4.1-mini": {
        "evaluation_folder": "evaluations-gpt41-mini",
        "git_table_result": "alt_results_gpt41-mini.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "gpt-4.1": {
        "evaluation_folder": "evaluations-gpt41",
        "git_table_result": "alt_results_gpt41.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
}


def get_evaluation_folder(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_folder"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_folder"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["evaluation_folder"]
    elif "claude-sonnet-4" in evaluating_model_name:
        return ALL_JUDGES["claude-sonnet-4-20250514"]["evaluation_folder"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["evaluation_folder"]
    elif "gpt-4.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1"]["evaluation_folder"]

def get_git_table_result(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["git_table_result"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["git_table_result"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["git_table_result"]
    elif "claude-sonnet-4" in evaluating_model_name:
        return ALL_JUDGES["claude-sonnet-4-20250514"]["git_table_result"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["git_table_result"]
    elif "gpt-4.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1"]["git_table_result"]

def get_evaluation_api_url(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_api_url"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["evaluation_api_url"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["evaluation_api_url"]
    elif "claude-sonnet-4" in evaluating_model_name:
        return ALL_JUDGES["claude-sonnet-4-20250514"]["evaluation_api_url"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["evaluation_api_url"]
    elif "gpt-4.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1"]["evaluation_api_url"]


def get_manual(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name:
        return False

    return False


def get_api_key(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    elif "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["api_key"]
    elif "grok-3" in evaluating_model_name:
        return ALL_JUDGES["grok-3"]["api_key"]
    elif "gemini-2.5-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.5-flash"]["api_key"]
    elif "claude-sonnet-4" in evaluating_model_name:
        return ALL_JUDGES["claude-sonnet-4-20250514"]["api_key"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["api_key"]
    elif "gpt-4.1" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1"]["api_key"]

#EVALUATION_FOLDER = get_evaluation_folder()
#TARGET_GIT_TABLE_RESULT = get_git_table_result()
#EVALUATION_API_URL = get_evaluation_api_url()
#MANUAL = get_manual()
