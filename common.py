ANSWERING_MODEL_NAME = "gemini-2.5-pro-preview-06-05"
EVALUATING_MODEL_NAME = "gpt-4.5-preview"
# EVALUATING_MODEL_NAME = "mistral-small-2503"
# EVALUATING_MODEL_NAME = "chatgpt-4o-latest"
# EVALUATING_MODEL_NAME = "grok-2-1212"
# EVALUATING_MODEL_NAME = "gemini-2.0-flash"
# EVALUATING_MODEL_NAME = "claude-3-5-sonnet-20241022"
# EVALUATING_MODEL_NAME = "gpt-4.1-mini-2025-04-14"


ALL_JUDGES = {
    "gpt-4.5-preview": {
        "evaluation_folder": "evaluations-gpt45",
        "git_table_result": "results_gpt_45.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "mistral-small-2503": {
        "evaluation_folder": "evaluations-mistral-small",
        "git_table_result": "alt_results_mistral-small-2503.md",
        "evaluation_api_url": "https://api.mistral.ai/v1/",
        "api_key": open("../api_mistral.txt", "r").read().strip(),
    },
    "gpt-4o": {
        "evaluation_folder": "evaluations-gpt4o",
        "git_table_result": "alt_results_gpt_4o.md",
        "evaluation_api_url": "https://api.x.ai/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    },
    "grok-2": {
        "evaluation_folder": "evaluations-grok2",
        "git_table_result": "alt_results_grok2.md",
        "evaluation_api_url": "https://api.x.ai/v1/",
        "api_key": open("../api_grok.txt", "r").read().strip(),
    },
    "gemini-2.0-flash": {
        "evaluation_folder": "evaluations-gemini2-flash",
        "git_table_result": "alt_results_gemini2_flash.md",
        "evaluation_api_url": "https://generativelanguage.googleapis.com/v1beta/",
        "api_key": open("../api_google.txt", "r").read().strip(),
    },
    "claude-3-5-sonnet-20241022": {
        "evaluation_folder": "evaluations-claude-35-sonnet",
        "git_table_result": "alt_results_claude-35-sonnet.md",
        "evaluation_api_url": "https://api.anthropic.com/v1/",
        "api_key": open("../api_anthropic.txt", "r").read().strip(),
    },
    "gpt-4.1-mini": {
        "evaluation_folder": "evaluations-gpt41-mini",
        "git_table_result": "alt_results_gpt41-mini.md",
        "evaluation_api_url": "https://api.openai.com/v1/",
        "api_key": open("../api_openai.txt", "r").read().strip(),
    }
}


def get_evaluation_folder(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["evaluation_folder"]
    elif "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_folder"]
    elif "gpt-4o" in evaluating_model_name:
        return ALL_JUDGES["gpt-4o"]["evaluation_folder"]
    elif "grok-2" in evaluating_model_name:
        return ALL_JUDGES["grok-2"]["evaluation_folder"]
    elif "gemini-2.0-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.0-flash"]["evaluation_folder"]
    elif "claude-3-5-sonnet" in evaluating_model_name:
        return ALL_JUDGES["claude-3-5-sonnet-20241022"]["evaluation_folder"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["evaluation_folder"]


def get_git_table_result(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["git_table_result"]
    elif "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["git_table_result"]
    elif "gpt-4o" in evaluating_model_name:
        return ALL_JUDGES["gpt-4o"]["git_table_result"]
    elif "grok-2" in evaluating_model_name:
        return ALL_JUDGES["grok-2"]["git_table_result"]
    elif "gemini-2.0-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.0-flash"]["git_table_result"]
    elif "claude-3-5-sonnet" in evaluating_model_name:
        return ALL_JUDGES["claude-3-5-sonnet-20241022"]["git_table_result"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["git_table_result"]


def get_evaluation_api_url(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name or "gpt-4o" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["evaluation_api_url"]
    elif "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["evaluation_api_url"]
    elif "grok-2" in evaluating_model_name:
        return ALL_JUDGES["grok-2"]["evaluation_api_url"]
    elif "gemini-2.0-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.0-flash"]["evaluation_api_url"]
    elif "claude-3-5-sonnet" in evaluating_model_name:
        return ALL_JUDGES["claude-3-5-sonnet-20241022"]["evaluation_api_url"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["evaluation_api_url"]


def get_manual(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name:
        return False
    return False


def get_api_key(evaluating_model_name=None):
    if evaluating_model_name is None:
        evaluating_model_name = EVALUATING_MODEL_NAME

    if "gpt-4.5" in evaluating_model_name or "gpt-4o" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.5-preview"]["api_key"]
    elif "mistral-small-2503" in evaluating_model_name:
        return ALL_JUDGES["mistral-small-2503"]["api_key"]
    elif "grok-2" in evaluating_model_name:
        return ALL_JUDGES["grok-2"]["api_key"]
    elif "gemini-2.0-flash" in evaluating_model_name:
        return ALL_JUDGES["gemini-2.0-flash"]["api_key"]
    elif "claude-3-5-sonnet" in evaluating_model_name:
        return ALL_JUDGES["claude-3-5-sonnet-20241022"]["api_key"]
    elif "gpt-4.1-mini" in evaluating_model_name:
        return ALL_JUDGES["gpt-4.1-mini"]["api_key"]


#EVALUATION_FOLDER = get_evaluation_folder()
#TARGET_GIT_TABLE_RESULT = get_git_table_result()
#EVALUATION_API_URL = get_evaluation_api_url()
#MANUAL = get_manual()
