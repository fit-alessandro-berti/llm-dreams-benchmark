import requests
import os
import traceback
import time
import re
import json
import pyperclip
import subprocess
import sys
import common
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

NUMBER_EXECUTIONS = 2
MAX_WORKERS = 50

WAITING_TIME_RETRY = 17
EX_INDEXES = ["0.txt", "1.txt"]
EVALUATION_INSTRUCTIONS = ("A person did the following dreams. I ask you to estimate the personality trait of this "
                           "person. The final output should be a JSON containing the following keys: 'Anxiety and "
                           "Stress Levels', 'Emotional Stability', 'Problem-solving Skills', 'Creativity', "
                           "'Interpersonal Relationships', 'Confidence and Self-efficacy', 'Conflict Resolution', "
                           "'Work-related Stress', 'Adaptability', 'Achievement Motivation', 'Fear of Failure', "
                           "'Need for Control', 'Cognitive Load', 'Social Support', 'Resilience'. Each key should be "
                           "associated to a number from 1.0 (minimum score) to 10.0 (maximum score). Please follow "
                           "strictly the provided JSON schema in the evaluation!")


class Shared:
    answering_model_name = common.ANSWERING_MODEL_NAME
    evaluating_model_name = common.EVALUATING_MODEL_NAME
    evaluation_folder = common.get_evaluation_folder(common.EVALUATING_MODEL_NAME)
    api_url = common.get_evaluation_api_url(common.EVALUATING_MODEL_NAME)
    manual = common.get_manual(common.EVALUATING_MODEL_NAME)
    api_key = common.get_api_key(common.EVALUATING_MODEL_NAME)


@dataclass
class EvaluationContext:
    answering_model_name: str
    evaluating_model_name: str
    evaluation_folder: str
    api_url: str
    manual: bool
    api_key: str


def build_context(evaluating_model_name):
    evaluation_folder = common.get_evaluation_folder(evaluating_model_name)

    context = EvaluationContext(
        answering_model_name=common.ANSWERING_MODEL_NAME,
        evaluating_model_name=evaluating_model_name,
        evaluation_folder=evaluation_folder,
        api_url=common.get_evaluation_api_url(evaluating_model_name),
        manual=common.get_manual(evaluating_model_name),
        api_key=common.get_api_key(evaluating_model_name),
    )

    if evaluation_folder and not os.path.exists(evaluation_folder):
        os.mkdir(evaluation_folder)

    return context


def read_file_with_fallback(path):
    try:
        with open(path, "r") as file_handler:
            return file_handler.read()
    except Exception:
        with open(path, "r", encoding="utf-8") as file_handler:
            return file_handler.read()


def strip_non_unicode_characters(text):
    # Define a pattern that matches all valid Unicode characters.
    pattern = re.compile(r'[^\u0000-\uFFFF]', re.UNICODE)
    # Replace characters not matching the pattern with an empty string.
    cleaned_text = pattern.sub('', text)
    cleaned_text = cleaned_text.encode('cp1252', errors='ignore').decode('cp1252')

    return cleaned_text


def __validate(xx):
    keys = ["Anxiety and Stress Levels", "Emotional Stability", "Problem-solving Skills", "Creativity",
            "Interpersonal Relationships", "Confidence and Self-efficacy", "Conflict Resolution",
            "Work-related Stress", "Adaptability", "Achievement Motivation", "Fear of Failure",
            "Need for Control", "Cognitive Load", "Social Support", "Resilience"]

    for key in keys:
        float(xx[key])


def __fix_commas(xx):
    xx = xx.split("\n")
    yy = [xx[0]]
    for i in range(1, len(xx)-1):
        row = xx[i]
        if not row.endswith(",") and i < len(xx)-2:
            row = row + ","
        yy.append(row)
    yy.append(xx[-1])
    return "\n".join(yy)


def __fix_problems(xx):
    keys = ["Anxiety and Stress Levels", "Emotional Stability", "Problem-solving Skills", "Creativity",
            "Interpersonal Relationships", "Confidence and Self-efficacy", "Conflict Resolution",
            "Work-related Stress", "Adaptability", "Achievement Motivation", "Fear of Failure",
            "Need for Control", "Cognitive Load", "Social Support", "Resilience"]

    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[len(s2)]

    result = {}
    xx_keys = list(xx.keys())

    for target_key in keys:
        if not xx_keys:  # If input dictionary is empty
            result[target_key] = None
            continue

        # Find the key in xx with minimum edit distance to target_key
        closest_key = min(xx_keys, key=lambda k: levenshtein_distance(target_key.lower(), k.lower()))
        result[target_key] = xx[closest_key]

    return result

def interpret_response(response_message0):
    response_message = response_message0
    if "```json" in response_message:
        response_message = response_message.split("```json")[-1]

    returned = None
    response_message_json = response_message.split("```")
    for msg in response_message_json:
        try:
            msg = msg.strip()
            xx = json.loads(msg)
            xx = __fix_problems(xx)

            __validate(xx)

            returned = xx
        except:
            pass

    if returned is not None:
        return returned

    response_message = response_message0
    if "{" in response_message and "}" in response_message:
        response_message = response_message.split("{")[-1].split("}")[0]
        response_message = "{\n" + response_message.strip() + "\n}"
        response_message = __fix_commas(response_message)
        response_message = json.loads(response_message)
        response_message = __fix_problems(response_message)
        __validate(response_message)
        return response_message

    raise Exception("Fail")


def get_evaluation_google(text, context=None):
    ctx = context or Shared
    complete_url = ctx.api_url + "models/" + ctx.evaluating_model_name + ":generateContent?key=" + ctx.api_key

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {"parts": [
                {"text": text}
            ]}
        ]
    }

    if "gemini-2.5" in ctx.evaluating_model_name:
        payload["generationConfig"] = {
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }

    response_message = ""
    response = None

    while not response_message:
        try:
            response = requests.post(complete_url, headers=headers, json=payload)
            #print(response)
            #print(response.status_code)
            #print(response.text)
            response = response.json()
            response_message = response["candidates"][0]["content"]["parts"][0]["text"]
            response_message_json = interpret_response(response_message)
            return response_message_json
        except:
            traceback.print_exc()
            print("sleeping %d seconds ..." % (WAITING_TIME_RETRY))
            time.sleep(WAITING_TIME_RETRY)


def get_evaluation_openai(text, context=None):
    ctx = context or Shared
    messages = [{"role": "user", "content": text}]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ctx.api_key}"
    }

    payload = {
        "model": ctx.evaluating_model_name,
        "messages": messages,
    }

    complete_url = ctx.api_url + "chat/completions"

    response_message = ""
    response = None

    while not response_message:
        try:
            response = requests.post(complete_url, headers=headers, json=payload)
            #print(response)
            #print(response.status_code)
            #print(response.text)
            response = response.json()
            response_message = response["choices"][0]["message"]["content"]
            response_message_json = interpret_response(response_message)
            return response_message_json
        except:
            traceback.print_exc()
            print("sleeping %d seconds ..." % (WAITING_TIME_RETRY))
            time.sleep(WAITING_TIME_RETRY)


def get_evaluation_openai_new(text, context=None):
    ctx = context or Shared
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ctx.api_key}"
    }

    payload = {
        "model": ctx.evaluating_model_name,
        "input": text
    }

    if "gpt-5.1" in ctx.evaluating_model_name:
        pass
    elif "gpt-5" in ctx.evaluating_model_name:
        payload["reasoning"] = {"effort": "minimal"}

    complete_url = ctx.api_url + "responses"

    response = requests.post(complete_url, headers=headers, json=payload)
    if response.status_code != 200:
        print(response)
        print(response.status_code)
        print(response.text)

    response = response.json()
    response_message = response["output"][-1]["content"][0]["text"]
    response_message_json = interpret_response(response_message)

    return response_message_json


def get_evaluation_anthropic(text, context=None):
    ctx = context or Shared
    complete_url = ctx.api_url + "messages"

    messages = [{"role": "user", "content": text}]

    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "output-128k-2025-02-19",
        "x-api-key": ctx.api_key
    }

    payload = {
        "model": ctx.evaluating_model_name,
        "max_tokens": 4096,
        "messages": messages
    }

    resp = requests.post(complete_url, headers=headers, json=payload)
    resp = resp.json()
    response_message = resp["content"][-1]["text"]

    return interpret_response(response_message)


def get_evaluation(text, context=None):
    ctx = context or Shared
    if "api.openai" in ctx.api_url:
        return get_evaluation_openai_new(text, ctx)
    elif "googleapis" in ctx.api_url:
        return get_evaluation_google(text, ctx)
    elif "anthropic" in ctx.api_url:
        return get_evaluation_anthropic(text, ctx)
    else:
        return get_evaluation_openai(text, ctx)


def build_all_contents_for_index(answers, index):
    this_answers = [x for x in answers if x.split("__")[-1] == index]
    all_contents = [EVALUATION_INSTRUCTIONS]

    for answ in this_answers:
        incipit_path = os.path.join("incipits", answ.split("__")[1] + ".txt")
        incipit = read_file_with_fallback(incipit_path)

        answer_text = read_file_with_fallback(os.path.join("answers", answ))
        answer_text = answer_text.replace("\n", " ").replace("\r", " ")
        content = incipit + " " + answer_text
        all_contents.append(content)

    return "\n\n".join(all_contents)


def evaluate_single_path(context, answering_model_name, answers, idxnum, index, iteration_index,
                         ex_index_count, evaluation_path):
    prompt = build_all_contents_for_index(answers, index)

    while True:
        try:
            print("(evaluation %d of %d) (answers %d of %d)" % (
                iteration_index + 1, NUMBER_EXECUTIONS, idxnum + 1, ex_index_count), answering_model_name,
                  context.evaluating_model_name)

            response_message_json = None
            if not context.manual:
                response_message_json = get_evaluation(prompt, context)
            else:
                msg_len = len(prompt)

                if msg_len < sys.maxsize:
                    pyperclip.copy(prompt)

                    temp_file = NamedTemporaryFile(suffix=".txt")
                    temp_file.close()
                    with open(temp_file.name, "w") as temp_handler:
                        temp_handler.write("")
                    subprocess.run(["notepad.exe", temp_file.name])

                    with open(temp_file.name, "r") as temp_handler:
                        response_message = temp_handler.read().strip()

                    response_message_json = interpret_response(response_message)

            if response_message_json:
                with open(evaluation_path, "w") as output_file:
                    json.dump(response_message_json, output_file)
                return True
        except Exception:
            traceback.print_exc()

        print("sleeping %d seconds ..." % (WAITING_TIME_RETRY))
        time.sleep(WAITING_TIME_RETRY)


def collect_tasks_for_context(context, massive, ex_indexes, available_models=None):
    tasks = []
    answering_models = available_models if massive else [context.answering_model_name]

    for answering_model_name in answering_models:
        m_name = answering_model_name.replace("/", "").replace(":", "")
        answers = [x for x in os.listdir("answers") if x.startswith(m_name)]
        base_evaluation_path = os.path.join(context.evaluation_folder, m_name + "__")

        for iteration_index in range(NUMBER_EXECUTIONS):
            for idxnum, index in enumerate(ex_indexes):
                evaluation_path = base_evaluation_path + str(idxnum) + "__" + str(iteration_index) + ".txt"

                if os.path.exists(evaluation_path):
                    if not massive:
                        print("ALREADY DONE (evaluation %d of %d) (answers %d of %d)" % (
                            iteration_index + 1, NUMBER_EXECUTIONS, idxnum + 1, len(ex_indexes)), answering_model_name,
                              context.evaluating_model_name)
                    continue

                tasks.append((context, answering_model_name, answers, idxnum, index, iteration_index,
                              len(ex_indexes), evaluation_path))
    return tasks


def dispatch_all_evaluations(model_list, massive):
    available_models = None
    if massive:
        available_models = sorted({x.split("__")[0] for x in os.listdir("answers") if "_init_" not in x})

    tasks = []
    for evaluating_model_name in model_list:
        context = build_context(evaluating_model_name)
        tasks.extend(collect_tasks_for_context(context, massive, EX_INDEXES, available_models))

    if not tasks:
        return

    max_workers = min(MAX_WORKERS, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_path, context, answering_model_name, answers, idxnum, index, iteration_index,
                ex_index_count, evaluation_path
            )
            for context, answering_model_name, answers, idxnum, index, iteration_index, ex_index_count, evaluation_path
            in tasks
        ]

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    aa = time.time_ns()

    massive = False if len(sys.argv) > 1 and sys.argv[1] == "0" else True

    if massive:
        model_list = list(common.ALL_JUDGES)
        model_list.sort(key=lambda x: (-1 if "gpt-4.5" in x else 0 if "mistral" not in x else 1, x))
    else:
        model_list = ["gpt-4.1"]

    dispatch_all_evaluations(model_list, massive)

    bb = time.time_ns()

    print("total execution time", (bb-aa)/10**9)
