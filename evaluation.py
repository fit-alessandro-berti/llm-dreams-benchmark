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

NUMBER_EXECUTIONS = 2

WAITING_TIME_RETRY = 17


class Shared:
    answering_model_name = common.ANSWERING_MODEL_NAME
    evaluating_model_name = common.EVALUATING_MODEL_NAME
    evaluation_folder = None
    api_url = None
    manual = None
    api_key = None


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


def get_evaluation_google(text):
    complete_url = Shared.api_url + "models/" + Shared.evaluating_model_name + ":generateContent?key=" + Shared.api_key

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

    if "gemini-2.5" in Shared.evaluating_model_name:
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


def get_evaluation_openai(text):
    messages = [{"role": "user", "content": text}]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Shared.api_key}"
    }

    payload = {
        "model": Shared.evaluating_model_name,
        "messages": messages,
    }

    complete_url = Shared.api_url + "chat/completions"

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


def get_evaluation_openai_new(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Shared.api_key}"
    }

    payload = {
        "model": Shared.evaluating_model_name,
        "input": text
    }

    if "gpt-5.1" in Shared.evaluating_model_name:
        pass
    elif "gpt-5" in Shared.evaluating_model_name:
        payload["reasoning"] = {"effort": "minimal"}

    complete_url = Shared.api_url + "responses"

    response = requests.post(complete_url, headers=headers, json=payload)
    if response.status_code != 200:
        print(response)
        print(response.status_code)
        print(response.text)

    response = response.json()
    response_message = response["output"][-1]["content"][0]["text"]
    response_message_json = interpret_response(response_message)

    return response_message_json


def get_evaluation_anthropic(text):
    complete_url = Shared.api_url + "messages"

    messages = [{"role": "user", "content": text}]

    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "output-128k-2025-02-19",
        "x-api-key": Shared.api_key
    }

    payload = {
        "model": Shared.evaluating_model_name,
        "max_tokens": 4096,
        "messages": messages
    }

    resp = requests.post(complete_url, headers=headers, json=payload)
    resp = resp.json()
    response_message = resp["content"][-1]["text"]

    return interpret_response(response_message)


def get_evaluation(text):
    if "api.openai" in Shared.api_url:
        return get_evaluation_openai_new(text)
    elif "googleapis" in Shared.api_url:
        return get_evaluation_google(text)
    elif "anthropic" in Shared.api_url:
        return get_evaluation_anthropic(text)
    else:
        return get_evaluation_openai(text)


def perform_evaluation(answering_model_name, massive):
    ret = False
    m_name = answering_model_name.replace("/", "").replace(":", "")

    #answers = [x for x in os.listdir("answers") if x.split("__")[0].startswith(m_name)]
    #ex_indexes = sorted(list(set(x.split("__")[-1] for x in answers)))

    answers = None

    ex_indexes = ["0.txt", "1.txt"]
    max_lenn = 0

    base_evaluation_path = Shared.evaluation_folder + "/" + m_name + "__"

    for i in range(NUMBER_EXECUTIONS):
        Is = str(i)
        for idxnum, index in enumerate(ex_indexes):
            this_suffix = str(idxnum) + "__" + Is + ".txt"
            evaluation_path = base_evaluation_path + this_suffix

            if not os.path.exists(evaluation_path):
                if answers is None:
                    answers = [x for x in os.listdir("answers") if x.startswith(m_name)]

                this_answers = [x for x in answers if x.split("__")[-1] == index]
                all_contents = [
                    "A person did the following dreams. I ask you to estimate the personality trait of this person. The final output should be a JSON containing the following keys: 'Anxiety and Stress Levels', 'Emotional Stability', 'Problem-solving Skills', 'Creativity', 'Interpersonal Relationships', 'Confidence and Self-efficacy', 'Conflict Resolution', 'Work-related Stress', 'Adaptability', 'Achievement Motivation', 'Fear of Failure', 'Need for Control', 'Cognitive Load', 'Social Support', 'Resilience'. Each key should be associated to a number from 1.0 (minimum score) to 10.0 (maximum score). Please follow strictly the provided JSON schema in the evaluation!"]

                for answ in this_answers:
                    incipit = open(os.path.join("incipits", answ.split("__")[1] + ".txt"), "r",
                                   encoding="utf-8").read()

                    try:
                        content = incipit + " " + open(os.path.join("answers", answ), "r").read().replace("\n",
                                                                                                          " ").replace(
                            "\r",
                            " ")
                    except:
                        content = incipit + " " + open(os.path.join("answers", answ), "r",
                                                       encoding="utf-8").read().replace(
                            "\n", " ").replace("\r", " ")

                    max_lenn = max(max_lenn, len(content))
                    all_contents.append(content)

                print("(evaluation %d of %d) (answers %d of %d)" % (
                    i + 1, NUMBER_EXECUTIONS, idxnum + 1, len(ex_indexes)), answering_model_name,
                      Shared.evaluating_model_name)

                all_contents = "\n\n".join(all_contents)

                response_message_json = None
                if not Shared.manual:
                    response_message_json = get_evaluation(all_contents)
                else:
                    msg_len = len(all_contents)

                    if msg_len < sys.maxsize:
                        pyperclip.copy(all_contents)

                        temp_file = NamedTemporaryFile(suffix=".txt")
                        temp_file.close()
                        F = open(temp_file.name, "w")
                        F.close()
                        subprocess.run(["notepad.exe", temp_file.name])

                        F = open(temp_file.name, "r")
                        response_message = F.read().strip()
                        F.close()

                        response_message_json = interpret_response(response_message)
                        #print(response_message_json)
                if response_message_json:
                    json.dump(response_message_json, open(evaluation_path, "w"))
                    ret = True
            else:
                if not massive:
                    print("ALREADY DONE (evaluation %d of %d) (answers %d of %d)" % (
                        i + 1, NUMBER_EXECUTIONS, idxnum + 1, len(ex_indexes)), answering_model_name,
                          Shared.evaluating_model_name)
    return ret


def main_execution(evaluating_model_name, massive):
    Shared.evaluating_model_name = evaluating_model_name

    Shared.evaluation_folder = common.get_evaluation_folder(Shared.evaluating_model_name)
    Shared.api_url = common.get_evaluation_api_url(Shared.evaluating_model_name)
    Shared.manual = common.get_manual(Shared.evaluating_model_name)
    Shared.api_key = common.get_api_key(Shared.evaluating_model_name)

    if not os.path.exists(Shared.evaluation_folder):
        os.mkdir(Shared.evaluation_folder)

    if massive:
        cont = True
        while cont:
            try:
                cont = False
                available_models = {x.split("__")[0] for x in os.listdir("answers") if not "_init_" in x}
                for m in available_models:
                    xy = perform_evaluation(m, massive)
                    cont = cont or xy
            except:
                traceback.print_exc()
                time.sleep(WAITING_TIME_RETRY)
                cont = True
    else:
        perform_evaluation(Shared.answering_model_name, massive)


if __name__ == "__main__":
    aa = time.time_ns()

    massive = False if len(sys.argv) > 1 and sys.argv[1] == "0" else True

    if massive:
        model_list = list(common.ALL_JUDGES)
        model_list.sort(key=lambda x: (-1 if "gpt-4.5" in x else 0 if "mistral" not in x else 1, x))
    else:
        model_list = ["gpt-4.1"]

    for m in model_list:
        main_execution(m, massive)

    bb = time.time_ns()

    print("total execution time", (bb-aa)/10**9)
