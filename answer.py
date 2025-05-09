import requests
import os
import traceback
import time
import re
import json
from common import ANSWERING_MODEL_NAME as MODEL_NAME


API_URL = "https://api.openai.com/v1/"
#API_URL = "http://127.0.0.1:11434/v1/"
#API_URL = "https://api.deepinfra.com/v1/openai/"
#API_URL = "https://api.mistral.ai/v1/"
#API_URL = "https://api.x.ai/v1/"
#API_URL = "https://generativelanguage.googleapis.com/v1beta/"
#API_URL = "https://api.groq.com/openai/v1/"
#API_URL = "https://api.deepseek.com/"
#API_URL = "https://api.hyperbolic.xyz/v1/"
#API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/"
#API_URL = "https://api.anthropic.com/v1/"
#API_URL = "https://integrate.api.nvidia.com/v1/"
#API_URL = "https://openrouter.ai/api/v1/"
#API_URL = "https://api.perplexity.ai/"

API_KEY = open("api_key.txt", "r").read()

NUMBER_EXECUTIONS = 2

WAITING_TIME_RETRY = 15

incipits = [x for x in os.listdir("incipits") if x.endswith("txt")]

m_name = MODEL_NAME.replace("/", "").replace(":", "")


def write_answer(response_message, answer_path):
    if not os.path.exists(answer_path):
        response_message = perform_query(
            "You are dreaming. Can you complete the following dream?\n\n" + open(dream_path, "r").read())

        if response_message:
            try:
                F = open(answer_path, "w")
                F.write(response_message)
            except:
                F = open(answer_path, "w", encoding="utf-8")
                F.write(response_message)
            F.close()


def strip_non_unicode_characters(text):
    # Define a pattern that matches all valid Unicode characters.
    pattern = re.compile(r'[^\u0000-\uFFFF]', re.UNICODE)
    # Replace characters not matching the pattern with an empty string.
    cleaned_text = pattern.sub('', text)
    cleaned_text = cleaned_text.encode('cp1252', errors='ignore').decode('cp1252')

    return cleaned_text


def perform_query_google_api(text):
    complete_url = API_URL + "models/" + MODEL_NAME + ":generateContent?key=" + API_KEY

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

    response_message = ""
    response = None

    while not response_message:
        try:
            response = requests.post(complete_url, headers=headers, json=payload).json()
            response_message = strip_non_unicode_characters(response["candidates"][0]["content"]["parts"][0]["text"])
            return response_message
        except:
            print(response)
            traceback.print_exc()
            print("sleeping %d seconds ..." % (WAITING_TIME_RETRY))
            time.sleep(WAITING_TIME_RETRY)


def perform_query_new_openai_api(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "input": text
    }

    complete_url = API_URL + "responses"

    response = requests.post(complete_url, headers=headers, json=payload)
    if response.status_code != 200:
        print(response)
        print(response.status_code)
        print(response.text)

    response = response.json()
    return response["output"][-1]["content"][0]["text"]


def perform_query_openai_api(text):
    messages = [{"role": "user",
                 "content": text}]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096
    }

    complete_url = API_URL + "chat/completions"

    response_message = ""

    while not response_message:
        try:
            streaming_enabled = True  # Example usage

            if streaming_enabled:
                payload["stream"] = True
                response_message = None
                response_message = ""
                chunk_count = 0

                # We add stream=True to requests so we can iterate over chunks
                with requests.post(complete_url, headers=headers, json=payload, stream=True) as resp:
                    #print(resp)
                    #print(resp.status_code)
                    #print(resp.text)
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        decoded_line = line.decode("utf-8")

                        # OpenAI-style streaming lines begin with "data: "
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[len("data: "):].strip()
                            if data_str == "[DONE]":
                                # End of stream
                                break
                            data_json = json.loads(data_str)
                            if "choices" in data_json:
                                # Each chunk has a delta with partial content
                                chunk_content = data_json["choices"][0]["delta"].get("content", "")
                                if chunk_content:
                                    response_message += chunk_content
                                    chunk_count += 1
                                    #print(chunk_count)
                                    if chunk_count % 10 == 0:
                                        #print(chunk_count, len(response_message), response_message.replace("\n", " ").replace("\r", "").strip())
                                        #print(chunk_count, len(response_message))
                                        pass
        except:
            traceback.print_exc()

            print("sleeping %d seconds ..." % (WAITING_TIME_RETRY))

            time.sleep(WAITING_TIME_RETRY)

        return response_message.split("</think>")[-1].split("</thought>")[-1]

def perform_query_anthropic_api(question):
    ANTHROPIC_THINKING_TOKENS = None

    complete_url = API_URL + "messages"

    messages = [{"role": "user", "content": question}]

    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "output-128k-2025-02-19",
        "x-api-key": API_KEY
    }

    payload = {
        "model": MODEL_NAME,
        "max_tokens": 4096,
        "messages": messages
    }

    if ANTHROPIC_THINKING_TOKENS is not None:
        payload["thinking"] = {"type": "enabled", "budget_tokens": ANTHROPIC_THINKING_TOKENS}
        payload["max_tokens"] += ANTHROPIC_THINKING_TOKENS
        payload["max_tokens"] = min(128000, payload["max_tokens"])
        print(payload)

    payload["stream"] = True
    response_message = ""
    chunk_count = 0

    # Make a streaming POST request
    with requests.post(complete_url, headers=headers, json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            # Decode the line
            decoded_line = line.decode("utf-8").strip()

            # Optionally check for a stream end marker (Anthropic may send "[DONE]")
            if "message_stop" in decoded_line:
                break

            if "message_start" in decoded_line:
                continue

            try:
                decoded_line = decoded_line.split("data: ")[-1].strip()
                if "text" in decoded_line:
                    chunk = decoded_line.split('"text":"')[-1].split('"')[0].replace("\\n", "\n")
                    response_message += chunk
                    chunk_count += 1
                    #print(chunk_count)

                    # You could add logging or progress updates here if desired
                    if chunk_count % 10 == 0:
                        #print(chunk_count, len(response_message), response_message)
                        pass

            except json.JSONDecodeError:
                # Skip any malformed lines
                traceback.print_exc()
                continue

    return response_message


def perform_query(text):
    if "api.openai" in API_URL:
        return perform_query_new_openai_api(text)
    elif "googleapis" in API_URL:
        return perform_query_google_api(text)
    elif "anthropic" in API_URL:
        return perform_query_anthropic_api(text)
    else:
        return perform_query_openai_api(text)


for i in range(NUMBER_EXECUTIONS):
    while True:
        try:
            for index, incipit in enumerate(incipits):
                print(incipit+" (ex. %d of %d) (incipit %d of %d)" % (i+1, NUMBER_EXECUTIONS, index+1, len(incipits)))

                dream_path = os.path.join("incipits", incipit)
                answer_path = os.path.join("answers", m_name+"__"+incipit.split(".")[0]+"__"+str(i)+".txt")

                if not os.path.exists(answer_path):
                    response_message = perform_query(
                        "You are dreaming. Can you complete the following dream?\n\n" + open(dream_path, "r").read())

                    write_answer(response_message, answer_path)
            break
        except:
            traceback.print_exc()
            time.sleep(WAITING_TIME_RETRY)
