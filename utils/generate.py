import os
import pyperclip
import subprocess


def read_contents(file_path):
    F = None
    content = None
    try:
        F = open(file_path, "r")
        content = F.read()
    except:
        F = open(file_path, "r", encoding="utf-8")
        content = F.read()
    F.close()
    return content


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
incipits_folder = os.path.join(base_path, "incipits")
answers_folder = os.path.join(base_path, "answers")

incipits = [x for x in os.listdir(incipits_folder) if x.endswith("txt")]

model_name = input("Give me the name of the model that you are testing -> ")

count = input("Start count (0) -> ")
count = count.strip()
if count:
    count = int(count)
else:
    count = 0

while True:
    for inc in incipits:
        dream_path = os.path.join(incipits_folder, inc)

        dream = "You are dreaming. Can you complete the following dream?\n\n"+open(dream_path, "r").read()
        pyperclip.copy(dream)

        response_path = os.path.join(answers_folder, model_name+"__"+inc.split(".")[0]+"__"+str(count)+".txt")
        print(response_path)

        if os.path.exists(response_path):
            try:
                content = read_contents(response_path).strip()
                if content:
                    continue
            except:
                import traceback
                traceback.print_exc()
                F = open(response_path, "w")
                F.close()
        else:
            F = open(response_path, "w")
            F.close()

        subprocess.run(["notepad.exe", response_path])

        #input("Press ENTER to continue")

    prompt = input("Continue? (Y/N)")

    if prompt != "Y":
        break

    count += 1
