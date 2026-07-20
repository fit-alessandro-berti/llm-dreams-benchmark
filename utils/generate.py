import os
import pyperclip
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file_utils import read_file_with_fallback


def read_contents(file_path):
    return read_file_with_fallback(file_path)


def open_text_editor(file_path):
    configured_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if configured_editor:
        subprocess.run(shlex.split(configured_editor) + [file_path])
        return

    if sys.platform.startswith("linux"):
        editor_candidates = ["mousepad", "xdg-open"]
    elif os.name == "nt":
        editor_candidates = ["notepad++.exe", "notepad.exe"]
    else:
        editor_candidates = ["open"]

    for editor in editor_candidates:
        if shutil.which(editor):
            subprocess.run([editor, file_path])
            return

    raise RuntimeError(
        "No supported text editor found. Install mousepad on Linux, "
        "Notepad++/Notepad on Windows, or set VISUAL/EDITOR."
    )


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

        dream = "You are dreaming. Can you complete the following dream?\n\n" + read_file_with_fallback(dream_path)
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

        open_text_editor(response_path)

        #input("Press ENTER to continue")

    prompt = input("Continue? (Y/N)")

    if prompt != "Y":
        break

    count += 1
