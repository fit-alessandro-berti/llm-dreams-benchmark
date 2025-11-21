import os


def do_renaming(base_path, original_name, novel_name):
    files = [x for x in os.listdir(base_path) if x.startswith(original_name)]

    for f in files:
        original_path = os.path.join(base_path, f)
        new_path = original_path.replace(original_name, novel_name)

        print(original_path, new_path, os.path.exists(original_path), os.path.exists(new_path))
        os.rename(original_path, new_path)


if __name__ == "__main__":
    original_name = "Gpt-5.1-20251120_"
    novel_name = "Gpt-5.1-Pro-20251120_"

    if not original_name.endswith("_"):
        raise Exception("error")

    if not novel_name.endswith("_"):
        raise Exception("error")

    base_path = ".."

    answer_directory = os.path.join(base_path, "answers")
    do_renaming(answer_directory, original_name, novel_name)

    evaluation_directories = [x for x in os.listdir(base_path) if "evaluations-" in x]

    for x in evaluation_directories:
        ev_dir = os.path.join(base_path, x)

        do_renaming(ev_dir, original_name, novel_name)
