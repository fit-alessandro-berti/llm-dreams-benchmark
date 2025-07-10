import os


def do_deletion(base_path, original_name):
    files = [x for x in os.listdir(base_path) if x.startswith(original_name)]

    for f in files:
        original_path = os.path.join(base_path, f)

        print(original_path, os.path.exists(original_path))

        if os.path.exists(original_path):
            os.remove(original_path)


if __name__ == "__main__":
    original_name = "baiduernie-4.5-300b-a47b_"

    if not original_name.endswith("_"):
        raise Exception("error")

    base_path = ".."

    answer_directory = os.path.join(base_path, "answers")
    do_deletion(answer_directory, original_name)

    evaluation_directories = [x for x in os.listdir(base_path) if "evaluations-" in x]

    for x in evaluation_directories:
        ev_dir = os.path.join(base_path, x)

        do_deletion(ev_dir, original_name)
