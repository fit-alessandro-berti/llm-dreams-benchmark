import os

target = "combined"

def __strip(x):
    return x.lower().replace("-", "").replace(" ", "_")


for x in ["size", "date"]:
    for y in ["MHS", "Anxiety and Stress Levels",
              "Emotional Stability",
              "Problem-solving Skills",
              "Creativity",
              "Interpersonal Relationships",
              "Confidence and Self-efficacy",
              "Conflict Resolution",
              "Work-related Stress",
              "Adaptability",
              "Achievement Motivation",
              "Fear of Failure",
              "Need for Control",
              "Cognitive Load",
              "Social Support",
              "Resilience"]:
        print(x, y)
        target_path = os.path.join(target, x+"__"+__strip(y)+".png")

        command = "python plot_overall_kde.py --x "+x+" --y \""+y+"\" --output \""+target_path+"\""

        os.system(command)
