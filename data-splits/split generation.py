import pandas as pd
from datasets import load_dataset
from collections import defaultdict

# 20 questions over 20 subjects, 5 splits
dataset = load_dataset("openlifescienceai/medmcqa", split="train")
shuffled_dataset = dataset.shuffle(seed=12)

questions = defaultdict(list)

for q in shuffled_dataset:
    if (sum(len(v) for v in questions.values()) > 2000):
        break
    if len(questions[q["subject_name"]]) < 100 and q["subject_name"] != "Unknown":
        questions[q["subject_name"]].append(q)
    

for i in range(5):
    # File generation
    split = [q for subject, qs in questions.items() for q in qs[i*20:(i+1)*20]]
    df_split = pd.DataFrame(split, columns=["id", "question", "opa", "opb", "opc", "opd", "cop", "choice_type", "exp", "subject_name", "topic_name"])
    df_split.to_csv(f"medmcaq_{i+1}.csv", index=False)

    # Check questions per subject
    subject_counts = df_split["subject_name"].value_counts()
    print(f"Split {i+1} subject counts:\n{subject_counts}\n")

# 20 questions over 12 subjects, 5 splits
subjects = [
    "anatomy",
    "clinical-knowledge",
    "college-biology",
    "college-chemistry",
    "college-medicine",
    "high-school-biology",
    "human-aging",
    "medical-genetics",
    "nutrition",
    "professional-medicine",
    "professional-psychology",
    "virology"]
questions = [[] for i in range(5)]

for s in subjects:
    dataset = load_dataset("ekacare/mmlu-medical-mcqs-evaluation-dataset", s, split = "test")
    shuffled_dataset = dataset.shuffle(seed=12)
    for i in range(90):
        questions[i % 5].append(shuffled_dataset[i])
        questions[i % 5][-1]["subject"] = s
        questions[i % 5][-1]["id"] = i

for i in range(5):
    # File generation
    df_split = pd.DataFrame(questions[i], columns=["centerpiece", "options", "correct_options", "correct_options_idx", "correct_options_literal", "subject", "id"])
    df_split.to_csv(f"mmlu_{i+1}.csv", index=False)

    # Check questions per subject
    subject_counts = df_split["subject"].value_counts()
    print(f"Split {i+1} subject counts:\n{subject_counts}\n")