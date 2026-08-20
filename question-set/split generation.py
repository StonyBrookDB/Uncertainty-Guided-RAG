"""
https://github.com/medmcqa/medmcqa 1000 subject-stratified test set
https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MedQA.md USMLE test set, dropped all questions with answer as choice
"""

import pandas as pd
from datasets import load_dataset
import json
import math

EXTRACT = ["question","a", "b", "c", "d", "answer", "source", "id"]
ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# ========== MedQA conversion ==========
with open("question-set/medqa_raw.jsonl", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

records = []
for line in lines:
    data = json.loads(line)
    opts = data["options"]
    answer_letter = data["answer"]
    answer_idx = ANSWER_MAP[data["answer_idx"]]
    
    record = {
        "question": data.get("question", ""),
        "a": opts.get("A", ""),
        "b": opts.get("B", ""),
        "c": opts.get("C", ""),
        "d": opts.get("D", ""),
        "e": opts.get("E", ""),
        "answer": answer_idx,
        "source": "MEDQA",
        "id": None
    }
    records.append(record)

df = pd.DataFrame(records)
df["id"] = range(len(df))

df.to_csv("medqa_1273.csv", index=False)
print(f"Saved {len(df)} questions to medqa_1273.csv")

# ========== MedMCQA 1000 ==========
print("Loading MedMCQA...")
medmcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
medmcqa_df = pd.DataFrame(medmcqa)

subject_counts = medmcqa_df["subject_name"].value_counts().sort_index()
total = len(medmcqa_df)

targets = {}
fractional = {}
allocated = 0
for subject, count in subject_counts.items():
    exact = count / total * 1000
    targets[subject] = math.floor(exact)
    fractional[subject] = exact - targets[subject]
    allocated += targets[subject]

remaining = 1000 - allocated
for subject in sorted(fractional, key=lambda x: fractional[x], reverse=True)[:remaining]:
    targets[subject] += 1

sampled = []
for subject, target in targets.items():
    subject_df = medmcqa_df[medmcqa_df["subject_name"] == subject]
    sampled.append(subject_df.sample(target))

medmcqa_sampled = pd.concat(sampled).sample(frac=1).reset_index(drop=True)
medmcqa_sampled["source"] = "MEDMCQA"
medmcqa_sampled["id"] = range(1000)
medmcqa_sampled["a"] = medmcqa_sampled["opa"]
medmcqa_sampled["b"] = medmcqa_sampled["opb"]
medmcqa_sampled["c"] = medmcqa_sampled["opc"]
medmcqa_sampled["d"] = medmcqa_sampled["opd"]
medmcqa_sampled["answer"] = medmcqa_sampled["cop"]
medmcqa_sampled["subject"] = medmcqa_sampled["subject_name"]

print(f"Total sampled: {len(medmcqa_sampled)}")
print(f"Per-subject counts:\n{medmcqa_sampled['subject_name'].value_counts().sort_index()}\n")

medmcqa_sampled = medmcqa_sampled[EXTRACT]
medmcqa_sampled.to_csv("medmcqa_1000.csv", index=False)

# ========== MedMCQA 20 ==========
print("Loading MedMCQA...")
medmcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
medmcqa_df = pd.DataFrame(medmcqa)
sample_20 = medmcqa_df.sample(n=20).reset_index(drop=True)
sample_20["source"] = "MEDMCQA"
sample_20["id"] = range(20)
sample_20["a"] = sample_20["opa"]
sample_20["b"] = sample_20["opb"]
sample_20["c"] = sample_20["opc"]
sample_20["d"] = sample_20["opd"]
sample_20["answer"] = sample_20["cop"]
sample_20["subject"] = sample_20["subject_name"]
sample_20 = sample_20[EXTRACT]

sample_20.to_csv("medmcqa_20.csv", index=False)
print(f"Saved {len(sample_20)} questions to medmcqa_20.csv")