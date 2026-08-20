"""
Benchmark: 54.69/54.74 MedMCQA/MedQA
https://openreview.net/pdf?id=ri3Si3GBOm
"""

# region imports, models, constants
import os
import torch
import pandas as pd
import time as t
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
model.eval()

option_tokens = {
    "A": tokenizer.encode("A", add_special_tokens=False)[0],
    "B": tokenizer.encode("B", add_special_tokens=False)[0],
    "C": tokenizer.encode("C", add_special_tokens=False)[0],
    "D": tokenizer.encode("D", add_special_tokens=False)[0],
    "E": tokenizer.encode("E", add_special_tokens=False)[0],
}

errors = 0
# endregion

def _run_model(messages):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    start = t.time()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    end = t.time()

    last_token_logits = outputs.logits[0, -1, :]
    return inputs, last_token_logits, end - start


def _check_confident(inputs, last_token_logits):
    global errors
    next_token_id = torch.argmax(last_token_logits).item()
    if next_token_id not in option_tokens.values():
        next_token_text = tokenizer.decode(next_token_id)
        print(f"Warning: model's top token is not an option: {next_token_text}")
        errors += 1
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=100)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
        return False
    return True


# returns [[probA, probB, probC, probD], time, confident flag]
def eval_medmcqa(q):
    messages = [
        {"role": "system", "content": "Output your answer as a single letter (A, B, C, D). No explanation"},
        {"role": "user", "content": f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
Answer: """}
    ]

    inputs, last_token_logits, elapsed = _run_model(messages)
    confident = _check_confident(inputs, last_token_logits)

    option_logits = [
        last_token_logits[option_tokens["A"]].item(),
        last_token_logits[option_tokens["B"]].item(),
        last_token_logits[option_tokens["C"]].item(),
        last_token_logits[option_tokens["D"]].item(),
    ]

    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
    return [probs, elapsed, confident]


# returns [[probA, probB, probC, probD, probE], time, confident flag]
def eval_medqa(q):
    messages = [
        {"role": "system", "content": "Output your answer as a single letter (A, B, C, D, E). No explanation"},
        {"role": "user", "content": f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
    E) {q["e"]}
Answer: """}
    ]

    inputs, last_token_logits, elapsed = _run_model(messages)
    confident = _check_confident(inputs, last_token_logits)

    option_logits = [
        last_token_logits[option_tokens["A"]].item(),
        last_token_logits[option_tokens["B"]].item(),
        last_token_logits[option_tokens["C"]].item(),
        last_token_logits[option_tokens["D"]].item(),
        last_token_logits[option_tokens["E"]].item(),
    ]

    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
    return [probs, elapsed, confident]

def write(res, OUT_FILE):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", header=not file_exists, index=False)


# Warmup
print("Warming up model")
warmup_messages = [
    {"role": "system", "content": "Output your answer as a single letter (A, B, C, D). No explanation"},
    {"role": "user", "content": """Question: What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Madrid
Answer: """}
]
for _ in range(5):
    _run_model(warmup_messages)
torch.cuda.synchronize()

# # MEDMCQA Test
# print("Evaluating MedMCQA test")
# results = []
# df = pd.read_csv("question-set/medmcqa_20.csv")
# for i, q in enumerate(df.to_dict("records")):
#     probs, time, confident = eval_medmcqa(q)
#     results.append({
#         "id": q["id"],
#         "source": q["source"],
#         "probA": probs[0],
#         "probB": probs[1],
#         "probC": probs[2],
#         "probD": probs[3],
#         "answer": q["answer"],
#         "time": time,
#         "error": not confident
#     })
#     print(results[-1])
#     if i % 100 == 0 and i != 0:
#         write(results, "medmcqa_results.csv")
#         results.clear()
#
# if results:
#     write(results, "medmcqa_results.csv")
#     results.clear()

# MEDMCQA
print("Evaluating MedMCQA test")
results = []
df = pd.read_csv("question-set/medmcqa_1000.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time, confident = eval_medmcqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "answer": q["answer"],
        "time": time,
        "error": not confident
    })
    if i % 100 == 0 and i != 0:
        write(results, "medmcqa_results.csv")
        results.clear()

if results:
    write(results, "medmcqa_results.csv")
    results.clear()

# MEDQA
print("Evaluating MedQA test")
results = []
df = pd.read_csv("question-set/medqa_1273.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time, confident = eval_medqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "probE": probs[4],
        "answer": q["answer"],
        "time": time,
        "error": not confident
    })
    if i % 100 == 0 and i != 0:
        write(results, "medqa_results.csv")
        results.clear()

if results:
    write(results, "medqa_results.csv")
    results.clear()

print(f"Total errors: {errors}")
