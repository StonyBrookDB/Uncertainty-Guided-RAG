"""
Benchmark: 57.32/68.18 MedMCQA/MedQA
https://openreview.net/pdf?id=ri3Si3GBOm

1 error, not enough tokens for CoT
"""

# region imports, models, constants
import os
import re
import torch
import pandas as pd
import time as t
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
model = torch.compile(model, mode="reduce-overhead")
model.eval()

option_tokens = {
    "A": tokenizer.encode("A", add_special_tokens=False)[0],
    "B": tokenizer.encode("B", add_special_tokens=False)[0],
    "C": tokenizer.encode("C", add_special_tokens=False)[0],
    "D": tokenizer.encode("D", add_special_tokens=False)[0],
    "E": tokenizer.encode("E", add_special_tokens=False)[0],
}

MAX_NEW_TOKENS = 512  # generous budget for CoT reasoning before the final answer

errors = 0
# endregion


def _run_model_cot(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Runs full generation (CoT) and returns the generated token ids/text,
    per-step logits (scores), and elapsed time."""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    start = t.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    end = t.time()

    generated_ids = outputs.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    scores = outputs.scores  # tuple of length num_generated_tokens, each [1, vocab]

    return generated_ids, generated_text, scores, end - start


def _extract_answer_letter_and_logits(generated_ids, scores):
    global errors

    for i in range(len(generated_ids) - 1, -1, -1):
        tok_text = tokenizer.decode([generated_ids[i]], skip_special_tokens=True).strip()
        if tok_text in option_tokens:
            return tok_text, scores[i][0]

    errors += 1
    return None, None


def _probs_from_logits(logits, valid_letters):
    option_logits = [logits[option_tokens[letter]].item() for letter in valid_letters]
    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
    return probs


# returns [[probA, probB, probC, probD], time, confident flag, cot_text, answer_letter]
def eval_medmcqa(q):
    messages = [
        {
            "role": "system",
            "content": (
                "You are answering a multiple-choice medical question. "
                "Think through the question step by step, briefly explaining your "
                "reasoning. After your reasoning, on a new final line, give your "
                "answer in EXACTLY this format: 'Answer: <letter>' where <letter> "
                "is one of A, B, C, or D. Do not include anything after that line."
            ),
        },
        {
            "role": "user",
            "content": f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}

Think step by step, then finish with 'Answer: <letter>'.""",
        },
    ]

    generated_ids, generated_text, scores, elapsed = _run_model_cot(messages)
    answer_letter, answer_logits = _extract_answer_letter_and_logits(generated_ids, scores)

    confident = answer_letter is not None
    if confident:
        probs = _probs_from_logits(answer_logits, ["A", "B", "C", "D"])
    else:
        print(f"Warning: could not parse an answer letter from generated text:\n{generated_text}")
        probs = [0.25, 0.25, 0.25, 0.25]

    return [probs, elapsed, confident]


# returns [[probA, probB, probC, probD, probE], time, confident flag, cot_text, answer_letter]
def eval_medqa(q):
    messages = [
        {
            "role": "system",
            "content": (
                "You are answering a multiple-choice medical question. "
                "Think through the question step by step, briefly explaining your "
                "reasoning. After your reasoning, on a new final line, give your "
                "answer in EXACTLY this format: 'Answer: <letter>' where <letter> "
                "is one of A, B, C, D, or E. Do not include anything after that line."
            ),
        },
        {
            "role": "user",
            "content": f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
    E) {q["e"]}

Think step by step, then finish with 'Answer: <letter>'.""",
        },
    ]

    generated_ids, generated_text, scores, elapsed = _run_model_cot(messages)
    answer_letter, answer_logits = _extract_answer_letter_and_logits(generated_ids, scores)

    confident = answer_letter is not None
    if confident:
        probs = _probs_from_logits(answer_logits, ["A", "B", "C", "D", "E"])
    else:
        print(f"Warning: could not parse an answer letter from generated text:\n{generated_text}")
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    return [probs, elapsed, confident]


def write(res, OUT_FILE):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", header=not file_exists, index=False)


# Warmup
print("Warming up model")
warmup_messages = [
    {
        "role": "system",
        "content": (
            "You are answering a multiple-choice question. Think step by step, "
            "then finish with 'Answer: <letter>'."
        ),
    },
    {
        "role": "user",
        "content": """Question: What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Madrid

Think step by step, then finish with 'Answer: <letter>'.""",
    },
]
for _ in range(5):
    _run_model_cot(warmup_messages, max_new_tokens=32)
torch.cuda.synchronize()

# MEDMCQA
print("Evaluating MedMCQA test")
results = []
df = pd.read_csv("question-set/medmcqa_1000.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time_taken, confident = eval_medmcqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "answer": q["answer"],
        "time": time_taken,
        "error": not confident,
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
    probs, time_taken, confident = eval_medqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "probE": probs[4],
        "answer": q["answer"],
        "time": time_taken,
        "error": not confident,
    })
    if i % 100 == 0 and i != 0:
        write(results, "medqa_results.csv")
        results.clear()

if results:
    write(results, "medqa_results.csv")
    results.clear()

print(f"Total errors: {errors}")