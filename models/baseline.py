"""
v1: 0 errrors
v2: 3 errors
v3: 1 error, good catch
v1 -> instructions
v2 -> no instructions
v3 -> chat template
"""

# region Imports
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time as t
# endregion

# region Models
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
model.eval()
# endregion

# region Constants
option_tokens = {"A": tokenizer.encode("A", add_special_tokens=False)[0],
                 "B": tokenizer.encode("B", add_special_tokens=False)[0],
                 "C": tokenizer.encode("C", add_special_tokens=False)[0],
                 "D": tokenizer.encode("D", add_special_tokens=False)[0]}

INSTRUCTIONS = """You are a helpful medical expert, and your task is to answer a multi-choice medical question. 
The question is provided below, along with four answer options labeled A, B, C, and D. 
Your goal is to select the most appropriate answer based on your medical knowledge and reasoning, as well as any additional context provided. """
errors = 0
# endregion

# Keys: ["id", "question", "opa", "opb", "opc", "opd", "cop", "choice_type", "exp", "subject_name", "topic_name"]
def eval_medmcqa(q):

    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"""Question: {q["question"]}
    A) {q["opa"]}
    B) {q["opb"]}
    C) {q["opc"]}
    D) {q["opd"]}
Answer only with the letter of the correct option. Answer: """}
    ]
    
    start = t.time()
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt = True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    end = t.time()
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens["A"]],
                                  last_token_logits[option_tokens["B"]],
                                  last_token_logits[option_tokens["C"]],
                                  last_token_logits[option_tokens["D"]]])

    # Safety Check: 
    next_token_id = torch.argmax(last_token_logits).item()
    next_token_text = tokenizer.decode(next_token_id)
    if next_token_id not in option_tokens.values():
        print(f"Error, next token is not an option: {next_token_text}")
        global errors
        errors += 1

        # Print model response
        with torch.no_grad():
            generated = model.generate(**inputs,max_new_tokens=100)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))

    return torch.softmax(option_logits, dim=0).tolist() + [end - start]

# Keys: ["centerpiece", "options", "correct_options", "correct_options_idx", "correct_options_literal", "subject", "id"]
def eval_mmlu(q):
    options = ast.literal_eval(q["options"])
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"""Question: {q["centerpiece"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}
Answer only with the letter of the correct option. Answer: """
        }
    ]
    start = t.time()
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt = True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    end = t.time()
        
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens["A"]],
                                  last_token_logits[option_tokens["B"]],
                                  last_token_logits[option_tokens["C"]],
                                  last_token_logits[option_tokens["D"]]])

    # Safety Check: 
    next_token_id = torch.argmax(last_token_logits).item()
    next_token_text = tokenizer.decode(next_token_id)
    if next_token_id not in option_tokens.values():
        print(f"Error, next token is not an option: {next_token_text}")
        global errors
        errors += 1

        with torch.no_grad():
            generated = model.generate(**inputs,max_new_tokens=100)
            print(tokenizer.decode(generated[0], skip_special_tokens=True))
                                  
    return torch.softmax(option_logits, dim=0).tolist() + [end - start]

# Warmup
print("Warming up model")
warmup_prompt = """
    Question: What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Madrid
    Provide the answer as a single letter (A, B, C, or D).
"""

warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    for _ in range(5):
        _ = model(**warmup_inputs, return_dict=True)
torch.cuda.synchronize()

# MEDMCQA
print("Evaluating medmcqa splits")
for i in range(1, 6):
    results = []
    df = pd.read_csv(f"data-splits/medmcqa_{i}.csv")
    for n, q in enumerate(df.to_dict("records")):
        if (n + 1) % 100 == 0:
            print(f"Evaluated {n + 1} questions")
        prob = eval_medmcqa(q)
        results.append({
            "id" : q["id"],
            "subject" : q["subject_name"],
            "probA" : prob[0],
            "probB" : prob[1],
            "probC" : prob[2],
            "probD" : prob[3],
            "result" : q["cop"] == max(range(4), key=lambda x: prob[x]),
            "answer" : ["A", "B", "C", "D"][q["cop"]],
            "time" : prob[4],
            "split" : f"medmcqa_{i}"
        })
    df_split = pd.DataFrame(results, columns=["id", "subject", "probA", "probB", "probC", "probD", "result", "answer", "time", "split"])
    df_split.to_csv(f"medmcqa_results_{i}.csv", index=False)

# MMLU
print("Evaluating mmlu splits")
for i in range(1, 6):
    results = []
    df = pd.read_csv(f"data-splits/mmlu_{i}.csv")
    for n, q in enumerate(df.to_dict("records")):
        if (n + 1) % 100 == 0:
            print(f"Evaluated {n + 1} questions")
        prob = eval_mmlu(q)
        results.append({
            "id" : q["id"],
            "subject" : q["subject"],
            "probA" : prob[0],
            "probB" : prob[1],
            "probC" : prob[2],
            "probD" : prob[3],
            "result" : int(q["correct_options_idx"][1]) == max(range(4), key=lambda x: prob[x]),
            "answer" : ["A", "B", "C", "D"][int(q["correct_options_idx"][1])],
            "time" : prob[4],
            "split" : f"mmlu_{i}"
        })
    df_split = pd.DataFrame(results, columns=["id", "subject", "probA", "probB", "probC", "probD", "result", "answer", "time", "split"])
    df_split.to_csv(f"mmlu_results_{i}.csv", index=False)
print(f"Total errors: {errors}")