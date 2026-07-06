import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time as t

# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
model.eval()

# Map options to token IDs
option_tokens = {'A': tokenizer.encode("A", add_special_tokens=False)[0],
                 'B': tokenizer.encode("B", add_special_tokens=False)[0],
                 'C': tokenizer.encode("C", add_special_tokens=False)[0],
                 'D': tokenizer.encode("D", add_special_tokens=False)[0]}

# Keys: ["centerpiece", "options", "correct_options", "correct_options_idx", "correct_options_literal", "subject", "id"]
def eval_mmlu(q): # [probA, probB, probC, probD]
    options = ast.literal_eval(q["options"])
    prompt = f"""
        Question: {q["centerpiece"]}
        A) {options[0]}
        B) {options[1]}
        C) {options[2]}
        D) {options[3]}
        Provide the answer as a single letter (A, B, C, or D).
        """
    print(prompt)
    start = t.time()
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    end = t.time()
        
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens['A']],
                                  last_token_logits[option_tokens['B']],
                                  last_token_logits[option_tokens['C']],
                                  last_token_logits[option_tokens['D']]])
                                  
    return torch.softmax(option_logits, dim=0).tolist() + [end - start]

print("Evaluating mmlu splits")
for i in range(1, 6):
    results = []
    df = pd.read_csv(f"data-splits/mmlu_{i}.csv")
    for n, q in enumerate(df.to_dict("records")):
        if (n + 1) % 100 == 0:
            print(f"Evaluated {n + 1} questions")
        prob = eval_mmlu(q)
        print(f"Correct answer: {q['correct_options_idx'][1]}, Predicted answer: {max(range(4), key=lambda x: prob[x])}, Probabilities: {prob[:4]}")
        results.append({
            "id" : q["id"],
            "subject" : q["subject"],
            "probA" : prob[0],
            "probB" : prob[1],
            "probC" : prob[2],
            "probD" : prob[3],
            "result" : int(q["correct_options_idx"][1]) == max(range(4), key=lambda x: prob[x]),
            "time" : prob[4]
        })