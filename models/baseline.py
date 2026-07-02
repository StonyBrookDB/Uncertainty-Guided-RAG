import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Map options to token IDs
option_tokens = {'A': tokenizer.encode("A", add_special_tokens=False)[0],
                 'B': tokenizer.encode("B", add_special_tokens=False)[0],
                 'C': tokenizer.encode("C", add_special_tokens=False)[0],
                 'D': tokenizer.encode("D", add_special_tokens=False)[0]}

# Keys: ["id", "question", "opa", "opb", "opc", "opd", "cop", "choice_type", "exp", "subject_name", "topic_name"]
def eval_medmcqa(q) -> list[float, float, float, float]: # [probA, probB, probC, probD]

    prompt = f"""
        Question: {q["question"]}
        A) {q["opa"]}
        B) {q["opb"]}
        C) {q["opc"]}
        D) {q["opd"]}
        Provide the answer as a single letter (A, B, C, or D).
        """

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens['A']],
                                  last_token_logits[option_tokens['B']],
                                  last_token_logits[option_tokens['C']],
                                  last_token_logits[option_tokens['D']]])
                                  
    return torch.softmax(option_logits, dim=0).tolist()

# Keys: ["centerpiece", "options", "correct_options", "correct_options_idx", correct_options_literal]
def eval_mmlu(q) -> list[float, float, float, float]: # [probA, probB, probC, probD]

    prompt = f"""
        Question: {q["centerpiece"]}
        A) {q["options"][0]}
        B) {q["options"][1]}
        C) {q["options"][2]}
        D) {q["options"][3]}
        Provide the answer as a single letter (A, B, C, or D).
        """

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens['A']],
                                  last_token_logits[option_tokens['B']],
                                  last_token_logits[option_tokens['C']],
                                  last_token_logits[option_tokens['D']]])
                                  
    return torch.softmax(option_logits, dim=0).tolist()





