# region imports, models, constants
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import pandas as pd
import time as t
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda", attn_implementation="sdpa"
)
model.eval()

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

client = MilvusClient(uri="http://localhost:19530")
client.load_collection("MedRAG_combined_collection")
COLLECTION = "MedRAG_combined_collection"

ALL_OPTION_LETTERS = ["A", "B", "C", "D", "E"]
option_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ALL_OPTION_LETTERS}

INSTRUCTIONS = (
    "Use provided context and reasoning to guide your answer. "
    "Output your answer as a single letter. No explanation."
)

SYSTEM_PROMPT_QUERYGEN = (
    "You are a medical information-retrieval assistant. Given a multiple-choice "
    "medical question and one specific answer choice, write ONE focused search "
    "query that would retrieve medical literature (mechanisms, guideline "
    "recommendations, clinical trial evidence, or textbook definitions) that "
    "supports that choice as the correct answer.\n"
    "Rules:\n"
    "- Name the specific condition, drug, or procedure explicitly; do not use "
    "vague pronouns like 'this' or 'it'.\n"
    "- Use precise clinical terminology a medical database would index on.\n"
    "- Do not mention or allude to the other answer choices.\n"
    "- Do not use negation words like \"excluding\" or \"not\".\n"
    "- Do not restate the full question verbatim.\n"
    "- Keep it to a single sentence, under 40 words.\n"
    "Respond with the search query text only — no preamble, no labels."
)

ONE_SHOT_EXAMPLE = (
    "Example:\n"
    "Medical Question: What is the most effective initial pharmacological therapy for stable angina?\n"
    "Options: A) Nitroglycerin B) Beta-blockers C) Calcium channel blockers D) Aspirin\n"
    "Choice to support: \"Beta-blockers\"\n"
    "Search Query: Guideline recommendations and randomized controlled trial evidence for "
    "beta-blockers as first-line antianginal therapy in stable angina, including comparisons "
    "with calcium channel blockers and nitrates for symptom control and mortality benefit.\n\n"
)

USE_RERANKER = True
errors = 0
# endregion


class StopOnSubstrings(StoppingCriteria):
    """Stops each row of a batch once it has produced a stop string, checked
    by decoding just that row's generated tail so far."""

    def __init__(self, tokenizer, stop_strings, prompt_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len
        self.done = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.done is None:
            self.done = [False] * input_ids.shape[0]
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            gen_tokens = input_ids[i, self.prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            if any(s in text for s in self.stop_strings):
                self.done[i] = True
        return all(self.done)


def _run_model(messages):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    torch.cuda.synchronize()
    start = t.time()
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, logits_to_keep=1)
    torch.cuda.synchronize()
    end = t.time()

    last_token_logits = outputs.logits[0, -1, :]
    return inputs, last_token_logits, end - start


def _check_confident(inputs, last_token_logits, valid_letters):
    global errors
    valid_token_ids = {option_tokens[letter] for letter in valid_letters}
    next_token_id = torch.argmax(last_token_logits).item()
    if next_token_id not in valid_token_ids:
        next_token_text = tokenizer.decode(next_token_id)
        print(f"Warning: model's top token is not an option: {next_token_text}")
        errors += 1
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=100)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
        return False
    return True


def get_context(prompt, use_reranker=USE_RERANKER):
    start = t.time()
    query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
    search_limit = 15 if use_reranker else 5

    search = client.search(
        collection_name=COLLECTION,
        data=[query_embedding],
        limit=search_limit,
        output_fields=["id", "source", "content"],
        search_params={"metric_type": "COSINE", "params": {}}
    )

    results = []
    for r in search[0]:
        results.append({
            "score": r["distance"],
            "id": r["id"],
            "source": r["entity"].get("source"),
            "content": r["entity"].get("content")
        })

    if use_reranker:
        pairs = [(prompt, r["content"]) for r in results]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=search_limit)
        for r, score in zip(results, rerank_scores):
            r["rerank_score"] = score
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:5]

    return "\n\n".join([f"{r['content']}" for r in results]), results, t.time() - start


def build_querygen_prompt(question, options, opt):
    """options: full list of option texts for this question (len 4 or 5).
    opt: the single option text to generate a supporting query for."""
    letters = ALL_OPTION_LETTERS[:len(options)]
    options_str = " ".join(f"{letter}) {text}" for letter, text in zip(letters, options))
    user_content = (
        ONE_SHOT_EXAMPLE
        + "Now generate:\n"
        + f"Medical Question: {question}\n"
        + f"Options: {options_str}\n"
        + f"Choice to support: \"{opt}\"\n"
        + "Search Query:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_QUERYGEN},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_query(question, options, confident_options):
    """One search query per option in confident_options, batched into a
    single generate() call."""
    if not confident_options:
        return []

    prompts = [build_querygen_prompt(question, options, opt) for opt in confident_options]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    prompt_len = batch["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([StopOnSubstrings(tokenizer, ["\n\n"], prompt_len)])

    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=120,
            do_sample=False,
            temperature=None,
            top_p=None,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
        )

    queries = []
    for i in range(outputs.shape[0]):
        gen_tokens = outputs[i, prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        queries.append(text.split("\n\n")[0].strip())
    return queries


def _iterative_answer(question, options, letters, context, initial_messages_fn, THRESHOLD):
    messages = initial_messages_fn(context)
    inputs, last_token_logits, elapsed_model = _run_model(messages)

    option_logits = torch.tensor([last_token_logits[option_tokens[letter]] for letter in letters])
    probs = torch.softmax(option_logits, dim=0).tolist()
    confident_indices = [i for i in range(len(letters)) if probs[i] >= max(probs) - THRESHOLD]

    add_search_results = [{"id": -1} for _ in range(5)]
    if len(confident_indices) != 1:
        add_search_results = []
        add_context = ""
        queries = get_query(question, options, [options[i] for i in confident_indices])
        for query in queries:
            curr_context, curr_search_results, _ = get_context(query)
            add_search_results.extend(curr_search_results)
            add_context += f"\n\n{curr_context}"

        combined_context = f"{add_context}\n\n{context}"
        messages = initial_messages_fn(combined_context)
        inputs, last_token_logits, elapsed_model_2 = _run_model(messages)
        elapsed_model += elapsed_model_2

    return inputs, last_token_logits, elapsed_model, confident_indices, add_search_results


# returns [[probA, probB, probC, probD], time, confident flag, search_results, confident_count, add_search_results]
def eval_medmcqa(q, THRESHOLD):
    letters = ALL_OPTION_LETTERS[:4]
    options = [q["a"], q["b"], q["c"], q["d"]]

    context, search_results, elapsed_retrieval = get_context(f"""Question: {q["question"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}""", use_reranker=USE_RERANKER)

    def initial_messages_fn(ctx):
        return [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"""Context:
{ctx}
Question: {q["question"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}
Answer: """},
        ]

    inputs, last_token_logits, elapsed_model, confident_indices, add_search_results = _iterative_answer(
        q["question"], options, letters, context, initial_messages_fn, THRESHOLD
    )
    confident = _check_confident(inputs, last_token_logits, letters)

    option_logits = [last_token_logits[option_tokens[letter]].item() for letter in letters]
    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()

    return [probs, elapsed_retrieval + elapsed_model, confident, search_results, len(confident_indices), add_search_results]


# returns [[probA, probB, probC, probD, probE], time, confident flag, search_results, confident_count, add_search_results]
def eval_medqa(q, THRESHOLD):
    letters = ALL_OPTION_LETTERS[:5]
    options = [q["a"], q["b"], q["c"], q["d"], q["e"]]

    context, search_results, elapsed_retrieval = get_context(f"""Question: {q["question"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}
    E) {options[4]}""", use_reranker=USE_RERANKER)

    def initial_messages_fn(ctx):
        return [
            {"role": "system", "content": INSTRUCTIONS.replace("(A, B, C, D)", "(A, B, C, D, E)")},
            {"role": "user", "content": f"""Context:
{ctx}
Question: {q["question"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}
    E) {options[4]}
Answer: """},
        ]

    inputs, last_token_logits, elapsed_model, confident_indices, add_search_results = _iterative_answer(
        q["question"], options, letters, context, initial_messages_fn, THRESHOLD
    )
    confident = _check_confident(inputs, last_token_logits, letters)

    option_logits = [last_token_logits[option_tokens[letter]].item() for letter in letters]
    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()

    return [probs, elapsed_retrieval + elapsed_model, confident, search_results, len(confident_indices), add_search_results]


def write(res, out_file):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(out_file)
    df.to_csv(out_file, mode="a", header=not file_exists, index=False)


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
for _ in range(10):
    _run_model(warmup_messages)
torch.cuda.synchronize()

for T in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    errors = 0
    # MEDMCQA
    print("Evaluating MedMCQA test")
    results = []
    df = pd.read_csv("question-set/medmcqa_1000.csv")
    for i, q in enumerate(df.to_dict("records")):
        if T == 0.4: continue
        probs, time_taken, confident, search_results, confident_count, add_search_results = eval_medmcqa(q, T)
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
            "sources": [r["source"] for r in search_results],
            "ids": [r["id"] for r in search_results],
            "similarity": [r["score"] for r in search_results],
            "confident": confident_count,
            "add_ids": [r["id"] for r in add_search_results],
        })
        if i % 100 == 0 and i != 0:
            write(results, f"medmcqa_results_{T}.csv")
            results.clear()

    if results:
        write(results, f"medmcqa_results_{T}.csv")
        results.clear()

    # MEDQA
    print("Evaluating MedQA test")
    results = []
    df = pd.read_csv("question-set/medqa_1273.csv")
    for i, q in enumerate(df.to_dict("records")):
        if T == 0.4 and i <= 700: continue
        probs, time_taken, confident, search_results, confident_count, add_search_results = eval_medqa(q, T)
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
            "sources": [r["source"] for r in search_results],
            "ids": [r["id"] for r in search_results],
            "similarity": [r["score"] for r in search_results],
            "confident": confident_count,
            "add_ids": [r["id"] for r in add_search_results],
        })
        if i % 100 == 0 and i != 0:
            write(results, f"medqa_results_{T}.csv")
            results.clear()

    if results:
        write(results, f"medqa_results_{T}.csv")
        results.clear()

    print(f"Total errors: {errors}")