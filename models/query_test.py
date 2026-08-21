# import warnings
# warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
import logging
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

# region Imports
import time as t
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient
# endregion

# region Milvus Connect
# client = MilvusClient(uri="http://localhost:19530")
# client.load_collection("MedRAG_textbook_collection")
# client.load_collection("MedRAG_statpearls_collection")
# client.load_collection("MedRAG_pubmed_collection")
# endregion

# region Models
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_name = "meta-llama/Llama-3.1-8B-Instruct"

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Left padding is required for batched generation: all sequences in a batch
# must end at the same index so `generate()` starts new tokens at one shared
# position for every row.
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", quantization_config=bnb_config)
model.eval()

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")
# endregion

# region Constants
COLLECTIONS = ["MedRAG_textbook_collection", "MedRAG_statpearls_collection", "MedRAG_pubmed_collection"]

SYSTEM_PROMPT = (
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
# endregion


class StopOnSubstrings(StoppingCriteria):
    """Stops generation (per-sequence-aware via post-hoc truncation) once any
    stop string has been produced. Simple substring check on the decoded
    tail; good enough for short generations like these."""

    def __init__(self, tokenizer, stop_strings, prompt_lens):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_lens = prompt_lens  # token length of each prompt (post-padding), to know where generation starts
        self.done = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.done is None:
            self.done = [False] * input_ids.shape[0]
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            gen_tokens = input_ids[i, self.prompt_lens[i]:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            if any(s in text for s in self.stop_strings):
                self.done[i] = True
        return all(self.done)


def get_context(prompt):
    query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
    search = []
    for c in COLLECTIONS:
        query = client.search(
            collection_name=c,
            data=[query_embedding],
            limit=10,
            output_fields=["id", "source", "content"],
            search_params={"metric_type": "COSINE", "params": {}},
        )
        search.extend(query[0])

    results = []
    for r in search:
        results.append({
            "score": r["distance"],
            "id": r["id"],
            "source": r["entity"].get("source"),
            "content": r["entity"].get("content"),
        })

    pairs = [(prompt, r["content"]) for r in results]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=len(COLLECTIONS) * 10)
    for r, score in zip(results, rerank_scores):
        r["rerank_score"] = score

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:5]
    return "\n\n".join([f"{r['content']}" for r in results]), results


def build_prompt(question, options, opt):
    user_content = (
        ONE_SHOT_EXAMPLE
        + "Now generate:\n"
        + f"Medical Question: {question}\n"
        + f"Options: A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}\n"
        + f"Choice to support: \"{opt}\"\n"
        + "Search Query:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_query(question, options, confident_options):
    """Generate one search query per option in `confident_options`, batched
    into a single generate() call instead of one call per option."""
    prompts = [build_prompt(question, options, opt) for opt in confident_options]

    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    prompt_len = batch["input_ids"].shape[1]  # identical for every row post-padding (left padding)
    prompt_lens = [prompt_len] * len(prompts)

    stopping_criteria = StoppingCriteriaList([
        StopOnSubstrings(tokenizer, ["\n\n"], prompt_lens)
    ])

    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=120,
            do_sample=False,       # greedy: deterministic, focused queries
            temperature=None,
            top_p=None,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
        )

    queries = []
    for i in range(outputs.shape[0]):
        gen_tokens = outputs[i, prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        text = text.split("\n\n")[0].strip()
        queries.append(text)
    return queries


results = []
times = []
df = pd.read_csv("question-set/medmcqa_20.csv")
for n, q in enumerate(df.to_dict("records")):
    if n != 0 and n % 10 == 0:
        print(f"Processed {n} questions")
    print(f"""Question: {q["question"]}
A) {q["a"]}
B) {q["b"]}
C) {q["c"]}
D) {q["d"]}""")
    start = t.time()
    queries = get_query(q["question"], [q["a"], q["b"], q["c"], q["d"]],
                         [q["a"], q["b"], q["c"], q["d"]])
    end = t.time()
    times.append(end - start)
    for query in queries:
        print(query)

print(f"Average time per question: {sum(times) / len(times)} seconds")
print(f"Max time for a question: {max(times)} seconds")
print(f"Min time for a question: {min(times)} seconds")