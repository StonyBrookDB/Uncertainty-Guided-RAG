import pandas as pd

# Load CSV
base_df = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_baseline/1.2_results/medmcqa_results_{i}.csv")
    df['source'] = 'MEDMCQA'
    df['split'] = i
    base_df = pd.concat([base_df, df], ignore_index=True)

for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_baseline/1.2_results/mmlu_results_{i}.csv")
    df['source'] = 'MMLU'
    df['split'] = i
    base_df = pd.concat([base_df, df], ignore_index=True)
base_df["confident"] = 1
base_df["add_ids"] = "[-1, -1, -1, -1, -1]"


iterated_df = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_iterative_v2/0.5_results/medmcqa_results_{i}.csv")
    df['source'] = 'MEDMCQA'
    df['split'] = i
    iterated_df = pd.concat([iterated_df, df], ignore_index=True)

for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_iterative_v2/0.5_results/mmlu_results_{i}.csv")
    df['source'] = 'MMLU'
    df['split'] = i
    iterated_df = pd.concat([iterated_df, df], ignore_index=True)

# Margin 
base_df["top1"] = base_df[["probA", "probB", "probC", "probD"]].max(axis=1)
base_df["top2"] = base_df[["probA", "probB", "probC", "probD"]].apply(lambda x: sorted(x)[2], axis=1)
base_df["margin"] = base_df["top1"] - base_df["top2"]

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    filtered_df = base_df.mask(base_df["margin"] < threshold, iterated_df)
    filtered_df.to_csv(f"results-and-visualizations/ablation/results_{threshold}.csv", index=False)
