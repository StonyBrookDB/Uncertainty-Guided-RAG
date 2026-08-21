import ast
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RAG_VISUALIZATIONS = True

# Read both dataframes
mcqa_df = pd.read_csv("medmcqa_results.csv")   # 4 options (probA‑D)
qa_df   = pd.read_csv("medqa_results.csv")     # 5 options (probA‑E)

# Add missing probE column to the 4‑option dataframe
mcqa_df['probE'] = 0.0

# Combine into one DataFrame
df = pd.concat([mcqa_df, qa_df], ignore_index=True)

# ---- Compute derived columns ----
prob_cols = ['probA', 'probB', 'probC', 'probD', 'probE']
THRESHOLD = [0, 0.40, 0.70, 0.90, 1.01]
labels = ['Low', 'Medium', 'High', 'Very High']

df['confidence'] = pd.cut(df[prob_cols].max(axis=1),
                          bins=THRESHOLD, labels=labels, right=False)
df['choice'] = df[prob_cols].idxmax(axis=1).str.extract(r'prob(\w)')[0]
df['result'] = df[prob_cols].values.argmax(axis=1) == df['answer']

answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
df['answer_letter'] = df['answer'].map(answer_map)
df['p_top1'] = df[prob_cols].max(axis=1)
df['p_ans'] = df[prob_cols].values[range(len(df)), df['answer'].values]
df['margin'] = df['p_top1'] - df['p_ans']

# Split by source (adjust source names as needed)
medmcqa_df = df[df['source'].str.lower() == 'medmcqa']
medqa_df   = df[df['source'].str.lower() == 'medqa']   # or 'mmlu' if that's the case

# ---- 1. Pie Charts (Answers & Predictions) ----
data_list = [
    (medmcqa_df['answer_letter'].value_counts().reindex(['A','B','C','D','E'], fill_value=0),
     "Answers (MEDMCQA)"),
    (medqa_df['answer_letter'].value_counts().reindex(['A','B','C','D','E'], fill_value=0),
     "Answers (MEDQA)"),
    (medmcqa_df['choice'].value_counts().reindex(['A','B','C','D','E'], fill_value=0),
     "Predictions (MEDMCQA)"),
    (medqa_df['choice'].value_counts().reindex(['A','B','C','D','E'], fill_value=0),
     "Predictions (MEDQA)"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']  # added 5th colour

for ax, (counts, title) in zip(axes, data_list):
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors[:len(counts)])
    ax.set_title(f"{title} (n={sum(counts)})")
    ax.axis('equal')

plt.tight_layout()
plt.savefig('answer_pie_charts.png', bbox_inches='tight')
plt.close()

# ---- 2. Error by Confidence (KDE for each source + stacked bar overall) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MEDMCQA
ax = axes[0]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = medmcqa_df[medmcqa_df['result'] == outcome]['p_top1']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('MEDMCQA - Confidence Distribution')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

# MEDQA
ax = axes[1]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = medqa_df[medqa_df['result'] == outcome]['p_top1']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('MEDQA - Confidence Distribution')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

# Overall stacked bar by confidence level
ax = axes[2]
crosstab = pd.crosstab(df['confidence'], df['result'], normalize='index') * 100
crosstab = crosstab.rename(columns={True: 'Correct', False: 'Incorrect'})
crosstab[['Incorrect', 'Correct']].plot(kind='bar', stacked=True, ax=ax,
                                        color=['red', 'blue'], edgecolor='black')
ax.set_xlabel('Confidence Level')
ax.set_ylabel('Percentage (%)')
ax.set_title('Outcome by Confidence Level')
ax.legend(title='Outcome')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax.grid(axis='y', alpha=0.3)

for i, conf in enumerate(crosstab.index):
    total = len(df[df['confidence'] == conf])
    ax.text(i, 102, f'n={total}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ---- 3. Probability Density of Margin (overall) ----
fig, ax = plt.subplots(figsize=(10, 6))
for outcome, color, label in [(True, '#2ecc71', 'Correct'), (False, '#e74c3c', 'Incorrect')]:
    subset = df[df['result'] == outcome]['margin']
    if not subset.empty and subset.nunique() > 1:
        sns.kdeplot(subset, ax=ax, color=color, label=label,
                    linewidth=2.5, fill=True, alpha=0.25)

ax.set_xlabel('p(top1) − p(answer)')
ax.set_ylabel('Relative Density')
ax.set_title('Relative Density of p(top1) − p(answer)')
ax.set_xlim(0, 1)
ax.legend(title='Prediction')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('margin_density.png', dpi=300, bbox_inches='tight')
plt.close()

# ---- 4. Source Breakdown (RAG only) ----
if RAG_VISUALIZATIONS:
    df['sources'] = df['sources'].apply(ast.literal_eval)
    df['similarity'] = df['similarity'].apply(ast.literal_eval)

    overall = defaultdict(float)
    medmcqa_sources = defaultdict(float)
    medqa_sources = defaultdict(float)

    for _, row in df.iterrows():
        for s, sim in zip(row['sources'], row['similarity']):
            overall[s] += sim
            if row['source'].lower() == 'medmcqa':
                medmcqa_sources[s] += sim
            else:
                medqa_sources[s] += sim

    source_keys = sorted(overall.keys())  # e.g. ['PBMD', 'STAT', 'TEXT']
    data = [(overall, 'Overall'), (medmcqa_sources, 'MEDMCQA'), (medqa_sources, 'MEDQA')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

    for ax, (totals, title) in zip(axes, data):
        vals = [totals.get(k, 0) for k in source_keys]
        ax.pie(vals, labels=source_keys, autopct='%1.1f%%',
               startangle=90, colors=colors[:len(source_keys)])
        ax.set_title(title)
        ax.axis('equal')

    plt.tight_layout()
    plt.savefig('source_pies.png', dpi=300, bbox_inches='tight')
    plt.close()

# Top‑k accuracy table (Top‑1 to Top‑5)
def compute_topk_ranks(df, prob_cols):
    probs = df[prob_cols].values
    answers = df['answer'].values
    ranks = []
    for i in range(len(probs)):
        sorted_idx = np.argsort(probs[i])[::-1]  # descending
        rank = np.where(sorted_idx == answers[i])[0][0] + 1
        ranks.append(rank)
    return ranks

# MCQA (4 options)
mcqa_ranks = compute_topk_ranks(mcqa_df, ['probA','probB','probC','probD'])
mcqa_top1 = (np.array(mcqa_ranks) <= 1).mean() * 100
mcqa_top2 = (np.array(mcqa_ranks) <= 2).mean() * 100
mcqa_top3 = (np.array(mcqa_ranks) <= 3).mean() * 100
mcqa_top4 = (np.array(mcqa_ranks) <= 4).mean() * 100
mcqa_top5 = (np.array(mcqa_ranks) <= 5).mean() * 100  # always 100%
mcqa_time = mcqa_df['time'].mean()

# QA (5 options)
qa_ranks = compute_topk_ranks(qa_df, ['probA','probB','probC','probD','probE'])
qa_top1 = (np.array(qa_ranks) <= 1).mean() * 100
qa_top2 = (np.array(qa_ranks) <= 2).mean() * 100
qa_top3 = (np.array(qa_ranks) <= 3).mean() * 100
qa_top4 = (np.array(qa_ranks) <= 4).mean() * 100
qa_top5 = (np.array(qa_ranks) <= 5).mean() * 100  # always 100%
qa_time = qa_df['time'].mean()

# Build table with 7 columns
table_data = [
    ['MCQA', f"{mcqa_top1:.2f}", f"{mcqa_top2:.2f}", f"{mcqa_top3:.2f}", 
     f"{mcqa_top4:.2f}", f"{mcqa_top5:.2f}", f"{mcqa_time:.2f}"],
    ['QA',   f"{qa_top1:.2f}",   f"{qa_top2:.2f}",   f"{qa_top3:.2f}",
     f"{qa_top4:.2f}",   f"{qa_top5:.2f}",   f"{qa_time:.2f}"]
]
table_df = pd.DataFrame(table_data, 
                        columns=['Dataset', 'Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5', 'Avg Time (s)'])

fig, ax = plt.subplots(figsize=(10, 2.5))  # slightly wider for 7 columns
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                 cellLoc='center', loc='center',
                 colColours=['#f0f0f0'] * len(table_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.8, 2)
plt.savefig('topk_accuracy_table.png', bbox_inches='tight', dpi=300)
plt.close()