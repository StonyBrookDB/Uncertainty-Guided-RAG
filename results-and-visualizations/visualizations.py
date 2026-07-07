"""
llama_baseline: 3 errors
mistral_baseline: 15 errors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
all_dfs = []
for i in range(1, 6):
    df = pd.read_csv(f"medmcqa_results_{i}.csv")
    df['source'] = 'MEDMCQA'
    df['split'] = i
    all_dfs.append(df)

for i in range(1, 6):
    df = pd.read_csv(f"mmlu_results_{i}.csv")
    df['source'] = 'MMLU'
    df['split'] = i
    all_dfs.append(df)
df = pd.concat(all_dfs, ignore_index=True)

# Confidence Classifier
THRESHOLD = [0, 0.40, 0.70, 0.90, 1.01]
labels = ['Low', 'Medium', 'High', 'Very High']
df['confidence'] = pd.cut(df[['probA', 'probB', 'probC', 'probD']].max(axis=1), bins=THRESHOLD, labels=labels, right=False)

# Choice Classifier
prob_cols = ['probA', 'probB', 'probC', 'probD']
df['choice'] = df[prob_cols].idxmax(axis=1).str.extract(r'prob(\w)')[0]

# Maximum probability
df['max_prob'] = df[['probA', 'probB', 'probC', 'probD']].max(axis=1)

# Pie Charts for answer choices
data_list = [
    (df['answer'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Answers (Overall)"),
    (df[df['source']=='MEDMCQA']['answer'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Answers (MEDMCQA)"),
    (df[df['source']=='MMLU']['answer'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Answers (MMLU)"),
    (df['choice'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Predictions (Overall)"),
    (df[df['source']=='MEDMCQA']['choice'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Predictions (MEDMCQA)"),
    (df[df['source']=='MMLU']['choice'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Predictions (MMLU)"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

for ax, (counts, title) in zip(axes, data_list):
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors)
    ax.set_title(f"{title} (n={sum(counts)})")
    ax.axis('equal')

plt.tight_layout()
plt.savefig('answer_pie_charts.png', bbox_inches='tight')
plt.show()

# Accuracy by Subject
medmcqa_df = df[df['source'] == 'MEDMCQA']
mmlu_df = df[df['source'] == 'MMLU']

medmcqa_split_acc = medmcqa_df.groupby(['subject', 'split'])['result'].mean().reset_index()
medmcqa_split_acc.columns = ['subject', 'split', 'acc']
medmcqa_stats = medmcqa_split_acc.groupby('subject')['acc'].agg(['mean', 'std']).reset_index()
medmcqa_stats.columns = ['subject', 'mean', 'std']

mmlu_split_acc = mmlu_df.groupby(['subject', 'split'])['result'].mean().reset_index()
mmlu_split_acc.columns = ['subject', 'split', 'acc']
mmlu_stats = mmlu_split_acc.groupby('subject')['acc'].agg(['mean', 'std']).reset_index()
mmlu_stats.columns = ['subject', 'mean', 'std']

medmcqa_avg = medmcqa_stats['mean'].mean()
mmlu_avg = mmlu_stats['mean'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

subjects1 = medmcqa_stats['subject']
means1 = medmcqa_stats['mean']
stds1 = medmcqa_stats['std']
ax1.barh(subjects1, means1, xerr=stds1, color='steelblue', alpha=0.8, capsize=3)
ax1.axvline(x=medmcqa_avg, color='red', linestyle='--', label=f'Average = {medmcqa_avg:.3f}')
ax1.set_title('MEDMCQA')
ax1.set_xlabel('Accuracy')
ax1.set_ylabel('Subject')
ax1.set_xlim(0, 1)
ax1.legend()

subjects2 = mmlu_stats['subject']
means2 = mmlu_stats['mean']
stds2 = mmlu_stats['std']
ax2.barh(subjects2, means2, xerr=stds2, color='darkorange', alpha=0.8, capsize=3)
ax2.axvline(x=mmlu_avg, color='red', linestyle='--', label=f'Average = {mmlu_avg:.3f}')
ax2.set_title('MMLU')
ax2.set_xlabel('Accuracy')
ax2.set_ylabel('Subject')
ax2.set_xlim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig('accuracy_by_subject.png', dpi=300)
plt.show()

# Error by confidence
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = df[(df['source'] == 'MEDMCQA') & (df['result'] == outcome)]['max_prob']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('MEDMCQA - Confidence Distribution')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = df[(df['source'] == 'MMLU') & (df['result'] == outcome)]['max_prob']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('MMLU - Confidence Distribution')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
crosstab = pd.crosstab(df['confidence'], df['result'], normalize='index') * 100
crosstab = crosstab.rename(columns={True: 'Correct', False: 'Incorrect'})
crosstab[['Incorrect', 'Correct']].plot(kind='bar', stacked=True, ax=ax, color=['red', 'blue'], edgecolor='black')
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
plt.show()

# Performance across splits
split_acc = df.groupby(['source', 'split'])['result'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(data=split_acc, x='split', y='result', hue='source', marker='o', ax=ax)
ax.set_title('Performance Across Splits')
ax.set_xlabel('Split')
ax.set_ylabel('Accuracy')
ax.set_xticks(range(1, 6))
ax.set_ylim(0, 1)
for source in split_acc['source'].unique():
    source_data = split_acc[split_acc['source'] == source]
    for _, row in source_data.iterrows():
        ax.text(row['split'], row['result'] + 0.02, f"{row['result']:.2f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('performance_across_splits.png', dpi=300, bbox_inches='tight')
plt.show()

# Time graph
medmcqa_df = df[df['source'] == 'MEDMCQA']
mmlu_df = df[df['source'] == 'MMLU']

medmcqa_time_stats = medmcqa_df.groupby('subject')['time'].agg(['mean', 'std']).reset_index()
medmcqa_time_stats.columns = ['subject', 'mean', 'std']
mmlu_time_stats = mmlu_df.groupby('subject')['time'].agg(['mean', 'std']).reset_index()
mmlu_time_stats.columns = ['subject', 'mean', 'std']

medmcqa_avg = medmcqa_time_stats['mean'].mean()
mmlu_avg = mmlu_time_stats['mean'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

subjects1 = medmcqa_time_stats['subject']
means1 = medmcqa_time_stats['mean']
stds1 = medmcqa_time_stats['std']
ax1.barh(subjects1, means1, xerr=stds1, color='steelblue', alpha=0.8, capsize=3)
ax1.axvline(x=medmcqa_avg, color='red', linestyle='--', label=f'Average = {medmcqa_avg:.3f}s')
ax1.set_title('MEDMCQA')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Subject')
ax1.legend()

subjects2 = mmlu_time_stats['subject']
means2 = mmlu_time_stats['mean']
stds2 = mmlu_time_stats['std']
ax2.barh(subjects2, means2, xerr=stds2, color='darkorange', alpha=0.8, capsize=3)
ax2.axvline(x=mmlu_avg, color='red', linestyle='--', label=f'Average = {mmlu_avg:.3f}s')
ax2.set_title('MMLU')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Subject')
ax2.legend()

plt.tight_layout()
plt.savefig('time_by_subject.png', dpi=300)
plt.show()