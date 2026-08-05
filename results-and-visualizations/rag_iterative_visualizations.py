import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

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

# Maximum probability
df['max_prob'] = df[['probA', 'probB', 'probC', 'probD']].max(axis=1)

# Max similarity
df['max_similarity'] = df['similarity'].apply(lambda x: max(ast.literal_eval(x)) if isinstance(x, str) else max(x))

# Confidence & similarity classifier
THRESHOLD = [0, 0.40, 0.70, 0.90, 1.01]
labels = ['Low', 'Medium', 'High', 'Very High']
df['confidence'] = pd.cut(df['max_prob'], bins=THRESHOLD, labels=labels, right=False)
df['similarity_classify'] = pd.cut(df['max_similarity'], bins=THRESHOLD, labels=labels, right=False)

# Choice classifier
prob_cols = ['probA', 'probB', 'probC', 'probD']
df['choice'] = df[prob_cols].idxmax(axis=1).str.extract(r'prob(\w)')[0]


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

# Number of uncertain options
df['result'] = df['result'].astype(bool)
df['num_within'] = (df[['probA', 'probB', 'probC', 'probD']] >= (df['max_prob'] - 0.2).values[:, None]).sum(axis=1)
prob_cols = ['probA', 'probB', 'probC', 'probD']
letter_map = {'probA': 'A', 'probB': 'B', 'probC': 'C', 'probD': 'D'}

close_mask = df[prob_cols].ge(df['max_prob'] - 0.2, axis=0)
df['close_letters'] = close_mask.apply(lambda row: [letter_map[col] for col in close_mask.columns[row]], axis=1)

df['correct_in_close'] = df.apply(lambda row: row['answer'] in row['close_letters'], axis=1)

def categorize(row):
    if row['result']:
        return 'Correct'
    elif row['correct_in_close']:
        return 'Incorrect - Correct in close'
    else:
        return 'Incorrect - Correct not in close'

df['outcome_category'] = df.apply(categorize, axis=1)

grouped = (df.groupby(['num_within', 'outcome_category']).size().reset_index(name='count'))
grouped['percentage'] = grouped.groupby('num_within')['count'].transform(lambda x: x / x.sum() * 100)

pivot = grouped.pivot(index='num_within', columns='outcome_category', values='percentage').fillna(0)

cats = ['Correct', 'Incorrect - Correct in close', 'Incorrect - Correct not in close']
for cat in cats:
    if cat not in pivot.columns:
        pivot[cat] = 0
pivot = pivot[cats]

pivot = pivot.reindex(index=range(1, 5), fill_value=0)
fig, ax = plt.subplots(figsize=(8, 6))
pivot.plot(kind='bar', stacked=True, ax=ax, color=['blue', 'green', 'red'], edgecolor='black')

ax.set_xlabel('Number of options within 0.2 of maximum probability')
ax.set_ylabel('Percentage (%)')
ax.set_title('Outcome by Number of Close Options')
ax.legend(title='Outcome')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)
ax.set_xticklabels([str(i) for i in range(1, 5)], rotation=0)

for i, n in enumerate(range(1, 5)):
    total = len(df[df['num_within'] == n])
    ax.text(i, 102, f'n={total}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("uncertain_options_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot with trend line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='max_similarity', y='max_prob', hue='source', alpha=0.7)
sns.regplot(data=df, x='max_similarity', y='max_prob', scatter=False, color='red', label='Trend line (all data)')

plt.xlabel('Maximum Similarity')
plt.ylabel('Maximum Confidence (Probability)')
plt.title('Confidence vs. Retrieval Similarity')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('confidence_vs_similarity.png', dpi=300)
plt.show()

# Source Breakdown
df['sources'] = df['sources'].apply(ast.literal_eval)
df['similarity'] = df['similarity'].apply(ast.literal_eval)

overall = {'PBMD':0,'STAT':0,'TEXT':0}
medmcqa = {'PBMD':0,'STAT':0,'TEXT':0}
mmlu = {'PBMD':0,'STAT':0,'TEXT':0}

for _, row in df.iterrows():
    for s, sim in zip(row['sources'], row['similarity']):
        overall[s] += sim
        if row['source'] == 'MEDMCQA':
            medmcqa[s] += sim
        else:
            mmlu[s] += sim

data = [(overall, 'Overall'), (medmcqa, 'MEDMCQA'), (mmlu, 'MMLU')]
fig, axes = plt.subplots(1,3,figsize=(15,5))
colors = ['#ff9999','#66b3ff','#99ff99']

for ax, (totals, title) in zip(axes, data):
    vals = [totals[k] for k in ['PBMD','STAT','TEXT']]
    ax.pie(vals, labels=['PBMD','STAT','TEXT'], autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title(title)
    ax.axis('equal')

plt.tight_layout()
plt.savefig('source_pies.png')
plt.show()

# Similarity and accuracy
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = df[(df['source'] == 'MEDMCQA') & (df['result'] == outcome)]['max_similarity']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Similarity')
ax.set_ylabel('Density')
ax.set_title('MEDMCQA - Similarity Distribution')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = df[(df['source'] == 'MMLU') & (df['result'] == outcome)]['max_similarity']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Similarity')
ax.set_ylabel('Density')
ax.set_title('MMLU - Similarity Distribution')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
crosstab = pd.crosstab(df['similarity_classify'], df['result'], normalize='index') * 100
crosstab = crosstab.rename(columns={True: 'Correct', False: 'Incorrect'})
crosstab[['Incorrect', 'Correct']].plot(kind='bar', stacked=True, ax=ax, color=['red', 'blue'], edgecolor='black')
ax.set_xlabel('Similarity Level')
ax.set_ylabel('Percentage (%)')
ax.set_title('Outcome by Similarity Level')
ax.legend(title='Outcome')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax.grid(axis='y', alpha=0.3)

for i, conf in enumerate(crosstab.index):
    total = len(df[df['similarity_classify'] == conf])
    ax.text(i, 102, f'n={total}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('similarity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Distance from max probability

def get_prob(answer, row):
    return row[f'prob{answer}']
df['correct_prob'] = df.apply(lambda row: get_prob(row['answer'], row), axis=1)
df['prob_distance'] = df['max_prob'] - df['correct_prob']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
subsets = {
    'All Data': df[df['prob_distance'] > 0], 
    'MEDMCQA': df[(df['source'] == 'MEDMCQA') & (df['prob_distance'] > 0)], 
    'MMLU': df[(df['source'] == 'MMLU') & (df['prob_distance'] > 0)]
}
for ax, (label, data) in zip(axes, subsets.items()):
    sns.kdeplot(data=data, x='prob_distance', fill=True, color='steelblue', ax=ax)
    ax.set_title(label)
    ax.set_xlabel('Distance from max probability (max - prob_correct)')
    ax.set_ylabel('Relative density')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, data['prob_distance'].quantile(0.99))
plt.tight_layout()
plt.savefig("prob_distance.png", dpi=300, bbox_inches='tight')
plt.show()

# Frequency of sources
def get_id_freq_distribution(data_subset):
    id_counter = Counter()
    for ids_str in data_subset['ids']:
        ids_list = ast.literal_eval(ids_str) if isinstance(ids_str, str) else ids_str
        id_counter.update(ids_list)
    freq_count = Counter(id_counter.values())
    return freq_count

medmcqa_df = df[df['source'] == 'MEDMCQA']
mmlu_df = df[df['source'] == 'MMLU']

med_freq = get_id_freq_distribution(medmcqa_df)
mmlu_freq = get_id_freq_distribution(mmlu_df)
both_freq = get_id_freq_distribution(df)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, freq_dict, title in zip(axes, [med_freq, mmlu_freq, both_freq],
                                ['MEDMCQA', 'MMLU', 'Both']):
    x = sorted(freq_dict.keys())
    y = [freq_dict[k] for k in x]
    ax.bar(x, y, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Number of times ID appears')
    ax.set_ylabel('Number of unique IDs')
    ax.set_title(title)
    for i, v in enumerate(y):
        ax.text(x[i], v + 0.5, str(v), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('id_frequency_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

df['ids_parsed'] = df['ids'].apply(ast.literal_eval)
df['add_ids_parsed'] = df['add_ids'].apply(ast.literal_eval)

filter_condition = df['confident'] > 1
filtered_df = df[filter_condition].copy()

def count_differences(row):
    ids = row['ids_parsed']
    add_ids = row['add_ids_parsed']
    add_ids = [str(x) for x in add_ids]
    return sum(1 for i in ids if i not in add_ids)

filtered_df['num_diff'] = filtered_df.apply(count_differences, axis=1)
diff_counts = filtered_df['num_diff'].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 5))
diff_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Number of differing IDs')
plt.ylabel('Frequency (number of rows)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.savefig('different_ids_distribution.png', dpi=300, bbox_inches='tight')