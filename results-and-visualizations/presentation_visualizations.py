"""
Results table
Confidence vs accuracy
Ablation curve
Per fold analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# region Load CSV
BASELINE_DF = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/baseline/v1_results/medmcqa_results_{i}.csv")
    df['source'] = 'MEDMCQA'
    df['split'] = i
    BASELINE_DF = pd.concat([BASELINE_DF, df], ignore_index=True)
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/baseline/v1_results/mmlu_results_{i}.csv")
    df['source'] = 'MMLU'
    df['split'] = i
    BASELINE_DF = pd.concat([BASELINE_DF, df], ignore_index=True)

RAG_BASELINE_DF = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_baseline/1.2_results/medmcqa_results_{i}.csv")
    df['source'] = 'MEDMCQA'
    df['split'] = i
    RAG_BASELINE_DF = pd.concat([RAG_BASELINE_DF, df], ignore_index=True)
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_baseline/1.2_results/mmlu_results_{i}.csv")
    df['source'] = 'MMLU'
    df['split'] = i
    RAG_BASELINE_DF = pd.concat([RAG_BASELINE_DF, df], ignore_index=True)

FULL_DF = pd.read_csv("results-and-visualizations/ablation/results_1.0.csv")


STATIC_RAG_DF = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_iterative_v1/0.3_results/medmcqa_results_{i}.csv")
    df['source'] = 'MEDMCQA'
    df['split'] = i
    STATIC_RAG_DF = pd.concat([STATIC_RAG_DF, df], ignore_index=True)
for i in range(1, 6):
    df = pd.read_csv(f"results-and-visualizations/rag_iterative_v1/0.3_results/mmlu_results_{i}.csv")
    df['source'] = 'MMLU'
    df['split'] = i
    STATIC_RAG_DF = pd.concat([STATIC_RAG_DF, df], ignore_index=True)

DYNAMIC_RAG_DF = pd.read_csv("results-and-visualizations/ablation/results_0.3.csv")
# endregion Load CSV

models = {
    "baseline": BASELINE_DF,
    "RAG_baseline": RAG_BASELINE_DF,
    "static_iterative_RAG": STATIC_RAG_DF,
    "dynamic_RAG (τ = 0.3)": DYNAMIC_RAG_DF,
    "dynamic_RAG (τ = 1)": FULL_DF
}

# region Final Table
results = []
for name, df in models.items():
    folds = sorted(df['split'].unique())
    med_accs = []
    mmlu_accs = []
    avg_times = []
    for fold in folds:
        fold_df = df[df['split'] == fold]
        med_fold = fold_df[fold_df['source'] == 'MEDMCQA']
        med_accs.append(med_fold['result'].mean() * 100 if len(med_fold) > 0 else np.nan)
        mmlu_fold = fold_df[fold_df['source'] == 'MMLU']
        mmlu_accs.append(mmlu_fold['result'].mean() * 100 if len(mmlu_fold) > 0 else np.nan)
        avg_times.append(fold_df['time'].mean())
    
    med_mean = np.nanmean(med_accs)
    med_std = np.nanstd(med_accs)
    mmlu_mean = np.nanmean(mmlu_accs)
    mmlu_std = np.nanstd(mmlu_accs)
    time_mean = np.nanmean(avg_times)
    time_std = np.nanstd(avg_times)
    
    med_str = f"{med_mean:.2f} ± {med_std:.2f}"
    mmlu_str = f"{mmlu_mean:.2f} ± {mmlu_std:.2f}"
    time_str = f"{time_mean:.2f}"
    results.append([name, med_str, mmlu_str, time_str])

table_df = pd.DataFrame(results, columns=['Model', 'MEDMCQA Acc (%)', 'MMLU Acc (%)', 'Avg Time (s)'])
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center', colColours=['#f0f0f0']*len(table_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.6, 2)
plt.savefig('summary_table.png', bbox_inches='tight', dpi=300)
# endregion Final Table

# region Ablation Calibration
thresholds = []
accuracies = []
avg_times = []

for threshold in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    df = pd.read_csv(f"results-and-visualizations/ablation/results_{threshold}.csv")
    accuracy = (df['result'].sum() / len(df) * 100)
    avg_time = df['time'].mean()
    thresholds.append(threshold)
    accuracies.append(accuracy)
    avg_times.append(avg_time)

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(thresholds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Overall Accuracy (%)', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(68, 72)

ax2 = ax1.twinx()
ax2.plot(thresholds, avg_times, marker='o', linestyle='-', color='r', label='Avg Time')
ax2.set_ylabel('Average Time (s)', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax1.axvline(x=0.3, color='g', linestyle='--')
ax1.set_xticks(thresholds[::2])
ax1.set_xticklabels([f'{t:.1f}' for t in thresholds[::2]])
ax1.set_xlim(0, 1.0)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Calibration Curve', fontsize=14)
plt.xticks(thresholds, rotation=45)
plt.tight_layout()
plt.savefig('accuracy_vs_threshold_with_time.png', dpi=300)
plt.show()
# endregion Ablation Calibration

# region Confidence vs Accuracy
# for df in models.values():
#     df['max_prob'] = df[['probA', 'probB', 'probC', 'probD']].max(axis=1)

# fig, axes = plt.subplots(2, 1, figsize=(5, 10))
# axes = axes.flatten()
# i = 0

# for name, df in models.items():
#     ax = axes[i]
#     for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
#         subset = df[(df['result'] == outcome)]['max_prob']
#         if not subset.empty:
#             sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
#     ax.set_xlabel('Maximum Predicted Probability')
#     ax.set_ylabel('Density')
#     ax.set_title(name + ' - Confidence Distribution')
#     ax.legend()
#     ax.grid(alpha=0.3)
#     ax.set_xlim(0, 1)
#     i += 1

# plt.tight_layout()
# plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()
# endregion Confidence vs Accuracy