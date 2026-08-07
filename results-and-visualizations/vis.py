import pandas as pd

# Read your CSVs (adjust file paths)
for i in range(1, 6):
    df1 = pd.read_csv(f'results-and-visualizations/rag_iterative_v2/0.5_results/mmlu_results_{i}.csv')
    df2 = pd.read_csv(f'mmlu_results_{i}.csv')

    # Set a multi-index on the matching columns
    df1.set_index(['id', 'subject'], inplace=True)
    df2.set_index(['id', 'subject'], inplace=True)

    # Find common indices (rows that exist in both)
    common_idx = df1.index.intersection(df2.index)

    # Replace those rows entirely with the data from df2
    df1.loc[common_idx] = df2.loc[common_idx]

    # Reset index to bring 'id' and 'subject' back as columns
    df1.reset_index(inplace=True)

    # Save the updated CSV
    df1.to_csv(f'mmlu_{i}.csv', index=False)