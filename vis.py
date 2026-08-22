import glob
import re
import pandas as pd
import numpy as np

RESULTS_DIR = "."  # change if your csvs live elsewhere
FILENAME_RE = re.compile(r"(medmcqa|medqa)_results(?:_([\d.]+))?\.csv$")

PROB_COLUMNS_BY_LETTER = ["probA", "probB", "probC", "probD", "probE"]


def accuracy_for_file(path):
    df = pd.read_csv(path)

    prob_cols = [c for c in PROB_COLUMNS_BY_LETTER if c in df.columns]
    if not prob_cols:
        raise ValueError(f"No prob columns found in {path}")

    predicted = df[prob_cols].to_numpy().argmax(axis=1)
    answer = df["answer"].astype(int).to_numpy()
    correct = predicted == answer

    error_rate = df["error"].astype(str).str.lower().eq("true").mean() if "error" in df.columns else float("nan")
    avg_time = df["time"].mean() if "time" in df.columns else float("nan")

    return {
        "n": len(df),
        "accuracy": correct.mean(),
        "avg_time_sec": avg_time,
        "error_rate": error_rate,
    }


def main():
    rows = []
    for path in sorted(glob.glob(f"{RESULTS_DIR}/*_results*.csv")):
        m = FILENAME_RE.search(path)
        if not m:
            continue
        dataset, threshold = m.group(1), m.group(2)
        threshold = float(threshold) if threshold is not None else np.nan

        stats = accuracy_for_file(path)
        rows.append({"dataset": dataset, "threshold": threshold, "file": path, **stats})

    if not rows:
        print(f"No matching result files found under {RESULTS_DIR!r}.")
        return

    summary = pd.DataFrame(rows).sort_values(["dataset", "threshold"]).reset_index(drop=True)

    print("\n=== Summary (one row per dataset x threshold) ===")
    print(summary.to_string(index=False, formatters={
        "accuracy": "{:.4f}".format,
        "avg_time_sec": "{:.3f}".format,
        "error_rate": "{:.4f}".format,
    }))
    summary.to_csv("accuracy_summary.csv", index=False)

    pivot = summary.pivot_table(index="dataset", columns="threshold", values="accuracy")
    print("\n=== Accuracy by dataset x threshold ===")
    print(pivot.to_string(float_format="{:.4f}".format))
    pivot.to_csv("accuracy_pivot.csv")

    print("\nSaved accuracy_summary.csv and accuracy_pivot.csv")


if __name__ == "__main__":
    main()