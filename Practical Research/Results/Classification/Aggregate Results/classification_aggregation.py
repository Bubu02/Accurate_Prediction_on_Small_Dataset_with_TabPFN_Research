import pandas as pd

# Suppose you have all six result tables as CSVs
files = [
    "Comparing Results/australian_dataset_summary_raw_and_processed_combined.csv",
    "Comparing Results/blood_transfusion_raw_and_processed_combined.csv",
    "Comparing Results/chum_dataset_summary_raw_and_processed_combined.csv",
    "Comparing Results/cmc_dataset_summary_raw_and_processed_combined (1).csv",
    "Comparing Results/credit_g_dataset_summary_raw_and_processed_combined (1).csv",
    "Comparing Results/heart_dataset_summary_raw_and_processed_combined.csv"
]

# Load and stack
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, keys=[f"Exp{i+1}" for i in range(len(dfs))], names=["Experiment"])

# Compute mean and std across experiments per model
summary = combined.groupby("Model").agg(['mean','std'])

# Flatten multiindex columns
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]

# Format mean ± std
final = pd.DataFrame(index=summary.index)
for col in summary.columns:
    if col.endswith("_mean"):
        base = col.replace("_mean","")
        final[base] = summary[f"{base}_mean"].round(3).astype(str) + " ± " + summary[f"{base}_std"].round(2).astype(str)

# Save as CSV
final.to_csv("summary_results.csv")
