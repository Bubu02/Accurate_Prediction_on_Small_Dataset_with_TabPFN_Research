import re
import glob
import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
# 1) Where your six CSVs are:
INPUT_GLOB = "results_exp*.csv"       # e.g., results_exp1.csv ... results_exp6.csv

# 2) Which metrics to include in the WINS table (choose what matches your task)
#    For classification:
CLASSIFICATION_METRICS = ["ROC_AUC", "Accuracy", "F1", "CE", "ECE"]
#    For regression (uncomment and use these instead):
# REGRESSION_METRICS = ["MAE", "R2", "RMSE"]

# 3) Direction of better = higher (True) or lower (False)
#    Adjust if needed for your metrics.
METRIC_DIRECTION = {
    # classification
    "ROC_AUC": True,
    "Accuracy": True,
    "F1": True,
    "CE": False,
    "ECE": False,
    "Time_s": False,
    # regression
    "MAE": False,
    "R2": True,
    "RMSE": False,
}

# 4) Which set to use for wins (pick ONE line)
METRICS_FOR_WINS = CLASSIFICATION_METRICS
# METRICS_FOR_WINS = REGRESSION_METRICS

# 5) Output CSVs
SUMMARY_CSV = "summary_results.csv"
WINS_CSV    = "wins_table.csv"

# -----------------------
# HELPERS
# -----------------------
def normalize_col(c: str) -> str:
    """
    Normalize column names like:
      'Raw_Mean Normalized_ROC AUC' -> 'Raw_Mean_Normalized_ROC_AUC'
    and fix common variations.
    """
    c = c.strip()
    c = re.sub(r"\s+", "_", c)
    c = c.replace("-", "_")
    return c

def split_agg_col(c: str):
    """
    Try to parse a column into (split, agg, metric), e.g.
      'Raw_Mean_ROC_AUC'          -> ('Raw', 'Mean', 'ROC_AUC')
      'Processed_Mean_Normalized_ROC_AUC' -> ('Processed','Mean_Normalized','ROC_AUC')
    Returns None if doesn't match the pattern we care about.
    """
    c = normalize_col(c)
    # Accept Raw/Processed prefixes
    m = re.match(r"^(Raw|Processed)_(.+)$", c, flags=re.I)
    if not m:
        return None
    split = m.group(1).title()    # Raw/Processed
    rest  = m.group(2)

    # Try to peel off 'Mean_Normalized' or 'Mean'
    if rest.lower().startswith("mean_normalized_"):
        agg = "Mean_Normalized"
        metric = rest[len("Mean_Normalized_"):]
    elif rest.lower().startswith("mean_"):
        agg = "Mean"
        metric = rest[len("Mean_"):]
    elif rest.lower().startswith("wins_"):
        agg = "Wins"
        metric = rest[len("Wins_"):]
    else:
        # optionally you can capture other fields like times, etc.
        # We'll treat bare metrics under Mean by default
        agg = "Mean"
        metric = rest

    metric = normalize_col(metric)
    return split, agg, metric

def longify(df: pd.DataFrame, dataset_id: str):
    """
    Convert a wide one-dataset table into long format:
    columns -> Split / Agg / Metric, with value
    Keeps 'Model' as identifier.
    """
    df2 = df.copy()
    df2.columns = [normalize_col(c) for c in df2.columns]
    if "Model" not in df2.columns:
        raise ValueError("Input files must have a 'Model' column.")
    id_cols = ["Model"]
    long_rows = []
    for col in df2.columns:
        if col in id_cols:
            continue
        parsed = split_agg_col(col)
        if not parsed:
            # skip unknown columns safely
            continue
        split, agg, metric = parsed
        for model, val in zip(df2["Model"], df2[col]):
            long_rows.append({
                "Dataset": dataset_id,
                "Model": model,
                "Split": split,          # Raw / Processed
                "Agg": agg,              # Mean / Mean_Normalized / Wins
                "Metric": metric,        # ROC_AUC, Accuracy, ...
                "Value": val,
            })
    return pd.DataFrame(long_rows)

# -----------------------
# LOAD & COMBINE
# -----------------------
paths = sorted(glob.glob(INPUT_GLOB))
if not paths:
    raise SystemExit(f"No files matched {INPUT_GLOB}")

long_all = []
for p in paths:
    dataset_id = Path(p).stem    # use filename stem as dataset label
    wide = pd.read_csv(p)
    long_all.append(longify(wide, dataset_id))

long_df = pd.concat(long_all, ignore_index=True)

# -----------------------
# SUMMARY (mean ± std across datasets)
# -----------------------
# Only use Agg == 'Mean' for the mean±std table
mean_df = long_df[long_df["Agg"].eq("Mean")]

g = (mean_df
     .groupby(["Model","Split","Metric"])["Value"]
     .agg(['mean','std'])
     .reset_index())

# format mean ± std (3 decimals for mean, 2 for std)
g["mean_std"] = g["mean"].round(3).astype(str) + " ± " + g["std"].round(2).astype(str)

# Pivot to wide with columns like 'Processed|ROC_AUC', 'Raw|Accuracy', ...
summary_wide = g.pivot_table(index="Model",
                             columns=["Split","Metric"],
                             values="mean_std",
                             aggfunc="first").sort_index(axis=1)

# Flatten columns for CSV friendliness
summary_wide.columns = [f"{s}|{m}" for (s,m) in summary_wide.columns]
summary_wide.to_csv(SUMMARY_CSV)

# -----------------------
# WINS TABLE (per split & metric)
# -----------------------
# We compute wins per Dataset, Split, Metric on the 'Mean' values.
def compute_wins(df: pd.DataFrame, metrics):
    rows = []
    for split in ["Processed","Raw"]:
        for metric in metrics:
            sub = df[(df["Split"].eq(split)) &
                     (df["Metric"].eq(metric)) &
                     (df["Agg"].eq("Mean"))]
            if sub.empty:
                continue
            # find winners per dataset
            for ds, block in sub.groupby("Dataset"):
                # direction
                higher_is_better = METRIC_DIRECTION.get(metric, True)
                best_val = block["Value"].max() if higher_is_better else block["Value"].min()
                winners = block[block["Value"].eq(best_val)]["Model"].unique()
                for w in winners:
                    rows.append({"Model": w, "Split": split, "Metric": metric, "Win": 1})
    if not rows:
        return pd.DataFrame()
    wins = (pd.DataFrame(rows)
            .groupby(["Model","Split","Metric"])["Win"].sum()
            .reset_index())
    return wins

wins_long = compute_wins(long_df, METRICS_FOR_WINS)

# Pivot to wide layout like your screenshot (two blocks: Processed / Raw)
wins_wide = wins_long.pivot_table(index="Model",
                                  columns=["Split","Metric"],
                                  values="Win",
                                  fill_value=0,
                                  aggfunc="sum").sort_index(axis=1)

# Ensure missing combos appear as 0 columns (for consistent order)
all_cols = []
for split in ["Processed","Raw"]:
    for metric in METRICS_FOR_WINS:
        all_cols.append((split, metric))
for c in all_cols:
    if c not in wins_wide.columns:
        wins_wide[c] = 0
wins_wide = wins_wide[all_cols]

# Flatten columns as 'Processed|MAE', 'Raw|RMSE', etc., for CSV
wins_wide.columns = [f"{s}|{m}" for (s,m) in wins_wide.columns]
wins_wide.to_csv(WINS_CSV)
print(f"Saved:\n  - {SUMMARY_CSV}\n  - {WINS_CSV}")
