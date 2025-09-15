import re
import pandas as pd
from pathlib import Path

# ====== your list of CSVs ======
files = [
    "Comparing Results/australian_results.csv",
    "Comparing Results/blood_transfusion_results.csv",
    "Comparing Results/chum_results.csv",
    "Comparing Results/cmc_results.csv",
    "Comparing Results/credit_g_results.csv",
    "Comparing Results/heart_results.csv"
]

# ====== CONFIG (adjust only if needed) ======
# Choose which metrics to show in the WINS table
CLASSIFICATION_METRICS = ["ROC_AUC", "Accuracy", "F1", "CE", "ECE"]
REGRESSION_METRICS     = ["MAE", "R2", "RMSE"]
METRICS_FOR_WINS       = CLASSIFICATION_METRICS   # or REGRESSION_METRICS

# Direction: higher is better (True) / lower is better (False)
METRIC_DIRECTION = {
    "ROC_AUC": True, "Accuracy": True, "F1": True, "CE": False, "ECE": False,
    "Time_s": False, "MAE": False, "R2": True, "RMSE": False,
}

# ====== HELPERS ======
def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s

def _parse_col(c: str):
    """
    Accepts columns like:
      Raw_Mean_ROC_AUC
      Processed_Mean Normalized_ROC AUC
      Raw_Wins_RMSE
    Returns (Split, Agg, Metric) or None.
    """
    c = _norm(c)
    m = re.match(r"^(Raw|Processed)_(.+)$", c, flags=re.I)
    if not m:
        return None
    split = m.group(1).title()
    rest  = m.group(2)

    if rest.lower().startswith("mean_normalized_"):
        agg = "Mean_Normalized"; metric = rest[len("mean_normalized_"):]
    elif rest.lower().startswith("mean_"):
        agg = "Mean"; metric = rest[len("mean_"):]
    elif rest.lower().startswith("wins_"):
        agg = "Wins"; metric = rest[len("wins_"):]
    else:
        agg = "Mean"; metric = rest
    return split, agg, _norm(metric)

def _longify(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    if "Model" not in df.columns:
        raise ValueError(f"{dataset_id}: missing 'Model' column")

    rows = []
    for col in df.columns:
        if col == "Model":
            continue
        parsed = _parse_col(col)
        if not parsed:
            continue
        split, agg, metric = parsed
        for model, val in zip(df["Model"], df[col]):
            rows.append({
                "Dataset": dataset_id,
                "Model": model,
                "Split": split,     # Raw | Processed
                "Agg": agg,         # Mean | Mean_Normalized | Wins
                "Metric": metric,   # ROC_AUC | Accuracy | ...
                "Value": val,
            })
    return pd.DataFrame(rows)

# ====== LOAD & COMBINE ======
parts = []
for p in files:
    dataset_id = Path(p).stem
    df = pd.read_csv(p)
    parts.append(_longify(df, dataset_id))
long_df = pd.concat(parts, ignore_index=True)

# ====== SUMMARY (mean ± std across datasets) ======
# Use the “Mean” columns (switch to "Mean_Normalized" if you prefer)
mean_df = long_df[long_df["Agg"].eq("Mean")]

agg = (mean_df
       .groupby(["Model","Split","Metric"])["Value"]
       .agg(['mean','std'])
       .reset_index())
agg["mean_std"] = agg["mean"].round(3).astype(str) + " ± " + agg["std"].round(2).astype(str)

summary_wide = agg.pivot_table(index="Model",
                               columns=["Split","Metric"],
                               values="mean_std",
                               aggfunc="first").sort_index(axis=1)
summary_wide.columns = [f"{s}|{m}" for (s,m) in summary_wide.columns]
summary_out = "summary_results.csv"
summary_wide.to_csv(summary_out)

# ====== WINS TABLE (ties count as wins) ======
def _wins_table(df: pd.DataFrame, metrics):
    rows = []
    base = df[df["Agg"].eq("Mean")]   # or "Mean_Normalized"
    for split in ["Processed","Raw"]:
        for metric in metrics:
            sub = base[(base["Split"].eq(split)) & (base["Metric"].eq(metric))]
            if sub.empty:
                continue
            for ds, block in sub.groupby("Dataset"):
                higher = METRIC_DIRECTION.get(metric, True)
                best = block["Value"].max() if higher else block["Value"].min()
                winners = block[block["Value"].eq(best)]["Model"].unique()
                for w in winners:
                    rows.append({"Model": w, "Split": split, "Metric": metric, "Win": 1})
    if not rows:
        return pd.DataFrame(columns=["Model","Split","Metric","Win"])
    return (pd.DataFrame(rows)
            .groupby(["Model","Split","Metric"])["Win"].sum()
            .reset_index())

wins_long = _wins_table(long_df, METRICS_FOR_WINS)
wins_wide = wins_long.pivot_table(index="Model",
                                  columns=["Split","Metric"],
                                  values="Win",
                                  fill_value=0,
                                  aggfunc="sum").sort_index(axis=1)

# enforce column order: Processed block then Raw block (like your screenshot)
all_cols = [(s, m) for s in ["Processed","Raw"] for m in METRICS_FOR_WINS]
for c in all_cols:
    if c not in wins_wide.columns:
        wins_wide[c] = 0
wins_wide = wins_wide[all_cols]
wins_wide.columns = [f"{s}|{m}" for (s,m) in wins_wide.columns]
wins_out = "wins_table.csv"
wins_wide.to_csv(wins_out)

print(f"Saved:\n  - {summary_out}\n  - {wins_out}")