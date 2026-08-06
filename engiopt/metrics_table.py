import pandas as pd

# Load the data
df = pd.read_csv("engiopt/all_metrics.csv")

# Base metrics (always present)
base_metrics = ["cog", "iog", "mmd", "dpp", "viol"]

# Auto-detect lv_* columns (mmd, dpp, sigma, iw variants, active dims per threshold combination)
lv_cols = [c for c in df.columns if c.startswith("lv_") or c.startswith("lvae_n_active")]

all_metrics = base_metrics + lv_cols

# Group by model_id and problem_id, compute mean and std
agg = df.groupby(["model_id", "problem_id"])[all_metrics].agg(["mean", "std"])

agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
agg = agg.reset_index()

# Write out with scientific notation (6 decimal places)
agg.to_csv("aggregated_metrics_sci.csv", index=False, float_format="%.2e")
print(agg.head())  # or further process as needed
