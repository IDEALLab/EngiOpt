"""Pull the Phase 1 ladder's train/val reconstruction and dimension counts from W&B.

The gate reads *training* NMSE, so that is what the thresholds were anchored on;
val NMSE is the trust diagnostic that says whether the resulting instrument
generalises. Handoff 3.6: report both, gate on neither.
"""

import argparse
import collections
import json
import statistics

import pandas as pd

import wandb

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--project", default="engibench/engiopt-lvladder")
_ap.add_argument("--out", default="ladder_results.csv")
_args = _ap.parse_args()
project, out_csv = _args.project, _args.out

api = wandb.Api()
rows = []
PERF_DISABLED = 100.0
"""nmse_threshold_perf at or above this disables the performance constraint entirely."""

for r in api.runs(project):
    s = r.summary  # this script's runs log hyperparameters into the summary, not the config
    if s.get("val_nmse_rec") is None:
        print(f"skip {r.name}: no val_nmse_rec (state={r.state})")
        continue
    # Run names are "{problem}__{algo}__{seed}__{timestamp}".
    problem, _algo, seed_str, *_ = r.name.split("__")
    rows.append(
        {
            "problem": problem,
            "perf": "OFF" if float(s.get("nmse_threshold_perf")) >= PERF_DISABLED else "ON",
            "thr_rec": float(s.get("nmse_threshold_rec")),
            "thr_perf": float(s.get("nmse_threshold_perf")),
            "seed": int(seed_str),
            "train_nmse": float(s.get("nmse_rec")),
            "val_nmse": float(s.get("val_nmse_rec")),
            "train_perf": float(s.get("nmse_perf")),
            "val_perf": float(s.get("val_nmse_perf")),
            "dims": int(s.get("active_dims", s.get("vol_active", -1))),
            "w_vol": float(s.get("w_vol")),
            "hf_path": s.get("hf_config_package_path"),
            "hf_rev": s.get("hf_config_revision"),
            "run": r.name,
        }
    )

df = pd.DataFrame(rows).sort_values(["problem", "perf", "thr_rec", "seed"])
df.to_csv(out_csv, index=False)
print(f"wrote {len(df)} rows -> {out_csv}\n")

# Group by arm; median over seeds, and the val/train ratio as the overfit diagnostic.
g = df.groupby(["problem", "perf", "thr_rec"], sort=False)
summary = g.agg(
    train=("train_nmse", "median"),
    val=("val_nmse", "median"),
    dims_lo=("dims", "min"),
    dims_hi=("dims", "max"),
    w_vol=("w_vol", "median"),
    n=("seed", "count"),
).reset_index()
summary["gap"] = summary["val"] / summary["train"]
pd.set_option("display.width", 160)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# Per-arm seed spread of the dimension count: an instrument whose dimension
# depends on the seed is not an instrument.
print("\n=== dimension stability across seeds ===")
for (p, mode, thr), sub in g:
    d = sorted(sub["dims"])
    cv = statistics.pstdev(d) / statistics.mean(d) if statistics.mean(d) else 0
    print(f"{p:18s} {mode:3s} thr={thr:<8.4g} dims={d!s:16s} cv={cv:.2f}")

print("\n=== checkpoint paths for metric analysis ===")
paths = collections.defaultdict(list)
for _, r in df.iterrows():
    paths[(r["problem"], r["perf"], r["thr_rec"])].append({"seed": r["seed"], "path": r["hf_path"], "rev": r["hf_rev"]})
with open("ladder_checkpoints.json", "w") as fh:
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in paths.items()}, fh, indent=2)
print("wrote ladder_checkpoints.json")
