"""Per-fold comparison: early fusion MLP-32 vs early fusion linear vs expr-only ViT-S (BEAT).

Run: UV_ENV_FILE=.env uv run python scripts/eval/compare_linear_expr.py   (needs network). Selection rationale: docs/status.md §4.
"""
import pandas as pd
import wandb

ENTITY = "chuv"
PROJECT = {"cell_types": "xe-hne-fus-cell-v1", "hvg100": "xe-hne-fus-expr-v0", "hvg50": "xe-hne-fus-expr-v0"}
# Baselines: one run per fold (outer=0..3), split files cells/outer=N-inner=0-seed=0.parquet.
BASELINES = {
    ("cell_types", "early_mlp32"): ["40utmvmw", "j5x4suw2", "owzcohia", "fif5wuld"],  # not ryydvint (conch items, no metrics)
    ("cell_types", "expr_vit_s"): ["w0y2mwg5", "005s9g5a", "pbi7jz39", "u2l1eisi"],
    ("hvg100", "early_mlp32"): ["uaepyp93", "hdkim0ma", "z2shyn16", "5a6crdzl"],
    ("hvg100", "expr_vit_s"): ["3nh0f2lw", "fk4lzu7l", "buzc1v5e", "juonbqiu"],
    ("hvg50", "early_mlp32"): ["gl0w9chl", "6361tn0m", "bngpxrkn", "4nwbprmb"],
    ("hvg50", "expr_vit_s"): [],  # not run yet
}
# New runs: group early-fusion-linear, submitted 2026-09-24 from worktree ../xenium-hne-fusion-c1759b4.
LINEAR = {
    "cell_types": ["ot1pgfni", "jpjepvt8", "tlo50d9c", "l8vs7ols"],
    "hvg100": ["yougc371", "c44l6rqk", "42sp5wbq", "fhd8ooec"],
    "hvg50": ["n0c3o6fr", "jihqpk5k", "otx0gemw", "dt030tk6"],
}
METRICS = ["test/pearson_mean", "test/spearman_mean", "test/mse_mean", "val/pearson_mean"]

api = wandb.Api(timeout=120)
runs = [(t, m, i) for (t, m), ids in BASELINES.items() for i in ids] + [(t, "early_linear", i) for t, ids in LINEAR.items() for i in ids]
rows = []
for task, model, rid in runs:
    r = api.run(f"{ENTITY}/{PROJECT[task]}/{rid}")
    fold = int(r.config["data"]["metadata_path"].split("outer=")[1][0])
    rows.append({"task": task, "model": model, "fold": fold, "run_id": rid, "state": r.state,
                 "epoch": r.summary.get("epoch"), **{k: r.summary.get(k) for k in METRICS}})
df = pd.DataFrame(rows)
assert df.groupby(["task", "model"]).fold.apply(lambda f: f.is_unique).all(), "duplicate fold"
df.to_csv("compare_linear_expr.csv", index=False)

pd.set_option("display.width", 200)
print(df[["task", "model", "fold", "run_id", "state", "epoch"]].to_string(index=False))
for m in ["test/pearson_mean", "val/pearson_mean"]:
    t = df.pivot_table(index=["task", "model"], columns="fold", values=m)
    t["mean"] = t.mean(axis=1)
    print(f"\n## {m}\n{t.round(3).to_string()}")
# Paired per-fold deltas (test Pearson) vs early_linear
p = df.pivot_table(index=["task", "fold"], columns="model", values="test/pearson_mean")
for base in ["early_mlp32", "expr_vit_s"]:
    if base in p and "early_linear" in p:
        d = (p["early_linear"] - p[base]).unstack("fold")
        print(f"\n## test Pearson: early_linear - {base}\n{d.round(3).to_string()}")
