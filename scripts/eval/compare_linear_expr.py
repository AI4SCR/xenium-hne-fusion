"""Per-fold comparison: early fusion MLP-32 vs early fusion linear vs expr-only ViT-S (BEAT).

Run: UV_ENV_FILE=.env uv run python scripts/eval/compare_linear_expr.py   (needs network). Selection rationale: docs/status.md §4.

All arms of the controlled comparison were trained from commit c1759b4 on gpu-rtx with >= 35 epochs of budget:
  - W&B group `early-fusion-linear`: early fusion, expr ingestion Linear(G->384)
  - W&B group `c1759b4-baseline`:   stock early-fusion.yaml (MLP-32) and expr-token-vit.yaml (expr-only ViT-S)
LEGACY lists the older (wall-time truncated, older-commit) runs for reference only.
"""
import pandas as pd
import wandb

ENTITY = "chuv"
PROJECTS = ["xe-hne-fus-cell-v1", "xe-hne-fus-expr-v0"]
GROUPS = ["early-fusion-linear", "c1759b4-baseline"]
# Old baselines, one run per fold 0..3. Truncated by max_time (see docs/status.md §4); reference only.
LEGACY = {
    ("cell_types", "early_mlp32"): ["40utmvmw", "j5x4suw2", "owzcohia", "fif5wuld"],
    ("cell_types", "expr_vit_s"): ["w0y2mwg5", "005s9g5a", "pbi7jz39", "u2l1eisi"],
    ("hvg100", "early_mlp32"): ["uaepyp93", "hdkim0ma", "z2shyn16", "5a6crdzl"],
    ("hvg100", "expr_vit_s"): ["3nh0f2lw", "fk4lzu7l", "buzc1v5e", "juonbqiu"],
    ("hvg50", "early_mlp32"): ["gl0w9chl", "6361tn0m", "bngpxrkn", "4nwbprmb"],
}
METRICS = ["test/pearson_mean", "test/spearman_mean", "test/mse_mean", "val/pearson_mean"]


def task_of(cfg: dict) -> str:
    panel = cfg["data"]["panel_path"]
    if cfg["task"]["target"] == "cell_types":
        assert panel.endswith("default.yaml"), panel
        return "cell_types"
    n = len(cfg["data"]["target_panel"])
    assert n in (50, 100) and "/hvg-" in panel, (n, panel)
    return f"hvg{n}"


def model_of(cfg: dict) -> str:
    b = cfg["backbone"]
    if b["fusion_stage"] is None:
        assert b["expr_encoder_name"] == "vit_small_patch16_224" and b["morph_encoder_name"] is None, b
        return "expr_vit_s"
    assert b["fusion_stage"] == "early" and b["fusion_strategy"] == "add" and not b["freeze_morph_encoder"], b
    kws = b["expr_encoder_kws"]
    if kws["output_dim"] == 384 and kws["num_hidden_layers"] == 0 and not b["use_proj"]:
        return "early_linear"
    assert kws["output_dim"] == 32 and b["use_proj"], kws
    return "early_mlp32"


def row(r, source: str) -> dict:
    cfg = r.config
    assert "items/cells.json" in cfg["data"]["items_path"], cfg["data"]["items_path"]
    fold = int(cfg["data"]["metadata_path"].split("outer=")[1][0])
    return {"source": source, "task": task_of(cfg), "model": model_of(cfg), "fold": fold, "run_id": r.id,
            "state": r.state, "epoch": r.summary.get("epoch"), "runtime_h": r.summary.get("_runtime", 0) / 3600,
            **{k: r.summary.get(k) for k in METRICS}}


api = wandb.Api(timeout=120)
rows = []
for project in PROJECTS:
    for g in GROUPS:
        for lite in api.runs(f"{ENTITY}/{project}", filters={"group": g}):
            rows.append(row(api.run(f"{ENTITY}/{project}/{lite.id}"), "c1759b4"))  # lite runs have empty configs
for (task, model), ids in LEGACY.items():
    project = "xe-hne-fus-cell-v1" if task == "cell_types" else "xe-hne-fus-expr-v0"
    for rid in ids:
        rows.append(row(api.run(f"{ENTITY}/{project}/{rid}"), "legacy"))
df = pd.DataFrame(rows)
assert not df.duplicated(["source", "task", "model", "fold"]).any(), df[df.duplicated(["source", "task", "model", "fold"], keep=False)]
df.to_csv("compare_linear_expr.csv", index=False)

pd.set_option("display.width", 200)
print(df.sort_values(["source", "task", "model", "fold"])[["source", "task", "model", "fold", "run_id", "state", "epoch", "runtime_h"]].round(2).to_string(index=False))
done = df[df.state == "finished"]
for m in ["test/pearson_mean", "val/pearson_mean"]:
    t = done.pivot_table(index=["source", "task", "model"], columns="fold", values=m)
    t["mean"] = t.mean(axis=1)
    print(f"\n## {m} (finished runs only)\n{t.round(3).to_string()}")
p = done[done.source == "c1759b4"].pivot_table(index=["task", "fold"], columns="model", values="test/pearson_mean")
for base in ["early_mlp32", "expr_vit_s"]:
    if base in p and "early_linear" in p:
        print(f"\n## test Pearson, paired per fold: early_linear - {base}\n{(p['early_linear'] - p[base]).unstack('fold').round(3).to_string()}")
