"""
Fetch test scores for supervised training runs from W&B and plot per-setting boxplots.

Runs are grouped into "settings" (e.g. c_cells, d_cells, c_on_d, d_on_c) by matching
`data.metadata_path` against `plot.setting_pattern`. For each setting, one figure is
saved with a boxplot + black swarm overlay of `plot.metric`, one box per run name in
`plot.run_names`, pooling all outer folds together.

Usage:
    uv run python scripts/eval/boxplot.py \\
        --boxplot.project xe-hne-fus-expr-v0 \\
        --boxplot.name owkin \\
        --boxplot.data_dir $DATA_DIR
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

from xenium_hne_fusion.utils.getters import ManagedPaths


@dataclass
class FilterConfig:
    state: str = "finished"


@dataclass
class PlotConfig:
    metric: str = "test/spearman_mean"
    setting_pattern: str = r"(?P<setting>.+)/outer=\d+\.parquet$"
    run_names: list[str] = field(default_factory=lambda: ["vision", "early-fusion", "expr-resmlp", "expr-token-vit"])


@dataclass
class BoxplotConfig:
    entity: str = "chuv"
    project: str = None
    name: str = None
    data_dir: Path = None
    filter: FilterConfig = field(default_factory=FilterConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def extract_setting(metadata_path: str, pattern: str) -> str:
    setting = re.search(pattern, metadata_path)["setting"]
    return Path(setting).name


def build_dataframe(runs, name: str, metric: str, pattern: str) -> pd.DataFrame:
    records = []
    for run in runs:
        if run.config["data"]["name"] != name:
            continue
        records.append({
            "run_id": run.id,
            "run_name": run.config["wandb"]["name"],
            "setting": extract_setting(run.config["data"]["metadata_path"], pattern),
            "metric_value": run.summary[metric],
        })
    df = pd.DataFrame.from_records(records)
    assert not df.empty, f"no runs matched data.name={name!r}"
    return df


def plot_setting_boxplot(df_setting: pd.DataFrame, metric: str, run_names: list[str], out_path: Path) -> None:
    df_setting = df_setting[df_setting["run_name"].isin(run_names)]
    order = [r for r in run_names if r in set(df_setting["run_name"])]

    _, ax = plt.subplots(figsize=(max(6, len(order) * 1.5), 6))
    sns.boxplot(data=df_setting, x="run_name", y="metric_value", order=order, ax=ax)
    sns.swarmplot(data=df_setting, x="run_name", y="metric_value", order=order, color="black", ax=ax)
    ax.set_ylabel(metric)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)


def main(cfg: BoxplotConfig) -> int:
    api = wandb.Api()
    runs = api.runs(f"{cfg.entity}/{cfg.project}", filters={"state": cfg.filter.state})

    df = build_dataframe(runs, cfg.name, cfg.plot.metric, cfg.plot.setting_pattern)

    managed = ManagedPaths(data_dir=cfg.data_dir, name=cfg.name)
    out_dir = managed.figures_dir / "boxplot" / cfg.project
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir / "runs.parquet")
    for setting, df_setting in df.groupby("setting"):
        plot_setting_boxplot(df_setting, cfg.plot.metric, cfg.plot.run_names, out_dir / f"{setting}.png")

    return 0


def _build_parser():
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(BoxplotConfig, nested_key="boxplot")
    return parser


def cli(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = _build_parser()
    cfg = parser.parse_args(argv)
    init = parser.instantiate(cfg)
    return main(init.boxplot)


if __name__ == "__main__":
    import sys

    raise SystemExit(cli(sys.argv[1:]))
