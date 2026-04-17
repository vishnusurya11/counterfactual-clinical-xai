"""Generate all paper figures from aggregated CSVs.

Outputs both PDF (for LaTeX) and PNG (for review) to results/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Wrote %s (pdf+png)", name)


def _load(agg_dir: Path, name: str) -> pd.DataFrame | None:
    path = agg_dir / name
    if not path.exists():
        logger.warning("Missing %s", path)
        return None
    return pd.read_csv(path)


def figure_1_esi_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=df, x="model", y="esi_score", ax=ax, inner="quartile", cut=0)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="High-consistency threshold")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel("ESI (mean pairwise cosine similarity)")
    ax.set_title("Figure 1: Explanation Stability Index distribution by model")
    plt.xticks(rotation=20, ha="right")
    ax.legend(loc="lower right")
    _save(fig, out_dir, "figure1_esi_distribution")


def figure_2_ac_vs_esi(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="esi_score", y="answer_consistency", hue="model", alpha=0.7, ax=ax)
    ax.axvline(0.8, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("ESI (explanation stability)")
    ax.set_ylabel("Answer consistency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 2: High AC + low ESI = rationalization danger zone")
    _save(fig, out_dir, "figure2_ac_vs_esi")


def figure_3_ect(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    agg = df.groupby("model")["cfs_score"].agg(["mean", "std"]).reset_index()
    sns.barplot(data=df, x="model", y="cfs_score", ax=ax, errorbar=("ci", 95))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_title("Figure 3: Fraction of cited concepts that are causally important")
    plt.xticks(rotation=20, ha="right")
    _save(fig, out_dir, "figure3_ect")


def figure_4_pss_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["answer_stability", "explanation_stability", "concept_stability", "confidence_drift"]
    available = [m for m in metrics if m in df.columns]
    agg = df.groupby("model")[available].mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(agg, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, cbar=True, ax=ax)
    ax.set_title("Figure 4: Perturbation Stability Score heatmap")
    plt.xticks(rotation=20, ha="right")
    _save(fig, out_dir, "figure4_pss_heatmap")


def figure_5_counterfactual(df: pd.DataFrame, out_dir: Path) -> None:
    agg = df.groupby("model").agg(
        validity=("counterfactual_valid", "mean"),
        minimality=("minimality", "mean"),
        plausibility=("plausibility", "mean"),
    ).reset_index()
    melted = agg.melt(id_vars="model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Figure 5: Counterfactual validity / minimality / plausibility")
    plt.xticks(rotation=20, ha="right")
    _save(fig, out_dir, "figure5_counterfactual")


def figure_6_bias(df: pd.DataFrame, out_dir: Path) -> None:
    agg = df.groupby("model")[["edd_score", "n_stereotype_flags"]].mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(agg, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar=True)
    ax.set_title("Figure 6: Demographic bias — EDD + stereotype flag rate")
    plt.xticks(rotation=20, ha="right")
    _save(fig, out_dir, "figure6_bias")


def figure_7_reliability_quadrant(df_esi: pd.DataFrame, df_ect: pd.DataFrame, out_dir: Path) -> None:
    """HERO FIGURE: 2D scatter of ESI (x) vs CFS (y)."""
    merged = df_esi.merge(df_ect, on=["question_id", "model"], how="inner")
    if merged.empty:
        logger.warning("Cannot build fig 7 — no overlapping question_ids between exp1 and exp2")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=merged, x="esi_score", y="cfs_score", hue="model", alpha=0.7, ax=ax, s=60)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ESI (explanation stability)")
    ax.set_ylabel("CFS (causal faithfulness)")
    ax.set_title("Figure 7: The Reliability Quadrant")

    # Quadrant labels
    ax.text(0.75, 0.95, "Reliable", ha="center", fontsize=10, fontweight="bold", color="green")
    ax.text(0.25, 0.95, "Unstable but honest", ha="center", fontsize=10, color="darkorange")
    ax.text(0.75, 0.05, "Stably wrong reasoning", ha="center", fontsize=10, color="darkred")
    ax.text(0.25, 0.05, "Unreliable", ha="center", fontsize=10, color="red")
    _save(fig, out_dir, "figure7_reliability_quadrant")


def main(poc: bool = False) -> None:
    cfg = load_config(ROOT / "config.yaml", poc=poc)
    agg_dir = ROOT / cfg["storage"]["aggregated_dir"]
    fig_dir = ROOT / cfg["storage"]["figures_dir"]

    suffix = "_poc" if poc else ""

    df1 = _load(agg_dir, f"exp1_esi_results{suffix}.csv")
    df2 = _load(agg_dir, f"exp2_ect_results{suffix}.csv")
    df3 = _load(agg_dir, f"exp3_pss_results{suffix}.csv")
    df4 = _load(agg_dir, f"exp4_counterfactual_results{suffix}.csv")
    df5 = _load(agg_dir, f"exp5_bias_results{suffix}.csv")

    if df1 is not None:
        figure_1_esi_distribution(df1, fig_dir)
        figure_2_ac_vs_esi(df1, fig_dir)
    if df2 is not None:
        figure_3_ect(df2, fig_dir)
    if df3 is not None:
        figure_4_pss_heatmap(df3, fig_dir)
    if df4 is not None:
        figure_5_counterfactual(df4, fig_dir)
    if df5 is not None:
        figure_6_bias(df5, fig_dir)
    if df1 is not None and df2 is not None:
        figure_7_reliability_quadrant(df1, df2, fig_dir)

    logger.info("All figures written to %s", fig_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--poc", action="store_true")
    args = p.parse_args()
    main(poc=args.poc)
