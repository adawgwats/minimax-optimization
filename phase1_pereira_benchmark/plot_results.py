"""Visualize Phase 1 results.

Reads results/raw_results.csv and writes six plots into results/plots/:
  01_win_loss_summary.png      — grouped bar of WIN/TIE/LOSS counts vs each baseline
  02_heatmap_wins_vs_mice.png  — dataset x mechanism, colored by wins - losses
  03_win_rate_by_missing_rate  — line plot, one series per mechanism
  04_scatter_minimax_vs_mice   — per-cell log-log scatter, colored by mechanism
  05_method_rank_boxplot       — per-cell rank distribution across all 9 methods
  06_mse_diff_pct_by_mechanism — per-cell (minimax - mice)/mice distribution

Usage: python plot_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze import aggregate, win_loss_vs_baseline

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)

MINIMAX = "minimax_score"
MICE = "mice"
ERM = "erm_sgd"
CC = "complete_case"

METHOD_ORDER = [
    "oracle", "mice", "knn_impute", "mean_impute",
    "heckman", "ipw_estimated", "erm_sgd", "minimax_score", "complete_case",
]
MECH_ORDER = [
    "MBOV_Lower", "MBOV_Higher", "MBOV_Centered", "MBOV_Stochastic",
    "MBUV", "MBIR_Frequentist", "MBIR_Bayesian",
]

sns.set_theme(style="whitegrid", context="talk")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    agg = aggregate(raw)
    return raw, agg


def plot_win_loss_summary(agg: pd.DataFrame) -> None:
    baselines = [(MICE, "vs MICE"), (ERM, "vs ERM (same SGD)"), (CC, "vs Complete-case")]
    rows = []
    for b, label in baselines:
        wl = win_loss_vs_baseline(agg, baseline=b, method=MINIMAX)
        counts = wl.outcome.value_counts().to_dict()
        for outcome in ("WIN", "TIE", "LOSS"):
            rows.append({"comparison": label, "outcome": outcome, "count": counts.get(outcome, 0)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {"WIN": "#2ca02c", "TIE": "#bdbdbd", "LOSS": "#d62728"}
    sns.barplot(
        data=df, x="comparison", y="count", hue="outcome",
        order=[lbl for _, lbl in baselines],
        hue_order=["WIN", "TIE", "LOSS"], palette=palette, ax=ax,
    )
    total = 350
    for container in ax.containers:
        labels = [f"{int(v.get_height())}\n({v.get_height()/total*100:.0f}%)" for v in container]
        ax.bar_label(container, labels=labels, fontsize=11, padding=3)
    ax.set_ylabel(f"Cells (of {total})")
    ax.set_xlabel("")
    ax.set_title("Minimax score: statistical outcome per cell (95% CI)")
    ax.set_ylim(0, total * 0.75)
    ax.legend(title=None, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS / "01_win_loss_summary.png", dpi=150)
    plt.close(fig)


def plot_heatmap_vs_mice(agg: pd.DataFrame) -> None:
    wl = win_loss_vs_baseline(agg, baseline=MICE, method=MINIMAX)
    score = wl.outcome.map({"WIN": 1, "TIE": 0, "LOSS": -1}).astype(float)
    wl = wl.assign(score=score)
    mat = wl.pivot_table(index="dataset", columns="mechanism", values="score", aggfunc="sum")
    mat = mat.reindex(columns=[m for m in MECH_ORDER if m in mat.columns])

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        mat, annot=True, fmt=".0f", center=0, cmap="RdBu",
        vmin=-5, vmax=5,
        cbar_kws={"label": "wins − losses (over 5 missing-rates)"},
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Minimax vs MICE: net wins per (dataset, mechanism)\nsummed over 10/20/40/60/80% missing rates (max=±5)")
    ax.set_xlabel("MNAR mechanism")
    ax.set_ylabel("Dataset")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "02_heatmap_wins_vs_mice.png", dpi=150)
    plt.close(fig)


def plot_win_rate_by_rate(agg: pd.DataFrame) -> None:
    wl = win_loss_vs_baseline(agg, baseline=MICE, method=MINIMAX)
    rows = []
    for (mech, rate), sub in wl.groupby(["mechanism", "missing_rate_pct"]):
        total = len(sub)
        wins = (sub.outcome == "WIN").sum()
        losses = (sub.outcome == "LOSS").sum()
        rows.append({
            "mechanism": mech, "missing_rate_pct": rate,
            "win_rate": wins / total, "loss_rate": losses / total,
        })
    d = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, col, title, color in [
        (axes[0], "win_rate", "Win rate vs MICE", "#2ca02c"),
        (axes[1], "loss_rate", "Loss rate vs MICE", "#d62728"),
    ]:
        sns.lineplot(
            data=d, x="missing_rate_pct", y=col, hue="mechanism",
            hue_order=[m for m in MECH_ORDER if m in d.mechanism.unique()],
            marker="o", linewidth=2, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("Fraction of 10 datasets")
        ax.axhline(0.5, ls=":", color="gray", lw=1)
        if ax is axes[0]:
            ax.legend(fontsize=9, title="Mechanism", loc="upper left")
        else:
            ax.get_legend().remove()
    fig.suptitle("Minimax vs MICE as missingness grows", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_win_rate_by_missing_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_minimax_vs_mice(agg: pd.DataFrame) -> None:
    wide = agg.pivot_table(
        index=["dataset", "mechanism", "missing_rate_pct"],
        columns="method", values="mean_mse",
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 9))
    mechs = [m for m in MECH_ORDER if m in wide.mechanism.unique()]
    palette = sns.color_palette("tab10", n_colors=len(mechs))
    for mech, color in zip(mechs, palette):
        sub = wide[wide.mechanism == mech]
        ax.scatter(sub[MICE], sub[MINIMAX], label=mech, alpha=0.75, s=40, color=color, edgecolor="white", linewidth=0.5)

    lo, hi = 1e-5, 2.0
    x = np.geomspace(lo, hi, 100)
    ax.plot(x, x, "k--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("MICE test MSE  (per cell, mean of 10 seeds)")
    ax.set_ylabel("Minimax test MSE")
    ax.set_title("Per-cell MSE: points below the line = minimax wins\n(log-log)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "04_scatter_minimax_vs_mice.png", dpi=150)
    plt.close(fig)


def plot_method_rank(agg: pd.DataFrame) -> None:
    wide = agg.pivot_table(
        index=["dataset", "mechanism", "missing_rate_pct"],
        columns="method", values="mean_mse",
    )
    ranks = wide.rank(axis=1, method="average")
    melted = ranks.reset_index().melt(
        id_vars=["dataset", "mechanism", "missing_rate_pct"],
        var_name="method", value_name="rank",
    )
    order = [m for m in METHOD_ORDER if m in melted.method.unique()]
    medians = melted.groupby("method")["rank"].median().loc[order]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="method", y="rank", order=order, ax=ax, palette="Set3", showfliers=False)
    sns.stripplot(data=melted, x="method", y="rank", order=order, ax=ax, color="black", alpha=0.15, size=2)
    for i, m in enumerate(order):
        ax.text(i, 9.3, f"med={medians[m]:.1f}", ha="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_ylabel("Rank across 9 methods (1 = best MSE)")
    ax.set_xlabel("")
    ax.set_title("Per-cell method ranking across all 350 cells")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "05_method_rank_boxplot.png", dpi=150)
    plt.close(fig)


def plot_mse_diff_pct_by_mechanism(agg: pd.DataFrame) -> None:
    wl = win_loss_vs_baseline(agg, baseline=MICE, method=MINIMAX)
    wl["diff_clip"] = wl.mse_diff_pct.clip(-50, 200)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    mechs = [m for m in MECH_ORDER if m in wl.mechanism.unique()]
    sns.violinplot(data=wl, x="mechanism", y="diff_clip", order=mechs, ax=ax, inner=None, cut=0, color="#d9d9d9")
    sns.stripplot(data=wl, x="mechanism", y="diff_clip", order=mechs,
                  hue="outcome", hue_order=["WIN", "TIE", "LOSS"],
                  palette={"WIN": "#2ca02c", "TIE": "#555", "LOSS": "#d62728"},
                  dodge=False, ax=ax, size=4, alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("(minimax − MICE) / MICE   [%, clipped to ±50/200]")
    ax.set_xlabel("")
    ax.set_title("Per-cell relative MSE vs MICE, by MNAR mechanism\n(below 0 = minimax wins; clipped for visibility — CTG losses extend off-scale)")
    ax.legend(title=None, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "06_mse_diff_pct_by_mechanism.png", dpi=150)
    plt.close(fig)


def main() -> None:
    raw, agg = load()
    print(f"loaded {len(raw)} rows, {agg[['dataset','mechanism','missing_rate_pct']].drop_duplicates().shape[0]} cells")
    plot_win_loss_summary(agg)
    plot_heatmap_vs_mice(agg)
    plot_win_rate_by_rate(agg)
    plot_scatter_minimax_vs_mice(agg)
    plot_method_rank(agg)
    plot_mse_diff_pct_by_mechanism(agg)
    print(f"wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
