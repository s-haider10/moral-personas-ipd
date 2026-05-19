"""
E4 plots — reads CSVs from e4_analyze.py and saves PNG figures.

Inputs: csvs/trajectory_metrics.csv, csvs/round_metrics.csv,
        csvs/trajectory_text_metrics.csv, csvs/defection_curves.csv,
        csvs/keyword_timing.csv, csvs/vocab_fingerprint.csv

Outputs (figures/):
  fig1_d_star_heatmap.png            — model × opponent D* heatmap (deont only)
  fig2_temporal_curves.png           — defection rate per round per model
  fig3_justification_length.png      — mean justification length per persona per model
  fig4_mismatch_heatmap.png          — say-do mismatch by model × persona
  fig5_keyword_timing.png            — first appearance of keywords by model
  fig6_within_traj_diversity.png     — within-trajectory drift, deontologist vs AllD
  fig7_payoff_vs_d_star.png          — equifinality plot: payoff vs D* (deont vs AllD)
  fig8_persona_violin.png            — D_raw distribution per persona per model

Usage:
  python e4_plots.py
  python e4_plots.py --csv-dir csvs --fig-dir figures
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False


MODEL_META = {
    "gpt-4o":                   {"provider": "openai", "generation": "prior",   "tier": "flagship"},
    "gpt-4o-mini":              {"provider": "openai", "generation": "prior",   "tier": "cheap"},
    "gpt-5.5":                  {"provider": "openai", "generation": "current", "tier": "flagship"},
    "gpt-5.4-mini":             {"provider": "openai", "generation": "current", "tier": "cheap"},
    "gemini-2.5-pro":           {"provider": "google", "generation": "prior",   "tier": "flagship"},
    "gemini-2.5-flash":         {"provider": "google", "generation": "prior",   "tier": "cheap"},
    "gemini-3.1-pro-preview":   {"provider": "google", "generation": "current", "tier": "flagship"},
    "gemini-3-flash-preview":   {"provider": "google", "generation": "current", "tier": "cheap"},
}

MORAL_PERSONAS = ["deontologist", "utilitarian", "virtue_ethicist"]
ALL_PERSONAS = MORAL_PERSONAS + ["selfish", "neutral"]


def model_order(models):
    """Return models sorted: provider then prior→current then flagship→cheap."""
    def key(m):
        info = MODEL_META.get(m, {})
        return (info.get("provider", "z"),
                info.get("generation", "") == "current",
                info.get("tier", "") == "cheap")
    return sorted(models, key=key)


def model_color(m):
    info = MODEL_META.get(m, {})
    if info.get("provider") == "openai":
        return "#10a37f" if info.get("generation") == "prior" else "#0a6b52"
    if info.get("provider") == "google":
        return "#4285f4" if info.get("generation") == "prior" else "#1f4f9e"
    return "#888"


# --------- figure 1: D* heatmap ---------
def fig1_d_star_heatmap(traj, fig_dir):
    df = traj[traj["persona"] == "deontologist"].copy()
    if df.empty:
        return
    pivot = df.pivot_table(index="model", columns="opponent",
                            values="D_star", aggfunc="mean")
    models = [m for m in model_order(pivot.index)]
    pivot = pivot.loc[models]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(models) + 1.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    ax.set_title("Fig 1. D* — deontologist normative defection rate, by model × opponent")
    plt.colorbar(im, ax=ax, label="D*")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_d_star_heatmap.png")
    plt.close()


# --------- figure 2: temporal curves ---------
def fig2_temporal_curves(curves, fig_dir):
    df = curves[(curves["persona"] == "deontologist") & (curves["opponent"] == "AllD")].copy()
    if df.empty:
        return
    models = model_order(df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        sub = df[df["model"] == m].sort_values("round")
        ax.plot(sub["round"], sub["defection_rate"],
                marker="o", markersize=4, linewidth=1.5,
                label=m, color=model_color(m))
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of trajectories defecting")
    ax.set_title("Fig 2. Round-by-round defection rate — deontologist vs AllD")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_temporal_curves.png")
    plt.close()


# --------- figure 3: justification length ---------
def fig3_justification_length(text, fig_dir):
    df = text.copy()
    models = model_order(df["model"].unique())
    pivot = df.pivot_table(index="model", columns="persona",
                           values="mean_just_len", aggfunc="mean")
    pivot = pivot.reindex(models)
    pivot = pivot[[p for p in ALL_PERSONAS if p in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.15
    for i, persona in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[persona], width, label=persona)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Mean justification length (words)")
    ax.set_title("Fig 3. Justification length by model × persona")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_justification_length.png")
    plt.close()


# --------- figure 4: mismatch heatmap ---------
def fig4_mismatch_heatmap(text, fig_dir):
    df = text.dropna(subset=["mismatch_rate"]).copy()
    if df.empty:
        return
    pivot = df.pivot_table(index="model", columns="persona",
                           values="mismatch_rate", aggfunc="mean")
    pivot = pivot.reindex([m for m in model_order(pivot.index)])
    pivot = pivot[[p for p in ALL_PERSONAS if p in pivot.columns]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(pivot) + 1.5)))
    im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=max(0.3, pivot.values[~np.isnan(pivot.values)].max() if pivot.values.size else 0.3), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8)
    ax.set_title("Fig 4. Action–justification mismatch rate (says one, does other)")
    plt.colorbar(im, ax=ax, label="mismatch rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_mismatch_heatmap.png")
    plt.close()


# --------- figure 5: keyword timing ---------
def fig5_keyword_timing(kw, fig_dir):
    df = kw.dropna(subset=["first_round"]).copy()
    if df.empty:
        return
    df = df[df["persona"] == "deontologist"]
    df = df[df["opponent"].isin(["AllD"])] if "opponent" in df.columns else df
    pivot = df.pivot_table(index="model", columns="keyword_group",
                           values="first_round", aggfunc="mean")
    pivot = pivot.reindex([m for m in model_order(pivot.index)])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(pivot) + 1.5)))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > 10 else "black", fontsize=8)
    ax.set_title("Fig 5. Mean round of first keyword appearance (deont vs AllD)")
    plt.colorbar(im, ax=ax, label="round of first appearance")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_keyword_timing.png")
    plt.close()


# --------- figure 6: within-trajectory diversity ---------
def fig6_diversity(text, fig_dir):
    df = text.dropna(subset=["within_traj_div_mean"]).copy()
    if df.empty:
        return
    df = df[(df["persona"] == "deontologist") & (df["opponent"] == "AllD")]
    if df.empty:
        return
    models = model_order(df["model"].unique())
    means = df.groupby("model")["within_traj_div_mean"].mean().reindex(models)
    drifts = df.groupby("model")["within_traj_drift_1toT"].mean().reindex(models)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.4
    ax.bar(x - w/2, means.values, w, label="Mean pairwise distance",
           color=[model_color(m) for m in models])
    ax.bar(x + w/2, drifts.values, w, label="Round 1 → Round T drift",
           color=[model_color(m) for m in models], hatch="///", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Cosine distance (sentence-transformer embeddings)")
    ax.set_title("Fig 6. Within-trajectory justification diversity (deont vs AllD)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_within_traj_diversity.png")
    plt.close()


# --------- figure 7: payoff vs D* (equifinality) ---------
def fig7_equifinality(traj, fig_dir):
    df = traj[(traj["persona"] == "deontologist") & (traj["opponent"] == "AllD")].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in model_order(df["model"].unique()):
        sub = df[df["model"] == m]
        ax.scatter(sub["D_star"], sub["agent_total"], s=80, alpha=0.7,
                   color=model_color(m), edgecolor="black", linewidth=0.5,
                   label=m)
    ax.set_xlabel("D* (normative defection rate)")
    ax.set_ylabel("Agent cumulative payoff")
    ax.set_title("Fig 7. Equifinality — payoff vs D* for deontologist vs AllD")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_payoff_vs_d_star.png")
    plt.close()


# --------- figure 8: persona D distribution ---------
def fig8_persona_violin(traj, fig_dir):
    df = traj[traj["opponent"] == "AllD"].copy()
    if df.empty:
        return
    personas = [p for p in ALL_PERSONAS if p in df["persona"].unique()]
    models = model_order(df["model"].unique())
    fig, axes = plt.subplots(1, len(models),
                              figsize=(2.2 * len(models), 4),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        data = [df[(df["model"] == m) & (df["persona"] == p)]["D_raw"].values
                for p in personas]
        positions = np.arange(len(personas))
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                         patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(model_color(m))
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(personas, rotation=35, ha="right", fontsize=7)
        ax.set_title(m, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("D_raw (defection rate vs AllD)")
    fig.suptitle("Fig 8. D_raw distribution across personas, per model (vs AllD)", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_persona_distribution.png", bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", default="csvs")
    p.add_argument("--fig-dir", default="figures")
    args = p.parse_args()

    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def maybe_load(name):
        p = csv_dir / name
        return pd.read_csv(p) if p.exists() else None

    traj = maybe_load("trajectory_metrics.csv")
    text = maybe_load("trajectory_text_metrics.csv")
    curves = maybe_load("defection_curves.csv")
    kw = maybe_load("keyword_timing.csv")

    if traj is None:
        print("trajectory_metrics.csv not found; run e4_analyze.py first."); return

    print("Generating plots...")
    if traj is not None:
        fig1_d_star_heatmap(traj, fig_dir); print("  fig1_d_star_heatmap.png")
        fig7_equifinality(traj, fig_dir); print("  fig7_payoff_vs_d_star.png")
        fig8_persona_violin(traj, fig_dir); print("  fig8_persona_distribution.png")
    if curves is not None:
        fig2_temporal_curves(curves, fig_dir); print("  fig2_temporal_curves.png")
    if text is not None:
        fig3_justification_length(text, fig_dir); print("  fig3_justification_length.png")
        fig4_mismatch_heatmap(text, fig_dir); print("  fig4_mismatch_heatmap.png")
        fig6_diversity(text, fig_dir); print("  fig6_within_traj_diversity.png")
    if kw is not None:
        fig5_keyword_timing(kw, fig_dir); print("  fig5_keyword_timing.png")
    print(f"\nFigures saved to {fig_dir}/")


if __name__ == "__main__":
    main()