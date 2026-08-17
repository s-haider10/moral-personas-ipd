"""Faceted per-layer M3 figures.

The built-in M3 figures collapse across layers (group only by alpha/persona),
which hides the layer-sweep story. This script reads the combined M3 CSVs and
draws one panel per intervention layer so the dose-response can be compared
across layers 8 / 16 / 24.

Usage:
  python m3_facet_figure.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("csvs")
FIG_DIR = Path("figures")


def facet_steering(df):
    layers = sorted(df["layer_idx"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, L in zip(axes, layers):
        sub = df[df["layer_idx"] == L]
        for persona in sorted(sub["persona"].unique()):
            s = sub[sub["persona"] == persona]
            g = s.groupby("alpha")["D_raw"]
            mu, se = g.mean(), g.sem()
            ax.errorbar(mu.index, mu.values * 100, yerr=se.values * 100,
                        marker="o", capsize=3, label=persona)
        ax.axvline(0, ls="--", c="gray", alpha=0.5)
        ax.set_title(f"layer {L}")
        ax.set_xlabel("alpha (steering along v = integrity - phronesis)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Defection rate (%) vs AllD")
    axes[-1].legend()
    fig.suptitle("M3.1: Steering dose-response across intervention layers")
    fig.tight_layout()
    out = FIG_DIR / "fig_m3_steering_by_layer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


def facet_patching(df):
    layers = sorted(df["layer_idx"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, L in zip(axes, layers):
        sub = df[df["layer_idx"] == L]
        rows = (sub.groupby(["source_persona", "target_persona"])["D_raw"]
                .agg(["mean", "sem"]).reset_index())
        labels = [f"{r.source_persona[7:]}→{r.target_persona[7:]}"
                  for r in rows.itertuples()]
        x = list(range(len(rows)))
        ax.bar(x, rows["mean"].values * 100,
               yerr=rows["sem"].values * 100, capsize=4, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"layer {L}")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Defection rate (%) vs AllD")
    fig.suptitle("M3.2: Activation patching across intervention layers "
                 "(src→tgt persona)")
    fig.tight_layout()
    out = FIG_DIR / "fig_m3_patching_by_layer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


def main():
    s_path = CSV_DIR / "M3_steering.csv"
    p_path = CSV_DIR / "M3_patching.csv"
    if s_path.exists():
        facet_steering(pd.read_csv(s_path))
    else:
        print(f"missing {s_path}")
    if p_path.exists():
        facet_patching(pd.read_csv(p_path))
    else:
        print(f"missing {p_path}")


if __name__ == "__main__":
    main()
