"""
E6 — Validate linguistic fingerprint as a predictor of persona-stability.

The question E6 answers: do the linguistic features from A1-A5 (justification
length, within-trajectory diversity, keyword timing, lexical markers) predict
the trajectory's defection rate? If yes, the fingerprint is a quantitative
behavioral identification tool, not just descriptive.

Three sub-analyses:

  (a) Within-trajectory correlation: does within-trajectory cosine diversity
      correlate with D_raw across all moral-persona trajectories? (n ~ 180)

  (b) Cross-validated classifier: can we predict whether a trajectory has
      D_raw > 0.3 from text features alone, using leave-one-MODEL-out CV?
      If yes, the fingerprint generalizes beyond the training models.

  (c) Feature importance: which text features contribute the most signal?

No new data collection. This runs on the existing E4 trajectories.

Usage:
  python e6_fingerprint_validate.py
  python e6_fingerprint_validate.py --csv-dir csvs --fig-dir figures

Requires:
  pip install pandas scikit-learn matplotlib numpy
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from scipy.stats import pearsonr, spearmanr

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["font.size"] = 9


MORAL_PERSONAS = ["deontologist", "utilitarian", "virtue_ethicist"]


def load_data(csv_dir):
    """Load trajectory_text_metrics + trajectory_metrics, merge."""
    tt = pd.read_csv(csv_dir / "trajectory_text_metrics.csv")
    tm = pd.read_csv(csv_dir / "trajectory_metrics.csv")
    # join on identifying columns
    key = ["model", "persona", "opponent", "seed"]
    merged = tt.merge(tm[key + ["D_raw", "D_star", "L", "R", "agent_total"]], on=key)
    return merged


def analyze_correlations(df):
    """A6a — Correlate text features with D_raw across all moral trajectories."""
    print("\n" + "=" * 80)
    print("E6a — Linguistic feature correlations with D_raw (moral personas only)")
    print("=" * 80)
    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()
    if moral.empty:
        print("  no moral-persona data"); return

    feature_cols = [
        "mean_just_len", "max_just_len", "n_mismatches", "mismatch_rate",
        "within_traj_div_mean", "within_traj_drift_1toT",
        "kw_first_principle", "kw_first_exploitation", "kw_first_tournament",
        "kw_first_self_concern", "kw_first_consequence", "kw_first_rationalize",
    ]

    print(f"\n  Pearson and Spearman correlations with D_raw (n={len(moral)}):")
    print(f"  {'feature':<28}{'Pearson':>10}{'p':>10}{'Spearman':>12}{'p':>10}{'n':>6}")
    print("  " + "-" * 76)
    results = []
    for f in feature_cols:
        if f not in moral.columns:
            continue
        sub = moral.dropna(subset=[f, "D_raw"])
        if len(sub) < 5:
            continue
        try:
            r_p, p_p = pearsonr(sub[f], sub["D_raw"])
            r_s, p_s = spearmanr(sub[f], sub["D_raw"])
        except Exception:
            continue
        flag = "  *" if abs(r_p) > 0.3 else ""
        print(f"  {f:<28}{r_p:>+10.3f}{p_p:>10.4f}{r_s:>+12.3f}{p_s:>10.4f}{len(sub):>6}{flag}")
        results.append({"feature": f, "pearson_r": r_p, "pearson_p": p_p,
                         "spearman_r": r_s, "spearman_p": p_s, "n": len(sub)})

    # Vs AllD only — where signal should be strongest
    print(f"\n  Same, restricted to vs AllD (n={len(moral[moral['opponent']=='AllD'])}):")
    print(f"  {'feature':<28}{'Pearson':>10}{'p':>10}{'n':>6}")
    print("  " + "-" * 60)
    sub = moral[moral["opponent"] == "AllD"]
    for f in feature_cols:
        if f not in sub.columns:
            continue
        dat = sub.dropna(subset=[f, "D_raw"])
        if len(dat) < 5:
            continue
        try:
            r_p, p_p = pearsonr(dat[f], dat["D_raw"])
        except Exception:
            continue
        flag = "  *" if abs(r_p) > 0.3 else ""
        print(f"  {f:<28}{r_p:>+10.3f}{p_p:>10.4f}{len(dat):>6}{flag}")

    return results


def predict_hypocrisy(df, fig_dir):
    """A6b — Predict trajectory-level high-defection from text features."""
    print("\n" + "=" * 80)
    print("E6b — Predict D_raw > 0.3 from text features alone")
    print("=" * 80)
    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()

    feature_cols = [
        "mean_just_len", "max_just_len", "mismatch_rate",
        "within_traj_div_mean", "within_traj_drift_1toT",
        "kw_first_principle", "kw_first_exploitation", "kw_first_tournament",
        "kw_first_self_concern", "kw_first_consequence", "kw_first_rationalize",
    ]
    feature_cols = [c for c in feature_cols if c in moral.columns]
    moral = moral.dropna(subset=feature_cols + ["D_raw"])

    if len(moral) < 30:
        print(f"  Too few trajectories ({len(moral)}); skipping.")
        return

    X = moral[feature_cols].values
    y = (moral["D_raw"] > 0.3).astype(int).values
    groups = moral["model"].values

    print(f"\n  n trajectories: {len(moral)}")
    print(f"  n positive (D_raw > 0.3): {y.sum()} ({y.mean():.0%})")
    print(f"  features: {len(feature_cols)}")
    print(f"  groups (models): {sorted(set(groups))}")

    # Standard CV (within-distribution): generous baseline
    print("\n  Within-distribution 5-fold AUC:")
    for clf_name, clf in [("logistic", LogisticRegression(max_iter=500)),
                          ("random_forest", RandomForestClassifier(n_estimators=200, random_state=0))]:
        Xs = StandardScaler().fit_transform(X) if clf_name == "logistic" else X
        try:
            scores = cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc")
            print(f"    {clf_name:<20} AUC = {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"    {clf_name:<20} failed: {e}")

    # Leave-one-MODEL-out: tests whether the fingerprint generalizes across models
    print("\n  Leave-one-MODEL-out (does fingerprint generalize to unseen models?):")
    logo = LeaveOneGroupOut()
    for clf_name, clf in [("logistic", LogisticRegression(max_iter=500)),
                          ("random_forest", RandomForestClassifier(n_estimators=200, random_state=0))]:
        per_model_scores = []
        for train_idx, test_idx in logo.split(X, y, groups):
            held_out = groups[test_idx[0]]
            if len(set(y[test_idx])) < 2:
                continue  # can't compute AUC if all labels same
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            if clf_name == "logistic":
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            try:
                clf.fit(Xtr, ytr)
                proba = clf.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, proba)
                per_model_scores.append((held_out, auc))
            except Exception as e:
                per_model_scores.append((held_out, None))
        print(f"\n    {clf_name}:")
        for held_out, auc in per_model_scores:
            s = f"{auc:.3f}" if auc is not None else "(no variance in test labels)"
            print(f"      held out = {held_out:<30} AUC = {s}")
        valid = [a for _, a in per_model_scores if a is not None]
        if valid:
            print(f"      mean across models  AUC = {sum(valid)/len(valid):.3f}")

    # Feature importance from RF on full data
    print("\n  Feature importance (RandomForest, fit on full data):")
    rf = RandomForestClassifier(n_estimators=500, random_state=0)
    rf.fit(X, y)
    importances = sorted(zip(feature_cols, rf.feature_importances_),
                          key=lambda x: -x[1])
    for f, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"    {f:<28}{imp:.3f}  {bar}")

    # Save importance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fcs = [f for f, _ in importances]
    imps = [i for _, i in importances]
    ax.barh(range(len(fcs)), imps, color="#4285f4")
    ax.set_yticks(range(len(fcs)))
    ax.set_yticklabels(fcs)
    ax.invert_yaxis()
    ax.set_xlabel("RF feature importance")
    ax.set_title("E6 — Feature importance for predicting D_raw > 0.3")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_e6_feature_importance.png")
    plt.close()
    print(f"\n  Saved: {fig_dir / 'fig_e6_feature_importance.png'}")


def plot_diversity_vs_d(df, fig_dir):
    """Scatter: within-traj diversity vs D_raw, colored by model."""
    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()
    moral = moral.dropna(subset=["within_traj_div_mean", "D_raw"])
    if moral.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {
        "gpt-4o": "#10a37f",
        "gpt-4o-mini": "#7fc9b7",
        "gemini-2.5-pro": "#4285f4",
        "gemini-2.5-flash": "#a4c2f4",
    }
    markers = {"deontologist": "o", "utilitarian": "s", "virtue_ethicist": "^"}
    for model in moral["model"].unique():
        sub = moral[moral["model"] == model]
        for persona in sub["persona"].unique():
            ss = sub[sub["persona"] == persona]
            ax.scatter(ss["within_traj_div_mean"], ss["D_raw"],
                       label=f"{model} / {persona}",
                       color=palette.get(model, "#888"),
                       marker=markers.get(persona, "o"),
                       s=70, alpha=0.7, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Within-trajectory cosine diversity")
    ax.set_ylabel("D_raw (raw defection rate)")
    ax.set_title("E6 — Linguistic diversity vs defection rate (moral personas)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_e6_diversity_vs_d.png")
    plt.close()
    print(f"  Saved: {fig_dir / 'fig_e6_diversity_vs_d.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="csvs")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data(csv_dir)
    except FileNotFoundError as e:
        print(f"Missing CSV: {e}")
        print("Run e4_analyze.py first to generate trajectory_text_metrics.csv and trajectory_metrics.csv")
        return

    print(f"Loaded {len(df)} trajectories")

    analyze_correlations(df)
    predict_hypocrisy(df, fig_dir)
    plot_diversity_vs_d(df, fig_dir)

    print("\n=== E6 DONE ===")
    print("Key claims to look for:")
    print("  (a) Within-trajectory diversity correlates with D_raw r > 0.4")
    print("  (b) Leave-one-MODEL-out AUC > 0.7 (fingerprint generalizes)")
    print("  (c) Top features include within_traj_div_mean and kw_first_tournament")


if __name__ == "__main__":
    main()