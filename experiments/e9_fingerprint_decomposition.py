"""
E9 (scoped) — What does the linguistic fingerprint actually identify?

The question E9 answers: is the behavioral fingerprint B(π) primarily picking up
(a) the model's identity, or (b) the persona's operationalization? E6 showed
the fingerprint predicts D_raw across unseen models (AUC 0.91). E9 asks the
complementary question: within a fixed persona, can the fingerprint identify
which model produced a trajectory? If yes, models have their own signatures
independent of the prompt. If no, the fingerprint is dominated by what the
prompt makes models do, not by the model itself.

Three sub-analyses, all on existing E4 data (no new collection):

  A. Variance decomposition: for each behavioral feature, partition variance
     into (i) between-persona, (ii) between-model-within-persona, (iii) residual.
     Tells us which axis dominates.

  B. Two multi-class classifiers:
     - Model-ID classifier: predict model from text features, controlling for persona.
     - Persona-ID classifier: predict persona from text features, controlling for model.
     If the persona classifier outperforms the model classifier, the fingerprint
     is operationalization-driven.

  C. Cross-axis transfer: does a classifier trained on flagship models work on
     cheap models for the same persona? Does a classifier trained on one persona
     transfer to another? These tell us where the fingerprint generalizes.

Usage:
  python e9_fingerprint_decomposition.py
  python e9_fingerprint_decomposition.py --csv-dir csvs --fig-dir figures

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
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["font.size"] = 9


MORAL_PERSONAS = ["deontologist", "utilitarian", "virtue_ethicist"]
FEATURE_COLS = [
    "mean_just_len", "max_just_len", "mismatch_rate",
    "within_traj_div_mean", "within_traj_drift_1toT",
    "kw_first_principle", "kw_first_exploitation", "kw_first_tournament",
    "kw_first_self_concern", "kw_first_consequence", "kw_first_rationalize",
]


def load_data(csv_dir):
    tt = pd.read_csv(csv_dir / "trajectory_text_metrics.csv")
    tm = pd.read_csv(csv_dir / "trajectory_metrics.csv")
    key = ["model", "persona", "opponent", "seed"]
    return tt.merge(tm[key + ["D_raw", "D_star", "L", "R"]], on=key)


# ====================================================================
# Sub-analysis A — Variance decomposition
# ====================================================================

def variance_decomposition(df):
    """For each feature, partition variance into (persona, model|persona, residual)."""
    print("\n" + "=" * 90)
    print("E9a — Variance decomposition: how much variance is between-persona vs between-model?")
    print("=" * 90)

    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()
    print(f"\n  Working with {len(moral)} moral-persona trajectories across "
          f"{moral['model'].nunique()} models × {moral['persona'].nunique()} personas\n")

    print(f"  {'feature':<28}{'σ²_persona':>14}{'σ²_model|persona':>20}{'σ²_residual':>16}{'% persona':>12}")
    print("  " + "-" * 90)

    results = []
    for f in FEATURE_COLS:
        if f not in moral.columns:
            continue
        sub = moral.dropna(subset=[f])
        if len(sub) < 20:
            continue

        # Total variance
        total_var = sub[f].var(ddof=0)
        if total_var == 0:
            continue

        # Between-persona variance: variance of per-persona means
        persona_means = sub.groupby("persona")[f].mean()
        persona_var = persona_means.var(ddof=0)

        # Between-model-within-persona: avg variance of per-(model,persona) means around per-persona mean
        cell_means = sub.groupby(["persona", "model"])[f].mean().reset_index()
        merged = cell_means.merge(persona_means.rename("persona_mean"),
                                   left_on="persona", right_index=True)
        merged["dev_from_persona"] = merged[f] - merged["persona_mean"]
        model_var = (merged["dev_from_persona"] ** 2).mean()

        residual_var = max(0.0, total_var - persona_var - model_var)
        pct_persona = persona_var / total_var * 100 if total_var > 0 else 0

        print(f"  {f:<28}{persona_var:>14.3f}{model_var:>20.3f}{residual_var:>16.3f}{pct_persona:>11.0f}%")
        results.append({
            "feature": f,
            "var_persona": persona_var,
            "var_model_given_persona": model_var,
            "var_residual": residual_var,
            "pct_persona": pct_persona,
        })

    if results:
        avg_pct_persona = np.mean([r["pct_persona"] for r in results])
        print(f"\n  Mean %-variance attributable to persona axis: {avg_pct_persona:.0f}%")
        if avg_pct_persona > 50:
            print("  >>> The behavioral fingerprint is PERSONA-DOMINATED.")
            print("  >>> Features track the operationalization more than the model.")
        elif avg_pct_persona < 30:
            print("  >>> The behavioral fingerprint is MODEL-DOMINATED.")
            print("  >>> Features track model identity more than operationalization.")
        else:
            print("  >>> Mixed: persona and model contribute comparably.")

    return results


# ====================================================================
# Sub-analysis B — Two classifiers, head to head
# ====================================================================

def two_classifiers(df, fig_dir):
    """Compare a model-ID classifier vs a persona-ID classifier on the same data."""
    print("\n" + "=" * 90)
    print("E9b — Two classifiers: predict model vs predict persona, same features")
    print("=" * 90)

    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()
    moral = moral.dropna(subset=FEATURE_COLS + ["model", "persona"])
    X = moral[FEATURE_COLS].values
    y_model = moral["model"].values
    y_persona = moral["persona"].values

    n = len(moral)
    n_models = len(np.unique(y_model))
    n_personas = len(np.unique(y_persona))
    chance_model = 1.0 / n_models
    chance_persona = 1.0 / n_personas

    print(f"\n  n = {n}; {n_models} models, {n_personas} personas")
    print(f"  Chance accuracy: model = {chance_model:.0%}, persona = {chance_persona:.0%}\n")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def run(y, label):
        try:
            Xs = StandardScaler().fit_transform(X)
            log_scores = cross_val_score(
                LogisticRegression(max_iter=500),
                Xs, y, cv=skf, scoring="accuracy")
            rf_scores = cross_val_score(
                RandomForestClassifier(n_estimators=300, random_state=0),
                X, y, cv=skf, scoring="accuracy")
            print(f"  Classify {label}:")
            print(f"    logistic:      {log_scores.mean():.3f} ± {log_scores.std():.3f}")
            print(f"    random forest: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
            return log_scores.mean(), rf_scores.mean()
        except Exception as e:
            print(f"  Classify {label}: FAILED — {e}")
            return None, None

    log_model_acc, rf_model_acc = run(y_model, "MODEL (4-way)")
    print()
    log_persona_acc, rf_persona_acc = run(y_persona, "PERSONA (3-way)")

    # Normalize by chance to compare directly
    print(f"\n  Above-chance margin:")
    if log_model_acc is not None and log_persona_acc is not None:
        m_margin = log_model_acc - chance_model
        p_margin = log_persona_acc - chance_persona
        print(f"    logistic — model:   +{m_margin:.2f} above chance")
        print(f"    logistic — persona: +{p_margin:.2f} above chance")
        if p_margin > m_margin + 0.10:
            print("\n  >>> PERSONA is more identifiable than MODEL from text features.")
            print("  >>> Fingerprint is operationalization-dominated. Consistent with E9a.")
        elif m_margin > p_margin + 0.10:
            print("\n  >>> MODEL is more identifiable than PERSONA from text features.")
            print("  >>> Fingerprint has substantial model-specific signature.")
        else:
            print("\n  >>> Comparable: both axes carry signal.")

    # Per-class confusion matrix for model classifier (which models are confusable?)
    print("\n  Per-model accuracy (RandomForest, 5-fold):")
    try:
        from sklearn.model_selection import cross_val_predict
        rf = RandomForestClassifier(n_estimators=300, random_state=0)
        yhat = cross_val_predict(rf, X, y_model, cv=skf)
        labels = sorted(np.unique(y_model))
        cm = confusion_matrix(y_model, yhat, labels=labels)
        accs = cm.diagonal() / cm.sum(axis=1)
        for lab, acc in zip(labels, accs):
            print(f"    {lab:<28} acc = {acc:.0%}")

        # Save heatmap of confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm[i, j] / cm[i].sum() if cm[i].sum() else 0
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                         color="white" if v > 0.5 else "black", fontsize=8)
        ax.set_title("E9 — Model-ID confusion matrix (RF, 5-fold CV)")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_e9_model_confusion.png")
        plt.close()
        print(f"\n  Saved: {fig_dir / 'fig_e9_model_confusion.png'}")
    except Exception as e:
        print(f"  Confusion matrix failed: {e}")


# ====================================================================
# Sub-analysis C — Cross-axis transfer
# ====================================================================

def cross_axis_transfer(df):
    """Does a persona-classifier trained on flagships transfer to cheap models?"""
    print("\n" + "=" * 90)
    print("E9c — Cross-axis transfer: does the fingerprint generalize across tiers?")
    print("=" * 90)

    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()
    moral = moral.dropna(subset=FEATURE_COLS + ["persona", "model"])

    # Tier assignment
    tier = {
        "gpt-4o": "flagship", "gemini-2.5-pro": "flagship",
        "gpt-4o-mini": "cheap", "gemini-2.5-flash": "cheap",
    }
    moral["tier"] = moral["model"].map(tier)
    moral = moral.dropna(subset=["tier"])

    if moral.empty or moral["tier"].nunique() < 2:
        print("  Insufficient tier data."); return

    X = moral[FEATURE_COLS].values
    y_persona = moral["persona"].values

    # Train on flagships → test on cheap
    flag_idx = moral["tier"].values == "flagship"
    cheap_idx = moral["tier"].values == "cheap"
    if flag_idx.sum() < 10 or cheap_idx.sum() < 10:
        print("  Not enough trajectories per tier."); return

    sc = StandardScaler().fit(X[flag_idx])
    clf = LogisticRegression(max_iter=500)
    clf.fit(sc.transform(X[flag_idx]), y_persona[flag_idx])
    acc_flag_to_cheap = accuracy_score(y_persona[cheap_idx], clf.predict(sc.transform(X[cheap_idx])))

    sc = StandardScaler().fit(X[cheap_idx])
    clf = LogisticRegression(max_iter=500)
    clf.fit(sc.transform(X[cheap_idx]), y_persona[cheap_idx])
    acc_cheap_to_flag = accuracy_score(y_persona[flag_idx], clf.predict(sc.transform(X[flag_idx])))

    chance = 1.0 / 3
    print(f"\n  Persona classifier (3-way), cross-tier transfer:")
    print(f"    train flagship  → test cheap:    acc = {acc_flag_to_cheap:.0%} (chance = {chance:.0%})")
    print(f"    train cheap     → test flagship: acc = {acc_cheap_to_flag:.0%} (chance = {chance:.0%})")

    if acc_flag_to_cheap > 0.6 and acc_cheap_to_flag > 0.6:
        print("\n  >>> Fingerprint TRANSFERS across tiers — persona signature is tier-invariant.")
    elif acc_flag_to_cheap < 0.45 or acc_cheap_to_flag < 0.45:
        print("\n  >>> Fingerprint is TIER-DEPENDENT — flagships and cheap models express")
        print("  >>> the same persona through different linguistic patterns.")
    else:
        print("\n  >>> Partial transfer.")


# ====================================================================
# Main
# ====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", default="csvs")
    p.add_argument("--fig-dir", default="figures")
    args = p.parse_args()
    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data(csv_dir)
    except FileNotFoundError as e:
        print(f"Missing CSV: {e}"); return

    print(f"Loaded {len(df)} trajectories")

    variance_decomposition(df)
    two_classifiers(df, fig_dir)
    cross_axis_transfer(df)

    print("\n=== E9 DONE ===")
    print("Key claims to look for:")
    print("  (a) Mean % persona variance > 50: fingerprint is operationalization-dominated")
    print("  (b) Persona classifier accuracy > Model classifier accuracy (margin > 0.10)")
    print("  (c) Cross-tier transfer accuracy > 0.6 means fingerprint generalizes across tiers")


if __name__ == "__main__":
    main()