"""
E6 ablation — leakage check.

The `mismatch_rate` feature requires knowing the action sequence to compute
(it compares stated intent against chosen action). Predicting D_raw using
mismatch_rate is therefore partially circular.

This script reruns the E6 classifier excluding mismatch_rate (and n_mismatches)
to verify the non-circular fingerprint still predicts D_raw.

Three configurations:
  (a) Full feature set (E6 baseline, 11 features) — includes mismatch
  (b) Action-blind features only (9 features) — no action information
  (c) Just the strongest action-blind feature (within_traj_div_mean alone)

If (b) AUC stays high, the methodological claim survives.
If (b) AUC collapses to chance, the contribution depends on the circular feature.

Usage:
  python e6_ablation.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

MORAL_PERSONAS = ["deontologist", "utilitarian", "virtue_ethicist"]

FULL_FEATURES = [
    "mean_just_len", "max_just_len", "mismatch_rate",
    "within_traj_div_mean", "within_traj_drift_1toT",
    "kw_first_principle", "kw_first_exploitation", "kw_first_tournament",
    "kw_first_self_concern", "kw_first_consequence", "kw_first_rationalize",
]

# Action-blind: features computable from chain-of-thought alone, without
# observing actions. mismatch_rate requires comparing stated intent (CoT)
# to action (output), so it's excluded.
ACTION_BLIND_FEATURES = [
    "mean_just_len", "max_just_len",
    "within_traj_div_mean", "within_traj_drift_1toT",
    "kw_first_principle", "kw_first_exploitation", "kw_first_tournament",
    "kw_first_self_concern", "kw_first_consequence", "kw_first_rationalize",
]


def load_data(csv_dir):
    tt = pd.read_csv(csv_dir / "trajectory_text_metrics.csv")
    tm = pd.read_csv(csv_dir / "trajectory_metrics.csv")
    key = ["model", "persona", "opponent", "seed"]
    return tt.merge(tm[key + ["D_raw"]], on=key)


def eval_config(df, features, label):
    print(f"\n--- {label} ({len(features)} features) ---")
    moral = df[df["persona"].isin(MORAL_PERSONAS)].copy()
    moral = moral.dropna(subset=features + ["D_raw"])
    if len(moral) < 30:
        print(f"  too few trajectories ({len(moral)})"); return

    X = moral[features].values
    y = (moral["D_raw"] > 0.3).astype(int).values
    groups = moral["model"].values

    print(f"  n={len(moral)}, positives={y.sum()} ({y.mean():.0%})")

    # 5-fold within-distribution
    print("  5-fold CV:")
    for name, clf in [("logistic", LogisticRegression(max_iter=500)),
                      ("random_forest", RandomForestClassifier(n_estimators=300, random_state=0))]:
        Xs = StandardScaler().fit_transform(X) if name == "logistic" else X
        scores = cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc")
        print(f"    {name:<18} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # Leave-one-MODEL-out
    print("  Leave-one-MODEL-out:")
    logo = LeaveOneGroupOut()
    for name, clf in [("logistic", LogisticRegression(max_iter=500)),
                      ("random_forest", RandomForestClassifier(n_estimators=300, random_state=0))]:
        aucs = []
        for train_idx, test_idx in logo.split(X, y, groups):
            if len(set(y[test_idx])) < 2:
                continue
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            if name == "logistic":
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
            aucs.append((groups[test_idx[0]], roc_auc_score(yte, proba)))
        for m, a in aucs:
            print(f"    {name} held-out {m:<28} AUC = {a:.3f}")
        valid = [a for _, a in aucs]
        if valid:
            print(f"    {name} mean AUC = {sum(valid)/len(valid):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", default="csvs")
    args = p.parse_args()
    df = load_data(Path(args.csv_dir))
    print(f"Loaded {len(df)} trajectories")

    eval_config(df, FULL_FEATURES, "(a) FULL (11 features, includes mismatch_rate)")
    eval_config(df, ACTION_BLIND_FEATURES, "(b) ACTION-BLIND (10 features, no mismatch)")
    eval_config(df, ["within_traj_div_mean"], "(c) DIVERSITY ALONE (1 feature)")

    print("\n=== ABLATION DONE ===")
    print("Compare (a) vs (b): if AUC drops a lot, mismatch_rate was carrying signal.")
    print("Compare (b) vs (c): if (b) >> (c), keyword timing and length features add real value.")


if __name__ == "__main__":
    main()