"""
E6 strict leakage test.

The chain-of-thought often contains explicit statements of the action
("I will cooperate", "I will defect", "I choose C"). Any text-based predictor
of D_raw that operates over the full CoT is partially reading the answer.

This script reruns the diversity-alone predictor after stripping
action-declaration sentences from each justification. The pre-strip diversity
is computed on the full justification; the post-strip diversity is computed
on the reasoning portion only.

If post-strip AUC remains high, the linguistic fingerprint is detecting
something beyond stated intent — there's a real text signal independent of
the answer.

If post-strip AUC collapses to chance, E6's predictive claim was an artifact
of CoT containing the action verbatim.

Usage:
  python e6_strict_leakage.py
  python e6_strict_leakage.py --debug    # show 3 before/after examples
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Patterns that declare an action. Sentences containing any of these are stripped.
ACTION_DECLARATION_PATTERNS = [
    r"\bi (will|'ll|choose to|am going to|intend to|decide to) (cooperate|defect|play|choose)\b",
    r"\b(i will|i'll|i choose|i play|i pick|i select) (c|d)\b",
    r"\bi (cooperate|defect)\b",
    r"\b(cooperate|defect)(ing)? is (my|the right|the best|the correct)\b",
    r"\b(my|the) action (is|will be|should be) (c|d|cooperate|defect)\b",
    r"\b(choosing|picking|selecting) (c|d|cooperation|defection)\b",
    r"\bplay (c|d)\b",
    r"\b(continue to|continue) (cooperat|defect)",
    r"\b(stick with|stay with|maintain) (c|d|cooperation|defection)\b",
    r"\baction\s*:\s*(c|d)\b",
]
ACTION_RE = re.compile("|".join(ACTION_DECLARATION_PATTERNS), re.IGNORECASE)


def strip_action_sentences(text):
    """Remove sentences that explicitly state the action choice."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not ACTION_RE.search(s)]
    return " ".join(kept).strip()


def load_e4_trajectories(results_root):
    """Yield (model, persona, opponent, seed, justifications_list, D_raw)."""
    base = Path(results_root) / "E4"
    if not base.exists():
        print(f"  no E4 data at {base}")
        return
    for fp in sorted(base.rglob("*.jsonl")):
        meta = None
        justifications = []
        summary = None
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("type")
                if t == "meta":
                    meta = rec
                elif t == "round":
                    justifications.append(rec.get("justification") or "")
                elif t == "summary":
                    summary = rec
        if meta is None or summary is None:
            continue
        yield (meta["model"], meta["persona"], meta.get("opponent"),
               meta.get("seed"), justifications, summary["raw_defection_rate"])


def diversity(justifications, model):
    valid = [j for j in justifications if j and j.strip()]
    if len(valid) < 2:
        return None
    embs = model.encode(valid, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms
    sim = embs @ embs.T
    n = len(embs)
    triu = [1 - sim[i, j] for i in range(n) for j in range(i + 1, n)]
    return float(sum(triu) / len(triu)) if triu else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--debug", action="store_true",
                        help="show 3 before/after examples")
    args = parser.parse_args()

    print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    n_loaded = 0
    debug_shown = 0
    for model_name, persona, opponent, seed, justs, d_raw in load_e4_trajectories(args.results):
        if persona not in ("deontologist", "utilitarian", "virtue_ethicist"):
            continue
        n_loaded += 1
        # Compute pre-strip and post-strip diversity for this trajectory.
        stripped = [strip_action_sentences(j) for j in justs]

        # Debug print
        if args.debug and debug_shown < 3 and any(s != o for s, o in zip(stripped, justs)):
            print(f"\n--- DEBUG: {model_name} / {persona} / s{seed} ---")
            for i, (orig, strip) in enumerate(zip(justs[:3], stripped[:3])):
                print(f"  Round {i+1}:")
                print(f"    BEFORE: {orig[:200]}")
                print(f"    AFTER:  {strip[:200]}")
            debug_shown += 1

        # Length sanity: fraction of words removed
        before_words = sum(len((j or "").split()) for j in justs)
        after_words = sum(len(s.split()) for s in stripped)
        frac_kept = after_words / before_words if before_words > 0 else 0

        div_full = diversity(justs, emb_model)
        div_stripped = diversity(stripped, emb_model)

        rows.append({
            "model": model_name, "persona": persona, "opponent": opponent,
            "seed": seed, "D_raw": d_raw,
            "div_full": div_full, "div_stripped": div_stripped,
            "frac_words_kept": frac_kept,
        })

    if not rows:
        print(f"\nno data found"); return
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["div_full", "div_stripped", "D_raw"])
    print(f"\nLoaded {len(df)} moral-persona trajectories")

    # Word removal stats
    print(f"\nFraction of words kept after stripping action sentences:")
    print(f"  mean   = {df['frac_words_kept'].mean():.2f}")
    print(f"  median = {df['frac_words_kept'].median():.2f}")
    print(f"  min    = {df['frac_words_kept'].min():.2f}")

    # Correlation comparison
    from scipy.stats import pearsonr, spearmanr
    print("\n" + "=" * 80)
    print("CORRELATION OF DIVERSITY WITH D_RAW (moral personas)")
    print("=" * 80)
    print(f"  {'feature':<28}{'Pearson r':>14}{'p':>12}{'Spearman r':>14}{'p':>12}")
    print("  " + "-" * 80)
    for col, label in [("div_full", "diversity (full CoT)"),
                        ("div_stripped", "diversity (action-stripped)")]:
        r_p, p_p = pearsonr(df[col], df["D_raw"])
        r_s, p_s = spearmanr(df[col], df["D_raw"])
        print(f"  {label:<28}{r_p:>+14.3f}{p_p:>12.4f}{r_s:>+14.3f}{p_s:>12.4f}")

    # LOMO classifier comparison
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 80)
    print("LOMO CLASSIFIER (predicting D_raw > 0.3, single feature, logistic)")
    print("=" * 80)
    y = (df["D_raw"] > 0.3).astype(int).values
    groups = df["model"].values
    print(f"  n={len(df)}, positives={y.sum()} ({y.mean():.0%})")

    for col, label in [("div_full", "diversity (full CoT)"),
                        ("div_stripped", "diversity (action-stripped)")]:
        X = df[[col]].values
        logo = LeaveOneGroupOut()
        aucs = []
        for train_idx, test_idx in logo.split(X, y, groups):
            if len(set(y[test_idx])) < 2:
                continue
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
            proba = clf.predict_proba(sc.transform(Xte))[:, 1]
            aucs.append((groups[test_idx[0]], roc_auc_score(yte, proba)))
        print(f"\n  {label}:")
        for m, a in aucs:
            print(f"    held-out {m:<28} AUC = {a:.3f}")
        if aucs:
            mean_auc = sum(a for _, a in aucs) / len(aucs)
            print(f"    mean AUC = {mean_auc:.3f}")

    print("\n=== STRICT LEAKAGE TEST DONE ===")
    print("Interpretation:")
    print("  If div_stripped AUC stays high (>~0.80): linguistic signal is real, beyond stated intent.")
    print("  If div_stripped AUC drops to chance (~0.50): E6 was reading the answer in the CoT.")
    print("  Intermediate: partial leakage — methods claim needs explicit hedging.")


if __name__ == "__main__":
    main()