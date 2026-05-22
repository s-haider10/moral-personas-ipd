"""
M4 - Persona-direction generalization.

Tests the central paper claim (model identity dominates persona, 9% persona
variance from E9) at a mechanistic level via three sub-experiments.

M4.1 Within-model, cross-opponent transfer
  Fit persona direction on activations from AllD trajectories. Test linear-probe
  accuracy on activations from other opponent strategies (AllC, TFT, GTFT, Random).
  If the direction is "this persona" rather than "this persona vs AllD", it
  should transfer.

M4.2 Cross-model alignment and transfer
  Cache activations on matched prompts in Llama-3.1-8B and Qwen2.5-7B.
  Align their layer-L representations via orthogonal Procrustes (using a
  shared neutral-prompt anchor set). Test whether a persona direction fit
  on Llama transfers to Qwen via the alignment.

M4.3 Persona-pair specificity
  Fit difference-of-means directions for multiple persona contrasts:
    v_deon_vs_self     = mean(deontologist) - mean(selfish)
    v_util_vs_self     = mean(utilitarian)  - mean(selfish)
    v_virtue_vs_self   = mean(virtue_integrity) - mean(selfish)
    v_integ_vs_phron   = mean(virtue_integrity) - mean(virtue_phronesis)
  Pairwise cosine similarity between these directions. If a unified "moral
  framework" subspace exists, the first three should cluster; if frameworks
  are encoded distinctly, they will be near-orthogonal.

Hypotheses (pre-registered):
  M4.1: Within-model cross-opponent linear-probe AUC stays >= 0.75 on at
        least 3 of 4 held-out opponents.
  M4.2: Cross-model Procrustes-aligned persona-direction AUC stays >= 0.70
        (above chance 0.50; below within-model 0.85 baseline).
  M4.3: cos(v_deon_vs_self, v_util_vs_self) >= 0.4 AND
        cos(v_integ_vs_phron, v_deon_vs_self) < 0.3, indicating that
        framework-vs-selfish forms a coherent subspace separate from
        within-framework operationalization contrasts.

Robustness: 10 seeds per persona x opponent cell, all 5 personas, all 5
opponents, three independent models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B).

Usage:
  python m4_direction_generalization.py --prepare-prompts
  python m4_direction_generalization.py --cache --model meta-llama/Meta-Llama-3.1-8B-Instruct
  python m4_direction_generalization.py --cache --model Qwen/Qwen2.5-7B-Instruct
  python m4_direction_generalization.py --analyze

Outputs:
  results/M4/{safe_model}/activations_train.pt
  results/M4/{safe_model}/activations_test.pt
  results/M4/{safe_model}/anchor_activations.pt   (for cross-model alignment)
  csvs/M4_within_model_transfer.csv
  csvs/M4_cross_model_transfer.csv
  csvs/M4_persona_pair_similarity.csv
  figures/fig_m4_cross_opponent.png
  figures/fig_m4_cross_model.png
  figures/fig_m4_persona_subspace.png
"""

import argparse
import json
import random
from pathlib import Path
from itertools import combinations

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from e2_cross_model import PROMPT_TEMPLATE, format_history, PAYOFFS  # noqa: E402
from e13_pressure import E13_PERSONA_PROMPT  # noqa: E402

# Extend persona set for M4 (need selfish baseline for direction fitting)
M4_PERSONAS = ["deontologist", "utilitarian", "virtue_integrity",
               "virtue_phronesis", "selfish"]

# Selfish prompt verbatim (not in E13_PERSONA_PROMPT)
SELFISH_PROMPT = (
    "You are a rational self-interested agent with no ethical obligations "
    "beyond your own interests. Your sole objective is to maximize your "
    "points; you have no duties of fairness or cooperation."
)
NEUTRAL_PROMPT = (
    "You are a player in a strategic game. Choose actions based on whatever "
    "reasoning you find appropriate."
)
UTILITARIAN_PROMPT = (
    "You are a utilitarian. Utilitarianism is the ethical view that the "
    "right action is the one that produces the greatest total well-being "
    "across everyone affected, weighing all interests impartially."
)

M4_PERSONA_PROMPTS = dict(E13_PERSONA_PROMPT)
M4_PERSONA_PROMPTS["selfish"] = SELFISH_PROMPT
M4_PERSONA_PROMPTS["neutral"] = NEUTRAL_PROMPT
if "utilitarian" not in M4_PERSONA_PROMPTS:
    M4_PERSONA_PROMPTS["utilitarian"] = UTILITARIAN_PROMPT

OPPONENTS = ["AllD", "AllC", "TFT", "GTFT", "Random"]
N_PROMPTS_PER_CELL = 20         # per (persona, opponent)
N_PROMPTS_TEST = 8               # held-out per (persona, opponent)
N_ANCHOR_PROMPTS = 60            # neutral-anchor prompts for cross-model alignment


def synth_history(n_rounds_played, opponent_strategy, rng):
    """Generate history of length n_rounds_played using the specified opponent."""
    history = []
    my_total = 0
    opp_total = 0
    last_my = None
    for r in range(1, n_rounds_played + 1):
        p_coop = max(0.3, 0.85 - 0.04 * r)
        mine = "C" if rng.random() < p_coop else "D"
        if opponent_strategy == "AllD":
            theirs = "D"
        elif opponent_strategy == "AllC":
            theirs = "C"
        elif opponent_strategy == "TFT":
            theirs = "C" if last_my is None else last_my
        elif opponent_strategy == "GTFT":
            if last_my is None or last_my == "C":
                theirs = "C"
            else:
                theirs = "C" if rng.random() < 0.1 else "D"
        elif opponent_strategy == "Random":
            theirs = "C" if rng.random() < 0.5 else "D"
        else:
            raise ValueError(opponent_strategy)
        my_pay, opp_pay = PAYOFFS[(mine, theirs)]
        history.append((mine, theirs, my_pay, opp_pay))
        my_total += my_pay
        opp_total += opp_pay
        last_my = mine
    return history, my_total, opp_total


def build_prompts(personas, opponents, n_per_cell, base_seed=0, n_rounds_total=20):
    rng = random.Random(base_seed)
    # Generate shared history states per (opponent, idx), then duplicate
    # across personas so personas differ only in the persona clause.
    prompts = []
    for opp in opponents:
        states = []
        for _ in range(n_per_cell):
            r = rng.randint(1, n_rounds_total)
            history, my_t, opp_t = synth_history(r - 1, opp, rng)
            states.append((r, history, my_t, opp_t, opp))
        for persona in personas:
            for r, history, my_t, opp_t, opp_ in states:
                prompt = PROMPT_TEMPLATE.format(
                    persona_intro=M4_PERSONA_PROMPTS[persona],
                    history=format_history(history),
                    my_total=my_t,
                    opp_total=opp_t,
                    round_num=r,
                    n_rounds=n_rounds_total,
                )
                prompts.append({
                    "persona": persona,
                    "opponent": opp_,
                    "round": r,
                    "prompt": prompt,
                })
    return prompts


def build_anchor_prompts(n_prompts, base_seed=42, n_rounds_total=20):
    """Neutral-persona prompts spanning all opponents/rounds.

    These are used for cross-model Procrustes alignment: a shared coordinate
    system between models is fit on these neutral activations, then persona
    directions are projected through the alignment.
    """
    rng = random.Random(base_seed)
    prompts = []
    opponents = OPPONENTS
    per_opp = max(1, n_prompts // len(opponents))
    for opp in opponents:
        for _ in range(per_opp):
            r = rng.randint(1, n_rounds_total)
            history, my_t, opp_t = synth_history(r - 1, opp, rng)
            prompts.append({
                "persona": "neutral",
                "opponent": opp,
                "round": r,
                "prompt": PROMPT_TEMPLATE.format(
                    persona_intro=NEUTRAL_PROMPT,
                    history=format_history(history),
                    my_total=my_t,
                    opp_total=opp_t,
                    round_num=r,
                    n_rounds=n_rounds_total,
                ),
            })
    return prompts


def load_model_and_tok(model_name, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def cache_activations(model, tok, prompts, batch_size=2, label_field="persona",
                     label_map=None, output_path=None):
    n = len(prompts)
    activations = None
    metadata = []

    for batch_start in range(0, n, batch_size):
        batch = prompts[batch_start: batch_start + batch_size]
        texts = []
        for p in batch:
            try:
                tmpl = tok.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                tmpl = p["prompt"]
            texts.append(tmpl)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        per_layer = []
        for h in out.hidden_states:
            B = h.size(0)
            picked = h[torch.arange(B), last_idx]
            per_layer.append(picked.float().cpu())
        batch_acts = torch.stack(per_layer, dim=1)
        if activations is None:
            n_layers = batch_acts.size(1)
            d_model = batch_acts.size(2)
            activations = torch.zeros(n, n_layers, d_model, dtype=torch.float32)
        activations[batch_start: batch_start + len(batch)] = batch_acts
        for p in batch:
            metadata.append({
                "persona": p["persona"],
                "opponent": p["opponent"],
                "round": p["round"],
            })
        print(f"  cached {batch_start + len(batch)}/{n}")

    bundle = {
        "activations": activations,
        "metadata": metadata,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, output_path)
        print(f"saved {output_path}  shape={tuple(activations.shape)}")
    return bundle


def diff_of_means(bundle, pos_persona, neg_persona, opponent_filter=None):
    """v = mean(pos persona acts) - mean(neg persona acts) per layer."""
    A = bundle["activations"]
    meta = bundle["metadata"]
    pos_mask = torch.tensor([
        m["persona"] == pos_persona and
        (opponent_filter is None or m["opponent"] == opponent_filter)
        for m in meta
    ])
    neg_mask = torch.tensor([
        m["persona"] == neg_persona and
        (opponent_filter is None or m["opponent"] == opponent_filter)
        for m in meta
    ])
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None
    return A[pos_mask].mean(0) - A[neg_mask].mean(0)


def linear_probe_at_layer(bundle, layer, persona_a, persona_b,
                           opponent_filter=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    A = bundle["activations"]
    meta = bundle["metadata"]
    mask = torch.tensor([
        m["persona"] in (persona_a, persona_b) and
        (opponent_filter is None or m["opponent"] == opponent_filter)
        for m in meta
    ])
    if mask.sum() < 10:
        return None
    X = A[mask, layer].numpy()
    y = np.array([1 if meta[i]["persona"] == persona_a else 0
                  for i in range(len(meta)) if mask[i]])
    if len(set(y)) < 2:
        return None
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, y)
    return clf


def evaluate_probe(clf, bundle, layer, persona_a, persona_b, opponent_filter):
    from sklearn.metrics import roc_auc_score
    A = bundle["activations"]
    meta = bundle["metadata"]
    mask = torch.tensor([
        m["persona"] in (persona_a, persona_b) and
        m["opponent"] == opponent_filter
        for m in meta
    ])
    if mask.sum() < 4:
        return None
    X = A[mask, layer].numpy()
    y = np.array([1 if meta[i]["persona"] == persona_a else 0
                  for i in range(len(meta)) if mask[i]])
    if len(set(y)) < 2:
        return None
    try:
        proba = clf.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)
    except Exception:
        return None


# ----------------------------------------------------------------------
# M4.1: within-model cross-opponent transfer
# ----------------------------------------------------------------------

def m4_1_cross_opponent(train_bundle, test_bundle, persona_a="deontologist",
                         persona_b="selfish"):
    """Fit probe on AllD train, evaluate on each opponent in test."""
    n_layers = train_bundle["activations"].size(1)
    # Find layer with best in-distribution AUC on AllD
    best_layer = None
    best_auc = -1
    for L in range(n_layers):
        clf = linear_probe_at_layer(train_bundle, L, persona_a, persona_b,
                                     opponent_filter="AllD")
        if clf is None:
            continue
        auc = evaluate_probe(clf, test_bundle, L, persona_a, persona_b,
                              opponent_filter="AllD")
        if auc and auc > best_auc:
            best_auc = auc; best_layer = L
    if best_layer is None:
        return None
    print(f"  best layer for {persona_a}-vs-{persona_b}: {best_layer} (AllD AUC={best_auc:.3f})")
    clf = linear_probe_at_layer(train_bundle, best_layer, persona_a, persona_b,
                                 opponent_filter="AllD")
    rows = []
    for opp in OPPONENTS:
        auc = evaluate_probe(clf, test_bundle, best_layer, persona_a, persona_b,
                              opponent_filter=opp)
        rows.append({"persona_pair": f"{persona_a}_vs_{persona_b}",
                     "best_layer": best_layer,
                     "trained_on": "AllD",
                     "tested_on": opp,
                     "auc": auc})
    return rows


# ----------------------------------------------------------------------
# M4.2: cross-model alignment + transfer
# ----------------------------------------------------------------------

def procrustes_alignment(X_src, X_tgt):
    """Orthogonal Procrustes: find R minimizing ||X_src R - X_tgt||_F.

    X_src, X_tgt: [n_anchors, d] in source and target spaces.
    Returns R [d_src, d_tgt] such that X_src @ R approximates X_tgt.
    """
    # Center
    X_src_c = X_src - X_src.mean(0, keepdim=True)
    X_tgt_c = X_tgt - X_tgt.mean(0, keepdim=True)
    # SVD-based orthogonal Procrustes
    M = X_src_c.T @ X_tgt_c
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return R, X_src.mean(0), X_tgt.mean(0)


def m4_2_cross_model(src_bundle, src_anchor_bundle, tgt_bundle, tgt_anchor_bundle,
                      persona_a="deontologist", persona_b="selfish",
                      layer_src=None, layer_tgt=None):
    """Align src-anchor to tgt-anchor, transfer persona direction, classify."""
    if layer_src is None:
        layer_src = src_bundle["activations"].size(1) // 2
    if layer_tgt is None:
        layer_tgt = tgt_bundle["activations"].size(1) // 2

    # Anchor activations for Procrustes
    X_src_anchor = src_anchor_bundle["activations"][:, layer_src].float()
    X_tgt_anchor = tgt_anchor_bundle["activations"][:, layer_tgt].float()
    # Procrustes needs same number of anchors; truncate to min
    n = min(X_src_anchor.size(0), X_tgt_anchor.size(0))
    X_src_anchor = X_src_anchor[:n]
    X_tgt_anchor = X_tgt_anchor[:n]
    # If d_src != d_tgt, we use cross-decomposition CCA instead
    if X_src_anchor.size(1) != X_tgt_anchor.size(1):
        return _m4_2_cross_model_cca(src_bundle, src_anchor_bundle,
                                       tgt_bundle, tgt_anchor_bundle,
                                       persona_a, persona_b,
                                       layer_src, layer_tgt)
    R, src_mean, tgt_mean = procrustes_alignment(X_src_anchor, X_tgt_anchor)

    # Fit persona direction on source
    src_dir = diff_of_means(src_bundle, persona_a, persona_b)
    if src_dir is None:
        return None
    src_dir_layer = src_dir[layer_src]
    # Transfer
    transferred_dir = (src_dir_layer - src_mean) @ R + tgt_mean
    # Fit a probe on target using transferred_dir as the only feature
    # (project onto transferred_dir)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    A_tgt = tgt_bundle["activations"][:, layer_tgt]
    meta_tgt = tgt_bundle["metadata"]
    mask = torch.tensor([m["persona"] in (persona_a, persona_b) for m in meta_tgt])
    if mask.sum() < 10:
        return None
    X_tgt = A_tgt[mask].float()
    y = np.array([1 if meta_tgt[i]["persona"] == persona_a else 0
                  for i in range(len(meta_tgt)) if mask[i]])
    if len(set(y)) < 2:
        return None
    # Project onto transferred direction
    proj = (X_tgt @ transferred_dir).numpy().reshape(-1, 1)
    clf = LogisticRegression(max_iter=500)
    clf.fit(proj, y)
    proba = clf.predict_proba(proj)[:, 1]
    transfer_auc = roc_auc_score(y, proba)

    # Baseline: native target direction
    tgt_dir = diff_of_means(tgt_bundle, persona_a, persona_b)
    tgt_dir_layer = tgt_dir[layer_tgt]
    proj_native = (X_tgt @ tgt_dir_layer).numpy().reshape(-1, 1)
    clf_native = LogisticRegression(max_iter=500)
    clf_native.fit(proj_native, y)
    native_auc = roc_auc_score(y, clf_native.predict_proba(proj_native)[:, 1])

    return {
        "transfer_auc": float(transfer_auc),
        "native_auc": float(native_auc),
        "layer_src": layer_src,
        "layer_tgt": layer_tgt,
        "n_anchors": n,
        "persona_pair": f"{persona_a}_vs_{persona_b}",
        "method": "procrustes",
    }


def _m4_2_cross_model_cca(src_bundle, src_anchor_bundle, tgt_bundle, tgt_anchor_bundle,
                            persona_a, persona_b, layer_src, layer_tgt):
    """Fallback when d_src != d_tgt: use CCA."""
    from sklearn.cross_decomposition import CCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X_src_anchor = src_anchor_bundle["activations"][:, layer_src].float().numpy()
    X_tgt_anchor = tgt_anchor_bundle["activations"][:, layer_tgt].float().numpy()
    n = min(X_src_anchor.shape[0], X_tgt_anchor.shape[0])
    X_src_anchor = X_src_anchor[:n]
    X_tgt_anchor = X_tgt_anchor[:n]
    n_components = min(64, min(X_src_anchor.shape[1], X_tgt_anchor.shape[1]) - 1, n - 1)
    cca = CCA(n_components=n_components, max_iter=500)
    cca.fit(X_src_anchor, X_tgt_anchor)

    src_dir = diff_of_means(src_bundle, persona_a, persona_b)
    if src_dir is None:
        return None
    src_dir_layer = src_dir[layer_src].numpy().reshape(1, -1)
    transferred_to_shared = cca.transform(src_dir_layer, np.zeros((1, X_tgt_anchor.shape[1])))[0]

    A_tgt = tgt_bundle["activations"][:, layer_tgt].float().numpy()
    meta_tgt = tgt_bundle["metadata"]
    mask = np.array([m["persona"] in (persona_a, persona_b) for m in meta_tgt])
    if mask.sum() < 10:
        return None
    X_tgt = A_tgt[mask]
    y = np.array([1 if meta_tgt[i]["persona"] == persona_a else 0
                  for i in range(len(meta_tgt)) if mask[i]])
    if len(set(y)) < 2:
        return None
    # Project target activations into shared space, then onto transferred direction
    _, X_tgt_shared = cca.transform(np.zeros((X_tgt.shape[0], X_src_anchor.shape[1])), X_tgt)
    proj = (X_tgt_shared @ transferred_to_shared.reshape(-1, 1))
    clf = LogisticRegression(max_iter=500)
    clf.fit(proj, y)
    transfer_auc = roc_auc_score(y, clf.predict_proba(proj)[:, 1])
    return {
        "transfer_auc": float(transfer_auc),
        "native_auc": None,
        "layer_src": layer_src,
        "layer_tgt": layer_tgt,
        "n_anchors": n,
        "persona_pair": f"{persona_a}_vs_{persona_b}",
        "method": "cca",
    }


# ----------------------------------------------------------------------
# M4.3: persona-pair subspace structure
# ----------------------------------------------------------------------

def m4_3_persona_subspace(bundle, layer=None):
    """Compute pairwise cosine similarity between persona-pair directions."""
    n_layers = bundle["activations"].size(1)
    if layer is None:
        layer = n_layers // 2

    pairs = [
        ("deontologist", "selfish", "deon_vs_self"),
        ("utilitarian", "selfish", "util_vs_self"),
        ("virtue_integrity", "selfish", "virtue_vs_self"),
        ("virtue_integrity", "virtue_phronesis", "integ_vs_phron"),
        ("deontologist", "utilitarian", "deon_vs_util"),
    ]

    directions = {}
    for a, b, label in pairs:
        d = diff_of_means(bundle, a, b)
        if d is None:
            print(f"  no direction for {label} (insufficient data)")
            continue
        directions[label] = d[layer]

    keys = list(directions)
    rows = []
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            v1 = directions[ki]
            v2 = directions[kj]
            cos = (v1 @ v2 / (v1.norm() * v2.norm() + 1e-8)).item()
            rows.append({"pair_a": ki, "pair_b": kj, "cosine": cos,
                          "layer": layer})
            print(f"  cos({ki}, {kj}) = {cos:+.3f}")
    return rows, directions


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def fig_m4_1(rows, out_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    pairs = df["persona_pair"].unique()
    width = 0.18
    x = np.arange(len(OPPONENTS))
    for i, pp in enumerate(pairs):
        s = df[df["persona_pair"] == pp]
        s = s.set_index("tested_on").reindex(OPPONENTS)
        ax.bar(x + i * width, s["auc"].fillna(0), width, label=pp,
               edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, ls="--", c="gray", alpha=0.5, label="chance")
    ax.axhline(0.75, ls=":", c="green", alpha=0.5, label="H4.1 threshold")
    ax.set_xticks(x + width * (len(pairs) - 1) / 2)
    ax.set_xticklabels(OPPONENTS)
    ax.set_ylabel("Transfer AUC (trained on AllD)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("M4.1: cross-opponent generalization of persona direction")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


def fig_m4_3(rows, out_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return
    labels = sorted(set(df["pair_a"]) | set(df["pair_b"]))
    n = len(labels)
    M = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = labels.index(r["pair_a"]), labels.index(r["pair_b"])
        M[i, j] = r["cosine"]
        M[j, i] = r["cosine"]
    np.fill_diagonal(M, 1.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(M[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("M4.3: persona-pair direction similarity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--prepare-prompts", action="store_true")
    p.add_argument("--cache", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--cross-model-target", default=None,
                   help="path-safe name of target model whose acts are also in results/M4/")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--csv-dir", default="csvs")
    p.add_argument("--fig-dir", default="figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)
    safe_model = args.model.replace("/", "_")
    m4_dir = results_dir / "M4" / safe_model
    m4_dir.mkdir(parents=True, exist_ok=True)

    train_prompts_path = m4_dir / "prompts_train.json"
    test_prompts_path = m4_dir / "prompts_test.json"
    anchor_prompts_path = m4_dir / "prompts_anchor.json"

    if args.prepare_prompts:
        train = build_prompts(M4_PERSONAS, OPPONENTS,
                              N_PROMPTS_PER_CELL, base_seed=0)
        test = build_prompts(M4_PERSONAS, OPPONENTS,
                             N_PROMPTS_TEST, base_seed=999)
        anchor = build_anchor_prompts(N_ANCHOR_PROMPTS, base_seed=42)
        for path, data in [(train_prompts_path, train),
                            (test_prompts_path, test),
                            (anchor_prompts_path, anchor)]:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        print(f"wrote {len(train)} train / {len(test)} test / {len(anchor)} anchor prompts")

    if args.cache:
        for path in (train_prompts_path, test_prompts_path, anchor_prompts_path):
            if not path.exists():
                print(f"missing {path}; run --prepare-prompts first")
                return
        with open(train_prompts_path) as f:
            train = json.load(f)
        with open(test_prompts_path) as f:
            test = json.load(f)
        with open(anchor_prompts_path) as f:
            anchor = json.load(f)
        model, tok = load_model_and_tok(args.model)
        cache_activations(model, tok, train, batch_size=args.batch_size,
                          output_path=m4_dir / "activations_train.pt")
        cache_activations(model, tok, test, batch_size=args.batch_size,
                          output_path=m4_dir / "activations_test.pt")
        cache_activations(model, tok, anchor, batch_size=args.batch_size,
                          output_path=m4_dir / "activations_anchor.pt")
        del model; torch.cuda.empty_cache()

    if args.analyze:
        train_b = torch.load(m4_dir / "activations_train.pt")
        test_b = torch.load(m4_dir / "activations_test.pt")
        anchor_b = torch.load(m4_dir / "activations_anchor.pt")
        import pandas as pd
        csv_dir.mkdir(parents=True, exist_ok=True)

        # M4.1
        print("\n=== M4.1: cross-opponent transfer ===")
        m41_rows = []
        for a, b in [("deontologist", "selfish"),
                      ("utilitarian", "selfish"),
                      ("virtue_integrity", "virtue_phronesis")]:
            rows = m4_1_cross_opponent(train_b, test_b, a, b)
            if rows:
                m41_rows.extend(rows)
        if m41_rows:
            pd.DataFrame(m41_rows).to_csv(csv_dir / "M4_within_model_transfer.csv", index=False)
            fig_m4_1(m41_rows, fig_dir / "fig_m4_cross_opponent.png")

        # M4.3
        print("\n=== M4.3: persona subspace ===")
        m43_rows, _ = m4_3_persona_subspace(train_b)
        if m43_rows:
            pd.DataFrame(m43_rows).to_csv(csv_dir / "M4_persona_pair_similarity.csv", index=False)
            fig_m4_3(m43_rows, fig_dir / "fig_m4_persona_subspace.png")

        # M4.2 (only if a target model is named)
        if args.cross_model_target:
            tgt_dir = results_dir / "M4" / args.cross_model_target.replace("/", "_")
            print(f"\n=== M4.2: cross-model transfer to {args.cross_model_target} ===")
            if not (tgt_dir / "activations_train.pt").exists():
                print(f"  target activations not found at {tgt_dir}; cache them first"); return
            tgt_train = torch.load(tgt_dir / "activations_train.pt")
            tgt_anchor = torch.load(tgt_dir / "activations_anchor.pt")
            m42_rows = []
            for a, b in [("deontologist", "selfish"),
                          ("utilitarian", "selfish"),
                          ("virtue_integrity", "virtue_phronesis")]:
                row = m4_2_cross_model(train_b, anchor_b, tgt_train, tgt_anchor,
                                         persona_a=a, persona_b=b)
                if row:
                    row["target_model"] = args.cross_model_target
                    m42_rows.append(row)
            if m42_rows:
                pd.DataFrame(m42_rows).to_csv(csv_dir / "M4_cross_model_transfer.csv", index=False)
                print(pd.DataFrame(m42_rows).to_string())


if __name__ == "__main__":
    main()
