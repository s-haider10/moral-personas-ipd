"""
M2 - Layer-wise activation analysis and difference-of-means direction.

For each persona (integrity, phronesis) we cache residual-stream activations at
each layer at a fixed token position (the final user-prompt token, i.e. the
position where the next-token distribution determines the response). Then:

  (a) Linear-probe accuracy at each layer for distinguishing integrity from
      phronesis-prompted contexts; identifies the layer where the operationalization
      contrast is most linearly separable.
  (b) Difference-of-means direction v_layer = mean(integrity) - mean(phronesis)
      at each layer, with its norm and cosine similarity profile.
  (c) Token-position sweep at the best layer to confirm the signal isn't an
      artifact of one position.

This is the foundation for M3 (patching + steering).

Inputs:
  A balanced set of (persona, history-state) prompts. We use:
    - 60 integrity prompts + 60 phronesis prompts (round-r contexts harvested
      from M1 trajectories, balanced across early/mid/late rounds)
    - Plus a held-out set for probe evaluation (20 + 20)

Outputs:
  results/M2/activations_{layer}.pt        per-layer activation tensors
  results/M2/diff_of_means.pt              v_layer for every layer
  results/M2/probe_accuracies.csv          layer-wise probe AUC and accuracy
  figures/fig_m2_layer_sweep.png

Usage:
  python m2_activation_probe.py --prepare-prompts   # build the balanced prompt set
  python m2_activation_probe.py --cache             # cache activations
  python m2_activation_probe.py --analyze           # probes, diff-of-means, figure
  python m2_activation_probe.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --prepare-prompts --cache --analyze
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from e2_cross_model import PROMPT_TEMPLATE, format_history  # noqa: E402
from e13_pressure import E13_PERSONA_PROMPT  # noqa: E402


N_PROMPTS_PER_PERSONA_TRAIN = 60
N_PROMPTS_PER_PERSONA_TEST = 20


def synth_history(n_rounds_played, rng):
    """Generate a plausible AllD-game history of length n_rounds_played.

    The agent is the LLM (varied C/D mix biased toward C in early rounds),
    the opponent is AllD (always D). This gives the model realistic round-N
    contexts to compare integrity-vs-phronesis activations at matched histories.
    """
    history = []
    my_total = 0
    opp_total = 0
    for r in range(1, n_rounds_played + 1):
        # Early-round bias toward C; later, mixed; matches typical moral-persona behavior
        p_cooperate = max(0.3, 0.9 - 0.05 * r)
        mine = "C" if rng.random() < p_cooperate else "D"
        theirs = "D"
        from e2_cross_model import PAYOFFS
        my_pay, opp_pay = PAYOFFS[(mine, theirs)]
        history.append((mine, theirs, my_pay, opp_pay))
        my_total += my_pay
        opp_total += opp_pay
    return history, my_total, opp_total


def build_prompts(personas, n_per_persona, seed, n_rounds_total=20):
    """Generate n_per_persona prompts per persona, sampling round indices uniformly.

    All prompts share the same history-distribution so the *only* systematic
    difference between integrity and phronesis prompt sets is the persona clause.
    This is the necessary control for treating activation differences as a
    persona effect.
    """
    rng = random.Random(seed)
    prompts = []  # list of (persona, prompt_text, round_num)

    # Sample history-states first, then duplicate across personas. Each state
    # gets a stable id so a probe can be cross-validated leave-one-history-out
    # (LOMO): holding out a state removes *both* its persona instances, so a
    # probe that merely memorizes history tokens cannot generalize.
    states = []
    for _ in range(n_per_persona):
        r = rng.randint(1, n_rounds_total)
        history, my_total, opp_total = synth_history(r - 1, rng)
        states.append((r, history, my_total, opp_total))

    for persona in personas:
        for state_id, (r, history, my_total, opp_total) in enumerate(states):
            prompt = PROMPT_TEMPLATE.format(
                persona_intro=E13_PERSONA_PROMPT[persona],
                history=format_history(history),
                my_total=my_total,
                opp_total=opp_total,
                round_num=r,
                n_rounds=n_rounds_total,
            )
            prompts.append({"persona": persona, "round": r,
                             "state_id": state_id, "prompt": prompt})
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
        output_hidden_states=True,  # we need hidden states from every layer
    )
    model.eval()
    return model, tok


@torch.no_grad()
def cache_activations(model, tok, prompts, batch_size=4, output_path=None):
    """Forward each prompt, capture hidden states at every layer at the FINAL token.

    Returns a dict:
      {
        "activations": [n_prompts, n_layers + 1, d_model]   (incl. embed layer)
        "personas":   [n_prompts]
        "rounds":     [n_prompts]
        "labels":     [n_prompts]  binary integrity=0, phronesis=1
      }
    """
    label_map = {"virtue_integrity": 0, "virtue_phronesis": 1, "deontologist": 2}
    n = len(prompts)
    activations = None
    personas_out = []
    rounds_out = []
    labels_out = []
    state_ids_out = []

    for batch_start in range(0, n, batch_size):
        batch = prompts[batch_start: batch_start + batch_size]
        texts = []
        for p in batch:
            try:
                tmpl = tok.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                tmpl = p["prompt"]
            texts.append(tmpl)

        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple of length (n_layers + 1), each [B, T, D]
        # We take the FINAL non-padding token per example.
        attn_mask = enc["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1  # [B]

        per_layer = []
        for h in out.hidden_states:
            # h: [B, T, D]
            B = h.size(0)
            picked = h[torch.arange(B), last_idx]  # [B, D]
            per_layer.append(picked.float().cpu())
        # stack into [B, n_layers, D]
        batch_acts = torch.stack(per_layer, dim=1)

        if activations is None:
            n_layers = batch_acts.size(1)
            d_model = batch_acts.size(2)
            activations = torch.zeros(n, n_layers, d_model, dtype=torch.float32)
        activations[batch_start: batch_start + len(batch)] = batch_acts

        for p in batch:
            personas_out.append(p["persona"])
            rounds_out.append(p["round"])
            labels_out.append(label_map[p["persona"]])
            state_ids_out.append(p.get("state_id", -1))

        print(f"  cached {batch_start + len(batch)}/{n}")

    bundle = {
        "activations": activations,
        "personas": personas_out,
        "rounds": rounds_out,
        "labels": torch.tensor(labels_out),
        "state_ids": torch.tensor(state_ids_out),
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, output_path)
        print(f"saved {output_path}  shape={tuple(activations.shape)}")
    return bundle


def linear_probe_each_layer(train, test):
    """Leave-one-history-out (LOMO) logistic probe at each layer.

    A plain train/test split saturates here: with d_model (~5120) >> n_prompts
    (~120) a logistic probe separates any labelling perfectly, so every layer
    reports AUC=1.0 and the sweep carries no localization signal.

    Robust design (matches the README's M2.1 "LOMO AUC" hypothesis):
      - pool train+test so every history-state is usable as a fold;
      - group by state_id and hold out one history-state per fold, which
        removes *both* its integrity and phronesis instances at once -- a probe
        that memorizes history tokens cannot score on the held-out fold;
      - PCA-reduce each layer to N_PCA components fit on the training fold
        only. This is the key control: with d_model (~4096) >> n a logistic
        probe separates *any* labelling, so a full-dim probe is uninformative.
        Reducing to d < n removes that trivial-separation artifact -- if the
        contrast survives in a low-dim subspace it is a real representation,
        if it collapses to chance it was pure d>>n overfitting;
      - strong L2 inside the reduced space as a second guard;
      - aggregate out-of-fold probabilities into a single AUC per layer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    acts, labels, states = [], [], []
    for b in (train, test):
        mask = (b["labels"] == 0) | (b["labels"] == 1)
        acts.append(b["activations"][mask].numpy())
        labels.append(b["labels"][mask].numpy())
        states.append(b["state_ids"][mask].numpy())
    X_all = np.concatenate(acts, axis=0)        # [n, L, D]
    y = np.concatenate(labels, axis=0)          # [n]

    # Make group ids unique per (bundle, state) so train and test states do
    # not collide into the same fold.
    offset = 0
    fixed_groups = []
    for g in states:
        fixed_groups.append(g + offset)
        offset += (int(g.max()) + 1) if len(g) else 0
    groups = np.concatenate(fixed_groups, axis=0)

    n_layers = X_all.shape[1]
    logo = LeaveOneGroupOut()
    N_PCA = 16  # d < n per fold; removes the d>>n trivial-separation artifact
    C = 0.5
    results = []
    for L in range(n_layers):
        XL = X_all[:, L, :]
        oof_proba = np.zeros(len(y), dtype=np.float64)
        oof_pred = np.zeros(len(y), dtype=np.int64)
        for tr_idx, te_idx in logo.split(XL, y, groups):
            n_comp = min(N_PCA, len(tr_idx) - 1, XL.shape[1])
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_comp, random_state=0),
                LogisticRegression(max_iter=2000, C=C),
            )
            clf.fit(XL[tr_idx], y[tr_idx])
            oof_proba[te_idx] = clf.predict_proba(XL[te_idx])[:, 1]
            oof_pred[te_idx] = clf.predict(XL[te_idx])
        acc = float((oof_pred == y).mean())
        try:
            auc = float(roc_auc_score(y, oof_proba))
        except Exception:
            auc = float("nan")
        results.append({"layer": L, "acc": acc, "auc": auc})
        print(f"  layer {L:3d}: LOMO acc={acc:.3f}  auc={auc:.3f}")
    return results


def diff_of_means_direction(bundle):
    """v_layer = mean(integrity activations) - mean(phronesis activations) per layer.

    Returns [n_layers, d_model].
    """
    integ = bundle["activations"][bundle["labels"] == 0]  # [n_i, L, D]
    phron = bundle["activations"][bundle["labels"] == 1]
    v = integ.mean(dim=0) - phron.mean(dim=0)  # [L, D]
    return v


def make_figure(probe_results, v_per_layer, out_path):
    import matplotlib.pyplot as plt
    layers = [r["layer"] for r in probe_results]
    accs = [r["acc"] for r in probe_results]
    aucs = [r["auc"] for r in probe_results]
    norms = torch.norm(v_per_layer, dim=1).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(layers, accs, marker="o", label="accuracy")
    ax1.plot(layers, aucs, marker="s", label="AUC", alpha=0.7)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("integrity vs phronesis classifier")
    ax1.set_ylim(0.4, 1.02)
    ax1.axhline(0.5, ls="--", c="gray", alpha=0.4)
    ax1.legend()
    ax1.set_title("Linear separability across layers")
    ax1.grid(True, alpha=0.3)

    ax2.plot(layers, norms, marker="o", color="darkred")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("||v_layer||")
    ax2.set_title("Norm of difference-of-means direction")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("M2: layer-wise integrity vs phronesis signal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--prepare-prompts", action="store_true")
    p.add_argument("--cache", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--csv-dir", default="csvs")
    p.add_argument("--fig-dir", default="figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)
    safe_model = args.model.replace("/", "_")
    m2_dir = results_dir / "M2" / safe_model
    m2_dir.mkdir(parents=True, exist_ok=True)

    train_prompts_path = m2_dir / "prompts_train.json"
    test_prompts_path = m2_dir / "prompts_test.json"

    if args.prepare_prompts:
        train = build_prompts(["virtue_integrity", "virtue_phronesis"],
                              N_PROMPTS_PER_PERSONA_TRAIN, seed=0)
        test = build_prompts(["virtue_integrity", "virtue_phronesis"],
                             N_PROMPTS_PER_PERSONA_TEST, seed=999)
        with open(train_prompts_path, "w") as f:
            json.dump(train, f, indent=2)
        with open(test_prompts_path, "w") as f:
            json.dump(test, f, indent=2)
        print(f"wrote {len(train)} train and {len(test)} test prompts")

    if args.cache:
        if not train_prompts_path.exists():
            print("run --prepare-prompts first"); return
        with open(train_prompts_path) as f:
            train = json.load(f)
        with open(test_prompts_path) as f:
            test = json.load(f)
        model, tok = load_model_and_tok(args.model)
        cache_activations(model, tok, train, batch_size=args.batch_size,
                          output_path=m2_dir / "activations_train.pt")
        cache_activations(model, tok, test, batch_size=args.batch_size,
                          output_path=m2_dir / "activations_test.pt")
        del model
        torch.cuda.empty_cache()

    if args.analyze:
        train_bundle = torch.load(m2_dir / "activations_train.pt")
        test_bundle = torch.load(m2_dir / "activations_test.pt")
        probe_results = linear_probe_each_layer(train_bundle, test_bundle)
        v = diff_of_means_direction(train_bundle)
        torch.save({"v_per_layer": v}, m2_dir / "diff_of_means.pt")

        import pandas as pd
        df = pd.DataFrame(probe_results)
        csv_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_dir / "M2_probe_accuracies.csv", index=False)

        make_figure(probe_results, v, fig_dir / "fig_m2_layer_sweep.png")

        # Identify best layer
        best = max(probe_results, key=lambda r: r["auc"])
        print(f"\nbest layer: {best['layer']}  acc={best['acc']:.3f}  auc={best['auc']:.3f}")


if __name__ == "__main__":
    main()
