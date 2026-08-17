"""
M3 - Causal interventions: activation patching and steering.

Two causal probes that the integrity-vs-phronesis behavioral effect (E5)
is mediated by a localized residual-stream direction.

Intervention A: Steering. Add a multiple of the M2 difference-of-means vector
  v = mean(integrity_acts) - mean(phronesis_acts) at every forward pass, at
  the best layer L* identified by M2's linear probe. Sweep alpha from
  negative (push toward phronesis) to positive (push toward integrity). The
  control persona is phronesis (the more-defecting side). If steering with
  alpha>0 reduces defection monotonically, the direction v is causally
  responsible for the behavior shift.

Intervention B: Activation patching (cross-prompt). For each round-context
  c, generate two forward passes:
    - integrity-prompted, layer L* activation cached
    - phronesis-prompted, layer L* activation replaced with the integrity one
  Measure whether the patched run produces the integrity-style action choice
  more often than the unpatched phronesis run.

Pre-registered:
  M3.1 (steering): D(phronesis, alpha=+3) < D(phronesis, alpha=0) by >= 0.10,
       with monotone shift across alpha in {-3, -1, 0, +1, +3}.
  M3.2 (patching): D(phronesis-patched-with-integrity) < D(phronesis) by >= 0.10.

Usage:
  python m3_intervention.py --steering --alphas -3,-1,0,1,3 --seeds 3
  python m3_intervention.py --patching --seeds 3
  python m3_intervention.py --analyze

Outputs:
  results/M3/{safe_model}/steering/alpha{a}_seed{N}_persona{p}.jsonl
  results/M3/{safe_model}/patching/seed{N}_source{p_src}_target{p_tgt}.jsonl
  csvs/M3_steering.csv
  csvs/M3_patching.csv
  figures/fig_m3_steering.png
  figures/fig_m3_patching.png
"""

import argparse
import json
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from e2_cross_model import (  # noqa: E402
    PROMPT_TEMPLATE,
    PAYOFFS,
    format_history,
    parse_response,
    opponent_action,
)
from e13_pressure import E13_PERSONA_PROMPT, normative_action_e13  # noqa: E402


OPPONENT = "AllD"
N_ROUNDS = 20


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


def get_layer_module(model, layer_idx):
    """Find the residual-stream module at layer_idx.

    For Llama / Mistral / Qwen this is `model.model.layers[layer_idx]`. The
    output of this module's forward is the residual stream entering the next
    layer. We hook the OUTPUT of this module.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # Some models nest under .transformer
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError(
        "Could not find layer-list attribute on model. Tried model.model.layers "
        "and model.transformer.h. Inspect model architecture and add a branch."
    )


# ------------------------------------------------------------------
# Steering hook
# ------------------------------------------------------------------

@contextmanager
def steering_hook(model, layer_idx, direction, alpha):
    """Add alpha * direction to the residual stream at every forward pass.

    direction: [d_model] tensor, will be broadcast.
    Applied at the output of the specified transformer block, to every token
    position.
    """
    layer = get_layer_module(model, layer_idx)
    direction = direction.to(model.device, dtype=next(model.parameters()).dtype)

    def hook(module, inputs, outputs):
        # outputs is either Tensor or tuple; the first element is the hidden state
        if isinstance(outputs, tuple):
            h = outputs[0]
            h = h + alpha * direction
            return (h,) + outputs[1:]
        return outputs + alpha * direction

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ------------------------------------------------------------------
# Patching hook
# ------------------------------------------------------------------

class PatchState:
    """Carrier for the activation cached on the source run and replayed on
    the target run."""
    def __init__(self):
        self.cached = None
        self.last_token_idx = None  # which token position to patch


@contextmanager
def cache_hook(model, layer_idx, state: PatchState):
    """Record layer_idx output at the FINAL non-pad token of the current forward."""
    layer = get_layer_module(model, layer_idx)

    def hook(module, inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        # We cache the final-token slice. state.last_token_idx must be set
        # before calling forward.
        B, T, D = h.shape
        idx = state.last_token_idx if state.last_token_idx is not None else T - 1
        state.cached = h[:, idx:idx + 1, :].detach().clone()
        return outputs

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def replay_hook(model, layer_idx, state: PatchState):
    """Replace the final-token activation at layer_idx with state.cached."""
    layer = get_layer_module(model, layer_idx)
    if state.cached is None:
        raise RuntimeError("nothing cached; run cache_hook first")

    def hook(module, inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        h = outputs[0] if is_tuple else outputs
        B, T, D = h.shape
        idx = state.last_token_idx if state.last_token_idx is not None else T - 1
        # Only patch the prompt-end token, not the generated tokens (so
        # subsequent generations are downstream effects of the single patch).
        cached = state.cached.to(h.device, dtype=h.dtype)
        h_new = h.clone()
        h_new[:, idx:idx + 1, :] = cached.expand(B, -1, -1)
        if is_tuple:
            return (h_new,) + outputs[1:]
        return h_new

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ------------------------------------------------------------------
# Generation utilities
# ------------------------------------------------------------------

def encode_prompt(tok, prompt, model_device):
    try:
        text_in = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text_in = prompt
    ids = tok(text_in, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(model_device)


@torch.no_grad()
def generate_with_optional_hook(model, tok, prompt, max_new_tokens=256,
                                 temperature=1.0, seed=0):
    """Plain generation; any hooks should already be installed."""
    torch.manual_seed(seed)
    ids = encode_prompt(tok, prompt, model.device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------------
# A single steered trajectory
# ------------------------------------------------------------------

def run_steered_trajectory(model, tok, model_name, persona, alpha, direction,
                            layer_idx, seed, results_dir, temperature=1.0):
    rng = random.Random(seed)
    persona_intro = E13_PERSONA_PROMPT[persona]
    safe_model = model_name.replace("/", "_")
    out_dir = results_dir / "M3" / safe_model / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer{layer_idx}_alpha{alpha:+.1f}_seed{seed}_persona{persona}.jsonl"

    meta = {
        "type": "meta",
        "experiment": "M3_steering",
        "model": model_name,
        "persona": persona,
        "alpha": alpha,
        "layer_idx": layer_idx,
        "seed": seed,
        "n_rounds": N_ROUNDS,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }

    history = []
    my_total = 0
    opp_total = 0

    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")

        with steering_hook(model, layer_idx, direction, alpha):
            for r in range(1, N_ROUNDS + 1):
                prompt = PROMPT_TEMPLATE.format(
                    persona_intro=persona_intro,
                    history=format_history(history),
                    my_total=my_total,
                    opp_total=opp_total,
                    round_num=r,
                    n_rounds=N_ROUNDS,
                )
                text = generate_with_optional_hook(
                    model, tok, prompt,
                    temperature=temperature, seed=seed * 100 + r,
                )
                action, justification = parse_response(text)
                if action is None:
                    action = "C"
                opp = opponent_action(OPPONENT, history, rng)
                my_pay, opp_pay = PAYOFFS[(action, opp)]
                history.append((action, opp, my_pay, opp_pay))
                my_total += my_pay
                opp_total += opp_pay
                a_star = normative_action_e13(persona, history)
                f.write(json.dumps({
                    "type": "round",
                    "round": r,
                    "agent_action": action,
                    "opponent_action": opp,
                    "agent_payoff": my_pay,
                    "agent_total": my_total,
                    "a_star": a_star,
                    "normative_defection": action != a_star,
                    "justification": justification,
                    "raw_response": text,
                }) + "\n")
                f.flush()

        n_d = sum(1 for h in history if h[0] == "D")
        L = next((i + 1 for i, h in enumerate(history) if h[0] == "D"), None)
        f.write(json.dumps({
            "type": "summary",
            "raw_defection_rate": n_d / N_ROUNDS,
            "first_defection_round": L,
            "agent_total": my_total,
        }) + "\n")
        f.flush()
    print(f"  steering {persona} alpha={alpha:+.1f} seed{seed}: D={n_d}/{N_ROUNDS}")
    return n_d / N_ROUNDS


# ------------------------------------------------------------------
# A single patched trajectory
# ------------------------------------------------------------------

def run_patched_trajectory(model, tok, model_name, source_persona, target_persona,
                            layer_idx, seed, results_dir, temperature=1.0):
    """For each round: (1) run a forward pass under source_persona prompt to
    cache the final-token activation at layer_idx; (2) run the actual generation
    under target_persona prompt with that activation patched in. The agent's
    action and the history evolution use the *target* persona prompt with the
    source's residual stream at the prompt-end token."""
    rng = random.Random(seed)
    source_intro = E13_PERSONA_PROMPT[source_persona]
    target_intro = E13_PERSONA_PROMPT[target_persona]
    safe_model = model_name.replace("/", "_")
    out_dir = results_dir / "M3" / safe_model / "patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer{layer_idx}_seed{seed}_src{source_persona}_tgt{target_persona}.jsonl"

    meta = {
        "type": "meta",
        "experiment": "M3_patching",
        "model": model_name,
        "source_persona": source_persona,
        "target_persona": target_persona,
        "layer_idx": layer_idx,
        "seed": seed,
        "n_rounds": N_ROUNDS,
        "timestamp": datetime.now().isoformat(),
    }

    history = []
    my_total = 0
    opp_total = 0
    state = PatchState()

    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for r in range(1, N_ROUNDS + 1):
            src_prompt = PROMPT_TEMPLATE.format(
                persona_intro=source_intro,
                history=format_history(history),
                my_total=my_total, opp_total=opp_total,
                round_num=r, n_rounds=N_ROUNDS,
            )
            tgt_prompt = PROMPT_TEMPLATE.format(
                persona_intro=target_intro,
                history=format_history(history),
                my_total=my_total, opp_total=opp_total,
                round_num=r, n_rounds=N_ROUNDS,
            )

            # 1) Cache source activation
            src_ids = encode_prompt(tok, src_prompt, model.device)
            state.last_token_idx = src_ids.size(1) - 1
            with cache_hook(model, layer_idx, state):
                with torch.no_grad():
                    _ = model(src_ids)

            # 2) Generate under target prompt with the cached activation
            #    patched in at the final-token position.
            tgt_ids = encode_prompt(tok, tgt_prompt, model.device)
            # last_token_idx in the target prompt may differ; recompute
            state.last_token_idx = tgt_ids.size(1) - 1
            torch.manual_seed(seed * 100 + r)
            with replay_hook(model, layer_idx, state):
                with torch.no_grad():
                    out = model.generate(
                        tgt_ids,
                        max_new_tokens=256,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_p=0.95,
                        pad_token_id=tok.eos_token_id,
                    )
            text = tok.decode(out[0, tgt_ids.size(1):], skip_special_tokens=True)
            action, justification = parse_response(text)
            if action is None:
                action = "C"

            opp = opponent_action(OPPONENT, history, rng)
            my_pay, opp_pay = PAYOFFS[(action, opp)]
            history.append((action, opp, my_pay, opp_pay))
            my_total += my_pay
            opp_total += opp_pay
            a_star = normative_action_e13(target_persona, history)
            f.write(json.dumps({
                "type": "round",
                "round": r,
                "agent_action": action,
                "opponent_action": opp,
                "agent_payoff": my_pay,
                "a_star": a_star,
                "normative_defection": action != a_star,
                "justification": justification,
                "raw_response": text,
            }) + "\n")
            f.flush()

        n_d = sum(1 for h in history if h[0] == "D")
        L = next((i + 1 for i, h in enumerate(history) if h[0] == "D"), None)
        f.write(json.dumps({
            "type": "summary",
            "raw_defection_rate": n_d / N_ROUNDS,
            "first_defection_round": L,
            "agent_total": my_total,
        }) + "\n")
    print(f"  patching src={source_persona} tgt={target_persona} seed{seed}: D={n_d}/{N_ROUNDS}")
    return n_d / N_ROUNDS


# ------------------------------------------------------------------
# Analysis and figures
# ------------------------------------------------------------------

def load_steering_results(results_dir, model_name):
    safe_model = model_name.replace("/", "_")
    root = results_dir / "M3" / safe_model / "steering"
    if not root.exists():
        return None
    import pandas as pd
    rows = []
    for fp in sorted(root.glob("*.jsonl")):
        meta, summary = None, None
        with open(fp) as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                if rec.get("type") == "meta": meta = rec
                elif rec.get("type") == "summary": summary = rec
        if meta and summary:
            rows.append({
                "model": meta["model"], "persona": meta["persona"],
                "alpha": meta["alpha"], "seed": meta["seed"],
                "layer_idx": meta["layer_idx"],
                "D_raw": summary["raw_defection_rate"],
            })
    return pd.DataFrame(rows)


def load_patching_results(results_dir, model_name):
    safe_model = model_name.replace("/", "_")
    root = results_dir / "M3" / safe_model / "patching"
    if not root.exists():
        return None
    import pandas as pd
    rows = []
    for fp in sorted(root.glob("*.jsonl")):
        meta, summary = None, None
        with open(fp) as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                if rec.get("type") == "meta": meta = rec
                elif rec.get("type") == "summary": summary = rec
        if meta and summary:
            rows.append({
                "model": meta["model"],
                "source_persona": meta["source_persona"],
                "target_persona": meta["target_persona"],
                "layer_idx": meta["layer_idx"],
                "seed": meta["seed"],
                "D_raw": summary["raw_defection_rate"],
            })
    return pd.DataFrame(rows)


def make_steering_figure(df, out_path):
    import matplotlib.pyplot as plt
    if df is None or df.empty:
        print("no steering data"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    for persona in df["persona"].unique():
        s = df[df["persona"] == persona]
        g = s.groupby("alpha")["D_raw"]
        mu = g.mean()
        se = g.sem()
        ax.errorbar(mu.index, mu.values * 100, yerr=se.values * 100,
                    marker="o", capsize=3, label=persona)
    ax.axvline(0, ls="--", c="gray", alpha=0.5)
    ax.set_xlabel("alpha (steering strength along v = integrity - phronesis)")
    ax.set_ylabel("Defection rate (%) vs AllD")
    ax.set_ylim(-5, 105)
    ax.set_title("M3.1: Steering causally shifts defection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


def make_patching_figure(df, baseline_df, out_path):
    """baseline_df: defection by persona under no intervention (from M1)."""
    import matplotlib.pyplot as plt
    if df is None or df.empty:
        print("no patching data"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = df.groupby(["source_persona", "target_persona"])["D_raw"].agg(["mean", "sem"]).reset_index()
    labels = [f"src={r['source_persona']}\ntgt={r['target_persona']}" for _, r in rows.iterrows()]
    means = rows["mean"].values * 100
    sems = rows["sem"].values * 100
    x = list(range(len(rows)))
    ax.bar(x, means, yerr=sems, capsize=4, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Defection rate (%) vs AllD")
    ax.set_ylim(0, 105)
    ax.set_title("M3.2: Activation patching at L*")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


def hypothesis_m3_1(df):
    print("\n=== M3.1 (steering) ===")
    if df is None or df.empty:
        print("  no data"); return
    for persona in df["persona"].unique():
        s = df[df["persona"] == persona]
        g = s.groupby("alpha")["D_raw"].mean().sort_index()
        print(f"  {persona}: " + ", ".join(f"a={a:+.1f}->D={d:.2f}" for a, d in g.items()))
        if len(g) >= 2:
            a_lo, a_hi = g.index.min(), g.index.max()
            delta = g[a_hi] - g[a_lo]  # large alpha minus small alpha
            # v points from phronesis to integrity, so positive alpha should reduce D
            if persona == "virtue_phronesis":
                passes = (g[a_lo] - g[a_hi]) >= 0.10
                print(f"  {persona}: alpha effect = D(a_lo) - D(a_hi) = {(g[a_lo] - g[a_hi]):+.2f}  "
                      f"{'PASS (>=0.10)' if passes else 'FAIL'}")


def hypothesis_m3_2(df):
    print("\n=== M3.2 (patching) ===")
    if df is None or df.empty:
        print("  no data"); return
    g = df.groupby(["source_persona", "target_persona"])["D_raw"].agg(["mean", "sem"])
    print(g.to_string())
    # The key contrast: src=integrity, tgt=phronesis (the patch)
    # should yield lower D than the no-patch phronesis baseline.
    patched = df[(df["source_persona"] == "virtue_integrity") &
                 (df["target_persona"] == "virtue_phronesis")]["D_raw"]
    if len(patched):
        print(f"  src=integrity tgt=phronesis: D = {patched.mean():.2f}  (compare to baseline phronesis)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--m2-dir", default=None,
                   help="path to results/M2/{safe_model}/diff_of_means.pt; default infers from --model")
    p.add_argument("--layer", type=int, default=None,
                   help="layer index to intervene at; if omitted, choose best layer from M2 probe csv")
    p.add_argument("--steering", action="store_true")
    p.add_argument("--patching", action="store_true")
    p.add_argument("--alphas", default="-3,-1,0,1,3")
    p.add_argument("--steering-personas", nargs="+",
                   default=["virtue_phronesis", "virtue_integrity"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--csv-dir", default="csvs")
    p.add_argument("--fig-dir", default="figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)
    safe_model = args.model.replace("/", "_")

    # Resolve M2 directory
    m2_dir = Path(args.m2_dir) if args.m2_dir else (results_dir / "M2" / safe_model)
    dom_path = m2_dir / "diff_of_means.pt"

    if args.steering or args.patching:
        if not dom_path.exists():
            print(f"need {dom_path}; run M2 first"); return
        v_bundle = torch.load(dom_path)
        v_per_layer = v_bundle["v_per_layer"]

        if args.layer is None:
            probe_csv = csv_dir / "M2_probe_accuracies.csv"
            if probe_csv.exists():
                import pandas as pd
                probe = pd.read_csv(probe_csv)
                best_layer = int(probe.loc[probe["auc"].idxmax(), "layer"])
            else:
                # Fall back to the middle layer
                best_layer = v_per_layer.size(0) // 2
                print(f"no probe CSV, defaulting to middle layer {best_layer}")
        else:
            best_layer = args.layer
        v_star = v_per_layer[best_layer]
        # Normalize for interpretability of alpha
        v_star_unit = v_star / (v_star.norm() + 1e-8)
        # We use the UNIT direction so alpha is in units of activation magnitude
        v = v_star_unit * v_per_layer[best_layer].norm()  # keep typical magnitude
        # In practice, sweeping alpha relative to ||v|| is most interpretable.
        # We'll feed v_star_unit and let alpha control the magnitude in std-dev units.
        print(f"using layer L* = {best_layer}; ||v||={v_star.norm().item():.2f}")
        from m2_activation_probe import load_model_and_tok
        model, tok = load_model_and_tok(args.model)

    if args.steering:
        alphas = [float(a) for a in args.alphas.split(",")]
        # Normalize direction; alpha is in "unit-direction" steps
        v_unit = v_per_layer[best_layer]
        v_unit = v_unit / (v_unit.norm() + 1e-8)
        for persona in args.steering_personas:
            for alpha in alphas:
                for s in range(args.seeds):
                    tgt = (results_dir / "M3" / safe_model / "steering" /
                           f"layer{best_layer}_alpha{alpha:+.1f}_seed{s}_persona{persona}.jsonl")
                    if tgt.exists() and '"type": "summary"' in tgt.read_text():
                        print(f"SKIP {tgt}"); continue
                    run_steered_trajectory(
                        model, tok, args.model, persona, alpha, v_unit,
                        best_layer, s, results_dir,
                    )

    if args.patching:
        # Two contrasts:
        #   (intervention)  src=integrity, tgt=phronesis  - should reduce D
        #   (control)       src=phronesis, tgt=phronesis  - should be similar to baseline
        contrasts = [
            ("virtue_integrity", "virtue_phronesis"),
            ("virtue_phronesis", "virtue_phronesis"),  # null control
            ("virtue_phronesis", "virtue_integrity"),  # reverse direction
        ]
        for src, tgt in contrasts:
            for s in range(args.seeds):
                target_file = (results_dir / "M3" / safe_model / "patching" /
                               f"layer{best_layer}_seed{s}_src{src}_tgt{tgt}.jsonl")
                if target_file.exists() and '"type": "summary"' in target_file.read_text():
                    print(f"SKIP {target_file}"); continue
                run_patched_trajectory(model, tok, args.model, src, tgt,
                                       best_layer, s, results_dir)

    if args.steering or args.patching or args.analyze:
        s_df = load_steering_results(results_dir, args.model)
        p_df = load_patching_results(results_dir, args.model)
        if s_df is not None and not s_df.empty:
            csv_dir.mkdir(parents=True, exist_ok=True)
            s_df.to_csv(csv_dir / "M3_steering.csv", index=False)
            make_steering_figure(s_df, fig_dir / "fig_m3_steering.png")
            hypothesis_m3_1(s_df)
        if p_df is not None and not p_df.empty:
            csv_dir.mkdir(parents=True, exist_ok=True)
            p_df.to_csv(csv_dir / "M3_patching.csv", index=False)
            make_patching_figure(p_df, None, fig_dir / "fig_m3_patching.png")
            hypothesis_m3_2(p_df)


if __name__ == "__main__":
    main()
