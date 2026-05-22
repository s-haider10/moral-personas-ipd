"""
M1 - Behavioral replication of E5 on an open model.

Before any mech interp work, we MUST verify that the integrity-vs-phronesis
operationalization effect replicates on the open model we plan to probe.
If it doesn't, the interp claims have nothing to attach to.

We use Llama-3.1-8B-Instruct as the default open model:
  - fits on a single A6000 24GB at fp16 (16 GB) leaving 3 GPUs free
  - well-supported by transformer_lens and TransformerLens-style hooks
  - instruction-following good enough to follow persona prompts
  - frontier-comparable enough to make replication interpretable

Pre-registered kill switch (M1):
  If |D(phronesis) - D(integrity)| < 0.20 on Llama-3.1-8B, we stop the interp
  pipeline here and either (a) try a different open model, or (b) treat M2/M3
  as a separate behavioral-vs-mechanism dissociation paper.

Usage:
  python m1_open_model_replicate.py --grid
  python m1_open_model_replicate.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --seeds 5

Outputs:
  results/M1/{safe_model}/{persona}/seed{N}_oppAllD.jsonl
  csvs/M1_trajectory_metrics.csv
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from e2_cross_model import (  # noqa: E402
    PROMPT_TEMPLATE,
    PAYOFFS,
    format_history,
    parse_response,
    opponent_action,
)
from e13_pressure import (  # noqa: E402
    E13_PERSONA_PROMPT,
    normative_action_e13,
)

# Personas to test on the open model (same as E5)
PERSONAS = ["virtue_integrity", "virtue_phronesis", "deontologist"]
OPPONENT = "AllD"
N_ROUNDS = 20
N_SEEDS_DEFAULT = 5  # 5 not 3, because open-model variance is higher


def load_hf_model(model_name, device_map="auto", dtype=torch.float16):
    """Load a HuggingFace causal-LM model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def generate(model, tok, prompt, max_new_tokens=256, temperature=1.0, seed=0):
    """Generate a completion using HF generate with a chat template if available."""
    torch.manual_seed(seed)
    # Try chat template first (Llama-3.1, Qwen2.5)
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to(model.device)
    except Exception:
        inputs = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
    return text


def run_trajectory(model, tok, model_name, persona, seed, results_dir, temperature=1.0):
    import random
    rng = random.Random(seed)

    persona_intro = E13_PERSONA_PROMPT[persona]
    safe_model = model_name.replace("/", "_")
    out_dir = results_dir / "M1" / safe_model / persona
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{seed}_opp{OPPONENT}.jsonl"

    meta = {
        "type": "meta",
        "experiment": "M1",
        "provider": "huggingface",
        "model": model_name,
        "persona": persona,
        "opponent": OPPONENT,
        "seed": seed,
        "n_rounds": N_ROUNDS,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }

    history = []
    my_total = 0
    opp_total = 0
    parse_failures = 0

    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")

        for r in range(1, N_ROUNDS + 1):
            prompt = PROMPT_TEMPLATE.format(
                persona_intro=persona_intro,
                history=format_history(history),
                my_total=my_total,
                opp_total=opp_total,
                round_num=r,
                n_rounds=N_ROUNDS,
            )
            text = generate(model, tok, prompt, temperature=temperature, seed=seed * 100 + r)
            action, justification = parse_response(text)
            if action is None:
                parse_failures += 1
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
                "opp_payoff": opp_pay,
                "agent_total": my_total,
                "a_star": a_star,
                "normative_defection": action != a_star,
                "justification": justification,
                "raw_response": text,
                "prompt_at_round": prompt,
            }) + "\n")
            f.flush()

        n_d = sum(1 for h in history if h[0] == "D")
        L = next((i + 1 for i, h in enumerate(history) if h[0] == "D"), None)
        summary = {
            "type": "summary",
            "agent_total": my_total,
            "raw_defection_count": n_d,
            "raw_defection_rate": n_d / N_ROUNDS,
            "first_defection_round": L,
            "parse_failures": parse_failures,
        }
        f.write(json.dumps(summary) + "\n")

    print(f"  {model_name} {persona} seed{seed}: D={n_d}/{N_ROUNDS} (L={L})")
    return summary


def analyze(results_dir, csv_dir):
    import pandas as pd
    root = results_dir / "M1"
    if not root.exists():
        print("no M1 trajectories")
        return
    rows = []
    for fp in sorted(root.rglob("*.jsonl")):
        meta, summary = None, None
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("type") == "meta":
                    meta = rec
                elif rec.get("type") == "summary":
                    summary = rec
        if meta and summary:
            rows.append({
                "model": meta["model"],
                "persona": meta["persona"],
                "seed": meta["seed"],
                "D_raw": summary["raw_defection_rate"],
                "L": summary["first_defection_round"],
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_dir / "M1_trajectory_metrics.csv", index=False)
    print(f"\nloaded n={len(df)} trajectories")
    g = df.groupby(["model", "persona"])["D_raw"].agg(["mean", "std", "count"]).round(3)
    print("\nPer-cell defection rates:")
    print(g.to_string())

    # Kill switch
    print("\n=== M1 KILL SWITCH ===")
    for m in df["model"].unique():
        i = df[(df["model"] == m) & (df["persona"] == "virtue_integrity")]["D_raw"]
        p = df[(df["model"] == m) & (df["persona"] == "virtue_phronesis")]["D_raw"]
        if len(i) == 0 or len(p) == 0:
            continue
        delta = p.mean() - i.mean()
        verdict = "PASS, proceed to M2" if abs(delta) >= 0.20 else "FAIL, stop interp pipeline"
        print(f"  {m}: D(phronesis) - D(integrity) = {delta:+.3f}   [{verdict}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    p.add_argument("--personas", nargs="+", default=PERSONAS)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--csv-dir", default="csvs")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)

    if args.grid:
        model, tok = load_hf_model(args.model)
        for persona in args.personas:
            for s in range(args.seeds):
                safe_model = args.model.replace("/", "_")
                tgt = results_dir / "M1" / safe_model / persona / f"seed{s}_opp{OPPONENT}.jsonl"
                if tgt.exists():
                    print(f"SKIP {tgt}")
                    continue
                run_trajectory(model, tok, args.model, persona, s, results_dir)
        del model
        torch.cuda.empty_cache()

    if args.analyze or args.grid:
        analyze(results_dir, csv_dir)


if __name__ == "__main__":
    main()
