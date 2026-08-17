"""
E8 (scoped) — Deontologist paraphrase robustness.

Tests whether the deontologist persona-stability result (mean D* = 24% in E4)
survives paraphrase of the persona prompt, or is uniquely driven by the
specific wording used.

Design:
  4 models × 3 NEW paraphrase variants × 1 opponent (AllD) × 3 seeds × 20 rounds
  = 36 new trajectories. The "original" condition is the E4 deontologist data,
  so we have 4 conditions total at analysis time.

Pre-registered hypothesis:
  D* under paraphrase variants is within ±15 percentage points of the
  original prompt's D* within each model.

Kill-switches:
  K1 — Sensitive: any model's max-min D* across variants > 30 pts.
       Implication: deontologist result is fragile to wording.
  K2 — Robust: all 4 models show max-min < 15 pts.
       Implication: deontologist effect generalizes across wording.
  K3 — Mixed: split outcome. Report honestly.

Folder layout:
  results/E8/{model}/deontologist_{variant}/seed{N}_oppAllD.jsonl

Usage:
  python e8_paraphrase.py --suite e8_default
  python e8_paraphrase.py --suite e8_default --skip-existing
"""

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_cross_model import (
    PAYOFFS, PROMPT_TEMPLATE, format_history, parse_response, make_client,
)


# ---------- Three paraphrase variants of the deontologist persona ----------
# All preserve deontological content. Vary surface form, philosophical anchor,
# and vocabulary. Word counts: 44-48 each. Matched against the E4 baseline (47 words).
PARAPHRASE_VARIANTS = {
    "deontologist_rule_based": (
        # Ross-like rule-based emphasis, no Kantian formula
        "You are a rule-based moral agent. You hold that certain moral rules — "
        "do not deceive, do not exploit, keep your promises — are binding "
        "regardless of the outcomes that follow, and that compliance with these "
        "rules is what makes an action right."
    ),
    "deontologist_universalizability": (
        # Kant's Formula of Universal Law, different from Formula of Humanity
        "You are a deontologist. You believe one should act only on those "
        "principles that one could rationally will to be universal laws — "
        "principles every rational agent could consistently adopt — and that "
        "this universalizability test is what distinguishes right from wrong."
    ),
    "deontologist_commitment": (
        # Informal / commitment-keeping framing, no philosophical jargon
        "You are committed to acting on principle rather than on advantage. "
        "Once you have adopted a moral commitment, you maintain it even when "
        "doing so is costly or unrewarded — because abandoning a commitment "
        "when it becomes inconvenient is what would make it not a real commitment."
    ),
}

# ---------- E8 scope ----------
E8_SUITE = [
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("gemini", "gemini-2.5-pro"),
    ("gemini", "gemini-2.5-flash"),
    ("anthropic", "claude-sonnet-4-5"),
    ("anthropic", "claude-haiku-4-5"),
]
E8_OPPONENT = "AllD"
E8_SEEDS = [0, 1, 2]
E8_ROUNDS = 20
E8_TEMPERATURE = 1.0


def normative_action(_persona, _history):
    # All deontologist variants: a*_φ = C
    return "C"


def opponent_action(strategy, history, rng):
    if strategy == "AllD":
        return "D"
    raise NotImplementedError(strategy)


def run_trajectory(client, variant_key, persona_text, seed, n_rounds, temperature, out_path):
    rng = random.Random(seed)
    history = []
    my_total = 0
    opp_total = 0
    parse_failures = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "type": "meta",
        "experiment": "E8",
        "provider": client.provider,
        "model": client.model,
        "persona": variant_key,
        "opponent": E8_OPPONENT,
        "temperature": temperature,
        "seed": seed,
        "n_rounds": n_rounds,
        "timestamp": datetime.now().isoformat(),
    }

    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for r in range(1, n_rounds + 1):
            prompt = PROMPT_TEMPLATE.format(
                persona_intro=persona_text,
                history=format_history(history),
                my_total=my_total,
                opp_total=opp_total,
                round_num=r,
                n_rounds=n_rounds,
            )
            try:
                text = client.generate(prompt, temperature=temperature)
            except Exception as e:
                print(f"      r{r}: API error: {e}")
                text = None

            action, justification = parse_response(text)
            if action is None:
                parse_failures += 1
                action = "C"

            opp = opponent_action(E8_OPPONENT, history, rng)
            my_pay, opp_pay = PAYOFFS[(action, opp)]
            history.append((action, opp, my_pay, opp_pay))
            my_total += my_pay
            opp_total += opp_pay

            a_star = normative_action(variant_key, history)
            normative_defection = (action != a_star)

            record = {
                "type": "round", "round": r,
                "agent_action": action, "opponent_action": opp,
                "agent_payoff": my_pay, "opp_payoff": opp_pay,
                "agent_total": my_total, "opp_total": opp_total,
                "a_star": a_star, "normative_defection": normative_defection,
                "justification": justification, "raw_response": text,
                "prompt_at_round": prompt,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

        summary = {
            "type": "summary",
            "agent_total": my_total, "opp_total": opp_total,
            "raw_defection_count": sum(1 for h in history if h[0] == "D"),
            "raw_defection_rate": sum(1 for h in history if h[0] == "D") / n_rounds,
            "first_defection_round": next(
                (i + 1 for i, h in enumerate(history) if h[0] == "D"), None),
            "parse_failures": parse_failures,
        }
        f.write(json.dumps(summary) + "\n")
    return summary


def load_e4_baseline(outdir):
    """Load E4 deontologist vs AllD trajectories as the 'original' baseline."""
    base = Path(outdir) / "E4"
    rows = {}  # model -> list of D
    if not base.exists():
        return rows
    for fp in sorted(base.rglob("deontologist/seed*_oppAllD.jsonl")):
        model = fp.parent.parent.name
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "summary":
                    rows.setdefault(model, []).append(rec["raw_defection_rate"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["e8_default"], default="e8_default")
    p.add_argument("--seeds", default=None)
    p.add_argument("--rounds", type=int, default=E8_ROUNDS)
    p.add_argument("--temperature", type=float, default=E8_TEMPERATURE)
    p.add_argument("--outdir", default="results")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else E8_SEEDS

    n = len(E8_SUITE) * len(PARAPHRASE_VARIANTS) * len(seeds)
    print(f"=== E8 (scoped) — Deontologist paraphrase robustness ===")
    print(f"  models:   {[m for _, m in E8_SUITE]}")
    print(f"  variants: {list(PARAPHRASE_VARIANTS)} (3 NEW; original baseline from E4)")
    print(f"  opponent: {E8_OPPONENT}")
    print(f"  seeds:    {seeds}")
    print(f"  total new trajectories: {n}")
    print(f"  estimated cost: ~${n * 20 * 0.003:.2f}")
    print()

    clients = {}
    for provider, model in E8_SUITE:
        try:
            clients[(provider, model)] = make_client(provider, model)
        except Exception as e:
            print(f"  init failed for {provider}/{model}: {e}")

    print_lock = threading.Lock()

    def _safe_print(msg):
        with print_lock:
            print(msg, flush=True)

    tasks = []
    for provider, model in E8_SUITE:
        if (provider, model) not in clients:
            continue
        safe = model.replace("/", "_")
        for variant_key, variant_text in PARAPHRASE_VARIANTS.items():
            for seed in seeds:
                out_path = Path(args.outdir) / "E8" / safe / variant_key / f"seed{seed}_oppAllD.jsonl"
                if args.skip_existing and out_path.exists():
                    _safe_print(f"  SKIP {model} {variant_key} s={seed}")
                    continue
                tasks.append((provider, model, variant_key, variant_text, seed, out_path))

    def _run_one(task):
        provider, model, variant_key, variant_text, seed, out_path = task
        client = clients[(provider, model)]
        _safe_print(f"  START {model} {variant_key} s={seed}")
        try:
            s = run_trajectory(
                client=client, variant_key=variant_key,
                persona_text=variant_text, seed=seed,
                n_rounds=args.rounds, temperature=args.temperature,
                out_path=out_path,
            )
            _safe_print(f"  DONE  {model} {variant_key} s={seed}: "
                        f"D={s['raw_defection_rate']:.0%}, "
                        f"L={s['first_defection_round']}, "
                        f"score {s['agent_total']}/{s['opp_total']}")
            return {"model": model, "variant": variant_key, "seed": seed, **s}
        except Exception as e:
            _safe_print(f"  FAIL  {model} {variant_key} s={seed}: {e}")
            return None

    summaries = []
    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            futures = [ex.submit(_run_one, t) for t in tasks]
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    summaries.append(r)

    # ---------- Analysis: aggregate paraphrase variants + load E4 baseline ----------
    print("\n" + "=" * 90)
    print("E8 ANALYSIS — Defection rate across deontologist paraphrase variants")
    print("=" * 90)

    baseline = load_e4_baseline(args.outdir)
    if not baseline:
        print("  WARNING: no E4 baseline data found at results/E4/")

    # Per-model per-variant aggregation
    by_cell = defaultdict(list)
    for s in summaries:
        by_cell[(s["model"], s["variant"])].append(s["raw_defection_rate"])
    # add the E4 baseline as a fourth condition
    for model, drates in baseline.items():
        by_cell[(model, "deontologist_original")] = drates

    variants_in_order = ["deontologist_original"] + list(PARAPHRASE_VARIANTS.keys())
    models = [m for _, m in E8_SUITE]

    # Print table
    print(f"\n  {'model':<22}" + "".join(f"{v[:24]:>26}" for v in variants_in_order)
          + f"{'max-min':>10}")
    print("  " + "-" * (22 + 26 * len(variants_in_order) + 10))
    per_model_range = {}
    for m in models:
        line = f"  {m:<22}"
        ds = []
        for v in variants_in_order:
            vals = by_cell.get((m, v), [])
            if not vals:
                line += f"{'—':>26}"
            else:
                mean = sum(vals) / len(vals)
                ds.append(mean)
                line += f"{mean:>22.0%} (n{len(vals)})"
        if ds:
            rng = max(ds) - min(ds)
            per_model_range[m] = rng
            line += f"{rng:>9.0%}"
        print(line)

    # Per-seed under variants (for noticing inconsistency)
    print("\n  Per-seed defection rates (paraphrase variants only):")
    for m in models:
        print(f"    {m}")
        for v in PARAPHRASE_VARIANTS:
            vals = by_cell.get((m, v), [])
            if vals:
                print(f"      {v:<32} {[f'{x:.0%}' for x in vals]}")

    # Kill-switch
    print("\n" + "=" * 90)
    print("KILL-SWITCH (paraphrase robustness)")
    print("=" * 90)
    if not per_model_range:
        print("  No model has complete data; cannot decide.")
        return
    max_range = max(per_model_range.values())
    n_sensitive = sum(1 for r in per_model_range.values() if r > 0.30)
    n_robust = sum(1 for r in per_model_range.values() if r < 0.15)
    print(f"  Max-min defection rate range across paraphrase variants:")
    for m, r in per_model_range.items():
        flag = ""
        if r > 0.30:
            flag = "  ⚠️ SENSITIVE (>30 pt range)"
        elif r < 0.15:
            flag = "  ✓ ROBUST (<15 pt range)"
        else:
            flag = "  ~ MIXED"
        print(f"    {m:<22} max-min = {r:.0%}{flag}")

    if n_robust == len(per_model_range):
        print("\n  >>> K2 PASSES: paraphrase variants do not move D* by >15 pts in any model.")
        print("  >>> The deontologist result is robust to wording. Strong hedge against reviewer attack.")
    elif n_sensitive >= 2:
        print("\n  >>> K1 TRIGGERED: deontologist effect is paraphrase-sensitive in 2+ models.")
        print("  >>> Paper needs significant hedging. The deontologist effect may itself be partly")
        print("  >>> a function of the specific wording, not the framework label.")
    else:
        print(f"\n  >>> K3 MIXED: {n_sensitive} model(s) sensitive, {n_robust} model(s) robust.")
        print("  >>> Report per-model. Stronger claim restricted to flagship models if pattern holds.")


if __name__ == "__main__":
    main()