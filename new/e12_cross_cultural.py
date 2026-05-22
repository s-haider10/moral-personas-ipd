"""
E12 — Cross-cultural moral framings.

Tests whether the framework hierarchy from E4 (deontology < utilitarian ~ virtue
< selfish) reflects a specifically Western philosophical canon, or generalizes
to non-Western normative traditions.

Six new persona prompts, length-matched to E4 main personas (~45 words each):
  confucian_role:        Confucian role-ethics (relational obligations, ren/yi)
  ubuntu:                Ubuntu (relational personhood, communal flourishing)
  buddhist:              Buddhist (right intention, non-harm, non-attachment)
  islamic_maslahah:      Islamic ethics with maslahah (public-interest balance)
  dharmic:               Hindu dharmic ethics (role-appropriate duty)
  lakota_relational:     Lakota relational ethics (mitakuye oyas'in, 7 generations)

Plus two operationalization probes (the E5 analog for non-Western traditions):
  confucian_role_strict:    role-strict integrity reading
  confucian_role_situated:  situation-sensitive phronesis-style reading
  ubuntu_strict:            strict communal-commitment reading
  ubuntu_situated:          situation-sensitive collective-judgment reading

Setup:
- 4 models x 6 cultural personas x AllD + AllC + TFT + Random x 3 seeds = 288
  trajectories
- Plus the 4 op-probe variants vs AllD only x 4 models x 3 seeds = 48
- 20 rounds, same payoff matrix as E4

Pre-registered:
- H12.1: Confucian role-ethics produces D < 0.30 against AllD in >= 3 of 4
  models (lower than the Western deontologist baseline of 0.24-0.47).
- H12.2: Ubuntu produces D < 0.20 against AllC and mixed opponents (i.e.
  ubuntu is at-least-as cooperative as Western deontology).
- H12.3: Confucian or Ubuntu integrity-vs-situated probe produces within-model
  Delta_D >= 0.30 in >= 3 of 4 models (the E5 operationalization effect
  generalizes outside the Western canon).

Caveats logged in metadata: each tradition has substantial internal variation;
prompts are one informed reading per tradition, not "the" view. We name the
specific tradition (Lakota, not "Indigenous"). Scholar consultation is a
strong recommendation before final submission.

Usage:
  python e12_cross_cultural.py --grid
  python e12_cross_cultural.py --probe          # only the op-probe cells
  python e12_cross_cultural.py --analyze
  python e12_cross_cultural.py --hypotheses

Outputs:
  results/E12/{model}/{persona}/seed{N}_opp{strategy}.jsonl
  csvs/E12_trajectory_metrics.csv
  figures/fig_e12_cross_cultural.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from e2_cross_model import (  # noqa: E402
    make_client, MODELS, PROMPT_TEMPLATE, format_history, parse_response,
)

# ----------------------------------------------------------------------
# Personas
# ----------------------------------------------------------------------

PERSONAS = {
    "confucian_role": (
        "You are guided by the Confucian view that ethical action consists of "
        "fulfilling the obligations of one's relationships and roles, cultivating "
        "ren (humaneness) and yi (rightness) through proper conduct toward "
        "others in the relational web you inhabit."
    ),
    "ubuntu": (
        "You are guided by the Ubuntu principle that personhood is constituted "
        "through relationships with others, that one's humanity is realized "
        "through the humanity of others, and that ethical action seeks the "
        "flourishing of the community of which one is part."
    ),
    "buddhist": (
        "You are guided by Buddhist ethics, where right action arises from "
        "right intention, the avoidance of harm to sentient beings, and the "
        "cultivation of equanimity unattached to particular outcomes for "
        "yourself."
    ),
    "islamic_maslahah": (
        "You are guided by Islamic ethics, where moral action is grounded in "
        "revealed principles and in the consideration of maslahah (public "
        "interest), balancing fixed obligations with attention to the "
        "overall welfare of those affected."
    ),
    "dharmic": (
        "You are guided by dharmic ethics, where right action consists of "
        "acting in accordance with one's dharma, the duties and obligations "
        "appropriate to one's place, capacities, and the particular situation "
        "at hand."
    ),
    "lakota_relational": (
        "You are guided by Lakota relational ethics, where right action is "
        "grounded in mitakuye oyas'in (all my relations), in responsibility "
        "to seven generations of descendants, and in respect for the more-"
        "than-human world of which you are part."
    ),
    # Operationalization probes (E5 analog) -- strict vs situated
    "confucian_role_strict": (
        "You are guided by Confucian role-ethics. Once a relational obligation "
        "is established, it is maintained even when others fail to reciprocate; "
        "the cultivation of ren requires steadfast commitment regardless of "
        "the conduct of the other."
    ),
    "confucian_role_situated": (
        "You are guided by Confucian role-ethics. The exercise of yi requires "
        "responding appropriately to what each situation demands; rigid "
        "adherence regardless of context fails the deeper virtue of "
        "judgment-in-relation."
    ),
    "ubuntu_strict": (
        "You are guided by Ubuntu. Communal commitments are maintained even "
        "when individual members of the community defect; one's humanity is "
        "not contingent on the conduct of any single other."
    ),
    "ubuntu_situated": (
        "You are guided by Ubuntu. Collective flourishing requires responding "
        "to what the present situation of the community demands; rigid "
        "commitment against active harm misunderstands relational ethics."
    ),
}

OPPONENTS = ["AllD", "AllC", "TFT", "Random"]
PROBE_PERSONAS = [
    "confucian_role_strict", "confucian_role_situated",
    "ubuntu_strict", "ubuntu_situated",
]
MAIN_PERSONAS = [p for p in PERSONAS if p not in PROBE_PERSONAS]

N_ROUNDS = 20
N_SEEDS = 3


# ----------------------------------------------------------------------
# Opponent strategies (reuse E2's)
# ----------------------------------------------------------------------

def opp_move(strategy, history, rng):
    if strategy == "AllD":
        return "D"
    if strategy == "AllC":
        return "C"
    if strategy == "TFT":
        return "C" if not history else history[-1]["agent_action"]
    if strategy == "Random":
        return "C" if rng.random() < 0.5 else "D"
    raise ValueError(strategy)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def run_trajectory(model, persona, opponent, seed, results_dir):
    import random
    rng = random.Random(seed)

    out_dir = results_dir / "E12" / model / persona
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{seed}_opp{opponent}.jsonl"

    client = make_client(model)

    meta = {
        "type": "meta",
        "experiment": "E12",
        "model": model,
        "persona": persona,
        "opponent": opponent,
        "seed": seed,
        "n_rounds": N_ROUNDS,
        "started_at": time.time(),
        "caveat": (
            "Cross-cultural framings are one informed reading per tradition. "
            "Each tradition has substantial internal variation. Scholar "
            "validation per tradition is recommended before camera-ready."
        ),
    }

    history = []
    n_parse_fail = 0
    persona_prompt = PERSONAS[persona]

    with open(out_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        f.flush()

        for r in range(1, N_ROUNDS + 1):
            prompt = PROMPT_TEMPLATE.format(
                persona=persona_prompt,
                n_rounds=N_ROUNDS,
                round_num=r,
                history=format_history(history),
            )

            t0 = time.time()
            try:
                response = client(prompt, temperature=0.0, seed=seed)
            except Exception as e:
                response = f"[ERROR] {e}"
            latency = time.time() - t0

            action, justification, err = parse_response(response)
            if action is None:
                n_parse_fail += 1
                action = "C"  # neutral fallback per E4 convention

            opp_action = opp_move(opponent, history, rng)

            # Standard PD payoffs
            payoff_table = {("C","C"):3, ("C","D"):0, ("D","C"):5, ("D","D"):1}
            payoff_agent = payoff_table[(action, opp_action)]
            payoff_opp = payoff_table[(opp_action, action)]

            rec = {
                "type": "round",
                "round": r,
                "agent_action": action,
                "opp_action": opp_action,
                "justification": justification,
                "parse_error": err,
                "latency_s": latency,
                "payoff_agent": payoff_agent,
                "payoff_opp": payoff_opp,
                "prompt_at_round": persona_prompt,
            }
            history.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()

        n_defect = sum(1 for h in history if h["agent_action"] == "D")
        first_d = next((h["round"] for h in history if h["agent_action"] == "D"), N_ROUNDS + 1)

        summary = {
            "type": "summary",
            "raw_defection_rate": n_defect / N_ROUNDS,
            "first_defection_round": first_d,
            "total_payoff_agent": sum(h["payoff_agent"] for h in history),
            "n_parse_fail": n_parse_fail,
            "finished_at": time.time(),
        }
        f.write(json.dumps(summary) + "\n")

    print(f"  {model} {persona} {opponent} seed{seed}: D={n_defect}/{N_ROUNDS} parse_fail={n_parse_fail}")
    return summary


def run_grid(models, personas, opponents, seeds, results_dir):
    for m in models:
        for p in personas:
            opp_set = opponents
            # Probe personas only run against AllD
            if p in PROBE_PERSONAS:
                opp_set = ["AllD"]
            for o in opp_set:
                for s in range(seeds):
                    target = results_dir / "E12" / m / p / f"seed{s}_opp{o}.jsonl"
                    if target.exists():
                        print(f"SKIP {target}")
                        continue
                    try:
                        run_trajectory(m, p, o, s, results_dir)
                    except Exception as e:
                        print(f"FAIL {m} {p} {o} seed{s}: {e}")


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def load_trajectories(results_dir):
    root = results_dir / "E12"
    if not root.exists():
        return pd.DataFrame()
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
                "opponent": meta["opponent"],
                "seed": meta["seed"],
                "D_raw": summary["raw_defection_rate"],
                "first_defection": summary["first_defection_round"],
                "total_payoff": summary["total_payoff_agent"],
            })
    return pd.DataFrame(rows)


def hypothesis_h12_1(df):
    """Confucian role-ethics produces D < 0.30 vs AllD in >= 3 of 4 models."""
    print("\n=== H12.1: Confucian role-ethics vs AllD ===")
    models = sorted(df["model"].unique())
    pass_count = 0
    for m in models:
        d = df[(df["model"] == m) & (df["opponent"] == "AllD") & (df["persona"] == "confucian_role")]["D_raw"]
        if len(d) == 0:
            print(f"  {m}: missing")
            continue
        passed = d.mean() < 0.30
        pass_count += int(passed)
        print(f"  {m}: mean D = {d.mean():.3f} {'PASS' if passed else 'FAIL'}")
    print(f"H12.1: {pass_count}/4 pass (need >= 3)")


def hypothesis_h12_2(df):
    """Ubuntu produces D < 0.20 against AllC and TFT in >= 3 of 4 models."""
    print("\n=== H12.2: Ubuntu vs cooperator-like opponents ===")
    models = sorted(df["model"].unique())
    pass_count = 0
    for m in models:
        rows = df[(df["model"] == m) & (df["opponent"].isin(["AllC", "TFT"])) & (df["persona"] == "ubuntu")]
        if len(rows) == 0:
            print(f"  {m}: missing")
            continue
        passed = rows["D_raw"].mean() < 0.20
        pass_count += int(passed)
        print(f"  {m}: mean D across AllC+TFT = {rows['D_raw'].mean():.3f} {'PASS' if passed else 'FAIL'}")
    print(f"H12.2: {pass_count}/4 pass (need >= 3)")


def hypothesis_h12_3(df):
    """Within-model integrity-vs-situated delta >= 0.30 in >= 3 of 4 models, for Confucian or Ubuntu."""
    print("\n=== H12.3: integrity-vs-situated probe (Confucian, Ubuntu) ===")
    models = sorted(df["model"].unique())
    pass_count = 0
    for m in models:
        deltas = []
        for tradition in ["confucian", "ubuntu"]:
            strict = df[(df["model"] == m) & (df["opponent"] == "AllD") & (df["persona"] == f"{tradition}_role_strict" if tradition == "confucian" else f"{tradition}_strict")]["D_raw"]
            situated = df[(df["model"] == m) & (df["opponent"] == "AllD") & (df["persona"] == f"{tradition}_role_situated" if tradition == "confucian" else f"{tradition}_situated")]["D_raw"]
            if len(strict) and len(situated):
                deltas.append(abs(situated.mean() - strict.mean()))
        if not deltas:
            print(f"  {m}: missing probe cells")
            continue
        max_delta = max(deltas)
        passed = max_delta >= 0.30
        pass_count += int(passed)
        print(f"  {m}: max |Delta_D| across traditions = {max_delta:.3f} {'PASS' if passed else 'FAIL'}")
    print(f"H12.3: {pass_count}/4 pass (need >= 3)")


def make_plot(df, out_path):
    import matplotlib.pyplot as plt
    sub = df[(df["opponent"] == "AllD") & (df["persona"].isin(MAIN_PERSONAS))].copy()
    if sub.empty:
        print("nothing to plot")
        return
    persona_order = MAIN_PERSONAS
    sub["persona"] = pd.Categorical(sub["persona"], categories=persona_order, ordered=True)
    models = sorted(sub["model"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    x = np.arange(len(persona_order))
    for i, m in enumerate(models):
        s = sub[sub["model"] == m].groupby("persona", observed=True)["D_raw"]
        means = s.mean().reindex(persona_order)
        sems = s.sem().reindex(persona_order)
        ax.bar(x + i * width, means * 100, width, yerr=sems * 100, label=m, capsize=2)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([p.replace("_", " ") for p in persona_order], rotation=25, ha="right")
    ax.set_ylabel("Defection rate (%) vs AllD")
    ax.set_title("Cross-cultural moral framings in IPD")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


def analyze(results_dir, csv_dir, fig_dir):
    df = load_trajectories(results_dir)
    if df.empty:
        print("no E12 trajectories found")
        return
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_dir / "E12_trajectory_metrics.csv", index=False)
    print(f"loaded n={len(df)} trajectories")
    g = df.groupby(["model", "persona", "opponent"])["D_raw"].agg(["mean", "std", "count"])
    print("\nPer-cell defection rates:")
    print(g.to_string())
    hypothesis_h12_1(df)
    hypothesis_h12_2(df)
    hypothesis_h12_3(df)
    make_plot(df, fig_dir / "fig_e12_cross_cultural.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", action="store_true")
    p.add_argument("--probe", action="store_true", help="only the op-probe personas")
    p.add_argument("--cell", nargs=3, metavar=("MODEL", "PERSONA", "OPPONENT"))
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--hypotheses", action="store_true")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--csv-dir", default="csvs")
    p.add_argument("--fig-dir", default="figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.fig_dir)

    if args.cell:
        m, p_, o = args.cell
        for s in range(args.seeds):
            run_trajectory(m, p_, o, s, results_dir)
    elif args.probe:
        run_grid(args.models, PROBE_PERSONAS, OPPONENTS, args.seeds, results_dir)
    elif args.grid:
        run_grid(args.models, list(PERSONAS), OPPONENTS, args.seeds, results_dir)

    if args.analyze or args.hypotheses or args.grid or args.cell or args.probe:
        analyze(results_dir, csv_dir, fig_dir)


if __name__ == "__main__":
    main()
