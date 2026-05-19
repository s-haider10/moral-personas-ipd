"""Quick E5 analyzer: defection rate per (model, variant) and Δ."""
import json
from collections import defaultdict
from pathlib import Path

MODELS = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-pro", "gemini-2.5-flash"]
VARIANTS = ["virtue_integrity", "virtue_phronesis"]
ROOT = Path("results/E5")


def summary_of(path):
    with open(path) as f:
        last = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "summary":
                return obj
            last = obj
    return last


by_cell = defaultdict(list)
first_def = defaultdict(list)
missing = []

for model in MODELS:
    for variant in VARIANTS:
        d = ROOT / model / variant
        if not d.exists():
            missing.append(f"{model}/{variant}")
            continue
        for f in sorted(d.glob("seed*_oppAllD.jsonl")):
            s = summary_of(f)
            if s is None:
                continue
            by_cell[(model, variant)].append(s["raw_defection_rate"])
            first_def[(model, variant)].append(s.get("first_defection_round"))

print(f"\n  {'model':<22}{'integrity':>12}{'phronesis':>12}{'Δ (phron-integ)':>20}{'  L_integ':>10}{'  L_phron':>10}")
print("  " + "-" * 86)
deltas = []
for m in MODELS:
    i_d = by_cell.get((m, "virtue_integrity"), [])
    p_d = by_cell.get((m, "virtue_phronesis"), [])
    if not i_d or not p_d:
        print(f"  {m:<22}{'—':>12}{'—':>12}{'—':>20}{'—':>10}{'—':>10}")
        continue
    mi, mp = sum(i_d)/len(i_d), sum(p_d)/len(p_d)
    d = mp - mi
    deltas.append(d)
    li = [x for x in first_def[(m, "virtue_integrity")] if x is not None]
    lp = [x for x in first_def[(m, "virtue_phronesis")] if x is not None]
    li_s = f"{sum(li)/len(li):.1f}" if li else "—"
    lp_s = f"{sum(lp)/len(lp):.1f}" if lp else "—"
    print(f"  {m:<22}{mi:>11.0%}{mp:>11.0%}{d:>+19.0%}{li_s:>10}{lp_s:>10}")

print("\n  per-seed defection rates:")
for m in MODELS:
    for v in VARIANTS:
        vals = by_cell.get((m, v), [])
        if vals:
            print(f"    {m:<22} {v:<18} {['%.0f%%' % (x*100) for x in vals]}")

print("\nKILL-SWITCH:")
if not deltas:
    print("  no data")
elif all(d < 0.10 for d in deltas):
    print("  >>> TRIGGERED: framing doesn't separate variants — GPT-4o anomaly NOT from interpretation.")
elif all(d >= 0.30 for d in deltas):
    print("  >>> CLEAN PASS: phronesis >> integrity in all models — interpretation hypothesis confirmed.")
else:
    print(f"  >>> MIXED: deltas {min(deltas):+.0%} to {max(deltas):+.0%}")

if missing:
    print(f"\n  missing: {missing}")
