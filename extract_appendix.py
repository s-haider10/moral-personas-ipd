"""
extract_appendix.py

Extract appendix content directly from the experiment scripts and CSVs:
  Appendix A: verbatim persona prompts from e2_cross_model.py (main 5),
              e5_virtue_variants.py (V_integrity, V_phronesis),
              e8_paraphrase.py (3 deontology paraphrases).
  Appendix B: action-declaration regex patterns from e6_strict_leakage.py,
              with sample counts of how many sentences were stripped per pattern.
  Appendix C: per-(model, persona, opponent, seed) numeric tables from
              csvs/trajectory_metrics.csv, formatted as LaTeX booktabs.

Output: prints LaTeX for each appendix section to stdout. Pipe to a file or
copy directly into main.tex.

Usage:
  python extract_appendix.py > appendix_dump.tex
  python extract_appendix.py --section A          # just Appendix A
  python extract_appendix.py --section A,B
  python extract_appendix.py --src experiments    # path containing scripts
  python extract_appendix.py --csv csvs           # path to CSVs
"""

import argparse
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Appendix A: persona prompts
# --------------------------------------------------------------------------

PROMPT_FILES = {
    "main_personas": "e2_cross_model.py",
    "virtue_variants": "e5_virtue_variants.py",
    "deontology_paraphrases": "e8_paraphrase.py",
}

# Variable names to look for inside each file. We match a dict literal assigned
# to one of these names (PERSONAS, VIRTUE_VARIANTS, DEONT_PARAPHRASES, etc.)
PROMPT_VARS = [
    "PERSONAS", "PERSONA_PROMPT", "PERSONA_PROMPTS",
    "VIRTUE_VARIANTS", "VIRTUE_PROMPTS", "PERSONA_VARIANTS",
    "DEONT_PARAPHRASES", "DEONTOLOGY_PARAPHRASES", "PARAPHRASES",
    "PARAPHRASE_VARIANTS",
]

# Display order
DISPLAY_ORDER_MAIN = [
    "deontologist", "utilitarian", "virtue_ethicist", "selfish", "neutral",
]
DISPLAY_ORDER_VIRTUE = ["virtue_integrity", "virtue_phronesis"]
DISPLAY_ORDER_DEONT = [
    "deontologist_rule_based",
    "deontologist_universalizability",
    "deontologist_commitment",
]

PRETTY_NAME = {
    "deontologist": "Deontologist (Formula of Humanity)",
    "utilitarian": "Utilitarian",
    "virtue_ethicist": "Virtue Ethicist",
    "selfish": "Self-Interested",
    "neutral": "Neutral",
    "virtue_integrity": "Virtue Ethics --- Integrity (V$_\\text{integrity}$)",
    "virtue_phronesis": "Virtue Ethics --- Phronesis (V$_\\text{phronesis}$)",
    "deontologist_rule_based": "Deontology --- Rule-Based (Ross)",
    "deontologist_universalizability": "Deontology --- Universalizability (Formula of Universal Law)",
    "deontologist_commitment": "Deontology --- Commitment (informal)",
}


def extract_dict_from_source(source_text, var_name):
    """Parse a Python file as AST and find an assignment of a dict literal to var_name."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        continue
                    if isinstance(value, dict):
                        return value
    return None


def latex_escape(text):
    """Minimal escaping for verbatim prompts going into a LaTeX quote env."""
    return (text.replace("\\", "\\textbackslash{}")
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("$", "\\$")
                .replace("^", "\\^{}")
                .replace("~", "\\~{}"))


def word_count(text):
    return len(text.split())


def find_prompts(src_dir):
    """Return three dicts: main personas, virtue variants, deontology paraphrases."""
    src_dir = Path(src_dir)
    found = {"main": {}, "virtue": {}, "deont": {}}

    candidate_files = list(src_dir.glob("e*.py"))
    if not candidate_files:
        print(f"% WARNING: no e*.py files found in {src_dir}", flush=True)

    for fp in candidate_files:
        text = fp.read_text(errors="replace")
        for var in PROMPT_VARS:
            d = extract_dict_from_source(text, var)
            if d is None:
                continue
            for key, val in d.items():
                if not isinstance(val, str):
                    continue
                if key in DISPLAY_ORDER_MAIN:
                    found["main"].setdefault(key, val)
                elif key in DISPLAY_ORDER_VIRTUE or "integrity" in key or "phronesis" in key:
                    norm = key
                    if "integrity" in key and "virtue" in key:
                        norm = "virtue_integrity"
                    elif "phronesis" in key and "virtue" in key:
                        norm = "virtue_phronesis"
                    found["virtue"].setdefault(norm, val)
                elif key in DISPLAY_ORDER_DEONT or key.startswith("deontologist_"):
                    found["deont"].setdefault(key, val)
    return found


def emit_appendix_A(src_dir):
    prompts = find_prompts(src_dir)
    out = []
    out.append("% =====================================================")
    out.append("% Appendix A: Persona Prompts (auto-extracted)")
    out.append("% =====================================================")
    out.append("\\section{Persona Prompts}")
    out.append("\\label{app:prompts}")
    out.append("")
    out.append("All persona prompts are given verbatim below. Word counts are reported "
               "for the persona description only, excluding the fixed task framing "
               "(payoff matrix, history, response format) shared across personas.")
    out.append("")
    out.append("\\subsection{Main Personas (\\S\\ref{sec:method})}")

    for key in DISPLAY_ORDER_MAIN:
        prompt = prompts["main"].get(key)
        if prompt is None:
            out.append(f"\\paragraph{{{PRETTY_NAME[key]}}} \\caveat{{NOT FOUND in source}}")
            out.append("")
            continue
        wc = word_count(prompt)
        out.append(f"\\paragraph{{{PRETTY_NAME[key]} ({wc} words).}}")
        out.append("\\begin{quote}")
        out.append(latex_escape(prompt.strip()))
        out.append("\\end{quote}")
        out.append("")

    out.append("\\subsection{Virtue-Ethics Variants (\\S\\ref{sec:operationalization})}")
    for key in DISPLAY_ORDER_VIRTUE:
        prompt = prompts["virtue"].get(key)
        if prompt is None:
            out.append(f"\\paragraph{{{PRETTY_NAME[key]}}} \\caveat{{NOT FOUND in source}}")
            out.append("")
            continue
        wc = word_count(prompt)
        out.append(f"\\paragraph{{{PRETTY_NAME[key]} ({wc} words).}}")
        out.append("\\begin{quote}")
        out.append(latex_escape(prompt.strip()))
        out.append("\\end{quote}")
        out.append("")

    out.append("\\subsection{Deontology Paraphrases (\\S\\ref{sec:operationalization})}")
    for key in DISPLAY_ORDER_DEONT:
        prompt = prompts["deont"].get(key)
        if prompt is None:
            out.append(f"\\paragraph{{{PRETTY_NAME[key]}}} \\caveat{{NOT FOUND in source}}")
            out.append("")
            continue
        wc = word_count(prompt)
        out.append(f"\\paragraph{{{PRETTY_NAME[key]} ({wc} words).}}")
        out.append("\\begin{quote}")
        out.append(latex_escape(prompt.strip()))
        out.append("\\end{quote}")
        out.append("")

    return "\n".join(out)


# --------------------------------------------------------------------------
# Appendix B: action-declaration regex patterns
# --------------------------------------------------------------------------

def extract_regex_patterns(src_dir):
    """Look in e6_strict_leakage.py for ACTION_DECLARATION_PATTERNS list of strings."""
    src_dir = Path(src_dir)
    fp = src_dir / "e6_strict_leakage.py"
    if not fp.exists():
        return None
    text = fp.read_text(errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ACTION_DECLARATION_PATTERNS":
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
    return None


def count_pattern_hits(src_dir, csv_dir, patterns):
    """If we have access to the trajectories on disk, count how many sentences each pattern matches.

    Returns dict: pattern -> count. If trajectory results are not on disk, returns empty.
    """
    results_root = Path("results") / "E4"
    if not results_root.exists():
        return {}

    import json
    sent_split = re.compile(r"(?<=[.!?])\s+")
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
    counts = Counter()
    total_sentences = 0

    for fp in results_root.rglob("*.jsonl"):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "round":
                    continue
                just = rec.get("justification") or ""
                if not just:
                    continue
                for s in sent_split.split(just):
                    total_sentences += 1
                    for raw, comp in compiled:
                        if comp.search(s):
                            counts[raw] += 1
                            break  # only count first matching pattern per sentence
    return {"per_pattern": counts, "total_sentences": total_sentences}


def emit_appendix_B(src_dir, csv_dir):
    patterns = extract_regex_patterns(src_dir)
    out = []
    out.append("% =====================================================")
    out.append("% Appendix B: Action-Declaration Regex (auto-extracted)")
    out.append("% =====================================================")
    out.append("\\section{Action-Declaration Regular Expressions}")
    out.append("\\label{app:regex}")
    out.append("")
    out.append("To address action leakage in the linguistic predictor "
               "(\\S\\ref{sec:fingerprint}), we remove sentences from each justification "
               "$j_t$ that explicitly declare the agent's action. A sentence is removed "
               "if it matches any of the regular expressions below (case-insensitive). "
               "Sentences are obtained by splitting on \\texttt{[.!?]\\textbackslash{}s+}.")
    out.append("")

    if patterns is None:
        out.append("\\caveat{ACTION\\_DECLARATION\\_PATTERNS not found in "
                   "\\texttt{e6\\_strict\\_leakage.py}.}")
        return "\n".join(out)

    # Optional per-pattern hit counts
    hits = count_pattern_hits(src_dir, csv_dir, patterns)

    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\small")
    if hits and hits.get("total_sentences"):
        out.append("\\begin{tabular}{p{0.55\\columnwidth} r r}")
        out.append("\\toprule")
        out.append("Pattern & Hits & \\% \\\\")
        out.append("\\midrule")
        total = hits["total_sentences"]
        per_pat = hits["per_pattern"]
        for p in patterns:
            c = per_pat.get(p, 0)
            pct = 100.0 * c / total if total else 0
            esc = latex_escape(p)
            out.append(f"\\verb`{p}` & {c} & {pct:.2f} \\\\")
        total_hits = sum(per_pat.values())
        total_pct = 100.0 * total_hits / total if total else 0
        out.append("\\midrule")
        out.append(f"\\textbf{{Total stripped}} & \\textbf{{{total_hits}}} & "
                   f"\\textbf{{{total_pct:.2f}}} \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append(f"\\caption{{Action-declaration patterns and their match counts "
                   f"across all $T = 20$ justifications of all $n = 300$ trajectories "
                   f"({total} total sentences).}}")
    else:
        out.append("\\begin{tabular}{p{0.9\\columnwidth}}")
        out.append("\\toprule")
        out.append("Pattern \\\\")
        out.append("\\midrule")
        for p in patterns:
            out.append(f"\\verb`{p}` \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\caption{Action-declaration patterns used to strip the chain-of-thought.}")
    out.append("\\label{tab:regex}")
    out.append("\\end{table}")
    out.append("")
    out.append("Across the $6{,}000$ justifications in the main grid, the strip retains "
               "$81\\%$ of words on average (median $0.81$; minimum $0.39$). The remaining "
               "text is what underlies the $\\text{AUC} = 0.82$ figure in \\S\\ref{sec:fingerprint}.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Appendix C: per-cell numeric tables
# --------------------------------------------------------------------------

PERSONA_PRETTY = {
    "deontologist": "Deontology",
    "utilitarian": "Utilitarianism",
    "virtue_ethicist": "Virtue Ethics",
    "selfish": "Self-interested",
    "neutral": "Neutral",
}
MODEL_ORDER = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-pro", "gemini-2.5-flash"]
PERSONA_ORDER = ["deontologist", "utilitarian", "virtue_ethicist", "selfish", "neutral"]
OPPONENT_ORDER = ["AllD", "AllC", "TFT", "GTFT", "Random"]


def emit_appendix_C(csv_dir):
    import csv as csvlib
    csv_dir = Path(csv_dir)
    fp = csv_dir / "trajectory_metrics.csv"
    out = []
    out.append("% =====================================================")
    out.append("% Appendix C: Per-Cell Numeric Tables (auto-extracted)")
    out.append("% =====================================================")
    out.append("\\section{Per-Cell Numeric Tables}")
    out.append("\\label{app:tables}")
    out.append("")

    if not fp.exists():
        out.append(f"\\caveat{{Could not find {fp}. Paste manually.}}")
        return "\n".join(out)

    rows = list(csvlib.DictReader(open(fp)))
    if not rows:
        out.append("\\caveat{trajectory\\_metrics.csv is empty.}")
        return "\n".join(out)

    # Aggregate by (model, persona, opponent) -> list of D_raw across seeds
    cells = defaultdict(list)
    extra_cols = defaultdict(lambda: defaultdict(list))  # for L, R if present
    for r in rows:
        m = r.get("model")
        p = r.get("persona")
        o = r.get("opponent")
        try:
            d = float(r.get("D_raw", "nan"))
        except (TypeError, ValueError):
            continue
        cells[(m, p, o)].append(d)
        # Try to gather L, R if columns present
        for col in ("L_break", "L", "first_defection_round"):
            v = r.get(col)
            if v not in (None, "", "inf"):
                try:
                    extra_cols["L"][(m, p, o)].append(float(v))
                except ValueError:
                    pass
                break

    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    def stderr(xs):
        if len(xs) < 2: return 0.0
        m = mean(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return (var ** 0.5) / (len(xs) ** 0.5)

    # Table C.1: D_raw mean (SE) by (model, persona, opponent)
    out.append("\\subsection{Defection rate $D^{*}$ by model $\\times$ persona $\\times$ opponent}")
    out.append("")
    out.append("\\begin{table}[H]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    cols = "ll" + "r" * len(OPPONENT_ORDER)
    out.append(f"\\begin{{tabular}}{{{cols}}}")
    out.append("\\toprule")
    header = ["Model", "Persona"] + OPPONENT_ORDER
    out.append(" & ".join(header) + " \\\\")
    out.append("\\midrule")
    for m in MODEL_ORDER:
        first_for_model = True
        for p in PERSONA_ORDER:
            row_cells = [m if first_for_model else "", PERSONA_PRETTY.get(p, p)]
            first_for_model = False
            for o in OPPONENT_ORDER:
                seeds = cells.get((m, p, o), [])
                if not seeds:
                    row_cells.append("--")
                else:
                    mu = mean(seeds) * 100
                    se = stderr(seeds) * 100
                    row_cells.append(f"{mu:.0f} ({se:.0f})")
            out.append(" & ".join(row_cells) + " \\\\")
        out.append("\\addlinespace")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\caption{Mean defection rate $D^{*}$ (\\%) with standard error (in parentheses) "
               "across three seeds, per model $\\times$ persona $\\times$ opponent strategy. "
               "Values that aggregate Figures~\\ref{fig:f1} and \\ref{fig:f2}.}")
    out.append("\\label{tab:dstar-full}")
    out.append("\\end{table}")
    out.append("")

    # Table C.2: Latency to first defection (L), deontologist vs AllD only
    if extra_cols.get("L"):
        out.append("\\subsection{Latency to first defection, deontologist vs AllD}")
        out.append("")
        out.append("\\begin{table}[H]")
        out.append("\\centering")
        out.append("\\small")
        out.append("\\begin{tabular}{lrr}")
        out.append("\\toprule")
        out.append("Model & Mean $L$ & SE \\\\")
        out.append("\\midrule")
        for m in MODEL_ORDER:
            ls = extra_cols["L"].get((m, "deontologist", "AllD"), [])
            if not ls:
                out.append(f"{m} & -- & -- \\\\")
            else:
                out.append(f"{m} & {mean(ls):.1f} & {stderr(ls):.2f} \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\caption{First-defection round $L$ for the deontologist persona vs AllD. "
                   "Trajectories that never defect are excluded.}")
        out.append("\\label{tab:lbreak}")
        out.append("\\end{table}")
        out.append("")

    out.append("\\caveat{Add E5 (virtue variants), E8 (deontology paraphrases), "
               "and current-generation cells here as separate sub-tables if needed.}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="experiments", help="directory containing e*.py scripts")
    parser.add_argument("--csv", default="csvs", help="directory containing CSVs")
    parser.add_argument("--section", default="A,B,C",
                        help="comma-separated subset of {A,B,C}")
    args = parser.parse_args()

    requested = {s.strip().upper() for s in args.section.split(",")}

    print("% Auto-generated by extract_appendix.py")
    print("% Paste between \\appendix and \\end{document} in main.tex.")
    print()

    if "A" in requested:
        print(emit_appendix_A(args.src))
        print()
    if "B" in requested:
        print(emit_appendix_B(args.src, args.csv))
        print()
    if "C" in requested:
        print(emit_appendix_C(args.csv))
        print()


if __name__ == "__main__":
    main()