# Moral Personas IPD

Raw trajectories (`results/`) and aggregated metrics
(`csvs/`) are released separately on Hugging Face Datasets.

## Layout

```
.
├── figures.py           # unified CLI to build every figure
├── fig_style.py         # shared matplotlib style
├── experiments/         # one script per experiment
├── figures/             # rendered PNGs
├── csvs/                # aggregated metrics (HF, gitignored)
└── results/             # raw JSONL trajectories (HF, gitignored)
```

## Setup

```bash
uv pip install matplotlib numpy pandas scipy scikit-learn \
                openai google-genai
```

API keys (set whichever providers you need):

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

Tested with Python 3.13.

## Running experiments

Each script is self-contained and writes to `results/E*/`. They share
primitives (payoffs, prompt template, client wrappers) defined in
`experiments/e2_cross_model.py`.

```bash
# E2 — cross-model sweep, deontologist vs AllD
uv run python experiments/e2_cross_model.py --suite e2_default --skip-existing

# E4 — persona × opponent grid
uv run python experiments/e4_grid.py --suite e4_default --skip-existing

# E5 — virtue ethics variants (parallel)
uv run python experiments/e5_virtue_variants.py --suite e5_default --skip-existing

# E8 — deontologist paraphrases (parallel)
uv run python experiments/e8_paraphrase.py --suite e8_default --skip-existing
```

Append `--help` to any script for its full options. Each experiment has a
matching `eN_analyze.py` (where present) that prints summary tables from the
JSONL outputs.

## Building figures

```bash
uv run python figures.py all               # everything
uv run python figures.py summary           # paper headline figure
uv run python figures.py f1 f2 f3 f4 f5    # individual figures
uv run python figures.py e5                # E5 slope chart
```

Outputs land
in `figures/`. Reads from `csvs/` and `results/`.
