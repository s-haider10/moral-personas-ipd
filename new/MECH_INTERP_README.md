# Mechanistic Interpretability Pipeline

Three scripts (M1, M2, M3) plus an orchestrator to causally test the claim:

> Prompt operationalization causally controls behavior; this corresponds to
> a localized residual-stream direction.

## What each script does

| Script | Purpose | Hardware | Time |
|---|---|---|---|
| `m1_open_model_replicate.py` | Reproduce E5 (integrity vs phronesis) on Llama-3.1-8B-Instruct. Mandatory kill-switch: if effect doesn't transfer, stop. | 1 × A6000 | ~25 min |
| `m2_activation_probe.py` | Cache residual-stream activations at every layer at the prompt-end token for 80 balanced prompts per persona. Linear probe + difference-of-means direction `v_layer`. | 1 × A6000 | ~10 min |
| `m3_intervention.py` | (a) **Steering:** add α·v at the best layer at every forward pass and sweep α. (b) **Patching:** swap the integrity-prompted activation into the phronesis-prompted run at the prompt-end token. | 1 × A6000 | ~40 min |

Total ~75 min on a single A6000. With 4 GPUs, you can run M1/M2/M3a/M3b in
parallel (set `CUDA_VISIBLE_DEVICES=0`, `=1`, etc. per shell).

## Hypotheses

All four are pre-registered before running anything:

| Hypothesis | Statement | Threshold |
|---|---|---|
| **M1.1** | Llama-3.1-8B reproduces the E5 effect | \|D(phronesis) − D(integrity)\| ≥ 0.20 |
| **M2.1** | Some layer L\* exists where integrity and phronesis are linearly separable | LOMO AUC ≥ 0.85 at the best layer |
| **M3.1** | Steering with α along v\_L\* causally shifts defection | D(phronesis, α=−3) − D(phronesis, α=+3) ≥ 0.10, monotone across α |
| **M3.2** | Patching integrity → phronesis at L\* reduces phronesis defection | D(src=int, tgt=phr) < D(phr baseline) by ≥ 0.10 |

If **M1.1 fails** the pipeline stops — interp claims about Llama don't transfer
to the closed-model results in the paper, but we still have a publishable
negative result.

If **M1.1 passes** but **M2.1 fails**, the operationalization is encoded
non-linearly. Report the failure honestly; this is itself a finding.

If **M2.1 passes but M3.1/M3.2 fail**, the linear probe found a correlation
that wasn't causal. Report this too; it's evidence on the limits of
linear-probing-as-causal-evidence.

## Running

```bash
# Set HF token if Llama-3.1 requires gated access
export HF_TOKEN=hf_...

# Full pipeline
bash run_mech_interp.sh

# Or step by step
python m1_open_model_replicate.py --grid --analyze
python m2_activation_probe.py --prepare-prompts --cache --analyze
python m3_intervention.py --steering --alphas "-3,-1,0,1,3" --seeds 3
python m3_intervention.py --patching --seeds 3
python m3_intervention.py --analyze
```

## Choosing the model

Default is `meta-llama/Meta-Llama-3.1-8B-Instruct`. Alternatives that should
work without code changes (all have `model.model.layers` structure):

- `meta-llama/Llama-3.1-8B-Instruct` (same as default; newer alias)
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Larger models (Llama-3.1-70B-Instruct in 4-bit):

```bash
python m1_open_model_replicate.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --grid --analyze
```

Will need to add `BitsAndBytesConfig(load_in_4bit=True)` in the loader. The
steering/patching hooks all work identically.

## Sanity checks before running

1. Confirm GPU memory: `nvidia-smi`. Llama-3.1-8B in fp16 needs ~16 GB.
2. Confirm `transformers >= 4.45` so that the chat template works for Llama 3.1.
3. Confirm `sklearn`, `torch`, `matplotlib` are installed.
4. Confirm `e2_cross_model.py` and `e13_pressure.py` are in the same directory
   (M1/M2/M3 import from them).

## What each output is for in the AIES paper

| File | Used in paper as |
|---|---|
| `csvs/M1_trajectory_metrics.csv` | Section "open-model replication"; demonstrates the effect isn't an API-only artifact |
| `figures/fig_m2_layer_sweep.png` | Section "operationalization has a localized neural correlate"; shows the layer where the contrast concentrates |
| `csvs/M2_probe_accuracies.csv` | Appendix; supplements the figure with numbers |
| `figures/fig_m3_steering.png` | Section "causal evidence: steering"; the dose-response curve is the headline figure |
| `figures/fig_m3_patching.png` | Section "causal evidence: patching"; cross-contrast bars |
| `csvs/M3_steering.csv`, `csvs/M3_patching.csv` | Appendix per-cell numbers |
