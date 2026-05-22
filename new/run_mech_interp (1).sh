#!/bin/bash
# Full mech interp pipeline: M1 -> M2 -> M3 (steering, patching) -> M4 -> M5.
#
# Runs sequentially on a single GPU. For parallelism across 4xA6000, see
# the "parallel" section at the bottom of this file (commented).
#
# Total wall-clock estimate (Llama-3.1-8B on 1xA6000):
#   M1 (5 seeds, 3 personas):           ~30 min
#   M2 (cache + probe + diff-of-means):  ~15 min
#   M3a steering (5 alphas x 2 personas x 3 seeds): ~30 min
#   M3b patching (3 contrasts x 3 seeds):           ~25 min
#   M4 (cache 5 personas x 5 opponents x 28 prompts + analyze): ~30 min
#   M5 (5 personas x 5 opponents x 10 seeds x 20 rounds x 2 fwd): ~3-4 hours
# Total: ~5-6 hours on one A6000.
#
# With 4xA6000 you can parallelize across models or across blocks (see below).

set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
SEEDS_M1="${SEEDS_M1:-5}"
SEEDS_M5="${SEEDS_M5:-10}"

echo "============================================================"
echo "MODEL = $MODEL"
echo "============================================================"

echo
echo "============================================================"
echo "M1: Behavioral replication"
echo "============================================================"
python m1_open_model_replicate.py --model "$MODEL" --seeds "$SEEDS_M1" --grid --analyze

echo
echo "============================================================"
echo "M2: Activation probe + layer sweep"
echo "============================================================"
python m2_activation_probe.py --model "$MODEL" --prepare-prompts --cache --analyze

echo
echo "============================================================"
echo "M3a: Steering"
echo "============================================================"
python m3_intervention.py --model "$MODEL" --steering --alphas "-3,-1,0,1,3" --seeds 3

echo
echo "============================================================"
echo "M3b: Activation patching"
echo "============================================================"
python m3_intervention.py --model "$MODEL" --patching --seeds 3

echo
echo "============================================================"
echo "M4: Persona-direction generalization"
echo "============================================================"
python m4_direction_generalization.py --model "$MODEL" --prepare-prompts --cache --analyze

echo
echo "============================================================"
echo "M5: Decision-emergence (logit lens at action token)"
echo "============================================================"
python m5_decision_emergence.py --model "$MODEL" --seeds "$SEEDS_M5" --grid --analyze

echo
echo "============================================================"
echo "DONE. Inspect:"
echo "  csvs/M1_trajectory_metrics.csv"
echo "  csvs/M2_probe_accuracies.csv"
echo "  csvs/M3_steering.csv  csvs/M3_patching.csv"
echo "  csvs/M4_within_model_transfer.csv  csvs/M4_persona_pair_similarity.csv"
echo "  csvs/M5_per_round.csv  csvs/M5_decision_layers.csv"
echo "  figures/fig_m2_layer_sweep.png"
echo "  figures/fig_m3_steering.png  figures/fig_m3_patching.png"
echo "  figures/fig_m4_cross_opponent.png  figures/fig_m4_persona_subspace.png"
echo "  figures/fig_m5_layer_decision.png  figures/fig_m5_mismatch_vs_dec_layer.png"
echo "  figures/fig_m5_dec_layer_vs_cot.png"
echo "============================================================"


# ---------------------------------------------------------------------
# PARALLEL EXECUTION ACROSS 4 GPUs (uncomment to use)
# ---------------------------------------------------------------------
#
# # Run M5 on three different models in parallel for cross-model M5.4
# CUDA_VISIBLE_DEVICES=0 python m5_decision_emergence.py \
#   --model meta-llama/Meta-Llama-3.1-8B-Instruct --seeds 10 --grid --analyze &
# CUDA_VISIBLE_DEVICES=1 python m5_decision_emergence.py \
#   --model Qwen/Qwen2.5-7B-Instruct --seeds 10 --grid --analyze &
# CUDA_VISIBLE_DEVICES=2 python m5_decision_emergence.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --seeds 10 --grid --analyze &
# wait
#
# # M4 cross-model transfer requires activations cached for both source and target
# CUDA_VISIBLE_DEVICES=3 python m4_direction_generalization.py \
#   --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --cross-model-target Qwen/Qwen2.5-7B-Instruct \
#   --analyze
