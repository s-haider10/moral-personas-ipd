#!/bin/bash
# Mech interp pipeline: M1 -> M2 -> M3
# Designed for 4xA6000 24GB. Runs sequentially on one GPU; parallelize across
# alphas/seeds by hand if you want to use all four.
#
# Total wall-clock estimate (Llama-3.1-8B):
#   M1: 5 seeds x 3 personas x 20 rounds x ~5s/round = ~25 min
#   M2: ~200 prompts cached + probes = ~10 min
#   M3 steering: 5 alphas x 2 personas x 3 seeds x 20 rounds = ~25 min
#   M3 patching: 3 contrasts x 3 seeds x 20 rounds x 2 fwd = ~15 min
# Total: ~75 min for the complete pipeline.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
SEEDS="${SEEDS:-5}"

echo "============================================================"
echo "M1: Behavioral replication on $MODEL"
echo "============================================================"
python m1_open_model_replicate.py --model "$MODEL" --seeds "$SEEDS" --grid --analyze

echo
echo "============================================================"
echo "M2: Activation probe and layer sweep"
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
echo "Done. Inspect:"
echo "  csvs/M1_trajectory_metrics.csv"
echo "  csvs/M2_probe_accuracies.csv"
echo "  csvs/M3_steering.csv"
echo "  csvs/M3_patching.csv"
echo "  figures/fig_m2_layer_sweep.png"
echo "  figures/fig_m3_steering.png"
echo "  figures/fig_m3_patching.png"
echo "============================================================"
