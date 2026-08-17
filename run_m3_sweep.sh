#!/bin/bash
set -uo pipefail
cd /home/haider/moral-personas-ipd/new
export HF_TOKEN=$(grep HF_TOKEN /home/haider/moral-personas-ipd/.env | cut -d= -f2)
export CUDA_VISIBLE_DEVICES=2
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

for L in 16 8 24; do
  echo "============================================================"
  echo "M3 sweep: layer $L  (steering)"
  echo "============================================================"
  ../.venv/bin/python m3_intervention.py --model "$MODEL" \
    --layer "$L" --steering --alphas="-3,-1,0,1,3" --seeds 3
  echo "M3 sweep: layer $L  (patching)"
  ../.venv/bin/python m3_intervention.py --model "$MODEL" \
    --layer "$L" --patching --seeds 3
done

echo "============================================================"
echo "M3 sweep: combined analyze (all layers)"
echo "============================================================"
../.venv/bin/python m3_intervention.py --model "$MODEL" --analyze
echo "M3 SWEEP DONE"
