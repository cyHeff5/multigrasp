#!/usr/bin/env bash
# Nachtlauf 25./26.08.: je EIN Seed precision + power mit Probe-Confirm-Trigger
# (trigger_probe, RL_VERIFICATION 2e) auf den *_neutraining-Configs
# (Servo-Modell + contact_spring + korrigierte Fingerkuppen-Kollision 9.10).
# Diese beiden Policies sind fuer den Realtest mit der echten Hand bestimmt;
# weitere Seeds folgen danach auf pc-lw.
# Wartet zuerst, bis der laufende fixedtips-Lauf + dessen Eval fertig sind.
# Danach je Grifftyp: Training, Benchmark-Eval der Policy und dieselbe
# Benchmark-Eval fuer die always_close-Baseline (die fehlende Vergleichszeile).
# Aufruf:  nohup bash training/night_probetrig.sh > /tmp/night_probetrig.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
P=../sim2real/.venv/bin/python

while kill -0 1471863 2>/dev/null || kill -0 1471865 2>/dev/null; do
  sleep 120
done
echo "=== $(date '+%F %T') Vorlauf fertig, Nachtlauf startet ==="

for CFG in precision power; do
  echo "=== $(date '+%F %T') TRAINING ${CFG} seed 0 (probetrig) ==="
  $P -m training.train --config "configs/${CFG}_neutraining.yaml" \
      --seed 0 --tag probetrig --lr-schedule linear --n-envs 3
  echo "=== $(date '+%F %T') TRAINING ${CFG} exit=$? ==="

  M="artifacts/models/${CFG}/seed_0_probetrig/best/best_model"
  echo "=== $(date '+%F %T') EVAL ${CFG} Policy ==="
  $P -m eval.eval_sim --config "configs/${CFG}_neutraining.yaml" --model "$M" \
      --mode benchmark --trials 20 \
      --output "artifacts/eval_results/probetrig_${CFG}_policy"
  echo "=== $(date '+%F %T') EVAL ${CFG} always_close ==="
  $P -m eval.eval_sim --config "configs/${CFG}_neutraining.yaml" --model always_close \
      --mode benchmark --trials 20 \
      --output "artifacts/eval_results/probetrig_${CFG}_baseline"
done
echo "=== $(date '+%F %T') NACHTLAUF KOMPLETT ==="
ls -la artifacts/models/*/seed_0_probetrig/best/best_model.zip
