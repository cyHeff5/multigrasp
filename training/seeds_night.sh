#!/usr/bin/env bash
# Drei-Seed-Nachtlauf (2026-08-25): je 3 Seeds precision + power mit den
# *_neutraining-Configs (servo_model + contact_spring + rate_probe,
# trigger_mode auto). Adressiert RL_VERIFICATION.md 2.1 — Ein-Seed-Ergebnisse
# gelten seit Henderson et al. (2018) nicht als Evidenz.
# lr-schedule linear: ppo.yaml dokumentiert das gegen Late-Stage-Collapse, und
# der 300k-Pilot vom 25.08. abends zeigte genau den (7.05 -> 5.58 ep_rew).
# Seriell, weil der Server 4 Kerne hat und ein Lauf mit 3 Envs sie fast fuellt.
# Aufruf:  nohup bash training/seeds_night.sh > /tmp/seeds_night.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
P=../sim2real/.venv/bin/python

for CFG in precision power; do
  for SEED in 0 1 2; do
    echo "=== $(date '+%F %T') START ${CFG} seed ${SEED} ==="
    $P -m training.train --config "configs/${CFG}_neutraining.yaml" \
        --seed "$SEED" --tag seeds --lr-schedule linear --n-envs 3
    echo "=== $(date '+%F %T') DONE ${CFG} seed ${SEED} exit=$? ==="
  done
done
echo "=== $(date '+%F %T') ALLE LAEUFE FERTIG ==="
ls -la artifacts/models/*/seed_*_seeds/best/best_model.zip
