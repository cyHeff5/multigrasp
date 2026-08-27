#!/usr/bin/env bash
# Pilot-Nacht (2026-08-24): je ein voller Lauf precision + power mit dem
# uebergebenen Config-Suffix, danach best-Modelle force-adden und pushen,
# damit der Labor-Laptop sie morgen per git pull bekommt.
# Aufruf: bash training/pilot_night.sh <config_suffix>   # z.B. _neutraining
set -u
cd "$(dirname "$0")/.."
P=../sim2real/.venv/bin/python
SUF="${1:-_neutraining}"

for CFG in precision power; do
  echo "=== $(date '+%F %T') Pilot-Training ${CFG}${SUF} ==="
  $P -m training.train --config "configs/${CFG}${SUF}.yaml" \
      --seed 0 --tag pilot --lr-schedule linear --n-envs 3
  echo "=== $(date '+%F %T') ${CFG} exit=$? ==="
done

git add -f \
  artifacts/models/precision/seed_0_pilot/best/best_model.zip \
  artifacts/models/precision/seed_0_pilot/run_meta.yaml \
  artifacts/models/power/seed_0_pilot/best/best_model.zip \
  artifacts/models/power/seed_0_pilot/run_meta.yaml 2>&1
git commit -m "pilot-policies fuer labortest (seed_0_pilot, servo_model)

Nachtlauf 24./25.08., Configs *${SUF}.yaml, je 1 Seed. Nur fuer den
Hardware-Pilottest in Laborsession 1; die finalen 3-Seed-Laeufe folgen
mit den Labor-Messwerten.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  && git push origin main 2>&1
echo "=== $(date '+%F %T') Pilot-Nacht fertig ==="
