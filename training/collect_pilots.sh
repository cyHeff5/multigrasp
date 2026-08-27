#!/usr/bin/env bash
# Einsammel-Kette Pilot-Nacht (2026-08-24): wartet auf den done-Marker der
# PC-Trainings (training/train_night.bat auf pc-lw), kopiert die best-Modelle
# + run_meta per scp in dieses Repo, committet mit force-add und pusht —
# damit der Labor-Laptop morgen frueh alles per git pull bekommt.
set -u
cd "$(dirname "$0")/.."
PCROOT="C:/Users/leonp/projects/multigrasp"

echo "$(date '+%F %T') warte auf night_done.marker auf pc-lw ..."
until ssh -o BatchMode=yes -o ConnectTimeout=10 pc-lw \
      "if exist C:\\Users\\leonp\\projects\\multigrasp\\artifacts\\night_done.marker (echo DONE)" \
      2>/dev/null | grep -q DONE; do
  sleep 300
done
echo "$(date '+%F %T') Marker da — sammle Modelle ein."

added=0
for cfg in precision power; do
  for s in 0 1 2; do
    dir="artifacts/models/${cfg}/seed_${s}_pilot"
    mkdir -p "${dir}/best"
    if scp -o BatchMode=yes "pc-lw:${PCROOT}/artifacts/models/${cfg}/seed_${s}_pilot/best/best_model.zip" "${dir}/best/" \
       && scp -o BatchMode=yes "pc-lw:${PCROOT}/artifacts/models/${cfg}/seed_${s}_pilot/run_meta.yaml" "${dir}/"; then
      git add -f "${dir}/best/best_model.zip" "${dir}/run_meta.yaml"
      added=$((added+1))
      echo "  ${cfg} seed ${s}: eingesammelt"
    else
      echo "  ${cfg} seed ${s}: FEHLT (Lauf gescheitert?)"
    fi
  done
done

ssh -o BatchMode=yes pc-lw "type C:\\Users\\leonp\\projects\\multigrasp\\artifacts\\logs\\night_overview.log" 2>/dev/null

if [ "$added" -gt 0 ]; then
  git commit -m "pilot-policies fuer labortest (seed 0-2, servo_model + rate_probe)

Nachtlauf 24./25.08. auf pc-lw (train_night.bat), Configs *_neutraining.yaml,
${added}/6 Laeufen eingesammelt. Nur fuer den Hardware-Pilottest in
Laborsession 1; finale Laeufe folgen mit den Labor-Messwerten.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
  && git push origin main
  echo "$(date '+%F %T') fertig: ${added}/6 Modelle committet + gepusht."
else
  echo "$(date '+%F %T') NICHTS eingesammelt — Logs auf pc-lw pruefen."
fi
