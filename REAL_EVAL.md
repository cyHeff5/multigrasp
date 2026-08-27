# Real-Eval Runbook

Ablauf fuer die Realevaluierung: Kalibrierung (~20 Min) + gefilmte Griff-Trials.
Zwei Laptops: **Sawyer-Laptop** (drive_pregrasp) und **AR10-Laptop** (alles hier).

**Ohne Gamepad** laeuft `drive_pregrasp` ueber die Tastatur — SPACE=A,
BACKSPACE=B, X=X, Y=Y, A=LB, D=RB, ESC=Menu, Pfeiltasten=Stick. Fuer den
Ablauf heisst das: D/A blaettern durch die Greifpunkte, SPACE bestaetigt,
X gedrueckt halten faehrt an, Y ist die Feinjustierung, BACKSPACE waehlt neu,
ESC beendet. Die Tasten gehen an das PyBullet-Fenster; laeuft keins, macht
`hardware/gamepad.py` ein eigenes kleines Fenster auf, das den Fokus braucht.

## 0. Setup-Check (AR10-Laptop, 5 Min)

1. Hand an COM-Port, Port-Namen merken (Geraetemanager, z.B. `COM4`).
2. **Maestro Control Center oeffnen** und pro Kanal Min/Max ablesen.
   In `artifacts/calibration/servo_limits.yaml` eintragen (Anleitung steht in
   der Datei — Reihenfolge ist die Sim-Joint-Reihenfolge, nicht die Kanalnummer!).
   Wenn alle Kanaele 4200/7700 sind: nichts zu tun.

## 1. Kalibrierung (AR10-Laptop, ~15 Min)

```bash
# 1-NEU) q_delta-BASELINE fuer den ContactDetector (PFLICHT, ~2 Min, ohne Objekt):
python -m eval.baseline_calibration --config configs/precision.yaml --port COM4
#   10 Freilauf-CLOSE-Zyklen -> artifacts/calibration/qdelta_baseline.yaml.
#   VOR JEDER SESSION NEU FAHREN (Drift ueber Tage ungetestet; der Runner
#   warnt ab 12 h Alter und bricht ohne Datei hart ab).
#   Hintergrund: artifacts/analysis/SENSOR_ANALYSIS_FINDINGS.md — Kontakt wird
#   als Residuum gegen die Freilauf-Baseline erkannt (~0.012 statt 0.05).
#   Gilt fuer precision (contact_detector.enabled: true). Power laeuft weiter
#   auf dem alten Threshold-Pfad -> dafuer bleiben 1a-1c Pflicht. Fuer precision
#   sind 1a-1c nur noetig, wenn der Detektor abgeschaltet wird (enabled: false).

# 1a) OHNE Objekt — Rauschboden (Untergrenze, mean + 3*sigma):
python -m eval.calibration --config configs/precision.yaml --port COM4 --phase free

# 1b) MIT Objekt (Wuerfel ~5cm aufs Podest, Sawyer in Precision-Pregrasp):
python -m eval.calibration --config configs/precision.yaml --port COM4 --phase parallel
#   Phase A laeuft allein (stoppt beim Sim-Kontakt im PyBullet-Fenster).
#   Phase B: '+5' = 5 Schritte zu, '-2' = 2 auf, 'ok' sobald der Finger das
#            Objekt SICHTBAR beruehrt. Du bist der Kontaktsensor.
#   Phase C laeuft allein, druckt pro Finger den real_threshold.
#   Ergebnis: artifacts/calibration/real_threshold.yaml (pro Finger).

# 1c) Power-Transfer pruefen (Objekt liegt, Sawyer in Power-Pregrasp):
python -m eval.calibration --config configs/power.yaml --port COM4 --phase power-check
#   Jeder Finger soll den Threshold mit Faktor >= 2 ueberschreiten.
```

Falls 1b fehlschlaegt ("unter Rauschboden"): Pose pruefen, Phase B sorgfaeltiger,
Sweep wiederholen. Falls 1c fehlschlaegt: `--phase parallel` mit der power-Config
in Power-Pose wiederholen (nur Zeige+Mittel ausrichten, Rest uebernimmt Fallback).

## 2. Eval-Loop (gefilmt, kein Ergebnis-Tippen)

```bash
python -m eval.policy_runner --config configs/precision.yaml \
    --model artifacts/models/precision/seed_0_dr/best/best_model --port COM4

python -m eval.policy_runner --config configs/power.yaml \
    --model artifacts/models/power/seed_0_dr/best/best_model --port COM4
```

Pro Trial (10x pro Greifpunkt):
1. Sawyer-Laptop: `drive_pregrasp` -> Hand steht, ggf. nachjustieren.
2. AR10-Laptop: Label eingeben (nur beim Objektwechsel) + **Enter** -> Griff
   laeuft, stoppt selbst nach dem Trigger ("Finger eingefroren").
3. Sawyer hebt (Lift-Test fuers Video), dann **wieder absenken**.
4. AR10-Laptop: **Enter** -> Hand oeffnet. Objekt neu platzieren, weiter mit 1.
5. `q` beendet; Timestamps-Log liegt in `artifacts/eval_results/real_*.csv`
   (Zuordnung Trial <-> Video ueber die Uhrzeiten).

Achtung Duty-Cycle (20% laut Firgelli-Datenblatt): zwischen Trigger-Stopp und
Lift nicht lange warten — die Aktuatoren stehen unter Last.

## Woher die Zahlen kommen (Kurzbegruendung)

- **ContactDetector (precision)**: q_delta ist kein Rauschen, sondern ein
  reproduzierbarer Tracking-Fehler (Zyklus-Std ~0.003). Kontakt = Residuum
  gegen die Session-Baseline > max(0.012, 4*sigma) fuer 3 Steps (+ CUSUM fuer
  langsame Blockierung). Offline validiert: 0 False Positives auf 9 Freilauf-
  Zyklen, Kugel-Trigger bei q~0.2-0.8 statt ~0.87. Details + Grenzen:
  artifacts/analysis/SENSOR_ANALYSIS_FINDINGS.md.
- **step_dt = substeps/sim_hz (20.8 ms)**: Training, Kalibrierung und Deployment
  laufen auf derselben Kontrollrate; der Runner nutzt Absolutzeit-Scheduling und
  warnt, wenn die serielle I/O langsamer ist.
- **real_threshold (pro Finger)**: am Sim-Nominalwert verankert — das reale
  Kontakt-Bit kippt am selben physischen Ereignis wie im Training (Phase C),
  validiert gegen den Rauschboden (Phase free).
- **threshold_range [0.02, 0.08] im Training**: Untergrenze = 2x gemessener
  Sim-Freilauf-q_delta (0.0095); die Policy haengt dadurch nicht am exakten
  Kalibrierwert.
- **substeps_range [4, 7] im Training**: Robustheit gegen die real erreichte
  Loop-Rate (34-60 Hz).
- **Power-Transfer**: Precision hat die weichste Kontaktkette (Silikonspitze
  gegen gefederten Daumen); der harte Power-Stall gegen die Handflaeche trennt
  Kontakt vom Rauschen erst recht (power-check bestaetigt das empirisch).
