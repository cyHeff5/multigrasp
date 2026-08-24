# Laborsession 1 — Messungen (Stand 2026-08-24, Ziel: ~90 Min)

Zweck: alle Messwerte holen, die das Neutraining heute Abend braucht, und den
ContactDetector zum ersten Mal end-to-end auf Hardware testen. **Reihenfolge
einhalten** — jeder Block baut auf dem vorigen auf. Alles läuft auf dem
Windows-Laptop im Repo-Ordner (`python -m …`), Hand frei montiert, nichts im
Arbeitsraum.

## Vorher (zuhause, 5 Min)

- Repo auf den Laptop syncen — **Stand von heute!** Neu/geändert seit dem
  letzten Sync: `eval/maestro_limit_probe.py`, `hardware/input_recalibrate.py`,
  `eval/qsweep_check.py`, `assets/stroke_angle_curves.yaml`, `sim/hand.py`,
  `configs/*.yaml`, diese Datei.
- Die Referenzbilder `artifacts/analysis/qsweep/sim_q*.png` **und**
  `sim_q*_nonlin.png` aufs Handy (oder Laptop-Bildschirm reicht).
- Laptop-Python braucht: pyserial, numpy, pyyaml (wie bisher).

## Block A — Maestro-Limit-Probe (10 Min) — ZUERST, alles andere hängt daran

```bash
python -m eval.maestro_limit_probe --port COM4          # ~5 Min
```

- Druckt je Kanal den effektiv umgesetzten Pulsbereich und vergleicht mit den
  Werkssettings. **Vorhersage aus der archivierten Werksdatei:** falls das
  Board nie umkonfiguriert wurde, clippt ch10 (servo8) bei 4544 und ch11
  (servo9) bei 5056 — dann kann servo9 real nur bis q≈0.755 schließen.
- Bei geflaggten Kanälen einmal fein nachmessen: `--step 100` (Auflösung der
  Erkennung = Schrittweite).
- Danach `artifacts/calibration/servo_limits_suggested.yaml` anschauen und,
  wenn plausibel, als `artifacts/calibration/servo_limits.yaml` **kopieren**
  (wird ab dann von allem automatisch geladen).
- Nebenbefund prüfen: ist auf servo6/8 das effektive **Max** deutlich unter
  7700, erklärt das das „Öffnen bleibt bei q≈0.19 hängen" vom 08.07. Notieren.

## Block B — Input-Kalibrierung neu (10 Min)

```bash
python -m hardware.input_recalibrate --port COM4        # ~6 Min
```

- Ersetzt `hardware/joint_input_calibration.json` (Backup wird automatisch
  angelegt). Grund: servo9 hat −0.12 Gain-Fehler, die alte Kalibrierung ist vom
  Februar.
- Erfolgskriterium: **R² ≥ 0.99** je Joint. Deutlich darunter → Kanal notieren,
  nicht reparieren, weiter.
- Ab jetzt sind ALLE alten Baselines ungültig (neue q-Skala).

## Block C — Dynamik-Messungen (10 Min)

```bash
python -m eval.pause_ramp_check --config configs/precision.yaml --port COM4   # ~3 Min
python -m eval.baseline_calibration --config configs/precision.yaml --port COM4
python -m eval.baseline_calibration --config configs/power.yaml     --port COM4
```

- pause_ramp_check misst Wiederanlauf-Totzone und Kollaps-Zeit → die Ausgabe
  ersetzt heute Abend `restart_steps: 12` und die Restart-Totzone des
  Sim-Servo-Modells. Ausgabe-Dateien mitnehmen.
- Baselines: am Ende druckt jedes Skript effektive Thresholds je Joint.
  Normal: **0.012–0.02**. Deutlich drüber → einmal wiederholen.

## Block D — Foto-Sweep (10 Min) — entscheidet die Sim-Geometrie fürs Training

```bash
python -m eval.qsweep_check --config configs/power.yaml --port COM4 --settle 8
```

- Hält jeden q-Punkt (0 / 0.25 / 0.5 / 0.75 / 1.0) ~8 s: **an jedem Punkt ein
  Foto** aus der Referenz-Perspektive (Seitenansicht wie auf den sim-Bildern,
  Daumen rechts, Kamera auf Handhöhe).
- Der gedruckte Restfehler ist die Gegenprobe zu Block A+B: nach frischen
  Limits + Kalibrierung muss er ≈ 0 sein.
- Auswertung (vor Ort in 2 Min): Fotos bei **0.25 / 0.5 / 0.75** neben
  `sim_q*.png` (linear) und `sim_q*_nonlin.png` (Hersteller-Kennlinie) halten.
  Die Endpunkte 0.0/1.0 sehen in beiden Serien gleich aus — die Mitte
  unterscheidet: nichtlinear = Finger bei 0.5 noch deutlich gestreckter.
  **Welche Serie passt → notieren.** Das schaltet heute Abend
  `kinematics.stroke_angle_map` im Neutraining an oder aus.

## Block E — Detektor-End-to-End (30–45 Min)

Alte (eingefrorene) Precision-Policy mit dem frisch kalibrierten Detektor auf
den flachen Problemteilen — testet die Kontaktkette, nicht die Policy:

```bash
python -m eval.policy_runner --config configs/precision.yaml \
    --model artifacts/models/precision/seed_0_dr/best/best_model --port COM4
```

- Teil 5 und Teil 9, je 5 Trials. Nach den ersten 3 Trials auf `n_steps`
  schauen (Go/No-Go, wie REAL_EVAL_MORGEN.md):
  - **38–45** → Frühtrigger ist zurück: `startup_mask_q` in beiden Configs auf
    0.22, Baseline aus Block C neu fahren, nochmal.
  - **~120–200** → Detektor arbeitet. Weiter, gern mehr Teile.
  - **300 / nie** → Bit kippt nie: Baseline-Ausgabe prüfen, dokumentieren,
    weiter (kein Blocker für heute Abend).
- Ziel-Erkenntnis: sind Fehlmodus A (Trigger bei q≈0.2) und B (Anschlag) weg?

## Block F — Wiegen (5 Min, nebenbei)

Alle 14 Benchmarkteile wiegen und eintragen in
`artifacts/analysis/benchmark_masses.yaml` (neu anlegen), Format:

```yaml
1: 0.152   # part_id: masse_kg
2: 0.087
```

## Mitbringen (Sync zurück auf den Server)

`artifacts/calibration/` (servo_limits.yaml + neue Baselines),
`hardware/joint_input_calibration.json`, pause_ramp-Ausgaben +
`maestro_limit_probe_*.csv` in `artifacts/analysis/`, die
`benchmark_masses.yaml`, alle Fotos, Trial-CSVs aus Block E — und die zwei
Entscheidungen: **linear oder nichtlinear** (Block D) und **Frühtrigger
weg ja/nein** (Block E).

## Abbruchkriterien

- Block A+B zusammen > 30 Min oder Limit-Probe zeigt tote Kanäle → Session
  abbrechen, Befund melden. Ohne A+B ist der Rest wertlos.
- Block E scheitert → trotzdem C, D, F fertig machen. Das Neutraining heute
  Abend braucht C und D, nicht E.
