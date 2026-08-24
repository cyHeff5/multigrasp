# multigrasp (aktueller Arbeitsstand)

## Umgebung
- **Kein eigenes venv auf dem Server.** Sim/Analyse laufen mit
  `../sim2real/.venv/bin/python` (hat pybullet, sb3, torch, numpy, pandas, yaml).
- **Hardware läuft NICHT hier**, sondern auf einem Windows-Laptop im Labor
  (COM-Ports). Alles mit `--port` ist dort; ohne `--port` = Mock-Modus.

## Schnelltests (alle ohne Hardware)
```bash
P=../sim2real/.venv/bin/python
$P -m eval.test_detector_offline          # Detektor gegen die 08.07.-CSVs (muss BESTANDEN sein)
echo | $P -m eval.baseline_calibration --config configs/precision.yaml   # Mock
$P -m eval.pause_ramp_check --config configs/precision.yaml --cycles 1   # Mock
```

## Architektur-Kurzfassung
- Kontakt-Bits real: `hardware/contact_detector.py` (Baseline-korrigiert,
  Raten-Skalierung, Statik/Dynamik getrennt, Anlauf-Gate, Restart-Maske,
  CUSUM). Baseline pro Session: `eval/baseline_calibration.py` (liest im
  Runner-Timing: senden → Slot abwarten → lesen).
- Sim-Training: `sim/env.py` + `sim/hand.py`. `servo_model.enabled` (Config)
  aktiviert das gemessene AR10-Servo-Verhalten + denselben ContactDetector in
  der Observation (synthetische Baseline pro Episode, nativer PyBullet-Lag
  wird einmal pro Env im Freilauf vermessen). Default false =
  Checkpoint-kompatibler Alt-Pfad.
- `action.mode: rate_probe` = neuer Aktionsraum {Probe, langsam, schnell} fürs
  Neutraining (Probe nutzt das rauschfreie statische Regime der echten Hand).
- Messdaten + Herleitung ALLER Konstanten:
  `artifacts/analysis/SENSOR_ANALYSIS_FINDINGS.md` (§8 = Stand 2026-08-24).
  Wissenschaftliche Schwächen: `artifacts/analysis/RL_VERIFICATION.md`.
- Nächster Laborslot: **`LABOR_SESSION1.md`** (Messungen + Detektor-End-to-End;
  ersetzt für Session 1 das ältere `REAL_EVAL_MORGEN.md`, das bleibt Referenz
  für die Real-Benchmark-Eval = Session 2).
- AR10-Herstellerdoku (Datenblatt, Maestro-Werkssettings, Vendor-Software,
  zitierfähige Fakten): `~/projects/multigrasp/ar10-doku/README.md`.
  Wichtig: Werks-Maestro-Minima ch10=4544/ch11=5056 → möglicher stiller
  Clip von servo8/9 beim Schließen — messen mit `eval/maestro_limit_probe.py`.
- Neu (2026-08-24): `eval/maestro_limit_probe.py` (empirische Maestro-Limits,
  Zwei-Knickpunkt-Fit), `hardware/input_recalibrate.py` (Input-Kalibrierung
  neu, ersetzt joint_input_calibration.json mit Backup),
  `assets/stroke_angle_curves.yaml` + `kinematics.stroke_angle_map` in den
  Configs (Hub→Winkel-Nichtlinearität, default aus; Foto-Sweep entscheidet),
  `configs/{precision,power}_neutraining.yaml` (servo_model + rate_probe),
  `load_policy("always_close")` = Skript-Baseline für eval_sim,
  run_meta.yaml je Trainingslauf, `--n-envs` in training/train.py.
  Benchmark-Massen: `artifacts/analysis/benchmark_masses.yaml` (part_id→kg)
  wird, wenn vorhanden, in eval_sim/URDF-Objekte übernommen.
- Trainings-Durchsatz Server: ~260 fps (3 Envs, servo_model an) → 2M ≈ 2¼ h.
  tqdm+rich sind seit 2026-08-24 im sim2real-venv (progress_bar).

## Stolpersteine
- `hardware/contact_detector.py` darf sim.hand nur lazy importieren
  (Zirkularität mit sim/env.py).
- Alte Baselines (ohne `read_timing: pre_next_send` in meta) sind ~+0.005 zu
  hoch — neu kalibrieren, Runner warnt.
- Die CSVs vom 08.07. sind mit dem alten Lese-Timing aufgenommen (konsistent
  untereinander, absolut ~+0.005 hoch).
