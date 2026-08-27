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
  **Die Werkssettings-Vermutung ist widerlegt** (2026-08-25): das Board trägt
  keine reinen Werkslimits, die echten stehen gemessen in
  `artifacts/calibration/servo_limits.yaml` (§9.1).
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

## Stand nach Laborsession 1 (2026-08-25) — alles in SENSOR_ANALYSIS_FINDINGS.md §9
- **Maestro-Limits gemessen** → `artifacts/calibration/servo_limits.yaml`. Ohne
  die Datei ist die q-Skala in der MITTE um bis zu 0.30 verschoben (§9.2) —
  nicht an der Endlage, dort war immer alles einig.
- **Index-Bug in `hardware/ar10.py` gefixt** (§9.3): `_set_all_channel_targets`
  clippte kanalindizierte Targets gegen jointindizierte Limits. Latent, solange
  alle Limits gleich sind; scharf ab per-Kanal-Limits. Trifft nur ch10/ch11
  (= sim servo8/9).
- **Input-Kalibrierung neu**, R² 0.9966–1.0 auf allen zehn Joints. Baselines
  für precision + power neu gefahren (`qdelta_baseline*.yaml`).
- **`restart_steps` 12 → 15** in allen vier Configs. Gemessen, nicht geraten:
  Wiederanlauf-Totzone max 14 Steps, Peak 0.062 (§9.5). 12 war zu KLEIN.
- **`kinematics.stroke_angle_map` bleibt `null`** — Hersteller-Kennlinie per
  Foto-Sweep widerlegt (§9.6). `assets/stroke_angle_curves.yaml` ist Archiv.
- **Kollisionsgeometrie neu** (`rebuild_collision.py`, §9.0): Hand jetzt 191
  Shapes statt 48 und sauber; Benchmarkteile 4/10/11 weiterhin bis 9 mm
  aufgeblasen → vor Zitieren von Sim-Benchmarkzahlen nachbessern.
- **Offen und wichtig:** die echte Hand ist nachgiebig, die Sim starr (§9.8).
  Mit visueller Ground Truth nachgemessen (§9.9, `eval/probe_contact.py`):
  am sichtbaren Erstkontakt ist statisch UND dynamisch nichts messbar; das
  statische Probe-Residuum wächst erst ab ~0.25 q Überfahrweg verlässlich
  über residual_min (Feder, k ≈ 0.2–0.3), Bits kommen spät und FLACKERN im
  Stop-and-go (rasten nicht, anders als §9.7 vermutete). Die Sim braucht
  diese Feder im Servo-Modell, sonst lernt rate_probe einen Erstkontakt-
  Detektor, den es real nicht gibt. **Erledigt 2026-08-25 abends:**
  `servo_model.contact_spring` in `sim/hand.py` + beiden Neutraining-Configs
  (k 0.2–0.3, onset_q 0.10–0.25, Erstkontakt per `getContactPoints`).
- **Neu 2026-08-25 abends, nur in den `*_neutraining.yaml`:**
  `servo_model.contact_spring` (Kontaktnachgiebigkeit im Sensormodell, §9.9) und
  `trigger_mode: policy` (Stop-Aktion — die Policy schliesst den Griff selbst ab,
  Timeout ohne Lift-Test; Vorbedingungen `trigger_min_steps: 40` +
  `trigger_requires_contact: true` als Explorationsmaske). Die Skript-Baseline
  `always_close` traegt die alte Zustandsmaschine jetzt selbst und bleibt
  vergleichbar. Zahlen + Begruendung: RL_VERIFICATION.md §2d.
- **Fingerkuppen-Kollision gefixt (2026-08-25 abends, §9.10):** `fingertip1..4`
  hatten eine `<box>` statt eines Meshes und wurden vom §9.0-Rebuild nie erfasst
  (4.5 mm Geistermaterial an der Hauptkontaktflaeche). Jetzt VHACD-Huelle,
  0.22 mm. Dazu `physics.collision_margin_m` (default 0.0001) gegen Bullets
  1.00-mm-Mesh-Margin. Messstand: `eval/ghost_check.py`.
  **Jedes Training vor diesem Fix ist mit falscher Kuppengeometrie gelaufen.**
- **Probe-Confirm-Trigger (2026-08-25 nachts, RL_VERIFICATION §2e):**
  `trigger_probe` in beiden Neutraining-Configs — ein Finger-Bit startet eine
  25-Step-Halteprobe, bestaetigt wird ueber das statische Detektor-Residuum.
  Ersetzt die n=2-Maschine (feuert bei asymmetrischem Kontakt nie); naives n=1
  triggert auf fruehe Fehlbits (Replay der 25.08.-CSVs). Identisch in
  `sim/env.py::_probe_step` + `eval/policy_runner.py::run_real_episode`.
- **Block F (Benchmarkteile wiegen) ist NICHT erledigt** —
  `artifacts/analysis/benchmark_masses.yaml` fehlt weiterhin.

## Lernkurve lesen (ab 2026-08-26)
`ep_rew_mean` taugt nicht als Fortschrittsanzeige — der Episodenreward ist bimodal
(+10 vs −5), der Mittelwert streut um ±6. Die Grösse, an der man sieht, wann der
Griff sitzt, ist die Erfolgsrate:
```bash
P=../sim2real/.venv/bin/python
$P -m eval.plot_success_curve \
   artifacts/models/precision/seed_0_probetrig \
   artifacts/models/power/seed_0_probetrig \
   --labels "Precision (Tripod)" "Power (Medium Wrap)" \
   --out artifacts/eval_results/success_curve.png
```
Läufe ab dem `is_success`-Patch liefern `eval/success_rate` und
`rollout/success_rate` direkt in TensorBoard (letztere ein Punkt pro Rollout, also
zehnmal dichter als die 100k-Evals — dazuplotten mit `--tb <logdir>`). Für ältere
Läufe rechnet das Skript die Rate exakt aus den Einzel-Rewards zurück.
**Befund (`seed_0_probetrig`):** Precision 62 %→97 % in 500k Steps, danach
Plateau. Power startet bei 84 % und sackt über 2M langsam auf ~70 % ab.

**`eval_freq: 100000` ist zu grob für den Kurvenanfang** (26.08.): der erste
Messpunkt liegt bei 100k Steps, und da sind beide Policies längst konvergiert
(`seed_0_pilot`: Power 86 %, Precision 88 % am ersten Punkt) — die Kurve sieht
flach aus, obwohl der Lernvorgang stattgefunden hat, er liegt nur komplett im
ersten Messintervall. Deshalb gibt es zwei Log-Configs:
`configs/ppo_final.yaml` (2M, `eval_freq` 25k, 50 Eval-Episoden, Checkpoint alle
250k) für die reguläre Kurve und `configs/ppo_finegrain.yaml` (200k,
`eval_freq` 5k) für den Anfang. Beide ändern NUR die Protokollierung, keinen
Hyperparameter.

## `timeout_is_failure` (2026-08-26) — steht auf `true`, bewusst
Der Schalter wertet Timeout als Fehlschlag ohne Lift-Test und gibt dem
Griffabschluss damit überhaupt erst einen Gradienten (RL_VERIFICATION.md §2f);
ohne ihn kostet ein nie bestätigter Trigger nur ~3 Punkte Schrittstrafe.

**Entscheidung Leon 2026-08-26:** bleibt `true` in beiden `*_neutraining.yaml`.
Die Frage der Arbeit ist, OB das RL den Griffabschluss lernen kann. Dass es das
in den ersten 100k Steps nicht kann, ist Teil der Lernkurve und kein Defekt.

Zahlen zum Einordnen. Mit `true` stand `precision_neutraining` nach 90k Steps
bei 0 % Erfolg (alle 50 Eval-Episoden Timeout, `ep_len` konstant 300),
`power_neutraining` bei 54 % → 42 % fallend. Mit `false` erreicht
`seed_0_probetrig` auf derselben Config 62 % → 97 % (Precision) und
84 % → ~75 % (Power) — der Unterschied ist also gross, aber 90k sind 4,5 % von
2M und beantworten die Frage nicht.

**Offener Punkt für den Sim-Real-Vergleich:** `eval/policy_runner.py` hebt an
der echten Hand auch ohne bestätigten Trigger nach `max_steps` an und meldet
nur „Kein Trigger — Griff lief bis max_steps". Solange das so ist, misst die
Sim strenger als das Labor. Wer die Zahlen nebeneinanderstellt, muss entweder
den Runner angleichen oder den Unterschied dazuschreiben.

**Läufe vom 26.08. auf pc-lw:** `seed_0_timeoutfail` = mit Schalter (die
eigentliche Frage), `seed_0_final` = ohne Schalter (Absicherung für die
Realevaluierung am 27.08.). Beide 2M, `ppo_final.yaml`, `--lr-schedule linear`.
Welche Einstellung ein Lauf hatte, steht in seiner `run_meta.yaml`.

## Stolpersteine
- `hardware/contact_detector.py` darf sim.hand nur lazy importieren
  (Zirkularität mit sim/env.py).
- Alte Baselines (ohne `read_timing: pre_next_send` in meta) sind ~+0.005 zu
  hoch — neu kalibrieren, Runner warnt.
- Die CSVs vom 08.07. sind mit dem alten Lese-Timing aufgenommen (konsistent
  untereinander, absolut ~+0.005 hoch).
