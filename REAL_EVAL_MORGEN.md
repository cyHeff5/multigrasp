# Ablaufplan Realevaluierung — ein Tag, ein Versuch

Stand 2026-07-28. Ergänzt `REAL_EVAL.md` um Reihenfolge, Zeitbudget und Abbruchkriterien.
**Leitregel: Daten schlagen Perfektion.** Alles vor Block C ist zusammen auf 30 Minuten gedeckelt;
läuft etwas nicht, greift der Fallback und die Trials starten trotzdem.

## Was heute schon erledigt ist

- `configs/power.yaml` hat jetzt einen `contact_detector`-Block. Vorher stürzte
  `baseline_calibration.py --config configs/power.yaml` sofort mit `KeyError` ab und Power wäre auf
  dem alten Roh-Threshold gelaufen — der laut Offline-Test in **jedem** Zyklus schon bei q ≈ 0.06
  ein Bit setzt.
- Beide Pfade (Baseline-Kalibrierung + Policy-Runner) sind für Power im Mock-Modus durchgelaufen.
- `eval/qsweep_check.py` ist neu: misst in ~2 Minuten, ob q auf der echten Hand dasselbe bedeutet
  wie in der Sim.
- Sim-Referenzbilder liegen in `artifacts/analysis/qsweep/sim_q*.png` (Seitenansicht, Daumen rechts).
- Offline-Validierung des Detektors: 0/9 warme Freilaufzyklen mit False Positive.

---

## Block A — Setup (10 Min, Pflicht)

1. Hand an COM-Port, Port notieren.
2. **Maestro Control Center öffnen, pro Kanal Min/Max ablesen, in
   `artifacts/calibration/servo_limits.yaml` eintragen.** Steht bis heute vollständig
   auskommentiert auf den Defaults 4200/7700. Der Maestro clippt still — sind seine Grenzen enger,
   laufen q_target und echte Position dauerhaft auseinander und erzeugen Phantom-Kontakt.
   Das ist die einzige Sache in diesem Plan, die alles andere verfälschen kann.
3. Kamera aufstellen und laufen lassen (Erfolg/Misserfolg kommt aus dem Video, nicht aus der Eingabe).

## Block B — Diagnose + Kalibrierung (20 Min, harte Obergrenze)

**Update 2026-08-24:** Detektor hat jetzt Raten-Skalierung + Anlauf-Gate +
Statik/Dynamik-Trennung, und die Kalibrierung liest im Runner-Timing (alte
Baselines sind ~+0.005 zu hoch — B2 ist also PFLICHT, alte YAMLs löschen).
Details: `SENSOR_ANALYSIS_FINDINGS.md` §8.

```bash
# B0  NEU: Pausen-Rampe, OHNE Objekt (~3 Min) — misst Wiederanlauf-Totzone und
#     Kollaps-Zeit nach Stopp; danach restart_steps in den Configs auf den
#     gemessenen Wert setzen (steht konservativ auf 12)
python -m eval.pause_ramp_check --config configs/precision.yaml --port COM4

# B1  q-Normierung, OHNE Objekt (~2 Min)
python -m eval.qsweep_check --config configs/power.yaml --port COM4
```
Fährt q = 0 / 0.25 / 0.5 / 0.75 / 1.0, hält jeden Punkt und liest q_measured. Im Freilauf muss der
Restfehler ≈ 0 sein. Deutung: Fehler nur bei q = 1.0 → Pulsspanne/Anschlag; Fehler schon bei
0.25–0.75 → Hub-zu-Winkel ist nichtlinear. Danach ein Handyfoto bei q = 1.0 aus der Perspektive von
`artifacts/analysis/qsweep/sim_q1.00.png`. **Nur messen, heute nichts daran reparieren.**

```bash
# B2  Baselines für den ContactDetector, OHNE Objekt (je ~2 Min)
python -m eval.baseline_calibration --config configs/precision.yaml --port COM4
python -m eval.baseline_calibration --config configs/power.yaml     --port COM4
```
Schreibt `qdelta_baseline.yaml` bzw. `qdelta_baseline_power.yaml`. Beide sind Pflicht — der Runner
bricht ohne sie hart ab. Am Ende gibt jedes Skript die effektiven Thresholds pro Joint aus; liegen
die im Bereich 0.012–0.02, ist alles normal.

**Abbruchkriterium:** Ist Block B nach 20 Minuten nicht sauber durch, in beiden Configs
`contact_detector.enabled: false` setzen, `eval/calibration.py --phase free` + `--phase parallel`
fahren (~10 Min) und mit dem alten Pfad evaluieren. Dann ist bekannt und dokumentiert, dass die
Kontakterkennung die schlechtere ist — aber die Trials existieren.

## Block C — Trials (der Rest des Tages)

```bash
python -m eval.policy_runner --config configs/precision.yaml \
    --model artifacts/models/precision/seed_0_dr/best/best_model --port COM4
python -m eval.policy_runner --config configs/power.yaml \
    --model artifacts/models/power/seed_0_dr/best/best_model --port COM4
```

### Go/No-Go nach den ersten 3 Trials — nicht überspringen

Der Runner druckt pro Trial `n_steps`. Vergleich mit den Referenzwerten:

| `n_steps` beim Trigger | Bedeutung | Handlung |
|---|---|---|
| ~38–45 | Frühtrigger im Servo-Anlauf — der Fehler vom 08.07. ist zurück | **Stopp.** `startup_mask_q` in der Config von 0.13 auf 0.22 anheben, Baseline neu fahren |
| ~120–200 | erwartet, Detektor arbeitet | weiter |
| 300 (kein Trigger) | Bit kippt nie | Baseline prüfen; im Zweifel weiterlaufen lassen und dokumentieren |

Drei Trials kosten zwei Minuten und schützen davor, 300 unbrauchbare Trials aufzuzeichnen.

### Reihenfolge der Greifpunkte

Insgesamt 19 Precision- + 13 Power-Greifpunkte × 10 = **320 Trials**. Bei 30–45 s pro Trial sind das
2.5–4 Stunden reine Trial-Zeit. Falls es eng wird, ist ein vollständiger Teilsatz mehr wert als ein
lückenhafter Gesamtsatz. Deshalb zuerst die Teile, die den wissenschaftlichen Gehalt tragen —
dort steckt der Sim2Real-Vergleich:

1. **Teil 5** (12.7 × 12.7 × 1.8 cm, flach) und **Teil 9** (8.8 × 2.8 × 1.3 cm, flach) — Sim sagt
   100 % / 95 %, real sind genau das die Problemfälle
2. **Teil 13** — Sim sagt precision 0 %, power 30 %; der einzige Punkt, an dem die Sim selbst scheitert
3. **Teil 7** — Sim 35 %
4. Rest in beliebiger Reihenfolge

### Nebenbei, kostet nichts

- Die realen Benchmarkteile **einmal wiegen**. In Sim haben alle 14 Teile 0.1 kg (Teil 3: 0.043 kg),
  weil `GraspObject.spawn` für URDF-Objekte die gesampelte Masse nie anwendet. Ohne die echten Massen
  ist der Sim↔Real-Vergleich konfundiert. Fünf Minuten.

---

## Danach

Die Trial-Logs liegen in `artifacts/eval_results/real_*.csv` mit `triggered`, `n_steps`,
`achieved_hz` und `q_final` pro Trial — zusammen mit dem Video reicht das für die Erfolgsraten-Tabelle
und für die Diagnose, falls etwas schiefgeht. Die offenen Punkte aus
`artifacts/analysis/RL_VERIFICATION.md` (ein Seed, fehlende Run-Metadaten, veraltetes LaTeX-Dokument)
brauchen keinen Roboter und können danach jederzeit nachgezogen werden.
