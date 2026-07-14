# AR10-Sensoranalyse: q_delta-Rauschstruktur und Kontakterkennung ohne FSR

**Datum der Analyse:** 2026-07-10 (Daten aufgezeichnet am 2026-07-08, precision-Config)
**Analysiert von:** Claude-Session, gemeinsam mit Leon
**Zweck:** Nachvollziehbare Dokumentation für spätere Sessions. Alle Zahlen stammen aus den
CSV-Dateien in diesem Ordner und sind mit den Skripten `analyze_sensors{1,2,3}.py` (ebenfalls
in diesem Ordner) reproduzierbar.

---

## 1. Ausgangsproblem

Kontakterkennung auf der echten Hand läuft über `q_delta = q_target − q_measured` pro Joint,
Kontakt-Bit = 1 wenn `max(0, q_delta) > threshold`. Der Threshold stand auf **0.05**, weil
niedrigere Werte (0.03 getestet) sofort False Positives erzeugten. Folge: Kontakt mit der
Kugel wird so spät erkannt, dass sie längst weggerollt ist. Würfel funktionieren (rollen
nicht). Verdacht vor der Analyse: q_delta kann Berührung grundsätzlich nicht erkennen,
einzige Lösung wären FSR-Sensoren.

**Ergebnis der Analyse: Der Verdacht stimmt nur halb.** Das Kontaktsignal der Kugel ist in
den Daten vorhanden — es ist nur 3–5× kleiner als der unkorrigierte Threshold. Der "Noise",
der den hohen Threshold erzwingt, ist **kein Rauschen, sondern systematisch und
reproduzierbar** — und lässt sich daher wegkalibrieren.

---

## 2. Datengrundlage (alle in `artifacts/analysis/`)

| Datei | Inhalt |
|---|---|
| `servo_analysis_precision_20260708_122810.csv` | **Freilauf-Baseline**, 10 Zyklen, alle 4 Joints (servo6/7/8/9), Phasen SETTLE/CLOSE/HOLD_CLOSED/OPEN/HOLD_OPEN/CLOSE_2/HOLD_2, 200 CLOSE-Steps/Zyklus, Rampe 0.005/Step |
| `contact_latency_precision_20260708_124132.csv` | **Kugel-Test**, 5 Zyklen, alle 4 Joints, gleiche Rampe, mit `*_baseline_delta`-Spalten (aus Zyklus 0 der Baseline) |
| `contact_latency_precision_20260708_133935.csv` | Kugel, **nur MCP** (servo6/8) — PIP wurde NICHT angesteuert (Finger gestreckt!) |
| `contact_latency_precision_20260708_134024.csv` | Würfel, nur MCP — ebenso |
| `position_noise_summary_precision_20260708_133816.csv` | Statisches Rauschen an 10 Positionen (q 0.0–0.9), 100 Samples/Position |
| `raw_adc_precision_20260708_133827.csv` | Rohe ADC-Werte (0–1023) während CLOSE-Rampe, servo6/8 |
| `servo_analysis_precision_20260708_133601.csv` | Zweite Freilauf-Baseline (nur servo6/8), normal 0.005/Step |
| `servo_analysis__temp_slow_20260708_133732.csv` | Freilauf **langsam** (delta_norm 0.003) |

**Wichtige Stolperfalle beim Einlesen:** Die `step`-Spalte der `servo_analysis`-Dateien ist
ein **globaler Zähler** über alle Phasen/Zyklen. Für Zyklus-Vergleiche muss ein relativer
Index gebildet werden: `rstep = groupby(cycle)[nach Phasenfilter].cumcount()`. In den
`contact_latency`-Dateien ist `step` dagegen bereits pro Zyklus (beginnt bei 0).

---

## 3. Kernbefund 1: q_delta hat drei Regime — und keins davon ist "zufälliges Rauschen"

### 3.1 Startup-Transient (Steps 0–25 jeder CLOSE-Rampe)

Die Servos haben eine Anlauf-Totzone: `q_measured` beginnt erst nach **~11–15 Steps** zu
steigen (servo7: Steps 11–14, servo9: Steps 11–15 über 10 Zyklen). Bis dahin läuft
`q_target` davon → q_delta-Peak in **jedem** Freilauf-Zyklus:

| Joint | max q_delta in Steps 0–25, pro Zyklus (10 Freilauf-Zyklen) |
|---|---|
| servo6 | 0.050–0.070 |
| servo7 | 0.045–0.058 |
| servo8 | 0.052–0.072 |
| servo9 | 0.054–0.061 |

**Konsequenz:** Selbst Threshold 0.05 wird in 9–10 von 10 Freilauf-Zyklen überschritten —
der heutige Detektor feuert auch OHNE Objekt kurz nach Start der Schließbewegung. Das ist
der eigentliche Grund, warum der Threshold nicht gesenkt werden konnte. Jeder bessere
Detektor muss die Steps < ~25 maskieren oder den Startup separat behandeln.

Randnotiz: In der Baseline starten die Zyklen 1–9 mit `q_measured > 0.02` (Hand nach OPEN
nicht vollständig offen), Zyklus 0 startet sauber bei 0. Im Kugel-Test (1 s Settle nach
Öffnen) starten alle Zyklen bei ~0. Deshalb sind Residuen an Step 0–15 **Alignment-Artefakte**
(scheinbares Residuum bis +0.18 an Step 0!) und dürfen nicht als Signal gewertet werden.

### 3.2 Tracking-Regime (während der Rampe, nach Startup)

q_delta liegt für MCP-Joints im Mittel bei ~0.03 — aber dieser Offset ist **positionsabhängig
und hochgradig reproduzierbar**. Zyklus-zu-Zyklus-Streuung am selben rstep:

| Joint | q_delta mean (CLOSE) | Zyklus-zu-Zyklus-Std (median über Steps) |
|---|---|---|
| servo6 | +0.026 | **0.0033** |
| servo7 | +0.019 | **0.0034** |
| servo8 | +0.028 | **0.0032** |
| servo9 | −0.020 | **0.0021** |

→ Nach Abzug der mittleren Baseline-Trajektorie bleibt nur ±0.003 Streuung. Ein
baseline-korrigierter 3–4σ-Threshold liegt bei **~0.010–0.013** statt 0.05.

Run-zu-Run-Drift: Baseline 12:28 vs. Kugel-Test 12:41 (13 min Abstand) — Residuen im
Tracking-Regime blieben innerhalb ±0.01. Drift über eine Session ist also klein, aber die
Baseline sollte pro Session frisch kalibriert werden.

### 3.3 Statisches Regime (Target wird gehalten)

Aus `position_noise_summary` (100 Samples pro Position, Hand steht):

| Position q | servo6 mean / std | servo8 mean / std |
|---|---|---|
| 0.0 | −0.000 / 0.0007 | 0.000 / 0.0000 |
| 0.3 | −0.010 / 0.0010 | −0.003 / 0.0021 |
| 0.6 | −0.019 / 0.0010 | −0.018 / 0.0010 |
| 0.9 | −0.025 / 0.0012 | −0.023 / 0.0011 |

**Statisches Sensorrauschen ist praktisch null (std ~0.001, 3σ ≈ 0.003).** Der Settle-Wert
selbst ist positionsabhängig (0 → −0.025) und muss aus dieser Tabelle nachgeschlagen werden.

Settle-Geschwindigkeit (Übergang CLOSE→HOLD_CLOSED, Freilauf): q_delta kollabiert innerhalb
von **~5–15 Steps** (≈100–300 ms bei 48 Hz) auf den statischen Wert. servo8: von +0.028 (h0)
auf +0.014 (h5) auf ~+0.008 (Steps 15+).

### 3.4 Raw-ADC: Auflösung ist NICHT der Engpass

servo6: ADC-Range 67–642 über q 0→1 → **1 ADC-Count ≈ 0.0017 q-Einheiten**. Die Rampe
(0.005/Step) entspricht 2.9 Counts/Step. Quantisierung liegt weit unter allen relevanten
Signalen. Multi-Read-Averaging (`adc_reads`) und EMA (`ema_alpha` in `hardware/ar10.py`)
brachten deshalb nichts bzw. verschlechterten es (EMA-Lag erhöht q_delta während Bewegung
um 8–28 % auf den PIP-Joints).

---

## 4. Kernbefund 2: Das Kugel-Kontaktsignal existiert — in zwei Formen

Analysiert auf `contact_latency_precision_20260708_124132.csv` (5 Zyklen, alle 4 Joints),
Residuum = q_delta − Baseline-Mean-Trajektorie(rstep) aus den 10 Freilauf-Zyklen.

### 4.1 Erstkontakt: transienter Buckel bei q_target ≈ 0.2 (servo8) — MIT VORBEHALT

> **Korrektur (Update 2026-07-10, nach Detektor-Offline-Validierung):** Dieser
> Buckel ist durch einen **Kalt-/Warmstart-Confound** belastet. In der
> Freilauf-Baseline startet nur Zyklus 0 "kalt" (q_measured = 0); die Zyklen 1–9
> starten "warm" (q_measured ≈ 0.02–0.05, Hand nach OPEN nicht ganz zurück).
> Die Kugel-Zyklen starten alle KALT (1 s Settle + Operator-Pause). Kalte Starts
> haben in Steps ~25–45 systematisch höheres q_delta als warme — der Freilauf-
> Zyklus 0 erzeugt gegen die warm-dominierte Baseline im SELBEN Fenster
> (Steps 33–38) dasselbe "Signal" wie die Kugel-Zyklen. Ob der Buckel Kontakt
> oder Startzustand ist, lässt sich aus diesen Daten NICHT entscheiden.
> `eval/baseline_calibration.py` startet deshalb jeden Kalibrier-Zyklus kalt
> (wie die echten Episoden); die Frage klärt sich beim nächsten Hardware-Test.
> Die späte Blockierungs-Detektion (4.2) ist vom Confound NICHT betroffen.

servo8 q_delta, Steps 28–70, Baseline (mean [min,max] über 10 Freilauf-Zyklen) vs. die 5
Kugel-Zyklen:

```
step  Baseline mean [min,max]      Kugel Z0..Z4
 28   +0.044 [+0.027,+0.048]   0.025 0.029 0.036 0.020 0.027   <- unter Baseline
 36   +0.031 [+0.027,+0.036]   0.045 0.046 0.043 0.045 0.050   <- ALLE 5 drüber!
 40   +0.034 [+0.027,+0.044]   0.046 0.042 0.042 0.051 0.051   <- ALLE 5 drüber
 48   +0.039 [+0.034,+0.042]   0.035 0.032 0.037 0.039 0.037   <- wieder in Baseline
 56   +0.037 [+0.034,+0.041]   0.037 0.035 0.037 0.034 0.035   <- unauffällig
 68   +0.036 [+0.035,+0.038]   0.047 0.042 0.047 0.045 0.045   <- steigt wieder
```

Bei Steps ~36–42 (q_target ≈ 0.18–0.21) liegen **alle 5 Kugel-Zyklen** um +0.010–0.015 über
der Baseline, die dort in 10 Freilauf-Zyklen nie so hoch war. Danach verschwindet das Signal
für ~15 Steps wieder → klassische Signatur "Finger berührt Kugel, Kugel gibt nach/rollt,
Blockierung weg". Ein z-Score-Detektor (Residuum / Zyklus-Std, Alarm bei z>3 für 3 Steps in
Folge) fand diesen Punkt in 4 von 5 Zyklen bei Steps 34–39.

**Das ist der physische Erstkontakt — sichtbar, aber nur mit Baseline-Subtraktion.**

### 4.2 Späte Blockierung: langsam wachsendes MCP-Residuum ab q_target ≈ 0.65

Wenn die Hand die Kugel geometrisch einklemmt, wächst das MCP-Residuum stetig:
servo8-Residuum (Zyklus 1): +0.011 @s119 → +0.017 @s135 → +0.031 @s159 → +0.046 @s183 →
+0.064 @s199. servo6 analog, etwas schwächer (bis +0.042).

Ein CUSUM-Detektor auf dem Residuum (drift=0.003, Alarmschwelle = 1.2 × max-Freilauf-CUSUM,
kalibriert auf 0 False Positives über die 10 Freilauf-Zyklen) erkennt das bei Steps 118–156
(q_target 0.60–0.79). Der heutige 0.05-Threshold (auf rohem q_delta, Startup-Treffer
ignoriert) schlägt erst bei Steps 173–175 an (q_target ≈ 0.87). **CUSUM ist ~50 Steps ≈ 1 s
früher; der transiente Erstkontakt (4.1) sogar ~135 Steps ≈ 2.7 s früher.**

### 4.3 PIP-Joints liefern KEIN anhaltendes Signal (bei pip_cap 0.5)

servo7/servo9 erreichen ihr gecapptes Ziel (0.5) **auch mit Kugel** vollständig
(q_measured servo7 → 0.507, servo9 → 0.558). q_delta nach dem Cap (Steps 120–199) ist mit
Kugel identisch zum Freilauf. Die Kugel blockiert die PIP-Gelenke bei Cap 0.5 schlicht nicht.

**Nebenbefund servo9:** konstanter Sensor-Offset von **−0.06** (q_measured läuft dem Target
um 0.06 voraus, auch statisch: HOLD-Werte um −0.060). Die Input-Kalibrierung dieses Kanals
stimmt nicht; das frisst einseitig Detektionsspielraum und sollte nachkalibriert werden.

### 4.4 Warum der MCP-only-Test (13:39/13:40) NICHTS zeigte

In diesen Tests war `finger_joints` auf MCP-only umgestellt → **die PIP-Joints wurden gar
nicht angesteuert, die Finger blieben gestreckt**. Mit gestreckten Fingern schiebt die
MCP-Bewegung die Kugel weg statt sie zu umschließen → keine Blockierung → Residuum bleibt
±0.005 (max +0.016 spät). Die damalige Schlussfolgerung "MCP sieht keinen Kontakt" gilt also
nur für diese (unrealistische) Geometrie. Im 4-Joint-Test sehen die MCP-Joints sehr wohl
Kontakt (4.1, 4.2).

### 4.5 Langsames Schließen hilft messbar

delta_norm 0.003 statt 0.005 (Vergleich `servo_analysis__temp_slow` vs. `_133601`, Steps ≥25):
Tracking-Mean sinkt von +0.031/+0.034 auf +0.018/+0.021 (servo6/8), Zyklus-zu-Zyklus-Std von
0.0031/0.0033 auf 0.0020/0.0020. Tracking-Fehler skaliert also ~proportional zur
Geschwindigkeit. Zusatznutzen: weniger Impuls auf die Kugel beim Erstkontakt.

---

## 5. Gescheiterte Ansätze (nicht wiederholen)

1. **Multi-Read-ADC-Averaging** (`adc_reads` in `hardware/ar10.py`): nur 5–15 % Reduktion —
   das Rauschen ist nicht ADC-Zufallsrauschen. Bei 8 Reads wird zudem das step_dt-Budget
   gesprengt.
2. **EMA-Filter** (`ema_alpha`): verschlechtert PIP-q_delta während Bewegung um 8–28 %
   (Temporallag). Beide Features sind noch im Code, aber ungenutzt (Defaults inaktiv).
3. **Threshold 0.03 auf rohem q_delta**: sofort False Positives (Startup-Peak + MCP-Baseline
   ~0.032 liegen darüber).
4. **MCP-only-`finger_joints`**: falsche Geometrie, siehe 4.4.
5. **Naiver Stall-Detektor** (Steigung von q_measured < Schwelle): feuert im Freilauf in
   10/10 Zyklen falsch — Startup-Totzone und ADC-Treppenstufen erzeugen ständig
   Null-Steigungs-Fenster. Nur mit Baseline-Gating brauchbar.

---

## 6. Schlussfolgerung: Kontakterkennung ohne FSR ist möglich

Empfohlene Detektor-Architektur (Reihenfolge = Priorität):

1. **Session-Kalibrierung:** ~10 Freilauf-CLOSE-Zyklen fahren → Tabelle
   `baseline_mean(joint, rstep)` + `baseline_std(joint, rstep)` (bzw. indexiert über
   q_target, robuster gegen abweichende Rampen). Startup-Steps < 25 maskieren.
2. **Primärdetektor (Erstkontakt):** `residuum = q_delta − baseline_mean` ; Kontakt wenn
   `residuum > max(0.012, 4·baseline_std)` für ≥3 aufeinanderfolgende Steps.
   Erwartung laut Daten: Kugel-Erstkontakt bei q_target ≈ 0.2 statt 0.87.
3. **Sekundärdetektor (langsame Blockierung):** CUSUM auf dem Residuum
   (drift 0.003, Schwelle auf 0 False Positives über die Kalibrier-Zyklen + 20 % Marge).
4. **Bestätigungsschicht vor Lift (optional):** Micro-Pause — Target 10 Steps halten,
   q_delta gegen statische Positions-Tabelle prüfen (Threshold 0.005–0.01; statische 3σ ≈
   0.003). Nahezu fehlalarmfrei, kostet ~200 ms.
5. **Flankierend:** delta_norm auf 0.003 senken (oder nur nahe dem Objekt), servo9
   nachkalibrieren, PIP-Cap überdenken (bei 0.5 tragen PIP-Joints nichts zur Detektion bei).

Wichtige Einschränkungen:
- Der Erstkontakt-Buckel ist **transient** (~10–15 Steps ≈ 250 ms Fenster). Detektor und
  Policy-Reaktion müssen innerhalb dieses Fensters greifen, sonst ist die Kugel trotzdem weg.
- Physikalische Grenze bleibt: Positionssensorik misst **Blockierung**, nicht Berührung.
  FSR-Sensoren (direkt an freie Maestro-Analogkanäle 0–9 anschließbar, Spannungsteiler mit
  10 kΩ, ~10 €) wären weiterhin die sauberere Lösung — aber kein Muss mehr.
- Alle Zahlen gelten für die precision-Config (Rampe 0.005/Step, pip_caps 0.5, 48 Hz).
  Für power oder andere Rampen: Baseline neu aufnehmen.
- Die CUSUM-/z-Schwellen wurden auf denselben Freilauf-Daten kalibriert, auf denen auch die
  False-Positive-Prüfung lief (optimistisch). Vor echtem Einsatz mit frischen
  Freilauf-Zyklen gegenvalidieren.

---

## 6b. Implementierung + Offline-Validierung (Update 2026-07-10)

Der Detektor aus Abschnitt 6 ist implementiert:

- `hardware/contact_detector.py` — `QDeltaBaseline` (Tabelle), `build_baseline()`
  (aus Freilauf-Zyklen), `ContactDetector` (Residuum + Persistenz + Hysterese +
  CUSUM + Startup-Maske + Settle-Maske)
- `eval/baseline_calibration.py` — Session-Kalibrierung (~2 min, 10 Zyklen,
  jeder Zyklus startet KALT) → `artifacts/calibration/qdelta_baseline.yaml`
- `eval/policy_runner.py` — `load_contact_detector()`; ersetzt `_binary_obs`
  wenn `contact_detector.enabled` (precision.yaml). Bricht hart ab ohne
  Baseline; prüft delta_norm-Match; warnt ab 12 h Baseline-Alter.
- `configs/precision.yaml` — `contact_detector:`-Block; `stabilization_steps`
  zurück auf 30 (Trainingswert — mit früher Detektion ist die
  Stabilisierungsphase wieder die Griffaufbau-Phase, siehe Chat-Verlauf)
- `eval/test_detector_offline.py` — Validierung auf den CSVs dieses Ordners

**Zwei zusätzliche Fallen, die erst die Offline-Validierung aufdeckte**
(beide gefixt, nicht wieder einbauen):

1. **Cap-Settle-Transient**: Erreicht ein Joint seinen pip_cap, fällt q_delta
   über ~10–15 Steps auf den Settle-Wert. Ein q_target-Bin, der Transient und
   Settled mischt, erzeugt am Cap systematische False Positives (vorher 10/10
   Freilauf-Zyklen!). Fix: Samples mit 0 < steps_since_target_move < settle_steps
   werden aus den Bins ausgeschlossen UND der Detektor maskiert dieselben Steps.
2. **CUSUM im Stand**: Der Settle-Wert am Cap driftet pro Zyklus um ±0.02
   (servo7, std 0.009) — ein CUSUM, der im Stand weiterläuft, summiert diesen
   Offset bis zum Fehlalarm auf. Fix: CUSUM akkumuliert nur bei bewegtem Target
   (steps_since_move == 0), in Kalibrierung und Detektor identisch.

**Validierungsergebnis** (`python -m eval.test_detector_offline`):
- Freilauf leave-one-out: **0/9 warme Zyklen mit False Positive** (Zyklus 0 =
  Kaltstart-Confound, zählt nicht — siehe 4.1-Korrektur).
- Kugel (Baseline aus allen 10 Freilauf-Zyklen, 13 min älter): Trigger
  (2 Finger, 3 Steps) bei q_target **0.21 / 0.55–0.80** je Zyklus — vs. alter
  0.05-Threshold bei 0.86–0.89 **oder fälschlich bei q≈0.07** (Startup-Peak;
  der alte Detektor triggerte in 3/5 Kugel-Zyklen im Anlauf, war also nicht nur
  spät, sondern auch unzuverlässig).
- Die frühen Trigger bei q≈0.21 (Zyklen 1(Bit), 3(Bit), 4(Trigger)) liegen im
  Confound-Fenster → erst nach frischer (kalt gestarteter) Kalibrierung belastbar.

## 7. Reproduktion

```
python analyze_sensors.py    # Regime-Statistiken, Baseline-Reproduzierbarkeit, Raw-ADC, MCP-only
python analyze_sensors2.py   # Residuum-Trajektorien, CUSUM, Vergleich mit 0.05-Threshold, PIP-Cap
python analyze_sensors3.py   # Statik-Tabelle, Settle-Verhalten, Slow-Close, Startup-Statistik
```
(Skripte liegen in diesem Ordner; Pfade zeigen auf `artifacts/analysis/`. Benötigt pandas/numpy.)
