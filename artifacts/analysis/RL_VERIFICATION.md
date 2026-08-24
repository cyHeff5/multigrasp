# RL-Verifizierung vor dem Freeze

**Datum:** 2026-07-28 · **Geprüft:** `configs/*.yaml`, `sim/`, `training/`, `eval/`, `hardware/`,
`artifacts/models/*/seed_0_dr/best`, alle CSVs in `artifacts/analysis/` und `artifacts/eval_results/`
**Zweck:** Vor dem Einfrieren der Policies festhalten, was wissenschaftlich trägt, was angreifbar ist
und was vor der Realevaluierung noch entschieden werden muss.

Kurzfassung: Das Design ist überdurchschnittlich sauber begründet — die meisten Parameter haben eine
Messung oder eine Quelle hinter sich, nicht nur ein Bauchgefühl. Die Angriffsfläche liegt nicht im
Reward oder im Observation-Design, sondern in **Evidenz** (ein Seed, keine Baseline zum Vergleich) und
in **Konsistenz** (LaTeX-Doc, Power-Detektor, Benchmark-Massen weichen vom Code ab).

---

## 1. Was trägt

Diese Punkte würde ich in einer Verteidigung offensiv vertreten:

**Reward ist potentialbasiert und formal korrekt.** `sim/reward.py` nutzt PBRS nach Ng, Harada &
Russell (1999): Φ(s) = w_contact · min(n_contact, n_target)/n_target, Shaping = γΦ(s′) − Φ(s). Das ist
die einzige Shaping-Form mit bewiesener Policy-Invarianz. Zwei Details sind richtig gemacht, die oft
falsch gemacht werden: γ im Shaping ist identisch mit dem PPO-γ (0.995, in beiden Configs explizit
kommentiert), und im Terminalzustand wird Φ(s′) = 0 gesetzt (`terminal=terminated` in `env.step`). Ohne
diese beiden Punkte gilt die Invarianz nicht. Der frühere naive "Bonus pro neuem Kontakt" hätte
Toggle-Farming erlaubt; PBRS bestraft Kontaktverlust symmetrisch.

**Das Observation-Design ist der eigentliche Sim2Real-Beitrag und ist konsequent durchgezogen.** Statt
Kraft/Drehmoment (in PyBullet und auf der AR10 systematisch verschieden) wird der Tracking-Fehler
q_delta = q_target − q_measured verwendet, binarisiert. Die Begründung — die Firgelli-Aktuatoren sind
nicht rückwärtsantreibbar, deshalb ist der kontinuierliche Zusammenhang zwischen q_delta und
Kontaktkraft nicht übertragbar, das binäre "blockiert ja/nein" aber schon — ist physikalisch sauber
und in README §4.1 dokumentiert.

**Die Kontakt-Latenz ist in Simulation gemessen, nicht behauptet.** `eval/sim_contact_latency.py`
vergleicht `getContactPoints` (Ground Truth) gegen den Bit-Flip bei sieben Thresholds. Für die 5-cm-Kugel:
physischer Kontakt bei Step 93 (middle) / 87 (index), Bit bei τ=0.02 nach 2–3 Steps, bei τ=0.05 nach
6–7, bei τ=0.08 nach 10–16 Steps. Damit ist `threshold_range [0.02, 0.08]` genau die Randomisierung
über die Latenzspanne, die real auftreten kann — das ist ein belegtes Argument, kein Handwedeln.

**Physikalisch verankerte Parameter statt Fantasiewerte.** `motor_force` 0.4–0.7 Nm ist aus Firgelli
40 N × 12 mm Hebel abgeleitet. Die Kontrollrate 48 Hz kommt aus dem ~50-Hz-PWM des Pololu Maestro.
Die Objektmasse wird dichtebasiert (0.3–1.5 g/cm³ × Volumen) gezogen statt aus einer festen
kg-Spanne — sonst hätten dünne Zylinder Dichten von 37 g/cm³ bekommen. Die Untergrenze
`threshold_range = 0.02` ist 2× das gemessene Sim-Freilauf-q_delta (0.0095).

**Die Kopplung der Precision-Finger ist begründet, nicht bequem.** Der Config-Kommentar hält fest,
warum servo6/8 und servo7/9 gekoppelt sind und der Daumen fix bleibt: der aktive Daumen flektierte
einseitig ins Objekt und kippte es (gemessen 0° → 47° → Drop). Eine blinde Policy, die nur Kontakt-Bits
sieht, kann die Opposition nicht selbst korrigieren — Symmetrie strukturell erzwingen statt lernen zu
lassen ist hier die richtige Entscheidung und lässt sich so auch schreiben.

**Die Sensoranalyse ist ehrliche Wissenschaft.** `SENSOR_ANALYSIS_FINDINGS.md` widerlegt eine eigene
frühere Schlussfolgerung (MCP-only-Test, §4.4), markiert einen Kalt-/Warmstart-Confound explizit als
nicht entscheidbar (§4.1-Korrektur) und listet gescheiterte Ansätze mit Begründung (§5). Dass die
CUSUM-Schwellen auf denselben Daten kalibriert wurden, auf denen die False-Positive-Prüfung lief, wird
selbst als optimistisch gekennzeichnet.

**Evaluations-Hygiene.** Env-Seeds pro Trainings-Seed disjunkt (`seed * 1000`), Eval-Env mit Abstand
(`+900`), und `eval_sim.py` evaluiert bei Seed 1000+ — also auf Episoden, die weder Training noch die
Best-Model-Auswahl gesehen haben. Über Gitterpunkte hinweg dieselbe Seed-Folge = Common Random Numbers,
das macht Vergleiche zwischen Objekten fairer.

---

## 2. Was angreifbar ist

Sortiert nach dem, was ein Gutachter zuerst fragen würde.

### 2.1 Ein einziger Seed — das ist die härteste Lücke

`artifacts/models/` enthält genau zwei Dateien: `power/seed_0_dr/best/best_model.zip` und
`precision/seed_0_dr/best/best_model.zip`. `training/sweep.py` und `eval/aggregate_seeds.py` existieren,
wurden für die finalen Policies aber nie benutzt.

Ein-Seed-Ergebnisse in RL gelten seit Henderson et al. (2018) nicht als Evidenz — die Varianz zwischen
Seeds ist bei PPO regelmäßig größer als die Differenz zwischen Methoden. Konkret hier: die Power-Policy
zeigt in der Sondierung ein **inkonsistentes Kontaktverhalten** (siehe 2.10) — pinky/ring stoppen bei
Kontakt, middle und index-PIP schließen stärker. Mit einem Seed lässt sich nicht sagen, ob das eine
Strategie ist oder Rauschen.

Das lässt sich lösen, ohne die eingefrorenen Policies anzufassen: 3–4 zusätzliche Seeds im Hintergrund
trainieren, während die Realevaluierung läuft. Die ausgelieferte Policy bleibt `seed_0_dr`; die
Zusatzläufe belegen nur, dass das Training reproduzierbar ist. Falls die Zeit dafür nicht reicht, gehört
der Satz "alle Ergebnisse stammen aus einem einzigen Trainingslauf" ausdrücklich in die Limitations.

### 2.2 Das Repo reproduziert die ausgelieferten Policies nicht

Im gespeicherten Modell steht `learning_rate` als **Funktion**, nicht als Float. Bei `lr_schedule:
constant` würde `training/train.py` einen Float übergeben — die beiden Policies wurden also mit
`--lr-schedule linear` trainiert, während `configs/ppo.yaml` `constant` sagt. Dazu: `num_timesteps` ist
1.400.000 (precision) bzw. 800.000 (power), nicht die 2.000.000 der Config; ob `--ent-coef` oder
`--timesteps` überschrieben wurden, ist nicht rekonstruierbar. TensorBoard-Logs und `eval_logs/` sind
gitignored und liegen nicht bei.

Damit ist die Frage "mit welchen Hyperparametern sind diese zwei Policies entstanden?" aus dem Repo
nicht beantwortbar — und die Lernkurve, die du für die Verteidigung bräuchtest, existiert nicht mehr als
Artefakt. Minimalfix: neben jedem `best_model.zip` eine `run_meta.yaml` mit dem exakten Kommando, den
Hyperparametern und der Schrittzahl; und falls die Logs lokal noch existieren, die
`eval_logs/evaluations.npz` mit `git add -f` sichern.

### 2.3 Das LaTeX-Dokument beschreibt eine Umgebung, die es nicht mehr gibt

`~/services/life-os/latex-docs/MultiGrasp - RL Mathematisch/main.tex` ist vom 22.05.2026 und damit
älter als die Reward-Umstellung, die Action-Gruppen und das Entfernen des Lock-Modells. Wenn daraus der
Methodenteil entsteht, beschreibt die Arbeit den falschen Algorithmus. Die Abweichungen:

| Thema | LaTeX-Doc | Code (Stand jetzt) |
|---|---|---|
| Reward | Delta-Bonus `w_c·max(0,n̂ₜ−n̂ₜ₋₁)/N` + Hold-Bonus `w_h·n̂ₜ/N` | PBRS `γΦ(s′)−Φ(s)`, kein `w_h` |
| `n_target` Precision | N = 1 | 2 |
| `trigger_n` Precision | 1 | 2 |
| Action-Space | ein Signal je Joint, servo0 mit 3 Optionen | ein Signal je **Gruppe**; Precision hat **kein** servo0-Signal |
| Obs-Dim Precision | 7 | **4** (2 Bits + 2 Gruppen-q) |
| Obs-Dim Power | 14 | 14 ✓ |
| Motorkraft | 𝒰(4.0, 6.0) Nm | 𝒰(0.4, 0.7) Nm |
| Objektmasse | 𝒰(0.05, 0.25) kg | Dichte × Volumen |
| Threshold | fest τ = 0.05 | 𝒰(0.02, 0.08) pro Episode |
| Substeps | fest 5 | 𝒰{4…7} pro Episode |
| Total timesteps | 10⁷ | 2·10⁶ (best bei 1.4·10⁶ / 0.8·10⁶) |
| Non-Backdrivable-Modell | Lock-Variable ℓⱼ + Hard-Reset pro Substep | **entfernt** — nur noch POSITION_CONTROL-Haltekraft |
| DIP-Kopplung | Hard-Reset pro Substep | JOINT_GEAR-Constraint |
| Podest-Magnet (k = 5 N/m) | ganzer Abschnitt | **existiert im Code nicht** |
| LR-Schedule | nicht erwähnt | linear (siehe 2.2) |

Die inhaltlich wichtigste Zeile ist das Lock-Modell: der LaTeX-Text stellt es als *den* Mechanismus dar,
der das Kontaktsignal überhaupt trägt. Im Code steht heute in `sim/hand.py` (Z. 186–191), dass genau
dieser Substep-Teleport die Finger bis zu 14 mm durch die Objekte trieb und deshalb entfernt wurde;
Non-Backdrivability wird jetzt nur noch über die Haltekraft der Positionsregelung approximiert. Das ist
eine **schwächere** Approximation — ein Gelenk mit force ≤ 0.7 Nm lässt sich von außen zurückdrücken,
das echte 100:1-Getriebe nicht. Ob das relevant ist, hängt davon ab, ob Kontaktmomente 0.7 Nm
überschreiten; das ist bisher nicht geprüft. Diese Änderung braucht in der Arbeit eine bewusste
Formulierung, weil sie den zentralen Sim2Real-Mechanismus betrifft.

### 2.4 Die Benchmark-Sim-Ergebnisse haben keine Massen-Randomisierung — und die berechnete Masse wird verworfen

In `sim/object.py` wird für `shape == "urdf"` per `loadURDF` gespawnt; `spec["mass_kg"]` wird **nie
angewendet**, nur `lateralFriction`. Nachgemessen in PyBullet: alle 14 Benchmarkteile haben 0.100 kg
(Teil 3: 0.043 kg) — unabhängig von der Größe, von 2.8 cm bis 12.7 cm Kantenmaß.

Gleichzeitig berechnet `env.reset()` bei **jedem** Reset eine Masse (Kommentar: "Masse und Reibung
werden immer neu gezogen — auch wenn obj_spec übergeben wurde") und wirft sie für URDF-Objekte weg.
Über den Würfel-Fallback in `object_volume_cm3` (Kantenlänge = AABB-Breite) käme dabei heraus:

| Teil | AABB (cm) | Sim-Masse real | berechnete Masse (verworfen) |
|---|---|---|---|
| 1 | 5.8 × 13.8 × 5.3 | 0.100 kg | 0.79–3.94 kg |
| 4 | 12.7 × 12.7 × 12.8 | 0.100 kg | 0.61–3.07 kg |
| 12 | 2.8 × 2.8 × 10.8 | 0.100 kg | 0.01–0.03 kg |

Drei Konsequenzen. Erstens: die Benchmark-Zahlen in `precision_benchmark_*` / `power_benchmark_*` gelten
für ein festes 100-g-Objekt, während README §7.2 sie neben die Formen-Evaluierung stellt, in der die
Masse randomisiert ist — das gehört unterschieden. Zweitens: für den Sim↔Real-Vergleich nächste Woche
(§7.2 vs. §7.3) ist die Masse ein Confounder. **Die realen Teile im Labor wiegen**, das kostet fünf
Minuten und entscheidet, ob der Vergleich hält. Drittens: der tote Berechnungspfad ist eine Falle — wer
ihn irgendwann "repariert", bekommt Kilogramm-Objekte und völlig andere Ergebnisse.

### 2.5 Power hat auf der echten Hand keinen kalibrierten Detektor

`configs/precision.yaml` hat den `contact_detector`-Block, `configs/power.yaml` nicht. Auf der echten
Hand fällt Power damit auf `real_threshold.yaml` zurück — abgeleitet aus der **Precision**-Kalibrierung.
Der gemessene Rauschboden dieser Kalibrierung steht in `_free_run_precision.yaml`:
`free_mean 0.0405`, `free_std 0.0125`, **`free_floor 0.0781`** (mean + 3σ).

Das heißt: der real nutzbare Roh-Threshold liegt bei ≥ 0.078, während das Training über [0.02, 0.08]
randomisiert hat — also am äußersten oberen Rand. Und `SENSOR_ANALYSIS_FINDINGS.md` §3.1 zeigt, dass der
Startup-Transient in jedem Freilaufzyklus bis 0.072 spikt, §6b, dass der alte 0.05-Detektor in **3 von 5**
Kugelzyklen im Anlauf falsch triggerte.

Praktisch bedeutet das: die Power-Ergebnisse nächste Woche messen eine andere und schlechtere
Kontaktkette als die Precision-Ergebnisse. Vor dem Labor zu entscheiden — entweder `contact_detector`
auch für Power aktivieren (`baseline_calibration.py` mit der power-Config, 8 statt 4 Joints, ~2 Min mehr
Kalibrierzeit) oder die Asymmetrie bewusst akzeptieren und in den Ergebnissen dazuschreiben. Der
zweite Weg ist vertretbar, aber nur wenn er benannt wird. REAL_EVAL.md erwähnt die Asymmetrie, zieht
aber die Konsequenz für die Vergleichbarkeit nicht.

### 2.6 Trainings- und Deployment-Detektor haben unterschiedliche Dynamik

In der Simulation ist das Kontakt-Bit gedächtnislos: `q_delta > τ`, jeden Step neu, es kann jederzeit
wieder auf 0 fallen. Auf der echten Hand liefert `ContactDetector` etwas anderes:
Baseline-Subtraktion, Persistenz über 3 Steps, Hysterese (`release_factor 0.5`), Startup-Maske,
Settle-Maske — und einen **CUSUM-Kanal, der latcht** (`_cusum_hit` wird innerhalb einer Episode nie
zurückgesetzt).

Die Domain Randomization deckt den *Threshold-Wert* ab, aber nicht die *Detektor-Dynamik*. Ein Bit, das
über CUSUM gesetzt wurde, fällt nie wieder — diesen Zustand hat die Policy im Training nie gesehen.
Das ist kein hypothetisches Problem: es ist ein plausibler Fehlermodus für nächste Woche (Bit hängt auf
1 → verfrühter Trigger → Griff friert ein, bevor er greift). Sauber wäre, die Detektor-Dynamik in der
Sim-Observation zu spiegeln und neu zu trainieren. Wenn nicht neu trainiert wird: bei der Realevaluierung
gezielt darauf achten (der Runner loggt `triggered` und `n_steps` — ein Trigger nach sehr wenigen Steps
ist das Warnsignal) und den Punkt in die Limitations aufnehmen.

### 2.7 q_norm bedeutet in Sim und Real nicht dasselbe

In der Simulation ist `q_norm` linear im **URDF-Gelenkwinkel** (`sim/hand.py::_norm_to_angle`:
`lo + norm·(hi−lo)`). Auf der echten Hand ist `q_norm` linear in der **Servo-Pulsweite** (4200–7700 µs,
`hardware/ar10.py::_to_servo`) beim Senden und in den **Potentiometer-ADC-Werten** beim Messen. Der
Firgelli ist ein Linearaktuator: Hub ↔ Gelenkwinkel läuft über die 4-Stab-Mechanik und ist nichtlinear.

Die beiden `q_norm` sind also nicht dieselbe Koordinate. Das betrifft beide Observation-Kanäle: das
Propriozeptions-`q_target` bedeutet geometrisch etwas anderes, und die q_delta-Skalen unterscheiden sich.
Die Messwerte bestätigen das: Sim-Freilauf-q_delta 0.0095 (worst case) gegen real 0.026–0.031 im
Tracking-Regime — Faktor ~3. **Das reale Freilauf-q_delta liegt damit über der unteren Grenze des
trainierten Threshold-Bereichs (0.02).**

Die Binarisierung plus Threshold-Randomisierung fängt einen Teil davon ab, und genau das ist ja das
Argument des Ansatzes. Aber die Abbildung selbst ist weder gemessen noch randomisiert, und sie ist
damit die größte verbliebene unmodellierte Lücke. Wenn dafür noch Zeit ist: für einen Joint die realen
Offen-/Geschlossen-Winkel nachmessen und die ADC-Winkel-Kurve aufnehmen. Wenn nicht: als Limitation
benennen — das ist die ehrliche Antwort auf "wie habt ihr den Gap geschlossen".

Nebenbefund aus derselben Analyse (§4.3), bis heute nicht behoben: **servo9 hat einen konstanten
Sensor-Offset von −0.06** — die Input-Kalibrierung dieses Kanals stimmt nicht. Das ist der Zeigefinger-PIP,
also ein kompletter Detektionskanal mit systematischem Bias.

**Bestätigt am 2026-07-28 (GUI-Vergleich Leon):** In der Sim treffen die Fingerkuppen den Daumen beim
freien Schließen deutlich besser als auf der echten Hand. Die Sendeseite ist damit als Verdächtiger
belegt. `hardware/ar10.py::_to_servo` nutzt für **alle** Joints dieselbe Pulsspanne 4200–7700, während
`joint_input_calibration.json` pro Joint einen anderen gemessenen Geschlossen-Puls enthält:

| Sim-Joint | gemessener Geschlossen-Puls | Kommando bei q = 1.0 | Überfahrung |
|---|---|---|---|
| servo1 (Daumen-Flexion) | 5504 | 4200 | **1304** (37 % der Spanne) |
| servo0 (Daumen-Abduktion) | 4864 | 4200 | 664 |
| servo9 (Zeige-PIP) | 4608 | 4200 | 408 |
| servo2/4/6/8 (alle MCP) | 4288 | 4200 | 88 (2.5 %) |
| servo3/5/7 (PIP) | 4200 | 4200 | 0 |

q = 1.0 heißt also je nach Gelenk etwas anderes; wo überfahren wird, stallt der Servo und q_delta bleibt
dauerhaft stehen — genau der Phantom-Kontakt, den `servo_limits.yaml` im Kommentar vorhersagt. Für die
Precision-Arbeitsgelenke servo6/servo8 sind es nur 2.5 %, das allein erklärt den sichtbaren
Geometrie-Unterschied nicht; der Hauptverdacht bleibt die **Nichtlinearität Hub → Gelenkwinkel** (Firgelli
Linearaktuator über die 4-Stab-Mechanik gegen die lineare Winkelabbildung des URDF), die nirgends
gemessen oder randomisiert ist. Für Power ist servo1 zusätzlich hart betroffen: dort ist es eine eigene
Action-Gruppe und kann bis 1.0 gefahren werden — 37 % über den Anschlag.

Messung, die das entscheidet (~10 Min, ohne Objekt): Hand auf q = 0.0 / 0.25 / 0.5 / 0.75 / 1.0 fahren,
je ein Foto aus fester Perspektive, gegen das Sim-Rendering derselben q-Werte halten. Weichen nur die
Endpunkte ab, ist es die Pulsspanne; weichen die Zwischenwerte ab, ist es die Nichtlinearität. Vorher
`servo_limits.yaml` ausfüllen — die Datei steht bis heute vollständig auskommentiert auf den Defaults,
obwohl REAL_EVAL.md Schritt 0 sie verlangt, und der Maestro clippt still.

### 2.8 Die Podest-Strafe kann den Terminal-Reward dominieren

`w_pedestal = 0.05` wird **pro Step** abgezogen, solange eine Fingerkuppe das Podest berührt. Bei
`max_steps = 300` sind das im Extremfall −15, gegen `r_lift_success = +10`. Anders als der Kontaktterm
ist die Podest-Strafe nicht potentialbasiert, verändert also die optimale Policy tatsächlich. Die
Relation 5 : 1 zum Step-Penalty ist nirgends begründet. Ob das praktisch beißt, ist unbekannt:
`info["pedestal_hit"]` existiert, wird von `eval_sim.py` aber nicht mitgeschrieben. Ein Lauf mit
geloggtem `pedestal_hit` würde die Frage in einer Stunde klären.

### 2.9 Was "das RL lernt" ist enger als es klingt

Die Policy hat keine Stop-Aktion und kann die Episode nicht beenden. Wann ein Griff fertig ist,
entscheidet eine handgeschriebene Zustandsmaschine (`trigger_n` Finger für
`trigger_confirmation_steps` Schritte → `stabilization_steps` → Lift). Gelernt wird die
**Finger-Koordination während des Schließens**; der Griffabschluss ist skriptet. Das ist völlig legitim
— aber die Formulierung in README §2.1 ("Die RL Policy lernt dann ausschließlich, wie die Finger
geschlossen werden müssen, um das Objekt stabil gegen die Hand zu drücken und anzuheben") liest sich
weiter, als das Design hergibt. Präziser: die Policy lernt die Schließ-Koordination unter binärem
Kontakt-Feedback; Trigger und Lift sind fest verdrahtet.

### 2.10 Die Policies sind nicht degeneriert — aber Precision ist sehr klein

Ich habe beide Policies direkt abgetastet, weil "lernt sie überhaupt etwas?" die naheliegende
Gutachterfrage ist.

**Precision** (Obs ∈ ℝ⁴, Action ∈ {0,1}², also 4 mögliche Aktionen): über 96 abgetastete Zustände
schließt die Policy in 78 Fällen beide Gruppen, in 18 Fällen nicht — und die Ausnahmen sind
kontaktgetrieben und plausibel. Beispiele: bei `c_middle = 1, c_index = 0` stoppt sie durchgehend die
MCP-Gruppe und fährt nur noch PIP; bei beiden Kontakten und hohem q_mcp stoppt sie ganz (`0,0`). Das ist
kein Always-Close-Automat, sondern echtes Kontakt-Feedback. Gut.

Trotzdem bleibt die Frage im Raum, ob ein skriptierter Regler (beide Gruppen schließen, bis der Trigger
feuert) dasselbe leisten würde — bei 4 Zuständen und 4 Aktionen ist das keine unfaire Frage. **Die
wertvollste einzelne Ergänzung für die Arbeit ist eine Baseline-Vergleichszeile:** `eval_sim.py` einmal
mit einem Dummy-"Policy"-Objekt laufen lassen, dessen `predict()` konstant `close` zurückgibt, auf
demselben Benchmark und denselben Seeds. Das sind ~20 Zeilen und ein Evaluierungslauf, und es beantwortet
"warum RL" mit einer Zahl statt mit einem Absatz. Wenn die Policy den Skript-Regler schlägt, ist der
Ansatz belegt; wenn nicht, weißt du es vor der Verteidigung statt in ihr.

**Power** (Obs ∈ ℝ¹⁴, Action ∈ {0,1,2}×{0,1}⁹) reagiert ebenfalls auf die Kontakt-Bits, aber nicht
einheitlich (4000 zufällige Zustände, P(close) je Joint, aufgeschlüsselt nach dem Kontakt-Bit des
zugehörigen Fingers):

| Finger | Joint | Bit = 0 | Bit = 1 | Δ |
|---|---|---|---|---|
| pinky | servo2 | 0.94 | 0.78 | −0.16 |
| pinky | servo3 | 0.58 | 0.30 | −0.28 |
| ring | servo4 | 0.78 | 0.41 | −0.37 |
| ring | servo5 | 0.89 | 0.53 | −0.36 |
| middle | servo6 | 0.79 | 1.00 | **+0.21** |
| middle | servo7 | 0.56 | 0.88 | **+0.32** |
| index | servo8 | 0.35 | 0.26 | −0.10 |
| index | servo9 | 0.48 | 0.94 | **+0.46** |

Pinky und Ring stoppen bei Kontakt, Mittelfinger und Zeige-PIP schließen *stärker*. Beides ist einzeln
erklärbar (Stoppen = nicht weiter gegen das Objekt drücken; stärker Schließen = Kraftaufbau, die
Loading-Phase bei Westling & Johansson), aber die Aufteilung wirkt idiosynkratisch statt prinzipiell.
Mit einem Seed lässt sich nicht sagen, ob das eine Strategie oder Seed-Rauschen ist — siehe 2.1.
Vorbehalt: die abgetasteten Zustände sind gleichverteilt gezogen und enthalten auch unerreichbare
Konfigurationen; die Tabelle zeigt die Tendenz, nicht das Verhalten auf der tatsächlichen Trajektorie.

### 2.11 Die Sim-Erfolgsraten sind hoch — als Obergrenze lesen

Benchmark: precision 86.6 % gesamt (Median je Greifpunkt 1.0), power 90.4 % (Median 1.0). Formen-Grid:
93.7 % / 88.4 %. Das entsteht unter günstigen Bedingungen: festes 100-g-Objekt (2.4), Hand starr am
JOINT_FIXED-Constraint (maxForce 500) statt an einem nachgiebigen Arm, und `numSolverIterations = 200`.
Die Zahlen sind nicht falsch, aber sie sind eine Obergrenze — der Sim→Real-Abfall nächste Woche wird
entsprechend groß ausfallen und sollte so eingeordnet werden, nicht als Überraschung.

Nachvollziehbar niedrige Punkte, die sich im Labor bestätigen sollten: Teil 13/gp_002 (precision 0 %),
Teil 7/gp_001 (35 %), Teil 13/gp_001 (power 30 %).

---

## 2b. Sonderfall: "Precision greift flache Objekte nicht" — nachgemessen

Beobachtung aus der Praxis: bei Precision schließen die Finger nicht so, dass Zeige-/Mittelfinger-Tips
dem Daumen gegenüberliegen; flache Objekte scheitern. Geprüft, ob das ein Lern-, Kinematik- oder
Hardware-Problem ist.

**In Simulation macht die Policy es richtig.** Auf den flachen Benchmarkteilen (Teil 5: 12.7 × 12.7 ×
1.8 cm, Teil 9: 8.8 × 2.8 × 1.3 cm) liegen beim Trigger Kontakte an genau den erwarteten Links:
`fingertip3` (Mittel), `fingertip4` (Zeige) **und** `thumbtip` — der Tripod steht. Erfolgsraten
20/20 bzw. 19/20. Die Finger stehen dabei bei q_mcp ≈ 0.82, q_pip = 0.50 (am Cap).

Zwei mögliche Erklärungen habe ich geprüft und **beide ausgeschlossen**:

- *PIP-Cap zu eng*: nein. Die Kuppen-Oppositions-Konfiguration liegt bei q_pip ≈ 0.45, also unterhalb
  des Caps von 0.5.
- *Finger durchdringen den Daumen, weil die Sim-Hand ohne Self-Collision geladen wird*
  (`env._build_world` ruft `loadURDF` ohne Flags): plausibel, aber falsch. Mit gezielt aktivierter
  Kollision nur zwischen Fingerkuppen und Daumen bleiben die Ergebnisse unverändert (10/10, 9/10,
  10/10). Nebenbefund: mit *pauschaler* `URDF_USE_SELF_COLLISION` brechen dieselben Teile auf 0/10
  ein — das kommt aber von Link-Paaren innerhalb der 4-Stab-Mechanik, die sich im URDF konstruktiv
  überlappen, und ist ein Artefakt, kein physikalischer Befund.

**Die Ursache steht in den echten Trial-Logs vom 08.07.** (`artifacts/eval_results/real_precision_*.csv`).
Die 43 Trials sind klar bimodal — und keiner der beiden Modi trifft den richtigen Moment:

| Modus | n_steps | q_final (MCP / PIP) | Deutung |
|---|---|---|---|
| A — viel zu früh | 38–44 | 0.19–0.22 / 0.19–0.22 | Trigger nach ~0.8 s, Finger kaum bewegt |
| B — zu spät / gar nicht | 200–300 | 1.00 / 0.50 (beide am Anschlag) | Bit kippt erst bei voll geschlossener Hand oder nie |

Modus A passt exakt auf den dokumentierten Servo-Anlauf-Transienten
(`SENSOR_ANALYSIS_FINDINGS.md` §3.1: q_delta spikt in **jedem** Freilaufzyklus auf 0.050–0.072, Servos
laufen erst nach 11–15 Steps an). Gegenrechnung: 43 Steps × delta_norm 0.005 = 0.215 — genau der
geloggte q_final. Die Finger sind also durchgehend gefahren und wurden vom Trigger gestoppt, nicht vom
Objekt. Modus B passt auf den zu hohen Roh-Threshold (gemessener Rauschboden 0.078, siehe 2.5).

Flache Objekte sind der Worst Case für beide Modi: sie brauchen den längsten Schließweg bis zur echten
Blockierung, also trifft der Frühtrigger sie am härtesten — und wenn die Hand doch ganz schließt, wird
eine 1.3–1.8 cm dünne Platte eher herausgedrückt als gepinched.

**Entscheidend: alle diese Logs sind vom 2026-07-08, der ContactDetector wurde am 2026-07-10 committet
(87c4165).** Sie liefen also sämtlich auf dem alten Roh-Threshold-Pfad. Der Detektor adressiert genau
diese beiden Modi — `startup_mask_q` gegen A, die Baseline-Subtraktion (~0.012 statt 0.078) gegen B —
und ist **auf echter Hardware noch nie gelaufen**.

Konsequenz: Das ist kein RL-Befund. Ein Neu-Training würde nichts ändern, weil die Policy in
Simulation nachweislich das Richtige tut. Der erste Test nächste Woche muss sein:
`baseline_calibration.py` fahren und die flachen Teile mit aktivem Detektor wiederholen.

Zwei Punkte dabei im Auge behalten. Erstens: `startup_mask_q = 0.13` deckt die dokumentierten
Startup-Steps 0–26 ab, die realen Fehltrigger lagen aber bei q ≈ 0.19–0.22, also **darüber**. Vor den
Trials `python -m eval.test_detector_offline` laufen lassen und prüfen, ob in den Freilaufzyklen
oberhalb q = 0.13 noch Bits gesetzt werden; wenn ja, die Maske auf ~0.22 anheben. Zweitens: Falls die
flachen Teile auch mit sauberem Detektor scheitern, bleiben mechanische Ursachen, die die Sim nicht
abbildet — der reale Daumen ist gefedert (REAL_EVAL.md), die Sim modelliert servo1 als starres Gelenk
bei 0.0.

---

## 3. Antwort auf "war das ein wissenschaftlicher Weg, den Gap zu schließen?"

Ja, die *Methode* ist sauber: das Beobachtungssignal wurde bewusst auf eine Größe eingeschränkt, die in
beiden Domänen existiert; die verbleibende Diskrepanz wurde binarisiert weggeworfen statt gelernt; über
die Restunsicherheit (Threshold, Kontrollrate) wird randomisiert; und die Schwelle wird auf der echten
Hand pro Session kalibriert. Diese Kette ist argumentierbar und in der Sim-Seite sogar vermessen
(§1, Kontakt-Latenz).

Was fehlt, ist die *Messung des Restgaps*. Es gibt keinen direkten Sim↔Real-Vergleich desselben Signals
unter derselben Rampe. Die Sim-Seite ist charakterisiert, die Real-Seite ist charakterisiert — aber die
reale Seite hat keinen Ground-Truth-Kontaktzeitpunkt, weil kein Kontaktsensor existiert
(`SENSOR_ANALYSIS_FINDINGS.md` §6 benennt das selbst: "Positionssensorik misst Blockierung, nicht
Berührung"). Die Behauptung "das Bit kippt am selben physischen Ereignis" ruht damit auf der
Parallel-Kalibrierung, in der ein Mensch per Auge den Kontakt bestätigt.

Das ist verteidigbar, aber es ist eine **Annahme, keine Messung**, und sie sollte als solche benannt
werden. Formulierungsvorschlag für die Arbeit: der Sim2Real-Transfer beruht auf der Annahme, dass die
Kalibrierung das binäre Kontakt-Bit an dasselbe physikalische Ereignis (Blockierung des Fingers) bindet
wie im Training; ohne Kontaktsensorik auf der Hand ist diese Annahme nicht direkt überprüfbar, und die
Realevaluierung testet sie implizit mit.

---

## 4. Was vor dem Freeze zu entscheiden ist

Nach Aufwand-Nutzen sortiert:

1. **Skript-Baseline in Sim** (~1 h). Beantwortet "warum RL" mit einer Zahl. Höchster Nutzen pro Aufwand.
2. **Power-Detektor entscheiden** (2.5) — muss vor dem Laborbesuch stehen, sonst messen Power und
   Precision nicht dasselbe.
3. **Reale Benchmarkteile wiegen** (2.4) — fünf Minuten im Labor, entscheidet über die Gültigkeit des
   Sim↔Real-Vergleichs.
4. **3–4 Zusatz-Seeds im Hintergrund** (2.1) — läuft parallel zur Realevaluierung, fasst die
   eingefrorenen Policies nicht an.
5. **`run_meta.yaml` je Modell + Logs sichern** (2.2), solange die Läufe lokal noch existieren.
6. **LaTeX-Doc auf den Codestand ziehen** (2.3) — steht ohnehin als Stretch-Goal auf dem Board, ist
   aber kein Komfort-Thema: das Dokument beschreibt aktuell den falschen Algorithmus.
7. **`pedestal_hit` in `eval_sim.py` mitloggen** (2.8) und einmal auswerten.

Ein Neu-Training ist aus dieser Prüfung heraus **nicht zwingend**. Der einzige Befund, der es
rechtfertigen würde, ist die Detektor-Dynamik-Lücke (2.6) — und die lässt sich auch als Limitation
tragen, solange die Realevaluierung gezielt darauf achtet.
