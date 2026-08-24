# Empirische Maestro-Limit-Probe: welche Puls-Targets setzt der Maestro
# WIRKLICH um? Der Maestro clippt Targets still auf seine gespeicherten
# Kanal-Limits. Die Werkssettings (ar10-doku/maestro-werkssettings-AR10v4.txt)
# haben z. B. ch10 min=4544 und ch11 min=5056 — unser Code kommandiert bis 4200.
# Traegt das Labor-Board noch Werkssettings, endet servo8 bei q~0.90 und servo9
# bei q~0.755, ohne dass man es dem Kommando ansieht.
#
# Vorgehen: je Joint (alle anderen offen) eine absteigende Puls-Treppe
# hi -> lo fahren, pro Stufe settlen und den rohen ADC lesen. Wo der ADC einer
# linearen Puls-ADC-Gerade folgt, wird das Target umgesetzt; flache Enden =
# Clipping (oder mechanischer Anschlag). Ausgabe: CSV aller Samples +
# artifacts/calibration/servo_limits_suggested.yaml (NICHT automatisch als
# servo_limits.yaml uebernommen — erst pruefen, dann kopieren).
#
# Klaert nebenbei das "Oeffnen bleibt bei q~0.19 haengen"-Problem (Run 13:36
# vom 08.07.): bleibt der ADC am oberen Puls-Ende flach, clippt der Maestro
# auch beim Oeffnen.
#
# Usage:
#   python -m eval.maestro_limit_probe --port COM4        # echte Hand, ~5 Min
#   python -m eval.maestro_limit_probe                    # Mock (Logik-Test)
from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

import numpy as np

from hardware.ar10 import AR10Interface, _CHANNELS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "artifacts" / "analysis"
_SUGG_FILE = _REPO_ROOT / "artifacts" / "calibration" / "servo_limits_suggested.yaml"

_OPEN_PULSE = 7700   # "sicher offen" fuer die nicht geprobten Joints

# Werksseitige Maestro-Limits (Viertel-µs) aus AR10v4 settings.txt,
# Sim-Joint-Reihenfolge servo0..servo9 (Kanal-Mapping siehe hardware/ar10.py).
_FACTORY_MIN = {ch: v for ch, v in zip(range(10, 20),
                [4544, 5056, 4288, 3968, 4288, 3968, 4288, 3968, 4288, 3968])}
_FACTORY_MAX = {ch: 8000 for ch in range(10, 20)}


class _MockHand:
    # Simuliert Maestro-Clipping auf die Werkslimits + linearen Poti (mit
    # Rauschen), damit die Auswertelogik ohne Hardware testbar ist.
    def __init__(self, input_cal: dict):
        self._cal = input_cal
        self._pulse = {ch: _OPEN_PULSE for ch in range(10, 20)}
        self._rng = np.random.default_rng(7)

    def set_pulses(self, pulses_by_channel: dict[int, int]) -> None:
        for ch, pulse in pulses_by_channel.items():
            self._pulse[ch] = max(_FACTORY_MIN[ch], min(_FACTORY_MAX[ch], int(pulse)))

    def read_adc(self, joint_idx: int) -> float:
        cal   = self._cal[joint_idx]
        ch    = _CHANNELS[joint_idx]
        # Poti linear in der (geclippten) Pulsweite: 7700 -> open_real, 4200 -> closed_real.
        frac  = (7700 - self._pulse[ch]) / (7700 - 4200)
        adc   = cal["open_real"] + frac * (cal["closed_real"] - cal["open_real"])
        return float(adc + self._rng.normal(0.0, 1.2))


class _RealHand:
    def __init__(self, port: str, lo: int, hi: int):
        # Weite eigene Limits, damit unser Code-Clip die Probe nicht verfaelscht.
        self.ar10 = AR10Interface(com_port=port,
                                  servo_min=[lo] * 10, servo_max=[hi] * 10)
        self._targets = [_OPEN_PULSE] * 10   # Index = Kanal - 10

    def set_pulses(self, pulses_by_channel: dict[int, int]) -> None:
        for ch, pulse in pulses_by_channel.items():
            self._targets[ch - 10] = int(pulse)
        self.ar10._set_all_channel_targets(list(self._targets))

    def read_adc(self, joint_idx: int) -> float:
        cal = self.ar10._input_cal[joint_idx]
        vals = [self.ar10._read_input_channel(cal["input_channel"]) for _ in range(3)]
        return float(sum(vals) / len(vals))

    def close(self):
        self.set_pulses({ch: _OPEN_PULSE for ch in range(10, 20)})
        time.sleep(2.0)
        self.ar10.close()


def _detect_effective_range(pulses: np.ndarray, adcs: np.ndarray) -> tuple[int, int, float]:
    # Zwei-Knickpunkt-Fit: das wahre Modell ist konstant-linear-konstant
    # (Clip unten, aktiver Bereich, Clip oben). Fuer jedes Segment [i..k] wird
    # eine Gerade gefittet, die Enden konstant fortgesetzt und die Summe der
    # Fehlerquadrate minimiert. Ein einfacher Innen-Fit kippt bei grossen
    # Clip-Bereichen (Werks-min 5056 auf ch11!) — dieser hier nicht.
    # Rueckgabe: (effektives Min, effektives Max, RMSE des besten Fits).
    order  = np.argsort(pulses)
    ps, ad = pulses[order].astype(float), adcs[order].astype(float)
    n      = len(ps)
    best   = (None, None, np.inf)
    for i in range(n - 3):
        for k in range(i + 3, n):
            coef = np.polyfit(ps[i:k + 1], ad[i:k + 1], 1)
            pred = np.polyval(coef, ps)
            pred[:i]     = pred[i]
            pred[k + 1:] = pred[k]
            sse = float(np.sum((ad - pred) ** 2))
            if sse < best[2]:
                best = (i, k, sse)
    i, k, sse = best
    rmse = (sse / n) ** 0.5
    return int(ps[i]), int(ps[k]), rmse


def main() -> None:
    ap = argparse.ArgumentParser(description="Empirische Maestro-Limit-Probe.")
    ap.add_argument("--port",   default=None, help="COM-Port; ohne = Mock-Modus.")
    ap.add_argument("--lo",     type=int, default=4000)
    ap.add_argument("--hi",     type=int, default=8000)
    ap.add_argument("--step",   type=int, default=250)
    ap.add_argument("--settle", type=float, default=1.2, help="Haltezeit je Stufe (s).")
    args = ap.parse_args()

    if args.port:
        hand = _RealHand(args.port, args.lo - 100, args.hi + 100)
        input_cal = hand.ar10._input_cal
        print(f"[limit-probe] ECHTE HAND an {args.port} — Arbeitsraum freihalten!")
    else:
        input_cal = AR10Interface()._input_cal
        hand = _MockHand(input_cal)
        print("[limit-probe] Mock-Modus (simulierte Werkslimits).")

    sweep = list(range(args.hi, args.lo - 1, -args.step))
    rows: list[dict] = []
    results: dict[int, tuple[int, int]] = {}

    try:
        for j in range(10):
            if j not in input_cal:
                print(f"  servo{j}: keine Input-Kalibrierung — uebersprungen")
                continue
            ch = _CHANNELS[j]
            # Alle anderen Joints offen halten, Probe-Joint an den Startpunkt.
            hand.set_pulses({c: _OPEN_PULSE for c in range(10, 20)})
            hand.set_pulses({ch: sweep[0]})
            time.sleep(3.0 * (args.settle / 1.2) if args.port else 0)

            ps, ad = [], []
            for pulse in sweep:
                hand.set_pulses({ch: pulse})
                if args.port:
                    time.sleep(args.settle)
                adc = hand.read_adc(j)
                ps.append(pulse); ad.append(adc)
                rows.append({"joint": f"servo{j}", "channel": ch,
                             "pulse": pulse, "adc": round(adc, 1)})
            eff_min, eff_max, tol = _detect_effective_range(np.array(ps, float), np.array(ad, float))
            results[j] = (eff_min, eff_max)
            fmin, fmax = _FACTORY_MIN[ch], _FACTORY_MAX[ch]
            note = []
            if eff_min > args.lo + args.step:
                note.append(f"CLIP unten (Werk: {fmin})")
            if eff_max < args.hi - args.step:
                note.append(f"CLIP oben (Werk: {fmax})")
            print(f"  servo{j} (ch{ch}): effektiv [{eff_min}, {eff_max}]  "
                  f"tol={tol:.1f}  {'  '.join(note) if note else 'kein Clipping im Probebereich'}")
            # Joint wieder oeffnen, bevor der naechste dran ist.
            hand.set_pulses({ch: _OPEN_PULSE})
            if args.port:
                time.sleep(2.0)
    finally:
        if args.port:
            hand.close()

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _OUT_DIR / f"maestro_limit_probe_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["joint", "channel", "pulse", "adc"])
        w.writeheader(); w.writerows(rows)

    # Vorschlag fuer servo_limits.yaml (Sim-Joint-Reihenfolge servo0..servo9).
    sugg_min = [results.get(j, (4200, 7700))[0] for j in range(10)]
    sugg_max = [min(results.get(j, (4200, 7700))[1], 7700) for j in range(10)]
    _SUGG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _SUGG_FILE.open("w", encoding="utf-8") as f:
        f.write(f"# Empirische Maestro-Limit-Probe {ts} (eval/maestro_limit_probe.py)\n")
        f.write(f"# Rohdaten: {csv_path.name}\n")
        f.write("# Pruefen und dann als servo_limits.yaml uebernehmen.\n")
        f.write(f"servo_min: {sugg_min}\n")
        f.write(f"servo_max: {sugg_max}\n")
    print(f"\n[limit-probe] Samples -> {csv_path}")
    print(f"[limit-probe] Vorschlag -> {_SUGG_FILE}  (pruefen, dann nach "
          f"artifacts/calibration/servo_limits.yaml kopieren)")


if __name__ == "__main__":
    main()
