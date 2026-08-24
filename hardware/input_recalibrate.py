# Input-Kalibrierung der AR10 neu aufnehmen (ersetzt joint_input_calibration.json).
#
# Warum: servo9 hat einen gemessenen Gain-Fehler von -0.12/q (SENSOR_ANALYSIS_
# FINDINGS.md §8.5) — sechsmal ausserhalb der Trainings-Randomisierung
# (sensor_gain_err 0.02). Die alte Kalibrierung ist vom 2026-02-19.
#
# WICHTIG: erst eval/maestro_limit_probe.py fahren und artifacts/calibration/
# servo_limits.yaml fuellen — die Kalibrierung faehrt auf q=0/1, und ohne
# korrekte Limits kalibriert sie gegen den Clip statt gegen die echte Endlage.
#
# Ablauf je Joint (alle anderen offen): oeffnen -> settlen -> ADC lesen,
# schliessen -> settlen -> ADC lesen, plus Zwischenpunkte fuer einen
# Linearitaets-Report (R² der Puls-ADC-Geraden). Beim Daumen-Tip (servo1) wird
# wie beim Hersteller der Daumen-Lower (servo0) halb angestellt, damit die
# Spitze nicht ans Gehaeuse stoesst (vgl. ar10-doku/AR10_calibrate.py).
#
# Usage:
#   python -m hardware.input_recalibrate --port COM4    # echte Hand, ~6 Min
#   python -m hardware.input_recalibrate                # Mock (Logik-Test)
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import time
from pathlib import Path

import numpy as np

from hardware.ar10 import AR10Interface, _CHANNELS, _OLD_CHANNELS

_CAL_FILE   = Path(__file__).resolve().parent / "joint_input_calibration.json"
_MOCK_OUT   = Path(__file__).resolve().parent.parent / "artifacts" / "analysis" / "joint_input_calibration_MOCK.json"
_MID_POINTS = [0.25, 0.5, 0.75]
_THUMB_TIP_JOINT   = 1     # sim servo1 (ch19): Daumen-Tip
_THUMB_LOWER_HOLD  = 0.63  # servo0 dabei auf ~Puls 5500 (Hersteller-Rezept)


def _read_adc(ar10: AR10Interface, joint: int, n: int = 8, mock_q: float | None = None) -> float:
    cal = ar10._input_cal[joint]
    if ar10._usb is None:
        # Mock: linearer Poti mit leichtem Gain-Fehler + Rauschen, damit die
        # Auswertung ohne Hardware testbar ist.
        rng  = np.random.default_rng(joint)
        gain = 1.0 + 0.03 * ((joint % 3) - 1)
        adc  = cal["open_real"] + gain * (mock_q or 0.0) * (cal["closed_real"] - cal["open_real"])
        return float(adc + rng.normal(0.0, 1.0))
    vals = [ar10._read_input_channel(cal["input_channel"]) for _ in range(n)]
    return float(sum(vals) / len(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="AR10 Input-Kalibrierung neu aufnehmen.")
    ap.add_argument("--port",   default=None, help="COM-Port; ohne = Mock-Modus.")
    ap.add_argument("--settle-end", type=float, default=4.0,
                    help="Haltezeit an den Endlagen (voller Hub ~2-3 s).")
    ap.add_argument("--settle-mid", type=float, default=2.0)
    args = ap.parse_args()

    ar10 = AR10Interface(com_port=args.port)
    mock = args.port is None
    if mock:
        print("[recal] Mock-Modus — Ergebnis geht nach", _MOCK_OUT.name)
    else:
        print(f"[recal] ECHTE HAND an {args.port} — Arbeitsraum freihalten!")

    def drive(joint: int, q: float) -> None:
        targets = [0.0] * 10
        if joint == _THUMB_TIP_JOINT:
            targets[0] = _THUMB_LOWER_HOLD
        targets[joint] = q
        ar10.send_q_target(targets)

    new_joints: dict = {}
    try:
        for j in range(10):
            if j not in ar10._input_cal:
                print(f"  servo{j}: kein Eintrag in alter Kalibrierung — uebersprungen")
                continue
            old_cal = ar10._input_cal[j]
            ch      = _CHANNELS[j]
            old_j   = _OLD_CHANNELS.index(ch)

            drive(j, 0.0)
            time.sleep(0.0 if mock else args.settle_end)
            adc_open = _read_adc(ar10, j, mock_q=0.0)

            mids = []
            for q in _MID_POINTS:
                drive(j, q)
                time.sleep(0.0 if mock else args.settle_mid)
                mids.append((q, _read_adc(ar10, j, n=4, mock_q=q)))

            drive(j, 1.0)
            time.sleep(0.0 if mock else args.settle_end)
            adc_closed = _read_adc(ar10, j, mock_q=1.0)

            drive(j, 0.0)
            time.sleep(0.0 if mock else 2.0)

            # Linearitaets-Report: R² der Geraden durch alle 5 Punkte.
            qs  = np.array([0.0] + [m[0] for m in mids] + [1.0])
            ads = np.array([adc_open] + [m[1] for m in mids] + [adc_closed])
            coef = np.polyfit(qs, ads, 1)
            ss_res = float(np.sum((ads - np.polyval(coef, qs)) ** 2))
            ss_tot = float(np.sum((ads - ads.mean()) ** 2)) or 1e-9
            r2 = 1.0 - ss_res / ss_tot

            # Gain-Aenderung relativ zur alten Kalibrierung: altes q am neuen
            # Geschlossen-Punkt (1.0 = keine Aenderung).
            denom_old = old_cal["closed_real"] - old_cal["open_real"]
            old_q_at_closed = ((adc_closed - old_cal["open_real"]) / denom_old
                               if denom_old else float("nan"))
            print(f"  servo{j} (ch{ch}, alt joint {old_j}): "
                  f"open {old_cal['open_real']:.0f}->{adc_open:.0f}  "
                  f"closed {old_cal['closed_real']:.0f}->{adc_closed:.0f}  "
                  f"R²={r2:.4f}  altes q@closed={old_q_at_closed:+.3f}")

            new_joints[str(old_j)] = {
                "joint":         old_j,
                "input_channel": old_cal["input_channel"],
                "opened": {"target": ar10._to_servo(0.0, j), "mapped_input": round(adc_open, 1)},
                "closed": {"target": ar10._to_servo(1.0, j), "mapped_input": round(adc_closed, 1)},
                "linearity_r2":  round(r2, 5),
                "mid_points":    [{"q": q, "adc": round(a, 1)} for q, a in mids],
            }
    finally:
        ar10.send_q_target([0.0] * 10)
        if not mock:
            time.sleep(2.0)
        ar10.close()

    out = {
        "created_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by":  "hardware/input_recalibrate.py",
        "servo_limits_min": ar10._servo_min,
        "servo_limits_max": ar10._servo_max,
        "joints": new_joints,
    }
    if mock:
        _MOCK_OUT.parent.mkdir(parents=True, exist_ok=True)
        _MOCK_OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\n[recal] Mock-Ergebnis -> {_MOCK_OUT}")
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = _CAL_FILE.with_suffix(f".json.bak_{ts}")
    shutil.copy2(_CAL_FILE, backup)
    _CAL_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[recal] Alte Kalibrierung -> {backup.name}")
    print(f"[recal] Neue Kalibrierung -> {_CAL_FILE}")
    print("[recal] Danach PFLICHT: Baselines neu fahren (eval/baseline_calibration.py), "
          "die alten passen nicht mehr zur neuen q-Skala.")


if __name__ == "__main__":
    main()
