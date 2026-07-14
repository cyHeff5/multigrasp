# Raw-ADC-Aufzeichnung: Zeichnet rohe ADC-Werte (0-1023) waehrend einer
# CLOSE-Rampe auf. Zeigt die echte Sensor-Aufloesung ohne Normalisierung.
#
# Usage:
#   python -m eval.raw_adc_test --config configs/precision.yaml --port COM4
from __future__ import annotations

import argparse
import csv
import datetime
import statistics
import time
from pathlib import Path

import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "artifacts" / "analysis"


def _watched_joints(cfg: dict) -> list[str]:
    joints = []
    for jlist in cfg["finger_joints"].values():
        for j in jlist:
            if j not in joints:
                joints.append(j)
    return joints


def _step_dt(cfg: dict) -> float:
    return float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])


def run(cfg: dict, port: str) -> Path:
    joints  = _watched_joints(cfg)
    j_idxs  = [CONTROL_JOINTS.index(j) for j in joints]
    dt      = _step_dt(cfg)
    rate    = float(cfg["action"]["delta_norm"])
    caps    = cfg["action"].get("pip_caps", {})
    max_close = int(1.0 / rate) + 20

    ar10 = AR10Interface(com_port=port)

    # Pregrasp
    q_target = [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]
    ar10.send_q_target(list(q_target))
    time.sleep(1.5)

    rows: list[dict] = []

    # CLOSE-Rampe
    t0 = time.perf_counter()
    for k in range(max_close):
        for j, idx in zip(joints, j_idxs):
            cap = caps.get(j, 1.0)
            q_target[idx] = min(cap, q_target[idx] + rate)
        ar10.send_q_target(list(q_target))

        # Raw ADC lesen (direkt ueber _read_input_channel)
        row: dict = {"step": k}
        for j, idx in zip(joints, j_idxs):
            if idx in ar10._input_cal:
                cal = ar10._input_cal[idx]
                raw_adc = ar10._read_input_channel(cal["input_channel"])
                q_meas  = ar10._normalize_input(raw_adc, cal["open_real"], cal["closed_real"])
                row[f"{j}_raw_adc"]    = raw_adc
                row[f"{j}_q_target"]   = round(q_target[idx], 5)
                row[f"{j}_q_measured"] = round(q_meas, 5)
                row[f"{j}_q_delta"]    = round(q_target[idx] - q_meas, 5)
        rows.append(row)

        if k % 20 == 0:
            adcs = "  ".join(f"{j}:{row.get(f'{j}_raw_adc', '?')}" for j in joints)
            print(f"\r  step={k:3d}  {adcs}   ", end="", flush=True)

        all_closed = all(
            q_target[idx] >= caps.get(j, 1.0) - 1e-9
            for j, idx in zip(joints, j_idxs)
        )
        if all_closed:
            break

        pause = t0 + (k + 1) * dt - time.perf_counter()
        if pause > 0:
            time.sleep(pause)

    # HOLD 30 Steps (statisch, fuer Noise-Messung am Cap)
    print(f"\n  HOLD (30 Steps am Cap)...")
    for k in range(30):
        row = {"step": len(rows)}
        for j, idx in zip(joints, j_idxs):
            if idx in ar10._input_cal:
                cal = ar10._input_cal[idx]
                raw_adc = ar10._read_input_channel(cal["input_channel"])
                q_meas  = ar10._normalize_input(raw_adc, cal["open_real"], cal["closed_real"])
                row[f"{j}_raw_adc"]    = raw_adc
                row[f"{j}_q_target"]   = round(q_target[idx], 5)
                row[f"{j}_q_measured"] = round(q_meas, 5)
                row[f"{j}_q_delta"]    = round(q_target[idx] - q_meas, 5)
        rows.append(row)
        time.sleep(dt)

    # Hand oeffnen
    ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
    time.sleep(0.5)
    ar10.close()

    # ADC-Statistik
    print(f"\n  ADC-Statistik:")
    for j in joints:
        adcs = [r[f"{j}_raw_adc"] for r in rows if f"{j}_raw_adc" in r]
        if adcs:
            mn, mx = min(adcs), max(adcs)
            unique = len(set(adcs))
            print(f"    {j}: range [{mn}-{mx}] = {mx-mn} ADC-Stufen, "
                  f"{unique} unique Werte in {len(adcs)} Reads")

    # CSV speichern
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(cfg.get("_config_path", "precision")).stem

    out = _OUT_DIR / f"raw_adc_{stem}_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [raw-adc] -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw-ADC-Aufzeichnung waehrend CLOSE-Rampe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port",   required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config

    print(f"[raw-adc] Joints: {_watched_joints(cfg)}")
    input("\nHand ist frei (KEIN Objekt)? Enter -> Start ...")
    run(cfg, args.port)


if __name__ == "__main__":
    main()
