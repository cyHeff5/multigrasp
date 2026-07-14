# Positions-abhaengiger Noise-Floor: Misst q_delta-Rauschen bei verschiedenen
# statischen Handpositionen (q=0.0, 0.1, ..., 0.9).
#
# Zeigt ob der Baseline-q_delta positionsabhaengig ist — wichtig fuer die
# Wahl des Kontakt-Thresholds.
#
# Usage:
#   python -m eval.position_noise_test --config configs/precision.yaml --port COM4
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


def run(cfg: dict, port: str, positions: list[float] | None = None,
        samples: int = 100) -> Path:
    joints  = _watched_joints(cfg)
    j_idxs  = [CONTROL_JOINTS.index(j) for j in joints]
    dt      = _step_dt(cfg)
    caps    = cfg["action"].get("pip_caps", {})

    if positions is None:
        positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ar10 = AR10Interface(com_port=port)

    all_rows: list[dict] = []
    summary: list[dict] = []

    for pos in positions:
        print(f"\n  Position q={pos:.1f}")

        # Alle watched joints auf pos setzen (mit caps beachten)
        q_target = [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]
        for j, idx in zip(joints, j_idxs):
            cap = caps.get(j, 1.0)
            q_target[idx] = min(cap, pos)
        ar10.send_q_target(list(q_target))
        time.sleep(1.0)  # Einschwingen

        measurements: dict[str, list[float]] = {j: [] for j in joints}

        for s in range(samples):
            q_meas = ar10.read_q_measured()
            row = {"position": pos, "sample": s}
            for j, idx in zip(joints, j_idxs):
                qt = q_target[idx]
                qm = q_meas[idx]
                qd = qt - qm
                measurements[j].append(qd)
                row[f"{j}_q_target"]   = round(qt, 5)
                row[f"{j}_q_measured"] = round(qm, 5)
                row[f"{j}_q_delta"]    = round(qd, 5)
            all_rows.append(row)
            time.sleep(dt)

        row_summary = {"position": pos}
        for j in joints:
            vals = measurements[j]
            m  = statistics.fmean(vals)
            s_ = statistics.pstdev(vals)
            row_summary[f"{j}_mean"]  = round(m, 5)
            row_summary[f"{j}_std"]   = round(s_, 5)
            row_summary[f"{j}_min"]   = round(min(vals), 5)
            row_summary[f"{j}_max"]   = round(max(vals), 5)
            row_summary[f"{j}_3sig"]  = round(abs(m) + 3.0 * s_, 5)
            print(f"    {j}: mean={m:+.5f}  std={s_:.5f}  3sig-thr={abs(m)+3*s_:.4f}")
        summary.append(row_summary)

    # Hand oeffnen
    ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
    time.sleep(0.5)
    ar10.close()

    # CSV speichern
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(cfg.get("_config_path", "precision")).stem

    out = _OUT_DIR / f"position_noise_{stem}_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    out2 = _OUT_DIR / f"position_noise_summary_{stem}_{ts}.csv"
    with out2.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n  [position-noise] Rohdaten -> {out}")
    print(f"  [position-noise] Summary  -> {out2}")
    return out2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Positions-abhaengiger Noise-Floor der AR10-Sensoren.")
    parser.add_argument("--config",   required=True)
    parser.add_argument("--port",     required=True)
    parser.add_argument("--samples",  type=int, default=100)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config

    print(f"[position-noise] Joints: {_watched_joints(cfg)}")
    input("\nHand ist frei (KEIN Objekt)? Enter -> Start ...")
    run(cfg, args.port, samples=args.samples)


if __name__ == "__main__":
    main()
