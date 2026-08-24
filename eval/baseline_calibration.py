# Session-Kalibrierung der q_delta-Freilauf-Baseline fuer den ContactDetector.
#
# Faehrt N CLOSE-Zyklen OHNE Objekt mit derselben Rampe, die die Policy maximal
# faehrt (delta_norm, pip_caps), und speichert pro Joint mean/std des q_delta
# ueber das q_target-Grid plus die CUSUM-Alarmschwelle nach
# artifacts/calibration/qdelta_baseline.yaml.
#
# VOR JEDER EVAL-SESSION NEU FAHREN (~2 min bei 10 Zyklen): die Baseline ist
# eine Eigenschaft der Servos/Sensorik am Messtag; Drift ueber Tage ist
# ungetestet. Herleitung: artifacts/analysis/SENSOR_ANALYSIS_FINDINGS.md.
#
# Usage:
#   python -m eval.baseline_calibration --config configs/precision.yaml --port COM4
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import yaml

from hardware.ar10             import AR10Interface
from hardware.contact_detector import build_baseline
from sim.hand                  import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT = Path(__file__).resolve().parent.parent

_SETTLE_S = 1.0   # Pause nach Oeffnen, damit jeder Zyklus sauber bei q=0 startet


def _watched_joints(cfg: dict) -> list[str]:
    joints: list[str] = []
    for jlist in cfg["finger_joints"].values():
        for j in jlist:
            if j not in joints:
                joints.append(j)
    return joints


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="q_delta-Baseline-Kalibrierung (Freilauf) fuer den ContactDetector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port",   default=None)
    parser.add_argument("--cycles", type=int, default=10,
                        help="Freilauf-CLOSE-Zyklen (default 10, min 3).")
    parser.add_argument("--out",    default=None,
                        help="Zielpfad (default: contact_detector.baseline_file aus der Config).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    det_cfg = cfg["contact_detector"]

    dt        = float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])
    joints    = _watched_joints(cfg)
    j_idxs    = [CONTROL_JOINTS.index(j) for j in joints]
    rate      = float(cfg["action"]["delta_norm"])
    caps      = cfg["action"].get("pip_caps", {})
    max_close = int(1.0 / rate) + 20

    out_path = Path(args.out) if args.out else _REPO_ROOT / det_cfg["baseline_file"]

    print(f"[baseline-cal] Joints: {joints}   rate={rate}  caps={caps}")
    print(f"[baseline-cal] step_dt={dt*1000:.1f} ms   Zyklen: {args.cycles}")
    print(f"[baseline-cal] Ziel: {out_path}")

    ar10 = AR10Interface(com_port=args.port)
    if args.port is not None:
        from eval.policy_runner import watched_joint_indices
        ar10.assert_input_calibration(watched_joint_indices(cfg))

    input("\nHand ist FREI (kein Objekt, nichts im Arbeitsraum)? Enter -> Start ...")

    # per_joint[j] = Liste pro Zyklus von (q_target-Array, q_delta-Array)
    per_joint: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {j: [] for j in joints}

    for cycle in range(args.cycles):
        q_target = _pregrasp_q()
        ar10.send_q_target(list(q_target))
        time.sleep(_SETTLE_S)

        qt_log: dict[str, list[float]] = {j: [] for j in joints}
        qd_log: dict[str, list[float]] = {j: [] for j in joints}

        def _log_static(n_steps: int) -> None:
            # Target steht — liefert die STATIK-Stuetzpunkte der Baseline
            # (Settle-Offset der Sensorik, nicht ratenabhaengig; noetig fuer
            # die Raten-Skalierung des Detektors). build_baseline() erkennt
            # stehende Samples selbst ueber steps_since_move.
            t0s = time.perf_counter()
            for ks in range(n_steps):
                pause_s = t0s + (ks + 1) * dt - time.perf_counter()
                if pause_s > 0:
                    time.sleep(pause_s)
                q_meas_s = ar10.read_q_measured()
                for j, idx in zip(joints, j_idxs):
                    qt_log[j].append(q_target[idx])
                    qd_log[j].append(q_target[idx] - q_meas_s[idx])

        _log_static(35)          # Statik bei q=0 (offen)

        t0 = time.perf_counter()
        for k in range(max_close):
            for j, idx in zip(joints, j_idxs):
                q_target[idx] = min(caps.get(j, 1.0), q_target[idx] + rate)
            ar10.send_q_target(list(q_target))

            # WICHTIG: erst den Step-Slot abwarten, DANN lesen — exakt wie der
            # Policy-Runner (liest am Anfang des naechsten Steps, ~dt nach dem
            # Senden). Vorher wurde direkt nach dem Senden gelesen; der Servo
            # holt waehrend dt ~einen Step auf, die Baseline lag dadurch
            # systematisch ~delta_norm (0.005) zu hoch — 40% des
            # Residuum-Budgets (Befund 2026-08-24).
            pause = t0 + (k + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)

            q_meas = ar10.read_q_measured()
            for j, idx in zip(joints, j_idxs):
                qt_log[j].append(q_target[idx])
                qd_log[j].append(q_target[idx] - q_meas[idx])

            deltas = "  ".join(f"{j}:{qd_log[j][-1]:+.4f}" for j in joints)
            print(f"\r  Zyklus {cycle+1}/{args.cycles}  step={k:3d}  {deltas}   ",
                  end="", flush=True)

            if all(q_target[idx] >= caps.get(j, 1.0) - 1e-9
                   for j, idx in zip(joints, j_idxs)):
                break

        _log_static(45)          # Statik am Cap (jeder Joint an SEINEM Cap)

        for j in joints:
            per_joint[j].append((np.array(qt_log[j]), np.array(qd_log[j])))
        print()

    ar10.send_q_target(_pregrasp_q())
    time.sleep(1.0)
    ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
    time.sleep(1.0)
    ar10.close()

    baseline = build_baseline(
        per_joint,
        startup_mask_q=float(det_cfg["startup_mask_q"]),
        settle_steps=int(det_cfg["settle_steps"]),
        cusum_drift=float(det_cfg["cusum_drift"]),
        cusum_margin=float(det_cfg["cusum_margin"]),
        meta={
            "created":    datetime.datetime.now().isoformat(timespec="seconds"),
            "config":     Path(args.config).stem,
            "delta_norm": rate,
            "read_timing": "pre_next_send",
            "pip_caps":   dict(caps),
            "n_cycles":   args.cycles,
            "step_dt_ms": round(dt * 1000, 2),
        },
    )
    baseline.save(out_path)

    print(f"\n[baseline-cal] Baseline -> {out_path}")
    print(f"\n-- Zusammenfassung (Grid-Punkte ab startup_mask_q="
          f"{det_cfg['startup_mask_q']}) --")
    for j in joints:
        grid = np.array(baseline.to_dict()["joints"][j]["q"])
        stds = np.array(baseline.to_dict()["joints"][j]["std"])
        sel  = grid >= float(det_cfg["startup_mask_q"])
        eff  = np.maximum(float(det_cfg["residual_min"]),
                          float(det_cfg["sigma_k"]) * stds[sel])
        print(f"  {j}: std median={np.median(stds[sel]):.4f}  "
              f"effektiver Threshold median={np.median(eff):.4f}  "
              f"max={eff.max():.4f}  CUSUM-Alarm={baseline.cusum_alarm(j):.4f}")


if __name__ == "__main__":
    main()
