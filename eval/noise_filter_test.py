# Vergleicht Sensor-Rauschen mit verschiedenen EMA-alpha-Werten
# unter realen Bedingungen (CLOSE-Rampe wie beim Greifen).
#
# Pro Einstellung: Hand oeffnet, faehrt CLOSE-Rampe, oeffnet wieder.
# Zeichnet q_delta pro Step auf und vergleicht Rauschen (std) + Glaettung.
#
# Usage:
#   python -m eval.noise_filter_test --config configs/precision.yaml --port COM4
#   python -m eval.noise_filter_test --config configs/precision.yaml --port COM4 --alphas 1.0 0.5 0.3
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


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def _step_dt(cfg: dict) -> float:
    return float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensor-Rauschen mit verschiedenen EMA-alpha vergleichen (CLOSE-Rampe).")
    parser.add_argument("--config",  required=True)
    parser.add_argument("--port",    required=True)
    parser.add_argument("--alphas",  type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2],
                        help="EMA-alpha-Werte (1.0=kein Filter, kleiner=staerker). Default: 1.0 0.5 0.3 0.2")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    joints    = _watched_joints(cfg)
    j_idxs    = [CONTROL_JOINTS.index(j) for j in joints]
    dt        = _step_dt(cfg)
    rate      = float(cfg["action"]["delta_norm"])
    caps      = cfg["action"].get("pip_caps", {})
    max_close = int(1.0 / rate) + 20

    print(f"[noise-test] Joints: {joints}")
    print(f"[noise-test] step_dt={dt*1000:.1f} ms ({1/dt:.0f} Hz)")
    print(f"[noise-test] CLOSE-Rampe: rate={rate}, max_steps={max_close}")
    print(f"[noise-test] EMA-alpha zu testen: {args.alphas}")
    input("\nHand ist frei (KEIN Objekt)? Enter -> Start ...")

    all_rows: list[dict] = []
    summary: list[dict] = []

    for alpha in args.alphas:
        print(f"\n{'='*60}")
        print(f"  EMA alpha = {alpha}")
        print(f"{'='*60}")

        ema = alpha if alpha < 1.0 else None
        ar10 = AR10Interface(com_port=args.port, ema_alpha=ema)

        # Pregrasp
        q_target = _pregrasp_q()
        ar10.send_q_target(list(q_target))
        time.sleep(1.5)

        # EMA einschwingen lassen (ein paar Reads im Stillstand)
        if ema is not None:
            for _ in range(10):
                ar10.read_q_measured()
                time.sleep(dt)

        # CLOSE-Rampe
        deltas_per_joint: dict[str, list[float]] = {j: [] for j in joints}

        t0 = time.perf_counter()
        for k in range(max_close):
            for j, idx in zip(joints, j_idxs):
                cap = caps.get(j, 1.0)
                q_target[idx] = min(cap, q_target[idx] + rate)
            ar10.send_q_target(list(q_target))

            q_meas = ar10.read_q_measured()

            row = {"ema_alpha": alpha, "step": k}
            for j, idx in zip(joints, j_idxs):
                qd = q_target[idx] - q_meas[idx]
                deltas_per_joint[j].append(qd)
                row[f"{j}_q_target"]   = round(q_target[idx], 5)
                row[f"{j}_q_measured"] = round(q_meas[idx], 5)
                row[f"{j}_q_delta"]    = round(qd, 5)
            all_rows.append(row)

            if k % 20 == 0:
                dstr = "  ".join(f"{j}:{deltas_per_joint[j][-1]:+.4f}" for j in joints)
                print(f"\r  step={k:3d}  {dstr}   ", end="", flush=True)

            all_closed = all(
                q_target[idx] >= caps.get(j, 1.0) - 1e-9
                for j, idx in zip(joints, j_idxs)
            )
            if all_closed:
                break

            pause = t0 + (k + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)

        # Hand oeffnen
        ar10.send_q_target(_pregrasp_q())
        time.sleep(1.0)
        ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
        time.sleep(0.5)
        ar10.close()

        # Statistik
        row_summary = {"ema_alpha": alpha}
        print()
        for j in joints:
            vals = deltas_per_joint[j]
            m   = statistics.fmean(vals)
            s   = statistics.pstdev(vals)
            mn  = min(vals)
            mx  = max(vals)
            row_summary[f"{j}_mean"]  = round(m, 5)
            row_summary[f"{j}_std"]   = round(s, 5)
            row_summary[f"{j}_min"]   = round(mn, 5)
            row_summary[f"{j}_max"]   = round(mx, 5)
            print(f"  {j}: mean={m:+.5f}  std={s:.5f}  [{mn:+.5f}, {mx:+.5f}]")
        summary.append(row_summary)

    # Vergleichstabelle
    print(f"\n{'='*60}")
    print(f"  VERGLEICH")
    print(f"{'='*60}")
    print(f"\n  {'alpha':>6s}", end="")
    for j in joints:
        print(f"  {j+'_std':>14s}", end="")
    print()

    base_stds = {j: summary[0][f"{j}_std"] for j in joints} if summary else {}
    for row in summary:
        print(f"  {row['ema_alpha']:6.2f}", end="")
        for j in joints:
            std = row[f"{j}_std"]
            if base_stds.get(j, 0) > 0 and row["ema_alpha"] != args.alphas[0]:
                pct = (1.0 - std / base_stds[j]) * 100
                print(f"  {std:.5f} ({pct:+.0f}%)", end="")
            else:
                print(f"  {std:.5f}       ", end="")
        print()

    # Minimaler Threshold
    print(f"\n  Minimaler Threshold (mean + 3*std):")
    for row in summary:
        print(f"    alpha={row['ema_alpha']:.2f}:", end="")
        for j in joints:
            thr = abs(row[f"{j}_mean"]) + 3.0 * row[f"{j}_std"]
            print(f"  {j}={thr:.4f}", end="")
        print()

    # CSV speichern
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.config).stem

    if all_rows:
        out = _OUT_DIR / f"noise_filter_{stem}_{ts}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[noise-test] Rohdaten -> {out}")

    if summary:
        out2 = _OUT_DIR / f"noise_filter_summary_{stem}_{ts}.csv"
        with out2.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print(f"[noise-test] Zusammenfassung -> {out2}")


if __name__ == "__main__":
    main()
