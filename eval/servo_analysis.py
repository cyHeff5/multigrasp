# Detaillierte Servo-Tracking-Analyse der echten AR10 Hand im Freilauf.
# Zeichnet q_target, q_measured, q_delta pro Joint und Step auf.
# Phasen: SETTLE -> CLOSE -> HOLD_CLOSED -> OPEN -> HOLD_OPEN -> CLOSE_2 -> HOLD_2
# Damit sieht man: Einschwingverhalten, Tracking-Fehler beim Schliessen/Oeffnen,
# Hysterese, Drift im Haltezustand, und Wiederholbarkeit.
#
# Usage:
#   python -m eval.servo_analysis --config configs/precision.yaml --port COM4
#   python -m eval.servo_analysis --config configs/power.yaml --port COM4
#
# Output: artifacts/analysis/servo_analysis_<config>_<timestamp>.csv
from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT   = Path(__file__).resolve().parent.parent
_OUT_DIR     = _REPO_ROOT / "artifacts" / "analysis"

# Phasen-Konfiguration (Schritte)
_SETTLE_STEPS      = 30   # Einschwingen nach Pregrasp
_HOLD_STEPS        = 60   # Halten in Endposition
_CLOSE2_ENABLED    = True # Zweiter Schliess-Zyklus fuer Wiederholbarkeit


def _step_dt(cfg: dict) -> float:
    return float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])


def _watched_joints(cfg: dict) -> list[str]:
    joints = []
    for jlist in cfg["finger_joints"].values():
        for j in jlist:
            if j not in joints:
                joints.append(j)
    return joints


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def _max_ramp_steps(cfg: dict) -> int:
    return int(1.0 / float(cfg["action"]["delta_norm"])) + 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Servo-Tracking-Analyse der AR10 im Freilauf.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port",   default=None)
    parser.add_argument("--cycles", type=int, default=10,
                        help="Anzahl Wiederholungen des gesamten Zyklus (default 10).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dt       = _step_dt(cfg)
    joints   = _watched_joints(cfg)
    j_idxs   = [CONTROL_JOINTS.index(j) for j in joints]
    rate     = float(cfg["action"]["delta_norm"])
    caps     = cfg["action"].get("pip_caps", {})
    max_close = _max_ramp_steps(cfg)
    n_cycles  = args.cycles

    ar10 = AR10Interface(com_port=args.port)
    if args.port is not None:
        from eval.policy_runner import watched_joint_indices
        ar10.assert_input_calibration(watched_joint_indices(cfg))

    print(f"[servo-analysis] Joints: {joints}")
    print(f"[servo-analysis] step_dt={dt*1000:.1f} ms ({1/dt:.0f} Hz)")
    print(f"[servo-analysis] rate={rate}  caps={caps}")
    print(f"[servo-analysis] Phasen: SETTLE({_SETTLE_STEPS}) -> CLOSE -> "
          f"HOLD({_HOLD_STEPS}) -> OPEN -> HOLD({_HOLD_STEPS})"
          f" -> CLOSE_2 -> HOLD_2({_HOLD_STEPS})")
    print(f"[servo-analysis] Zyklen: {n_cycles}")
    input("\nHand ist frei (KEIN Objekt)? Enter -> Start ...")

    # CSV vorbereiten
    fieldnames = ["cycle", "step", "t_ms", "phase"]
    for j in joints:
        fieldnames += [f"{j}_q_target", f"{j}_q_measured", f"{j}_q_delta"]
    rows: list[dict] = []

    q_target = _pregrasp_q()
    ar10.send_q_target(list(q_target))
    time.sleep(1.0)

    step_global = 0
    t0 = time.perf_counter()

    current_cycle = 0

    def record(phase: str) -> None:
        nonlocal step_global
        q_meas = ar10.read_q_measured()
        t_ms = (time.perf_counter() - t0) * 1000
        row = {"cycle": current_cycle, "step": step_global, "t_ms": round(t_ms, 1), "phase": phase}
        for j, idx in zip(joints, j_idxs):
            qt = q_target[idx]
            qm = q_meas[idx]
            row[f"{j}_q_target"]   = round(qt, 5)
            row[f"{j}_q_measured"] = round(qm, 5)
            row[f"{j}_q_delta"]    = round(qt - qm, 5)
        rows.append(row)
        # Live-Anzeige
        deltas = "  ".join(f"{j}:{row[f'{j}_q_delta']:+.4f}" for j in joints)
        print(f"\r  C{current_cycle:2d} [{phase:12s}] step={step_global:4d}  {deltas}   ", end="", flush=True)
        step_global += 1

    def sleep_until(k: int, phase_t0: float) -> None:
        pause = phase_t0 + (k + 1) * dt - time.perf_counter()
        if pause > 0:
            time.sleep(pause)

    # --- Phasen-Helfer ---
    def do_settle() -> None:
        print(f"\n  Phase: SETTLE")
        phase_t0 = time.perf_counter()
        for k in range(_SETTLE_STEPS):
            record("SETTLE")
            sleep_until(k, phase_t0)

    def do_close(phase_name: str) -> None:
        print(f"\n  Phase: {phase_name}")
        phase_t0 = time.perf_counter()
        for k in range(max_close):
            for j, idx in zip(joints, j_idxs):
                cap = caps.get(j, 1.0)
                q_target[idx] = min(cap, q_target[idx] + rate)
            ar10.send_q_target(list(q_target))
            # Slot abwarten, dann lesen — gleiche Lese-Phase wie der Policy-
            # Runner (Fix 2026-08-24; alte CSVs vom 08.07. lesen direkt nach
            # dem Senden und liegen dadurch ~+delta_norm hoeher).
            sleep_until(k, phase_t0)
            record(phase_name)
            all_closed = all(
                q_target[idx] >= caps.get(j, 1.0) - 1e-9
                for j, idx in zip(joints, j_idxs)
            )
            if all_closed:
                break

    def do_hold(phase_name: str) -> None:
        print(f"\n  Phase: {phase_name}")
        phase_t0 = time.perf_counter()
        for k in range(_HOLD_STEPS):
            record(phase_name)
            sleep_until(k, phase_t0)

    def do_open(phase_name: str) -> None:
        print(f"\n  Phase: {phase_name}")
        phase_t0 = time.perf_counter()
        for k in range(max_close):
            for j, idx in zip(joints, j_idxs):
                q_target[idx] = max(0.0, q_target[idx] - rate)
            ar10.send_q_target(list(q_target))
            sleep_until(k, phase_t0)
            record(phase_name)
            all_open = all(q_target[idx] <= 1e-9 for _, idx in zip(joints, j_idxs))
            if all_open:
                break

    # --- Zyklen ---
    for cycle in range(n_cycles):
        current_cycle = cycle
        print(f"\n{'='*60}")
        print(f"  Zyklus {cycle + 1}/{n_cycles}")
        print(f"{'='*60}")

        # Reset auf Pregrasp
        for j, idx in zip(joints, j_idxs):
            q_target[idx] = 0.0
        ar10.send_q_target(list(q_target))
        time.sleep(0.8)

        do_settle()
        do_close("CLOSE")
        do_hold("HOLD_CLOSED")
        do_open("OPEN")
        do_hold("HOLD_OPEN")
        do_close("CLOSE_2")
        do_hold("HOLD_2")

    # Hand oeffnen
    ar10.send_q_target(_pregrasp_q())
    time.sleep(1.0)
    ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
    time.sleep(1.0)
    ar10.close()

    # CSV speichern
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.config).stem
    out  = _OUT_DIR / f"servo_analysis_{stem}_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n[servo-analysis] {len(rows)} Samples, {n_cycles} Zyklen -> {out}")

    # Zusammenfassung
    import statistics
    print(f"\n-- Zusammenfassung pro Joint pro Phase (ueber {n_cycles} Zyklen) --")
    phases = []
    seen = set()
    for r in rows:
        if r["phase"] not in seen:
            phases.append(r["phase"])
            seen.add(r["phase"])
    for j in joints:
        print(f"\n  {j}:")
        for ph in phases:
            ph_rows = [r for r in rows if r["phase"] == ph]
            deltas  = [r[f"{j}_q_delta"] for r in ph_rows]
            if not deltas:
                continue
            mn  = min(deltas)
            mx  = max(deltas)
            avg = statistics.fmean(deltas)
            std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            print(f"    {ph:14s}  n={len(deltas):4d}  "
                  f"mean={avg:+.5f}  std={std:.5f}  min={mn:+.5f}  max={mx:+.5f}")


if __name__ == "__main__":
    main()
