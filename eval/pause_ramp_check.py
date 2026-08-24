# Pausen-Rampen-Messung: der EINE fehlende Baustein fuer das Servo-Modell.
#
# Alle Messungen vom 2026-07-08 fahren die CLOSE-Rampe am Stueck — die Policy
# faehrt aber stop-and-go, und das optimale Bewegungsmuster (Probe: Target
# einfrieren, blockierter Finger haelt q_delta, freier kollabiert) haengt an
# genau zwei ungemessenen Groessen:
#   1. KOLLAPS: wie schnell faellt q_delta nach einem Stopp mitten in der
#      Rampe auf den statischen Settle-Wert? (Erwartung aus Cap-Daten: 5-15
#      Steps — aber am Cap, nicht mitten im Hub.)
#   2. WIEDERANLAUF: wie gross ist der Anlauf-Transient, wenn das Target nach
#      einer kurzen Pause weiterfaehrt? (Kaltstart-Totzone: 11-15 Steps,
#      q_delta-Peak 0.05-0.075. Nach 10-Step-Pause: unbekannt -> deshalb steht
#      contact_detector.restart_steps konservativ auf 12.)
#
# Ablauf pro Zyklus: Hand offen (kalt), dann CLOSE-Rampe mit delta_norm, alle
# `--move-steps` Steps eine Pause von `--pause-steps`, bis alle Joints am Cap
# sind. OHNE Objekt. Dazu ein Referenz-Zyklus ohne Pausen.
# Lese-Timing wie im Policy-Runner: senden -> Step-Slot abwarten -> lesen.
#
# Auswertung am Ende:
#   - Kollaps-Zeitkonstante pro Pause (Steps bis q_delta < statisch + 0.005)
#   - Wiederanlauf: Steps bis q_measured wieder laeuft + q_delta-Peak danach
#     -> daraus contact_detector.restart_steps und servo_model-Restart-Totzone
#        ableiten und die konservativen Defaults ersetzen.
#
# Usage (Labor, ~3 min):
#   python -m eval.pause_ramp_check --config configs/precision.yaml --port COM4
# Mock-Smoke-Test (ohne Hardware, prueft nur den Ablauf):
#   python -m eval.pause_ramp_check --config configs/precision.yaml
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


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def _watched(cfg: dict) -> list[str]:
    joints: list[str] = []
    for jl in cfg["finger_joints"].values():
        for j in jl:
            if j not in joints:
                joints.append(j)
    return joints


def main() -> None:
    ap = argparse.ArgumentParser(description="CLOSE-Rampe mit periodischen Pausen (Freilauf).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", default=None)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--move-steps",  type=int, default=30)
    ap.add_argument("--pause-steps", type=int, default=15)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dt     = float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])
    rate   = float(cfg["action"]["delta_norm"])
    caps   = cfg["action"].get("pip_caps", {})
    joints = _watched(cfg)
    j_idxs = [CONTROL_JOINTS.index(j) for j in joints]

    ar10 = AR10Interface(com_port=args.port)
    mock = args.port is None
    print(f"[pause-ramp] joints={joints}  rate={rate}  dt={dt*1000:.1f} ms  "
          f"move={args.move_steps}  pause={args.pause_steps}  mock={mock}")
    if not mock:
        input("\nHand ist frei (KEIN Objekt)? Enter -> Start ...")

    fieldnames = ["cycle", "step", "t_ms", "phase", "pause_no"]
    for j in joints:
        fieldnames += [f"{j}_q_target", f"{j}_q_measured", f"{j}_q_delta"]
    rows: list[dict] = []

    for cycle in range(args.cycles + 1):          # letzter Zyklus = Referenz ohne Pausen
        reference = cycle == args.cycles
        q_target = _pregrasp_q()
        ar10.send_q_target(list(q_target))
        time.sleep(1.5)                            # kalt starten (wie echte Episoden)

        label = "REF" if reference else f"C{cycle}"
        print(f"\n  Zyklus {label} ({'ohne' if reference else 'mit'} Pausen)")
        t0 = time.perf_counter()
        step = 0
        pause_no = 0
        moved_since_pause = 0
        while True:
            in_pause = (not reference and moved_since_pause >= args.move_steps)
            if in_pause:
                phase = "PAUSE"
            else:
                for j, idx in zip(joints, j_idxs):
                    q_target[idx] = min(caps.get(j, 1.0), q_target[idx] + rate)
                phase = "MOVE"
                moved_since_pause += 1
            ar10.send_q_target(list(q_target))

            # Lese-Timing wie im Policy-Runner: Slot abwarten, dann lesen.
            pause_t = t0 + (step + 1) * dt - time.perf_counter()
            if pause_t > 0:
                time.sleep(pause_t)
            q_meas = ar10.read_q_measured()

            row = {"cycle": cycle, "step": step,
                   "t_ms": round((time.perf_counter() - t0) * 1000, 1),
                   "phase": phase, "pause_no": pause_no if in_pause else -1}
            for j, idx in zip(joints, j_idxs):
                row[f"{j}_q_target"]   = round(q_target[idx], 5)
                row[f"{j}_q_measured"] = round(q_meas[idx], 5)
                row[f"{j}_q_delta"]    = round(q_target[idx] - q_meas[idx], 5)
            rows.append(row)
            print(f"\r    step={step:4d} [{phase:5s}]  " +
                  "  ".join(f"{j}:{row[f'{j}_q_delta']:+.4f}" for j in joints),
                  end="", flush=True)

            if in_pause:
                # Pause zu Ende?
                pause_len = sum(1 for r in rows
                                if r["cycle"] == cycle and r["phase"] == "PAUSE"
                                and r["pause_no"] == pause_no)
                if pause_len >= args.pause_steps:
                    pause_no += 1
                    moved_since_pause = 0
            step += 1
            if all(q_target[idx] >= caps.get(j, 1.0) - 1e-9
                   for j, idx in zip(joints, j_idxs)):
                break
            if step > 600:
                break

    ar10.send_q_target(_pregrasp_q())
    time.sleep(1.0)
    ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
    if not mock:
        time.sleep(1.0)
    ar10.close()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _OUT_DIR / f"pause_ramp_{Path(args.config).stem}_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n\n[pause-ramp] {len(rows)} Samples -> {out}")

    # ── Auswertung ───────────────────────────────────────────────────────────
    print("\n-- Auswertung pro Joint --")
    for j in joints:
        collapse, restart_dead, restart_peak = [], [], []
        for cycle in range(args.cycles):
            cyc = [r for r in rows if r["cycle"] == cycle]
            for pn in sorted({r["pause_no"] for r in cyc if r["pause_no"] >= 0}):
                pause_rows = [r for r in cyc if r["pause_no"] == pn]
                if len(pause_rows) < 3:
                    continue
                settle = pause_rows[-1][f"{j}_q_delta"]
                k = next((i for i, r in enumerate(pause_rows)
                          if abs(r[f"{j}_q_delta"] - settle) < 0.005), None)
                if k is not None:
                    collapse.append(k)
                # Wiederanlauf: MOVE-Steps nach der Pause bis q_measured laeuft
                after = [r for r in cyc if r["step"] > pause_rows[-1]["step"]
                         and r["phase"] == "MOVE"][:25]
                if len(after) >= 5:
                    qm0 = after[0][f"{j}_q_measured"]
                    kd  = next((i for i, r in enumerate(after)
                                if r[f"{j}_q_measured"] > qm0 + 0.01), None)
                    if kd is not None:
                        restart_dead.append(kd)
                    restart_peak.append(max(r[f"{j}_q_delta"] for r in after))
        def _fmt(v):
            return (f"median {statistics.median(v):.3g}  max {max(v):.3g}  n={len(v)}"
                    if v else "keine Daten")
        print(f"  {j}:")
        print(f"    Kollaps auf Settle (Steps):   {_fmt(collapse)}")
        print(f"    Wiederanlauf-Totzone (Steps): {_fmt(restart_dead)}")
        print(f"    q_delta-Peak nach Pause:      {_fmt(restart_peak)}")
    print("\n  -> restart_steps in configs/*.yaml (contact_detector) auf den"
          "\n     gemessenen Wiederanlauf-Wert setzen (aktuell konservativ 12);"
          "\n     Kollaps-Steps bestaetigen settle_steps (aktuell 15).")


if __name__ == "__main__":
    main()
