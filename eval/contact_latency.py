# Kontakt-Latenz-Messung: Wie viele Steps zwischen physischem Kontakt
# und Threshold-Ueberschreitung?
#
# Laeuft identisch zur servo_analysis (gleiche Rampe), aber MIT Objekt.
# Zwei Signale fuer den echten Kontakt-Moment:
#   1. Automatisch: Abweichung von der Freilauf-Baseline (servo_analysis CSV)
#   2. Manuell: Leertaste druecken wenn Kontakt SICHTBAR ist
#
# Usage:
#   python -m eval.contact_latency --config configs/precision.yaml --port COM4 \
#       --baseline artifacts/analysis/servo_analysis_precision_<ts>.csv
#
# Output: artifacts/analysis/contact_latency_<config>_<timestamp>.csv
#         + Zusammenfassung mit Latenz in Steps und ms
from __future__ import annotations

import argparse
import csv
import datetime
import statistics
import threading
import time
from pathlib import Path

import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "artifacts" / "analysis"

_HOLD_STEPS = 60


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


def _load_baseline(path: str, joints: list[str]) -> dict[str, list[float]]:
    """Laedt die Freilauf-Baseline: pro Joint eine Liste von q_delta-Werten
    waehrend der CLOSE-Phase (nur Cycle 0, da alle Zyklen aehnlich sind)."""
    baseline: dict[str, list[float]] = {j: [] for j in joints}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["phase"] == "CLOSE" and row["cycle"] == "0":
                for j in joints:
                    baseline[j].append(float(row[f"{j}_q_delta"]))
    if not baseline[joints[0]]:
        raise ValueError(f"Keine CLOSE-Phase in Cycle 0 in {path} gefunden.")
    return baseline


def _baseline_stats(baseline: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """Pro Joint: (mean, std) des gesamten Freilauf-CLOSE q_delta."""
    out = {}
    for j, vals in baseline.items():
        out[j] = (statistics.fmean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return out


class KeyListener:
    """Wartet im Hintergrund auf Leertaste; setzt timestamp beim Druecken."""
    def __init__(self):
        self.contact_time: float | None = None
        self._stop = False

    def start(self):
        self.contact_time = None
        t = threading.Thread(target=self._listen, daemon=True)
        t.start()

    def _listen(self):
        try:
            import msvcrt
            while not self._stop:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b' ':
                        self.contact_time = time.perf_counter()
                        return
                time.sleep(0.005)
        except ImportError:
            # Fallback fuer nicht-Windows
            import sys, select
            while not self._stop:
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    sys.stdin.readline()
                    self.contact_time = time.perf_counter()
                    return

    def stop(self):
        self._stop = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kontakt-Latenz: Steps zwischen physischem Kontakt und Threshold.")
    parser.add_argument("--config",   required=True)
    parser.add_argument("--port",     default=None)
    parser.add_argument("--baseline", required=True,
                        help="Pfad zur servo_analysis CSV (Freilauf-Baseline).")
    parser.add_argument("--cycles",   type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Kontakt-Threshold (default: aus real_threshold.yaml oder Config).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dt        = _step_dt(cfg)
    joints    = _watched_joints(cfg)
    j_idxs    = [CONTROL_JOINTS.index(j) for j in joints]
    rate      = float(cfg["action"]["delta_norm"])
    caps      = cfg["action"].get("pip_caps", {})
    max_close = _max_ramp_steps(cfg)
    n_cycles  = args.cycles

    # Threshold laden
    if args.threshold is not None:
        threshold = args.threshold
    else:
        thr_path = _REPO_ROOT / "artifacts" / "calibration" / "real_threshold.yaml"
        if thr_path.exists():
            with thr_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            threshold = float(data.get("real_threshold", cfg["observation"]["threshold"]))
        else:
            threshold = float(cfg["observation"]["threshold"])

    # Baseline laden
    baseline = _load_baseline(args.baseline, joints)
    bl_stats = _baseline_stats(baseline)
    # Abweichungs-Schwelle: mean + 4*std der Freilauf-Baseline
    deviation_threshold = {j: bl_stats[j][0] + 4.0 * bl_stats[j][1] for j in joints}

    print(f"[contact-latency] Joints: {joints}")
    print(f"[contact-latency] step_dt={dt*1000:.1f} ms ({1/dt:.0f} Hz)")
    print(f"[contact-latency] Kontakt-Threshold: {threshold:.4f}")
    print(f"[contact-latency] Baseline (Freilauf CLOSE):")
    for j in joints:
        m, s = bl_stats[j]
        print(f"    {j}: mean={m:+.5f}  std={s:.5f}  "
              f"Abweichungs-Schwelle (mean+4*std)={deviation_threshold[j]:+.5f}")
    print(f"[contact-latency] Zyklen: {n_cycles}")
    print()

    ar10 = AR10Interface(com_port=args.port)
    if args.port is not None:
        from eval.policy_runner import watched_joint_indices
        ar10.assert_input_calibration(watched_joint_indices(cfg))

    # Finger-Zuordnung
    finger_joints = cfg["finger_joints"]
    fingers = list(finger_joints.keys())

    # CSV vorbereiten
    fieldnames = ["cycle", "step", "t_ms", "phase"]
    for j in joints:
        fieldnames += [f"{j}_q_target", f"{j}_q_measured", f"{j}_q_delta",
                       f"{j}_baseline_delta"]
    rows: list[dict] = []

    # Ergebnis-Sammlung pro Zyklus
    cycle_results: list[dict] = []

    key_listener = KeyListener()

    for cycle in range(n_cycles):
        print(f"\n{'='*60}")
        print(f"  Zyklus {cycle + 1}/{n_cycles}")
        print(f"  Objekt (Kugel) aufs Podest legen.")
        input(f"  Enter -> Hand schliesst. LEERTASTE druecken bei sichtbarem Kontakt.")
        print()

        q_target = _pregrasp_q()
        ar10.send_q_target(list(q_target))
        time.sleep(1.0)

        key_listener.start()
        t0 = time.perf_counter()

        # Pro Joint: Step bei dem Abweichung und Threshold erstmals ueberschritten wird
        deviation_step: dict[str, int | None] = {j: None for j in joints}
        threshold_step: dict[str, int | None] = {j: None for j in joints}
        human_step: int | None = None

        # CLOSE-Rampe
        for k in range(max_close):
            for j, idx in zip(joints, j_idxs):
                cap = caps.get(j, 1.0)
                q_target[idx] = min(cap, q_target[idx] + rate)
            ar10.send_q_target(list(q_target))

            q_meas = ar10.read_q_measured()
            t_ms = (time.perf_counter() - t0) * 1000

            row = {"cycle": cycle, "step": k, "t_ms": round(t_ms, 1), "phase": "CLOSE"}
            for j, idx in zip(joints, j_idxs):
                qt = q_target[idx]
                qm = q_meas[idx]
                qd = qt - qm
                # Baseline-Wert fuer denselben Step (falls vorhanden)
                bl = baseline[j][k] if k < len(baseline[j]) else baseline[j][-1]
                row[f"{j}_q_target"]       = round(qt, 5)
                row[f"{j}_q_measured"]     = round(qm, 5)
                row[f"{j}_q_delta"]        = round(qd, 5)
                row[f"{j}_baseline_delta"] = round(bl, 5)

                # Abweichungs-Erkennung: q_delta deutlich ueber Freilauf
                if deviation_step[j] is None and qd > deviation_threshold[j]:
                    deviation_step[j] = k

                # Threshold-Erkennung (wie Policy)
                if threshold_step[j] is None and max(0.0, qd) > threshold:
                    threshold_step[j] = k

            rows.append(row)

            # Human-Marker pruefen
            if human_step is None and key_listener.contact_time is not None:
                human_step = k

            # Live-Anzeige
            deltas = "  ".join(f"{j}:{row[f'{j}_q_delta']:+.4f}" for j in joints)
            print(f"\r  step={k:3d}  {deltas}   ", end="", flush=True)

            # Stopp: alle Joints am Cap oder Threshold ueberschritten
            all_closed = all(
                q_target[idx] >= caps.get(j, 1.0) - 1e-9
                for j, idx in zip(joints, j_idxs)
            )
            if all_closed:
                break

            pause = t0 + (k + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)

        key_listener.stop()

        # Hand oeffnen
        ar10.send_q_target(_pregrasp_q())
        time.sleep(0.8)

        # Ergebnis dieses Zyklus
        print(f"\n\n  Zyklus {cycle + 1} Ergebnis:")
        result = {"cycle": cycle, "human_step": human_step}
        for j in joints:
            ds = deviation_step[j]
            ts = threshold_step[j]
            result[f"{j}_deviation_step"] = ds
            result[f"{j}_threshold_step"] = ts
            if ds is not None and ts is not None:
                latency = ts - ds
                result[f"{j}_latency_steps"] = latency
                result[f"{j}_latency_ms"] = round(latency * dt * 1000, 1)
                print(f"    {j}: Abweichung@step {ds} -> Threshold@step {ts}  "
                      f"= {latency} Steps ({latency * dt * 1000:.0f} ms)")
            elif ds is not None:
                print(f"    {j}: Abweichung@step {ds}, aber Threshold NIE erreicht!")
                result[f"{j}_latency_steps"] = None
            else:
                print(f"    {j}: Keine Abweichung erkannt.")
                result[f"{j}_latency_steps"] = None

        if human_step is not None:
            print(f"    HUMAN: Kontakt gesehen @ step {human_step} ({human_step * dt * 1000:.0f} ms)")
        else:
            print(f"    HUMAN: Kein Tastendruck registriert.")
        cycle_results.append(result)

    # Hand schliessen / aufraeumen
    ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
    time.sleep(1.0)
    ar10.close()

    # CSV speichern
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.config).stem
    out  = _OUT_DIR / f"contact_latency_{stem}_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[contact-latency] Rohdaten -> {out}")

    # Zusammenfassung
    print(f"\n{'='*60}")
    print(f"  ZUSAMMENFASSUNG ueber {n_cycles} Zyklen")
    print(f"{'='*60}")
    for j in joints:
        latencies = [r[f"{j}_latency_steps"] for r in cycle_results
                     if r.get(f"{j}_latency_steps") is not None]
        if latencies:
            avg = statistics.fmean(latencies)
            mn, mx = min(latencies), max(latencies)
            print(f"  {j}: Latenz {avg:.1f} Steps avg  [{mn}-{mx}]  "
                  f"= {avg * dt * 1000:.0f} ms avg  [{mn * dt * 1000:.0f}-{mx * dt * 1000:.0f} ms]")
        else:
            dev_steps = [r[f"{j}_deviation_step"] for r in cycle_results
                         if r.get(f"{j}_deviation_step") is not None]
            if dev_steps:
                print(f"  {j}: Abweichung erkannt (steps {dev_steps}), "
                      f"aber Threshold nie erreicht!")
            else:
                print(f"  {j}: Kein Kontakt-Signal erkannt.")

    human_steps = [r["human_step"] for r in cycle_results if r["human_step"] is not None]
    if human_steps:
        avg_h = statistics.fmean(human_steps)
        print(f"\n  HUMAN: Kontakt gesehen @ step {avg_h:.0f} avg  "
              f"[{min(human_steps)}-{max(human_steps)}]  "
              f"= {avg_h * dt * 1000:.0f} ms avg")


if __name__ == "__main__":
    main()
