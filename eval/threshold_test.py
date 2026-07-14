# Kontakt-Test: Hand schliesst wie in der Kalibrierung (ohne Sim),
# stoppt sobald ein Finger den realen Threshold erreicht.
#
# Usage:
#   python -m eval.threshold_test --config configs/precision.yaml --port COM4
#   python -m eval.threshold_test --config configs/power.yaml --port COM4
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT = Path(__file__).resolve().parent.parent
_THRESHOLD_FILE = _REPO_ROOT / "artifacts" / "calibration" / "real_threshold.yaml"


def _step_dt(cfg: dict) -> float:
    return float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])


def _watched_fingers(cfg: dict) -> dict[str, list[int]]:
    return {f: [CONTROL_JOINTS.index(j) for j in joints]
            for f, joints in cfg["finger_joints"].items()}


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def _advance(q_target: list[float], cfg: dict) -> None:
    rate = float(cfg["action"]["delta_norm"])
    caps = cfg["action"].get("pip_caps", {})
    for joints in cfg["finger_joints"].values():
        for j in joints:
            idx = CONTROL_JOINTS.index(j)
            cap = caps.get(j, 1.0)
            q_target[idx] = min(cap, q_target[idx] + rate)


def _fully_closed(q_target: list[float], cfg: dict) -> bool:
    caps = cfg["action"].get("pip_caps", {})
    for joints in cfg["finger_joints"].values():
        for j in joints:
            if q_target[CONTROL_JOINTS.index(j)] < caps.get(j, 1.0) - 1e-9:
                return False
    return True


def _real_dq_per_finger(ar10: AR10Interface, q_target: list[float],
                        fingers: dict[str, list[int]]) -> dict[str, float]:
    q_meas = ar10.read_q_measured()
    return {f: max(max(0.0, q_target[j] - q_meas[j]) for j in idxs)
            for f, idxs in fingers.items()}


def _max_ramp_steps(cfg: dict) -> int:
    return int(1.0 / float(cfg["action"]["delta_norm"])) + 20


def _load_thresholds(cfg: dict) -> dict[str, float]:
    fingers = list(cfg["finger_joints"].keys())
    sim_thr = float(cfg["observation"]["threshold"])
    if not _THRESHOLD_FILE.exists():
        print(f"[threshold-test] WARN: {_THRESHOLD_FILE} fehlt — Sim-Threshold {sim_thr}")
        return {f: sim_thr for f in fingers}
    with _THRESHOLD_FILE.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    per_finger = data.get("per_finger") or {}
    default = float(data.get("real_threshold", sim_thr))
    return {f: float(per_finger.get(f, default)) for f in fingers}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kontakt-Test: Hand schliesst bis realer Threshold erreicht.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    thresholds = _load_thresholds(cfg)
    fingers = _watched_fingers(cfg)
    dt = _step_dt(cfg)

    ar10 = AR10Interface(com_port=args.port)
    if args.port is not None:
        from eval.policy_runner import watched_joint_indices
        ar10.assert_input_calibration(watched_joint_indices(cfg))

    print(f"[threshold-test] Thresholds: "
          + "  ".join(f"{f}={v:.4f}" for f, v in thresholds.items()))
    print(f"[threshold-test] step_dt={dt*1000:.1f} ms ({1/dt:.0f} Hz)")

    try:
        while True:
            input("\nEnter -> Hand schliesst bis Threshold erreicht (q = beenden): ")
            q_target = _pregrasp_q()
            ar10.send_q_target(list(q_target))
            time.sleep(1.0)

            triggered_finger = None
            t0 = time.perf_counter()
            for k in range(_max_ramp_steps(cfg)):
                _advance(q_target, cfg)
                ar10.send_q_target(list(q_target))
                dq = _real_dq_per_finger(ar10, q_target, fingers)
                line = "  ".join(f"{f}:dq={v:.4f}" for f, v in dq.items())
                print(f"\r  Step {k+1:3d}  {line}   ", end="", flush=True)

                for f, v in dq.items():
                    if v >= thresholds[f]:
                        triggered_finger = f
                        break
                if triggered_finger:
                    break
                if _fully_closed(q_target, cfg):
                    break
                pause = t0 + (k + 1) * dt - time.perf_counter()
                if pause > 0:
                    time.sleep(pause)

            elapsed = time.perf_counter() - t0
            print()
            if triggered_finger:
                print(f"  [KONTAKT] {triggered_finger} hat Threshold erreicht "
                      f"(dq={dq[triggered_finger]:.4f} >= {thresholds[triggered_finger]:.4f}) "
                      f"nach {k+1} Steps ({elapsed:.1f}s)")
                print("  Finger eingefroren. Enter -> Hand oeffnet.")
                input()
            else:
                print("  [KEIN KONTAKT] Voll geschlossen ohne Threshold-Trigger.")

            ar10.send_q_target(_pregrasp_q())
            time.sleep(0.8)

    except (KeyboardInterrupt, EOFError):
        print("\n[threshold-test] Beendet.")
    finally:
        ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
        time.sleep(1.0)
        ar10.close()


if __name__ == "__main__":
    main()
