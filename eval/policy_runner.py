"""Shared policy runner — used by sim and real eval scripts.

Sim:
  run_episode(env, policy, seed, options) — one Gymnasium episode.

Real (AR10-Laptop):
  run_real_episode(ar10, policy, cfg, thresholds, n_steps_max, step_dt)
    — one closed-loop grasp on the real AR10 hand.

Standalone entry point (AR10-Laptop), Film-Workflow ohne Ergebnis-Eingabe:
  python -m eval.policy_runner --config configs/precision.yaml \\
                               --model  artifacts/models/precision/best/best_model \\
                               --port   COM4

  Pro Trial: Enter (Sawyer steht in Pregrasp) -> Griff laeuft -> Enter nach
  Lift + Absenken -> Hand oeffnet -> naechster Trial. Erfolg/Fehlschlag wird
  NICHT abgefragt (wird gefilmt); gespeichert werden Timestamps pro Trial,
  um das Log mit dem Video zu synchronisieren.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

import numpy as np
import yaml

from hardware.ar10             import AR10Interface
from hardware.contact_detector import ContactDetector, QDeltaBaseline
from sim.hand                  import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT      = Path(__file__).resolve().parent.parent
_REAL_THRESHOLD = _REPO_ROOT / "artifacts" / "calibration" / "real_threshold.yaml"
_RESULTS_DIR    = _REPO_ROOT / "artifacts" / "eval_results"
_HAND_DOF       = len(CONTROL_JOINTS)


# ── Sim helpers ───────────────────────────────────────────────────────────────

class AlwaysClosePolicy:
    # Skript-Baseline fuer "warum RL": schliesst jede Gruppe in jedem Step
    # maximal (close_stay -> close, rate_probe -> schnell; servo0-Gruppe -> zu).
    # Trigger/Lift laufen wie immer ueber die Zustandsmaschine der Env.
    def __init__(self, cfg: dict):
        import numpy as np
        mode = cfg["action"].get("mode", "close_stay")
        if mode == "rate_probe":
            self._action = np.array([2] * len(cfg["action_groups"]))
        else:
            self._action = np.array(
                [2 if "servo0" in g else 1 for g in cfg["action_groups"]])

    def predict(self, obs, deterministic=True):
        return self._action, None


def load_policy(model_path: str, cfg: dict | None = None):
    """Load a stable-baselines3 PPO policy from disk.

    model_path "always_close" liefert stattdessen die Skript-Baseline
    (braucht cfg fuer den Action-Space)."""
    if model_path == "always_close":
        if cfg is None:
            raise ValueError("load_policy('always_close') braucht cfg.")
        return AlwaysClosePolicy(cfg)
    from stable_baselines3 import PPO
    zip_path = Path(model_path)
    if not zip_path.suffix:
        zip_path = Path(model_path + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Policy not found: {zip_path}")
    return PPO.load(str(zip_path.with_suffix("")),
                    custom_objects={"learning_rate": 3e-4,
                                    "lr_schedule": lambda _: 3e-4})


def run_episode(env, policy, seed: int, options: dict | None = None) -> dict:
    """Run a single sim episode with deterministic policy. Returns the final info dict."""
    obs, _ = env.reset(seed=seed, options=options)
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, _truncated, info = env.step(action)
        if terminated:
            return dict(info)


# ── Real-hand helpers ─────────────────────────────────────────────────────────

def load_real_thresholds(cfg: dict) -> dict[str, float]:
    """Threshold pro Finger aus real_threshold.yaml.

    Format (neu):  per_finger: {index: 0.031, middle: 0.028}
    Format (alt):  real_threshold: 0.03            (gilt fuer alle Finger)
    Fallback ohne Datei: Sim-Nominalwert observation.threshold fuer alle Finger.
    """
    fingers  = list(cfg["finger_joints"].keys())
    sim_thr  = float(cfg["observation"]["threshold"])
    if not _REAL_THRESHOLD.exists():
        print(f"[policy-runner] WARN: {_REAL_THRESHOLD} fehlt — Sim-Threshold {sim_thr} fuer alle Finger.")
        return {f: sim_thr for f in fingers}

    with _REAL_THRESHOLD.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    per_finger = data.get("per_finger") or {}
    default    = float(data.get("real_threshold", sim_thr))
    out = {f: float(per_finger.get(f, default)) for f in fingers}
    print(f"[policy-runner] Kalibrierte Thresholds: "
          + "  ".join(f"{f}={v:.4f}" for f, v in out.items()))
    return out


def load_contact_detector(cfg: dict) -> ContactDetector | None:
    """Baseline-korrigierter Detektor (siehe hardware/contact_detector.py).

    None wenn contact_detector.enabled false ist -> alter Threshold-Pfad.
    Faellt HART, wenn enabled aber Baseline fehlt oder mit anderer Rampe
    kalibriert wurde — ein stiller Fallback auf den 0.05-Threshold wuerde
    die Kugel-Grasps unbemerkt wieder kaputt machen.
    """
    det_cfg = cfg.get("contact_detector") or {}
    if not det_cfg.get("enabled", False):
        return None

    path = _REPO_ROOT / det_cfg["baseline_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"contact_detector.enabled, aber Baseline fehlt: {path}\n"
            f"Vor der Session fahren:  python -m eval.baseline_calibration "
            f"--config <config> --port <COM>")
    baseline = QDeltaBaseline.load(path)

    cal_rate = baseline.meta.get("delta_norm")
    cfg_rate = float(cfg["action"]["delta_norm"])
    if cal_rate is not None and abs(float(cal_rate) - cfg_rate) > 1e-9:
        raise ValueError(
            f"Baseline mit delta_norm={cal_rate} kalibriert, Config nutzt "
            f"{cfg_rate} — Baseline gilt nur fuer die kalibrierte Rampe. "
            f"eval/baseline_calibration.py neu fahren.")

    if baseline.meta.get("read_timing") != "pre_next_send":
        print("[policy-runner] WARN: Baseline stammt aus einer Kalibrierung mit "
              "altem Lese-Timing (direkt nach dem Senden gelesen) und liegt "
              "dadurch ~+delta_norm zu hoch — eval/baseline_calibration.py "
              "neu fahren (Fix 2026-08-24).")

    created = baseline.meta.get("created")
    if created:
        age_h = (datetime.datetime.now()
                 - datetime.datetime.fromisoformat(created)).total_seconds() / 3600
        if age_h > 12:
            print(f"[policy-runner] WARN: Baseline ist {age_h:.0f} h alt "
                  f"({created}) — Drift ueber Tage ungetestet, besser neu kalibrieren.")

    detector = ContactDetector(det_cfg, cfg["finger_joints"], baseline)
    print(f"[policy-runner] ContactDetector aktiv (Baseline {created}, "
          f"residual_min={det_cfg['residual_min']}, "
          f"startup_mask_q={det_cfg['startup_mask_q']}).")
    return detector


def default_step_dt(cfg: dict) -> float:
    # Reale Loop-Rate = nominale Sim-Kontrollrate (substeps/sim_hz, ~20.8 ms bei
    # 5/240). Damit laufen Training, Kalibrierung und Deployment auf derselben Rate.
    ep = cfg["episode"]
    return float(ep["substeps"]) / float(ep["sim_hz"])


def watched_joint_indices(cfg: dict) -> list[int]:
    idx: list[int] = []
    for joints in cfg["finger_joints"].values():
        for j in joints:
            i = CONTROL_JOINTS.index(j)
            if i not in idx:
                idx.append(i)
    return idx


def _binary_obs(ar10: AR10Interface, q_target: list[float],
                finger_joints: dict, fingers: list[str],
                thresholds: dict[str, float]) -> np.ndarray:
    q_meas = ar10.read_q_measured()
    out    = np.zeros(len(fingers), dtype=np.float32)
    for i, finger in enumerate(fingers):
        for joint_name in finger_joints[finger]:
            j = CONTROL_JOINTS.index(joint_name)
            if (q_target[j] - q_meas[j]) > thresholds[finger]:
                out[i] = 1.0
                break
    return out


def _apply_action(action: np.ndarray, q_target: list[float], cfg: dict) -> list[float]:
    """Mirror env._action_to_delta + PIP caps on the real hand."""
    delta = [0.0] * _HAND_DOF
    mode  = cfg["action"].get("mode", "close_stay")
    for i, group in enumerate(cfg["action_groups"]):
        a = int(action[i])
        if "servo0" in group:
            step = cfg["action"]["thumb_abduction_delta"]
            d = (-1.0 if a == 0 else 1.0 if a == 2 else 0.0) * step
        elif mode == "rate_probe":
            # {0: Probe/Halten, 1: langsam schliessen, 2: schnell schliessen}
            rates = cfg["action"]["rates"]
            d = (0.0 if a == 0 else
                 float(rates["slow"]) if a == 1 else float(rates["fast"]))
        else:
            d = a * cfg["action"]["delta_norm"]
        for joint_name in group:
            delta[CONTROL_JOINTS.index(joint_name)] = d

    new_q = [max(0.0, min(1.0, q + d)) for q, d in zip(q_target, delta)]

    rng_  = cfg["action"]["thumb_abduction_range"]
    idx0  = CONTROL_JOINTS.index("servo0")
    new_q[idx0] = max(rng_[0], min(rng_[1], new_q[idx0]))
    for joint_name, cap in cfg["action"]["pip_caps"].items():
        idx = CONTROL_JOINTS.index(joint_name)
        new_q[idx] = min(cap, new_q[idx])

    return new_q


def pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(_HAND_DOF)]


def run_real_episode(ar10: AR10Interface, policy, cfg: dict,
                     thresholds: dict[str, float], n_steps_max: int,
                     step_dt: float,
                     detector: ContactDetector | None = None) -> dict:
    """Open hand to pregrasp, run policy closed-loop, freeze grip. Returns stats.

    Spiegelt die Trigger-State-Machine aus env.step(): sobald trigger_n Finger
    fuer trigger_confirmation_steps Schritte in Folge Kontakt haben, gilt der
    Griff als getriggert; nach stabilization_steps wird das Schliessen GESTOPPT
    und die Finger bleiben eingefroren (Non-Backdrivable haelt die Position).
    Den eigentlichen Lift macht der Sawyer-Operator.

    Absolutzeit-Scheduling: Schritt k startet bei t0 + k*step_dt, damit die
    serielle I/O-Zeit (Poti-Reads + Send) die Loop-Rate nicht verlangsamt.
    Die tatsaechlich erreichte Rate steht im Rueckgabe-Dict — weicht sie stark
    von 1/step_dt ab, ist die I/O langsamer als das Zeitbudget.
    """
    fingers = list(cfg["finger_joints"].keys())
    groups  = cfg["action_groups"]

    trigger_n       = cfg["trigger_n"]
    confirm_steps   = cfg["trigger_confirmation_steps"]
    stabilize_steps = cfg["stabilization_steps"]

    q_target = pregrasp_q()
    ar10.send_q_target(list(q_target))
    time.sleep(1.0)

    if detector is not None:
        detector.reset()

    consecutive_contact = 0
    lift_triggered      = False
    stabilization_left  = 0
    steps_done          = 0
    _dbg_log: list[str] = []

    t0 = time.perf_counter()
    for k in range(n_steps_max):
        # Observation spiegelt EXAKT env._observation: [contact_bits, q_groups].
        if detector is not None:
            contact = detector.update(q_target, ar10.read_q_measured())
        else:
            contact = _binary_obs(ar10, q_target, cfg["finger_joints"], fingers, thresholds)
        q_groups = np.array([q_target[CONTROL_JOINTS.index(g[0])] for g in groups],
                            dtype=np.float32)
        obs       = np.concatenate([contact, q_groups])
        action, _ = policy.predict(obs, deterministic=True)
        q_target  = _apply_action(action, q_target, cfg)
        ar10.send_q_target(q_target)
        steps_done = k + 1
        if k < 10 or k % 50 == 0:
            _dbg_log.append(f"step={k} obs={obs.tolist()} action={action.tolist()} q6={q_target[CONTROL_JOINTS.index('servo6')]:.3f} q7={q_target[CONTROL_JOINTS.index('servo7')]:.3f}")

        n_contact = int(contact.sum())
        if n_contact >= trigger_n:
            consecutive_contact += 1
        else:
            consecutive_contact = 0

        if not lift_triggered and consecutive_contact >= confirm_steps:
            lift_triggered     = True
            stabilization_left = stabilize_steps
            print(f"\n  [trigger] {n_contact} Finger Kontakt -> "
                  f"{stabilize_steps} Stabilisierungs-Schritte, dann Stopp.")

        if lift_triggered:
            stabilization_left -= 1
            if stabilization_left <= 0:
                print("  [stop] Griff stabil — Finger eingefroren, bereit zum Heben.")
                break

        # Bis zum Absolutzeit-Slot des naechsten Schritts schlafen.
        t_next = t0 + (k + 1) * step_dt
        pause  = t_next - time.perf_counter()
        if pause > 0:
            time.sleep(pause)

    elapsed = time.perf_counter() - t0
    achieved_dt = elapsed / max(1, steps_done)
    if achieved_dt > step_dt * 1.25:
        print(f"  [warn] Loop-Rate {1.0/achieved_dt:.1f} Hz statt {1.0/step_dt:.1f} Hz "
              f"— serielle I/O langsamer als step_dt.")

    dbg_path = Path(__file__).resolve().parent.parent / "artifacts" / "_dbg_episode.log"
    dbg_path.parent.mkdir(parents=True, exist_ok=True)
    with dbg_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(_dbg_log))
    print(f"  [dbg] {len(_dbg_log)} Eintraege -> {dbg_path}")

    return {
        "q_final":     list(q_target),
        "triggered":   lift_triggered,
        "n_steps":     steps_done,
        "achieved_dt": achieved_dt,
    }


# ── Trial-Log (Video-Synchronisation) ─────────────────────────────────────────

def _save_trial_log(trials: list[dict], config_stem: str) -> None:
    """Timestamps + Griff-Metadaten pro Trial als CSV/YAML — Erfolg/Fehlschlag
    kommt aus dem Video, das Log liefert die Zuordnung Trial <-> Uhrzeit.
    Wird im finally aufgerufen -> auch bei Strg-C nichts verloren."""
    if not trials:
        print("[policy-runner] Keine Trials — nichts gespeichert.")
        return

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"real_{config_stem}_{ts}"

    fields = ["trial", "label", "t_start", "t_end", "triggered", "n_steps",
              "achieved_hz", "q_final"]
    with (_RESULTS_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)
    with (_RESULTS_DIR / f"{stem}.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({"mode": "real", "trials": trials}, f, default_flow_style=False)

    print(f"\n[policy-runner] {len(trials)} Trials geloggt -> {_RESULTS_DIR / stem}.{{csv,yaml}}")


# ── Standalone entry point (AR10-Laptop) ──────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AR10-Laptop: Greif-Loop fuer die gefilmte Real-Eval. "
                    "Koordination mit dem Sawyer-Laptop ueber Enter-Prompts."
    )
    parser.add_argument("--config",  required=True)
    parser.add_argument("--model",   required=True,
                        help="Path to trained model (with or without .zip).")
    parser.add_argument("--port",    default=None,
                        help="AR10 COM port (e.g. COM4). Omit for mock mode.")
    parser.add_argument("--step-dt", type=float, default=None,
                        help="Sekunden pro Policy-Step. Default: substeps/sim_hz "
                             "aus der Config (identisch zu Training + Kalibrierung).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detector    = load_contact_detector(cfg)
    # Alte Threshold-Logik nur noch als Fallback (contact_detector.enabled: false).
    thresholds  = {} if detector is not None else load_real_thresholds(cfg)
    step_dt     = args.step_dt if args.step_dt is not None else default_step_dt(cfg)
    n_steps_max = cfg["episode"]["max_steps"]

    policy = load_policy(args.model)
    ar10   = AR10Interface(com_port=args.port)
    ar10.assert_input_calibration(watched_joint_indices(cfg))
    if args.port is None:
        print("[policy-runner] Mock mode — keine echte Hand verbunden.\n")

    config_stem = Path(args.config).stem
    print(f"[policy-runner] config    = {config_stem}")
    print(f"[policy-runner] step_dt   = {step_dt*1000:.1f} ms ({1.0/step_dt:.0f} Hz)")
    print(f"[policy-runner] max_steps = {n_steps_max}")
    print("\nAblauf pro Trial:")
    print("  1. Sawyer-Laptop: drive_pregrasp -> Hand steht in Pregrasp")
    print("  2. Hier: Enter -> Griff laeuft, stoppt selbststaendig nach Trigger")
    print("  3. Sawyer hebt (Lift-Test), dann WIEDER ABSENKEN")
    print("  4. Hier: Enter -> Hand oeffnet, Objekt neu platzieren, weiter mit 1.")
    print("Label ist optional (fuers Video-Log). 'q' beendet und speichert.\n")

    trials: list[dict] = []
    last_label = ""
    try:
        while True:
            raw = input(f"[Trial {len(trials) + 1}] Label [{last_label or '-'}] "
                        f"| Enter = Start | q = Ende: ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            label = raw or last_label
            last_label = label

            t_start = datetime.datetime.now()
            stats   = run_real_episode(ar10, policy, cfg, thresholds, n_steps_max,
                                       step_dt, detector=detector)
            t_end   = datetime.datetime.now()
            if not stats["triggered"]:
                print("  [warn] Kein Trigger — Griff lief bis max_steps.")

            input("  Lift durch & Sawyer wieder abgesenkt? Enter -> Hand oeffnet: ")
            ar10.send_q_target(pregrasp_q())
            time.sleep(0.8)

            trials.append({
                "trial":       len(trials) + 1,
                "label":       label,
                "t_start":     t_start.strftime("%H:%M:%S.%f")[:-3],
                "t_end":       t_end.strftime("%H:%M:%S.%f")[:-3],
                "triggered":   stats["triggered"],
                "n_steps":     stats["n_steps"],
                "achieved_hz": round(1.0 / stats["achieved_dt"], 1),
                "q_final":     [round(v, 3) for v in stats["q_final"]],
            })
            print(f"  -> Trial {len(trials)} geloggt "
                  f"({stats['n_steps']} Steps, {trials[-1]['achieved_hz']} Hz)\n")

    except KeyboardInterrupt:
        print("\n[policy-runner] Abbruch.")
    finally:
        ar10.send_q_target([0.0] * _HAND_DOF)
        time.sleep(1.0)
        ar10.close()
        _save_trial_log(trials, config_stem)


if __name__ == "__main__":
    main()
