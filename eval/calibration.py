# Kalibrier-Pipeline fuer den q_delta-Kontakt-Threshold der echten Hand.
# Drei Phasen, je ein Aufruf (AR10-Laptop, Hand am COM-Port):
#
#   --phase free         OHNE Objekt. Deterministische Schliess-Rampe, misst das
#                        Freilauf-q_delta (Tracking-Rauschen). Liefert die
#                        UNTERGRENZE: mean + 3*std deckt ~99.7% des Rauschens ab,
#                        darunter sind Fehltrigger in der Luft ausgeschlossen.
#
#   --phase parallel     MIT Objekt (Wuerfel auf dem Podest, Hand per Sawyer in
#                        derselben Pregrasp wie bei der Eval). Sim und echte Hand
#                        schliessen synchron mit identischer Rampe:
#                          A) bis die Sim Erstkontakt meldet (getContactPoints,
#                             exakt und ghost-frei) -> Rampe stoppt.
#                          B) Operator schliesst NUR die echte Hand per Tastatur
#                             nach, bis der Kontakt SICHTBAR ist (das Auge ersetzt
#                             den fehlenden Kontaktsensor). Absorbiert nebenbei
#                             den Geometrie-/Platzierungsfehler.
#                          C) beide schliessen synchron weiter; in dem Schritt, in
#                             dem das Sim-q_delta eines Fingers den Trainings-
#                             Nominalwert (observation.threshold) kreuzt, wird das
#                             reale q_delta dieses Fingers festgehalten.
#                        Ergebnis: real_threshold PRO FINGER, semantisch verankert
#                        am selben physischen Ereignis wie im Training. Validiert
#                        gegen den Rauschboden aus --phase free.
#
#   --phase power-check  MIT Objekt, Hand in Power-Pregrasp. Reine Validierung
#                        des Threshold-Transfers Precision -> Power: schliesst
#                        alle Finger und prueft, dass jeder Finger den geladenen
#                        Threshold klar ueberschreitet (Faktor >= 2). Begruendung
#                        des Transfers: Precision hat die weichste Kontaktkette
#                        (Silikonspitze gegen gefederten Daumen); der harte
#                        Power-Stall gegen die Handflaeche trennt erst recht.
#
# q_delta ist ueberall max(0, q_target - q_measured), pro Finger das Maximum
# ueber seine Joints — identisch zur Kontakt-Bit-Logik in env._observation und
# policy_runner._binary_obs. Die Rampe (delta_norm, pip_caps, step_dt) spiegelt
# exakt den Policy-Loop, damit kalibriert wird, was die Policy spaeter sieht.
#
# Usage (morgen):
#   python -m eval.calibration --config configs/precision.yaml --port COM4 --phase free
#   python -m eval.calibration --config configs/precision.yaml --port COM4 --phase parallel
#   python -m eval.calibration --config configs/power.yaml     --port COM4 --phase power-check
from __future__ import annotations

import argparse
import copy
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, FINGERTIP_EE_MAP, SERVO0_INIT


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CAL_DIR   = _REPO_ROOT / "artifacts" / "calibration"
_OUT_PATH  = _CAL_DIR / "real_threshold.yaml"


# ── Gemeinsame Helfer ─────────────────────────────────────────────────────────

def _free_samples_path(config_stem: str) -> Path:
    return _CAL_DIR / f"_free_run_{config_stem}.yaml"


def _step_dt(cfg: dict) -> float:
    # Identisch zu policy_runner.default_step_dt: Kalibrierung, Training und
    # Deployment laufen auf derselben Kontrollrate.
    return float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])


def _watched_fingers(cfg: dict) -> dict[str, list[int]]:
    # Finger -> Joint-Indizes (CONTROL_JOINTS-Indexierung).
    return {f: [CONTROL_JOINTS.index(j) for j in joints]
            for f, joints in cfg["finger_joints"].items()}


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def _advance(q_target: list[float], cfg: dict) -> None:
    # Ein Rampen-Schritt: alle beobachteten Joints um delta_norm schliessen,
    # pip_caps respektieren — die Schliessdynamik des Policy-Loops.
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


# ── Phase FREE: Rauschboden ohne Objekt ───────────────────────────────────────

def run_free_phase(ar10: AR10Interface, cfg: dict, config_stem: str,
                   n_sweeps: int) -> None:
    fingers = _watched_fingers(cfg)
    dt      = _step_dt(cfg)
    print(f"\n=== PHASE FREE (Objekt ENTFERNT) — {n_sweeps} Sweeps @ {1/dt:.0f} Hz ===")

    samples: list[float] = []
    for s in range(n_sweeps):
        print(f"  Sweep {s + 1}/{n_sweeps}:")
        q_target = _pregrasp_q()
        ar10.send_q_target(list(q_target))
        time.sleep(0.8)
        t0 = time.perf_counter()
        for k in range(_max_ramp_steps(cfg)):
            _advance(q_target, cfg)
            ar10.send_q_target(list(q_target))
            dq  = _real_dq_per_finger(ar10, q_target, fingers)
            mx  = max(dq.values())
            samples.append(mx)
            print(f"\r    dq_max={mx:.4f}   ", end="", flush=True)
            if _fully_closed(q_target, cfg):
                break
            pause = t0 + (k + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)
        print()
        ar10.send_q_target(_pregrasp_q())
        time.sleep(0.8)

    mean  = statistics.fmean(samples) if samples else 0.0
    std   = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    floor = mean + 3.0 * std

    _CAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _free_samples_path(config_stem)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump({
            "config":     config_stem,
            "n_samples":  len(samples),
            "free_mean":  float(mean),
            "free_std":   float(std),
            "free_floor": float(floor),   # mean + 3*std
            "samples":    [float(x) for x in samples],
        }, f, default_flow_style=False)

    print(f"\n[free] {len(samples)} Samples  mean={mean:.4f}  std={std:.4f}")
    print(f"[free] Rauschboden (mean + 3*std) = {floor:.4f}  -> {path}")
    print("\nWeiter mit '--phase parallel' (Objekt aufs Podest, Sawyer in Pregrasp).")


# ── Phase PARALLEL: Sim-Anker-Kalibrierung ────────────────────────────────────

def _make_cal_cfg(cfg: dict) -> dict:
    # Deterministische Sim fuer die Kalibrierung: kein Pose-Jitter, keine
    # Episoden-Randomisierung, Domain-Randomization auf Mittelwerte fixiert.
    cal = copy.deepcopy(cfg)
    cal["pregrasp"]["jitter_xy_m"] = 0.0
    cal["pregrasp"]["jitter_z_m"]  = 0.0
    cal["observation"].pop("threshold_range", None)
    cal["episode"].pop("substeps_range", None)
    for key in ("motor_force", "fingertip_friction"):
        b = cal["physics"][key]
        mid = 0.5 * (b["min"] + b["max"])
        cal["physics"][key] = {"min": mid, "max": mid}
    b = cal["sampler"]["lateral_friction"]
    mid = 0.5 * (b["min"] + b["max"])
    cal["sampler"]["lateral_friction"] = {"min": mid, "max": mid}
    return cal


def _finger_ee_link(cfg: dict, finger: str) -> str:
    # Fingertip-EE-Link des Fingers (Kontaktmessung) — PIP-Joint ist der letzte
    # Eintrag in finger_joints.
    pip = cfg["finger_joints"][finger][-1]
    return FINGERTIP_EE_MAP[pip]


def _sim_contact_fingers(env, cfg: dict, fingers: list[str]) -> set[str]:
    import pybullet as p
    out = set()
    for f in fingers:
        ee = _finger_ee_link(cfg, f)
        cs = p.getContactPoints(
            env._hand_id, env._obj.object_id,
            linkIndexA=env._hand.joint_index[ee], physicsClientId=env._cid,
        )
        if cs:
            out.add(f)
    return out


def _sim_dq_per_finger(env, fingers: dict[str, list[int]]) -> dict[str, float]:
    dq = env._hand.q_delta_normalized()
    return {f: max(dq[j] for j in idxs) for f, idxs in fingers.items()}


def _jog_real_until_contact(ar10: AR10Interface, cfg: dict,
                            q_real: list[float],
                            fingers: dict[str, list[int]]) -> None:
    # Operator schliesst die echte Hand in delta_norm-Schritten nach, bis der
    # Kontakt SICHTBAR ist. Live-q_delta als Hinweis, entschieden wird per Auge.
    print("\n  --- Phase B: echte Hand nachschliessen bis SICHTBARER Kontakt ---")
    print("      +N / N = N Schritte zu   |  -N = N Schritte auf   |  ok = Kontakt bestaetigt")
    while True:
        dq  = _real_dq_per_finger(ar10, q_real, fingers)
        pos = "  ".join(f"{f}:dq={v:.4f}" for f, v in dq.items())
        raw = input(f"      [{pos}]  > ").strip().lower()
        if raw == "ok":
            return
        try:
            n = int(raw) if raw and raw != "+" else 1
        except ValueError:
            print("      Eingabe nicht verstanden ('+N', '-N' oder 'ok').")
            continue
        rate = float(cfg["action"]["delta_norm"])
        caps = cfg["action"].get("pip_caps", {})
        for joints in cfg["finger_joints"].values():
            for j in joints:
                idx = CONTROL_JOINTS.index(j)
                cap = caps.get(j, 1.0)
                q_real[idx] = min(cap, max(0.0, q_real[idx] + n * rate))
        ar10.send_q_target(list(q_real))
        time.sleep(0.15)


def run_parallel_phase(ar10: AR10Interface, cfg: dict, config_stem: str,
                       n_sweeps: int, cube_cm: float, gui: bool) -> None:
    from sim.env import GraspEnv

    fingers_map = _watched_fingers(cfg)
    finger_list = list(fingers_map.keys())
    sim_nominal = float(cfg["observation"]["threshold"])
    dt          = _step_dt(cfg)

    free_path = _free_samples_path(config_stem)
    free_floor = None
    if free_path.exists():
        with free_path.open(encoding="utf-8") as f:
            free_floor = float((yaml.safe_load(f) or {}).get("free_floor", 0.0))
    else:
        print(f"[parallel] WARN: {free_path} fehlt — erst '--phase free' laufen "
              "lassen, sonst keine Rauschboden-Validierung.")

    cal_cfg = _make_cal_cfg(cfg)
    env = GraspEnv(cal_cfg, render_mode="human" if gui else None)
    obj_spec = {"shape": "cube", "size_cm": float(cube_cm), "yaw_rad": 0.0}
    env.reset(seed=0, options={"obj_spec": obj_spec})
    if gui:
        # GUI sichtbar lassen, aber die Echtzeit-Sleeps der Env abschalten —
        # das Pacing macht unten der Absolutzeit-Scheduler (step_dt).
        env.render_mode = "human_no_sleep"

    print(f"\n=== PHASE PARALLEL — {n_sweeps} Sweep(s), Wuerfel {cube_cm} cm, "
          f"Sim-Nominal {sim_nominal}, {1/dt:.0f} Hz ===")
    print("Voraussetzung: Echte Hand steht per Sawyer in DERSELBEN Pregrasp-Pose,")
    print("Wuerfel auf dem Podest.\n")

    close_action = np.ones(len(cfg["action_groups"]), dtype=np.int64)
    stay_action  = np.zeros(len(cfg["action_groups"]), dtype=np.int64)

    results: dict[str, list[float]] = {f: [] for f in finger_list}
    for sweep in range(n_sweeps):
        if sweep > 0:
            input(f"\nSweep {sweep + 1}/{n_sweeps}: Objekt neu platzieren, "
                  "Sawyer in Pregrasp, dann Enter ...")
            env.reset(seed=0, options={"obj_spec": obj_spec})

        q_real = _pregrasp_q()
        ar10.send_q_target(list(q_real))
        time.sleep(1.0)

        # Phase A: synchron schliessen bis Sim-Erstkontakt.
        print("  --- Phase A: synchrones Schliessen bis Sim-Kontakt ---")
        t0 = time.perf_counter()
        for k in range(_max_ramp_steps(cfg)):
            env.step(close_action)
            _advance(q_real, cfg)
            ar10.send_q_target(list(q_real))
            touched = _sim_contact_fingers(env, cfg, finger_list)
            if touched:
                print(f"  [sim] Erstkontakt: {sorted(touched)} nach {k + 1} Schritten.")
                break
            if _fully_closed(q_real, cfg):
                print("  [warn] Voll geschlossen ohne Sim-Kontakt — Pose pruefen!")
                break
            pause = t0 + (k + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)

        # Phase B: Operator richtet die echte Hand am Kontakt aus. Die Sim haelt
        # ihre Pose (stay), Physik laeuft weiter.
        _jog_real_until_contact(ar10, cfg, q_real, fingers_map)
        env.step(stay_action)

        # Phase C: ab den ausgerichteten Ankern synchron weiter; realer q_delta
        # im Sim-Crossing-Schritt = Threshold des Fingers.
        print("  --- Phase C: synchrones Schliessen bis Sim-Threshold-Crossing ---")
        crossed: dict[str, float] = {}
        t0 = time.perf_counter()
        for k in range(_max_ramp_steps(cfg)):
            # Die Env fuehrt ihre Trigger-State-Machine mit: nach Trigger +
            # stabilization_steps wuerde sie den Lift-Test starten und das
            # Objekt bewegen -> Messung waere hin. Vorher abbrechen.
            _, _, terminated, _, _ = env.step(close_action)
            if terminated:
                print("\n  [warn] Sim-Episode terminiert (Lift-Test) — Sweep-Ende.")
                break
            _advance(q_real, cfg)
            ar10.send_q_target(list(q_real))
            sim_dq  = _sim_dq_per_finger(env, fingers_map)
            real_dq = _real_dq_per_finger(ar10, q_real, fingers_map)
            line = "  ".join(f"{f}: sim={sim_dq[f]:.3f} real={real_dq[f]:.3f}"
                             for f in finger_list)
            print(f"\r    {line}   ", end="", flush=True)
            for f in finger_list:
                if f not in crossed and sim_dq[f] >= sim_nominal:
                    crossed[f] = real_dq[f]
                    print(f"\n  [crossing] {f}: sim_dq={sim_dq[f]:.4f} -> "
                          f"real_threshold={real_dq[f]:.4f}")
            if len(crossed) == len(finger_list) or _fully_closed(q_real, cfg):
                break
            pause = t0 + (k + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)
        print()

        for f, v in crossed.items():
            results[f].append(v)
        missing = [f for f in finger_list if f not in crossed]
        if missing:
            print(f"  [warn] Kein Sim-Crossing fuer {missing} — Sweep fuer diese "
                  "Finger nicht gewertet.")

        ar10.send_q_target(_pregrasp_q())
        time.sleep(0.8)

    env.close()

    # Median ueber Sweeps, Validierung gegen den Rauschboden.
    per_finger = {f: float(statistics.median(v)) for f, v in results.items() if v}
    if not per_finger:
        print("\n[parallel] Keine gueltigen Messungen — nichts gespeichert.")
        return

    print("\n-- Ergebnis ------------------------------------------------")
    ok = True
    for f, thr in per_finger.items():
        note = ""
        if free_floor is not None:
            if thr <= free_floor:
                note = f"  << FEHLER: unter Rauschboden {free_floor:.4f}!"
                ok = False
            elif thr < 1.5 * free_floor:
                note = f"  (knapp ueber Rauschboden {free_floor:.4f})"
        print(f"  {f:<8s} real_threshold = {thr:.4f}{note}")
    print("-------------------------------------------------------------")

    if not ok:
        print("[parallel] NICHT gespeichert — Kontakt-Signal nicht vom Rauschen "
              "trennbar. Pose/Objekt pruefen, Phase B sorgfaeltiger ausrichten.")
        return

    _CAL_DIR.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as f:
        yaml.dump({
            "method":                "parallel_sim_anchor",
            "config":                config_stem,
            "sim_threshold_nominal": sim_nominal,
            "per_finger":            per_finger,
            # Fallback fuer Finger ohne eigene Kalibrierung (z.B. Power
            # pinky/ring): konservativ das Maximum der kalibrierten Finger.
            "real_threshold":        float(max(per_finger.values())),
            "free_floor":            free_floor,
            "sweeps_per_finger":     {f: [float(x) for x in v]
                                      for f, v in results.items()},
        }, f, default_flow_style=False)
    print(f"[parallel] gespeichert -> {_OUT_PATH}")


# ── Phase POWER-CHECK: Transfer-Validierung ───────────────────────────────────

def run_power_check(ar10: AR10Interface, cfg: dict) -> None:
    fingers = _watched_fingers(cfg)
    dt      = _step_dt(cfg)

    if not _OUT_PATH.exists():
        print(f"[power-check] {_OUT_PATH} fehlt — erst '--phase parallel' "
              "(precision) laufen lassen.")
        return
    with _OUT_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    per_finger = data.get("per_finger") or {}
    default    = float(data.get("real_threshold", 0.05))
    thr = {f: float(per_finger.get(f, default)) for f in fingers}

    print(f"\n=== POWER-CHECK — Hand in Power-Pregrasp ueber dem Objekt ===")
    print("Geladene Thresholds: " + "  ".join(f"{f}={v:.4f}" for f, v in thr.items()))
    input("Objekt liegt, Sawyer steht? Enter -> Finger schliessen ...")

    q_target = _pregrasp_q()
    ar10.send_q_target(list(q_target))
    time.sleep(1.0)

    peak: dict[str, float] = {f: 0.0 for f in fingers}
    t0 = time.perf_counter()
    for k in range(_max_ramp_steps(cfg)):
        _advance(q_target, cfg)
        ar10.send_q_target(list(q_target))
        dq = _real_dq_per_finger(ar10, q_target, fingers)
        for f, v in dq.items():
            peak[f] = max(peak[f], v)
        print("\r    " + "  ".join(f"{f}:dq={v:.3f}" for f, v in dq.items()) + "   ",
              end="", flush=True)
        # Stoppen sobald alle Finger klar drueber sind — nicht durchdruecken.
        if all(peak[f] > 2.0 * thr[f] for f in fingers) or _fully_closed(q_target, cfg):
            break
        pause = t0 + (k + 1) * dt - time.perf_counter()
        if pause > 0:
            time.sleep(pause)
    print()

    ar10.send_q_target(_pregrasp_q())

    print("\n-- Power-Check ----------------------------------------------")
    all_ok = True
    for f in fingers:
        factor = peak[f] / thr[f] if thr[f] > 0 else float("inf")
        verdict = "OK" if factor >= 2.0 else "ZU KNAPP"
        if factor < 2.0:
            all_ok = False
        print(f"  {f:<8s} peak={peak[f]:.4f}  threshold={thr[f]:.4f}  "
              f"Faktor {factor:.1f}x  [{verdict}]")
    print("--------------------------------------------------------------")
    print("Transfer OK — Precision-Thresholds fuer Power verwenden." if all_ok
          else "Transfer NICHT bestaetigt — Kontakt einzelner Finger zu schwach.\n"
               "Objekt/Pose pruefen; notfalls '--phase parallel' mit power-Config.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold-Kalibrierung der echten AR10 (free / parallel / power-check).")
    parser.add_argument("--config",  required=True)
    parser.add_argument("--phase",   required=True,
                        choices=["free", "parallel", "power-check"])
    parser.add_argument("--port",    default=None,
                        help="AR10 COM-Port (z.B. COM4). Weglassen = Mock (nur Ablauf-Test).")
    parser.add_argument("--sweeps",  type=int, default=2,
                        help="Wiederholungen (free/parallel). Default 2.")
    parser.add_argument("--cube-cm", type=float, default=5.0,
                        help="Kantenlaenge des Kalibrier-Wuerfels in cm (parallel).")
    parser.add_argument("--no-gui",  action="store_true",
                        help="PyBullet-Fenster in der parallel-Phase unterdruecken.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_stem = Path(args.config).stem

    ar10 = AR10Interface(com_port=args.port)
    if args.port is None:
        print("[calibration] Mock-Mode — q_delta ist immer 0 (nur Ablauf-Test).")
    else:
        from eval.policy_runner import watched_joint_indices
        ar10.assert_input_calibration(watched_joint_indices(cfg))

    try:
        if args.phase == "free":
            run_free_phase(ar10, cfg, config_stem, args.sweeps)
        elif args.phase == "parallel":
            run_parallel_phase(ar10, cfg, config_stem, args.sweeps,
                               args.cube_cm, gui=not args.no_gui)
        else:
            run_power_check(ar10, cfg)
    finally:
        ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
        time.sleep(1.0)
        ar10.close()


if __name__ == "__main__":
    main()
