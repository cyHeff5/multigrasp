# Sim-Kontakt-Latenz: Wie viele Steps zwischen getContactPoints (physischer
# Kontakt) und q_delta > threshold (was die Policy sieht)?
#
# Laeuft eine deterministische Close-Rampe in Simulation (wie Kalibrierung),
# zeichnet pro Step auf:
#   - getContactPoints pro Finger (Ground Truth)
#   - q_delta pro Joint
#   - ob das Kontakt-Bit flippen wuerde (q_delta > threshold)
#
# Mehrere Episoden mit verschiedenen Objekten/Thresholds fuer Statistik.
#
# Usage:
#   python -m eval.sim_contact_latency --config configs/precision.yaml
#   python -m eval.sim_contact_latency --config configs/precision.yaml --shape cube --size 5.0
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import statistics
from pathlib import Path

import numpy as np
import pybullet as p
import yaml

from sim.env  import GraspEnv
from sim.hand import CONTROL_JOINTS, FINGERTIP_EE_MAP


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "artifacts" / "analysis"


def _make_deterministic_cfg(cfg: dict) -> dict:
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
    pip = cfg["finger_joints"][finger][-1]
    return FINGERTIP_EE_MAP[pip]


def _watched_joints(cfg: dict) -> list[str]:
    joints = []
    for jlist in cfg["finger_joints"].values():
        for j in jlist:
            if j not in joints:
                joints.append(j)
    return joints


def _watched_fingers(cfg: dict) -> dict[str, list[int]]:
    return {f: [CONTROL_JOINTS.index(j) for j in joints]
            for f, joints in cfg["finger_joints"].items()}


def run_analysis(cfg: dict, obj_spec: dict, n_episodes: int, thresholds: list[float]):
    cal_cfg = _make_deterministic_cfg(cfg)
    env = GraspEnv(cal_cfg, render_mode=None)

    joints = _watched_joints(cfg)
    fingers = list(cfg["finger_joints"].keys())
    finger_map = _watched_fingers(cfg)
    rate = float(cfg["action"]["delta_norm"])
    caps = cfg["action"].get("pip_caps", {})
    max_steps = int(1.0 / rate) + 20

    close_action = np.ones(len(cfg["action_groups"]), dtype=np.int64)

    all_rows: list[dict] = []
    episode_results: list[dict] = []

    for ep in range(n_episodes):
        env.reset(seed=ep, options={"obj_spec": obj_spec})
        cid = env._cid
        hand = env._hand
        obj_id = env._obj.object_id
        hand_id = env._hand_id

        # Pro Finger: Step of first getContactPoints and first threshold crossing
        first_contact: dict[str, int | None] = {f: None for f in fingers}
        first_threshold: dict[str, dict[float, int | None]] = {
            f: {t: None for t in thresholds} for f in fingers
        }

        for k in range(max_steps):
            obs, _, terminated, _, _ = env.step(close_action)
            if terminated:
                break

            q_delta = hand.q_delta_normalized()

            # getContactPoints pro Finger (Ground Truth)
            contact_gt: dict[str, bool] = {}
            for finger in fingers:
                ee = _finger_ee_link(cfg, finger)
                ee_idx = hand.joint_index[ee]
                cs = p.getContactPoints(hand_id, obj_id, linkIndexA=ee_idx,
                                        physicsClientId=cid)
                contact_gt[finger] = len(cs) > 0
                if contact_gt[finger] and first_contact[finger] is None:
                    first_contact[finger] = k

            # q_delta pro Finger (max ueber Joints)
            finger_dq: dict[str, float] = {}
            for finger in fingers:
                dqs = [q_delta[j] for j in finger_map[finger]]
                finger_dq[finger] = max(dqs)

                for thr in thresholds:
                    if first_threshold[finger][thr] is None and max(dqs) > thr:
                        first_threshold[finger][thr] = k

            # Row aufzeichnen
            row: dict = {"episode": ep, "step": k, "shape": obj_spec["shape"]}
            for finger in fingers:
                row[f"{finger}_contact_gt"] = int(contact_gt[finger])
                row[f"{finger}_q_delta"] = round(finger_dq[finger], 5)
                for thr in thresholds:
                    row[f"{finger}_bit_{thr}"] = int(finger_dq[finger] > thr)
            for j in joints:
                idx = CONTROL_JOINTS.index(j)
                row[f"{j}_q_target"] = round(hand.q_target()[idx], 5)
                row[f"{j}_q_delta"] = round(q_delta[idx], 5)
            all_rows.append(row)

        # Episode-Ergebnis
        ep_result = {"episode": ep, "shape": obj_spec["shape"],
                     "size": obj_spec.get("size_cm", "")}
        for finger in fingers:
            fc = first_contact[finger]
            ep_result[f"{finger}_first_contact"] = fc
            for thr in thresholds:
                ft = first_threshold[finger][thr]
                ep_result[f"{finger}_first_thr_{thr}"] = ft
                if fc is not None and ft is not None:
                    ep_result[f"{finger}_latency_{thr}"] = ft - fc
                else:
                    ep_result[f"{finger}_latency_{thr}"] = None
        episode_results.append(ep_result)

    env.close()
    return all_rows, episode_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sim-Kontakt-Latenz: getContactPoints vs. q_delta threshold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shape", default=None,
                        help="Einzelne Form testen (sphere/cube/cylinder). Default: alle.")
    parser.add_argument("--size", type=float, default=5.0,
                        help="Objektgroesse in cm (sphere/cube).")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_threshold = float(cfg["observation"]["threshold"])
    thresholds = [0.02, 0.03, 0.04, sim_threshold, 0.06, 0.08, 0.10]

    if args.shape:
        shapes = [args.shape]
    else:
        shapes = ["sphere", "cube"]

    all_rows: list[dict] = []
    all_results: list[dict] = []

    for shape in shapes:
        obj_spec = {"shape": shape, "size_cm": args.size, "yaw_rad": 0.0}
        print(f"\n=== {shape} {args.size}cm — {args.episodes} Episoden ===")
        rows, results = run_analysis(cfg, obj_spec, args.episodes, thresholds)
        all_rows.extend(rows)
        all_results.extend(results)

        # Zusammenfassung
        fingers = list(cfg["finger_joints"].keys())
        for finger in fingers:
            contacts = [r[f"{finger}_first_contact"] for r in results
                        if r[f"{finger}_first_contact"] is not None]
            if not contacts:
                print(f"  {finger}: NIE Kontakt (getContactPoints)")
                continue
            avg_c = statistics.fmean(contacts)
            print(f"\n  {finger}: erster Kontakt (GT) @ step {avg_c:.1f} avg "
                  f"[{min(contacts)}-{max(contacts)}]")
            for thr in thresholds:
                latencies = [r[f"{finger}_latency_{thr}"] for r in results
                             if r.get(f"{finger}_latency_{thr}") is not None]
                if latencies:
                    avg_l = statistics.fmean(latencies)
                    print(f"    threshold={thr:.2f}: Latenz {avg_l:.1f} Steps avg "
                          f"[{min(latencies)}-{max(latencies)}]  "
                          f"= {avg_l * 5/240*1000:.0f} ms")
                else:
                    thr_steps = [r[f"{finger}_first_thr_{thr}"] for r in results
                                 if r.get(f"{finger}_first_thr_{thr}") is not None]
                    if not thr_steps:
                        print(f"    threshold={thr:.2f}: NIE erreicht")
                    else:
                        print(f"    threshold={thr:.2f}: erreicht aber kein GT-Kontakt?")

    # CSV speichern
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.config).stem

    if all_rows:
        out = _OUT_DIR / f"sim_contact_latency_{stem}_{ts}.csv"
        fields = list(all_rows[0].keys())
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[sim-latency] Rohdaten -> {out}")

    if all_results:
        out2 = _OUT_DIR / f"sim_contact_latency_summary_{stem}_{ts}.csv"
        fields2 = list(all_results[0].keys())
        with out2.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields2)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[sim-latency] Zusammenfassung -> {out2}")


if __name__ == "__main__":
    main()
