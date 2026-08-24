# Schliesst die trainierte Policy in FREILUFT (kein Objekt in Reichweite) und
# zeigt live, wie weit die Fingerkuppen an den Daumen kommen.
#
#   python gui_freeair.py --config configs/precision.yaml \
#       --model artifacts/models/precision/seed_0_dr/best/best_model
#
# Das Objekt wird nach dem Reset weit weggestellt und statisch gemacht, damit
# die Hand frei schliesst. Erwartung laut Sim: q_mcp faehrt auf 1.0, q_pip auf
# den Cap, und der Abstand Zeige-/Mittelkuppe zum Daumen geht gegen 0.
from __future__ import annotations

import argparse

import pybullet as p
import yaml

from eval.policy_runner import load_policy
from sim.env import GraspEnv
from sim.hand import CONTROL_JOINTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/precision.yaml")
    ap.add_argument("--model",  default="artifacts/models/precision/seed_0_dr/best/best_model")
    ap.add_argument("--seed",   type=int, default=1000)
    ap.add_argument("--no-gui", action="store_true", help="ohne Fenster (fuer Server/CI)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    policy = load_policy(args.model)
    env    = GraspEnv(cfg, render_mode=None if args.no_gui else "human")
    obs, _ = env.reset(seed=args.seed)

    # Objekt aus dem Weg raeumen: weit weg + statisch (mass=0), damit es weder
    # faellt noch den Drop-Check ausloest.
    p.resetBasePositionAndOrientation(env._obj.object_id, [1.0, 0.0, 0.5], [0, 0, 0, 1],
                                      physicsClientId=env._cid)
    p.changeDynamics(env._obj.object_id, -1, mass=0, physicsClientId=env._cid)

    link = {p.getJointInfo(env._hand_id, i, physicsClientId=env._cid)[12].decode(): i
            for i in range(p.getNumJoints(env._hand_id, physicsClientId=env._cid))}

    def gap(a: str, b: str) -> float:
        cps = p.getClosestPoints(env._hand_id, env._hand_id, distance=1.0,
                                 linkIndexA=link[a], linkIndexB=link[b],
                                 physicsClientId=env._cid)
        return min(c[8] for c in cps) * 100 if cps else float("nan")

    groups = [g[0] for g in cfg["action_groups"]]
    print(f"\nGruppen: {groups}   (Werte unten in derselben Reihenfolge)")
    print(f"{'step':>5} {'bits':>5} {'q je Gruppe':>28} {'Zeige<->Daumen':>15} {'Mittel<->Daumen':>16}")

    step = 0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _r, terminated, _t, info = env.step(action)
        step += 1
        if step % 25 == 0 or step == 1 or terminated:
            q = env._hand.q_target()
            qs = " ".join(f"{q[CONTROL_JOINTS.index(g)]:.3f}" for g in groups)
            print(f"{step:>5} {int(obs[:len(cfg['finger_joints'])].sum()):>5} {qs:>28} "
                  f"{gap('fingertip4','thumbtip'):>15.2f} {gap('fingertip3','thumbtip'):>16.2f}")
        if terminated:
            print(f"\nEnde nach {info['step_count']} Steps  "
                  f"(triggered={info['lift_triggered']}, lifted={info['lifted']})")
            print("q_target:", [round(v, 3) for v in env._hand.q_target()])
            print("q_meas  :", [round(v, 3) for v in env._hand.q_measured()])
            break

    if not args.no_gui:
        input("\nEnter zum Schliessen des Fensters: ")
    env.close()


if __name__ == "__main__":
    main()
