"""Testet, ob die Finger stark genug sind, ein Objekt zu greifen und anzuheben.

Umgeht die Policy: die Finger werden skriptiert voll geschlossen, danach laeuft der
eingebaute Lift-Test der Env. Gemeldet wird pro Masse die Lift-Erfolgsrate — so
findest du die schwerste noch hebbare Masse beim aktuellen motor_force.

Beispiele:
    python grip_test.py --shape cube     --size 5
    python grip_test.py --shape sphere   --size 6 --masses 0.1 0.2 0.3 0.5
    python grip_test.py --shape cylinder --thickness 2 --height 8.5 --motor 0.3
    python grip_test.py --shape cube --size 5 --gui        # zuschauen
"""
from __future__ import annotations

import argparse
import numpy as np
import pybullet as p
import yaml

from sim import GraspEnv


def build_spec(args) -> dict:
    spec = {"shape": args.shape, "yaw_rad": 0.0}
    if args.shape in ("sphere", "cube"):
        spec["size_cm"] = args.size
    if args.shape in ("cylinder", "rect_cylinder"):
        spec["thickness_cm"] = args.thickness
        spec["height_cm"]    = args.height
    if args.shape == "rect_cylinder":
        spec["width_cm"] = args.width
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description="Griff-/Hebekraft-Test der AR10-Finger.")
    ap.add_argument("--config", default="configs/precision.yaml")
    ap.add_argument("--shape",  default="cube",
                    choices=["sphere", "cube", "cylinder", "rect_cylinder"])
    ap.add_argument("--size",      type=float, default=5.0, help="cm (sphere/cube)")
    ap.add_argument("--thickness", type=float, default=2.0, help="cm (cylinder/rect)")
    ap.add_argument("--width",     type=float, default=2.0, help="cm (rect_cylinder)")
    ap.add_argument("--height",    type=float, default=8.5, help="cm (cylinder/rect)")
    ap.add_argument("--masses", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3, 0.5],
                    help="zu testende Massen in kg")
    ap.add_argument("--motor", type=float, default=None,
                    help="motor_force (Nm) ueberschreiben, sonst aus Config")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.motor is not None:
        cfg["physics"]["motor_force"] = {"min": args.motor, "max": args.motor}
    mf = cfg["physics"]["motor_force"]
    spec = build_spec(args)

    print(f"Objekt: {spec}")
    print(f"motor_force: {mf['min']}-{mf['max']} Nm   trials/Masse: {args.trials}\n")
    print(f"{'Masse(kg)':>9} {'Lift':>7} {'Drop':>6} {'Rate':>7}")

    for mass in args.masses:
        lift = drop = 0
        for t in range(args.trials):
            env = GraspEnv(cfg, render_mode="human" if args.gui else None)
            obs, _ = env.reset(seed=1000 + t, options={"obj_spec": dict(spec)})
            # feste Masse erzwingen (reset zieht sie sonst dichtebasiert neu)
            p.changeDynamics(env._obj.object_id, -1, mass=mass, physicsClientId=env._cid)
            close = np.array([n - 1 for n in env.action_space.nvec])  # max = schliessen
            for _ in range(cfg["episode"]["max_steps"]):
                obs, r, term, trunc, info = env.step(close)
                if term:
                    break
            lift += int(info["lifted"])
            drop += int(info["dropped"])
            env.close()
        print(f"{mass:>9.2f} {lift:>4}/{args.trials} {drop:>4}/{args.trials} {lift/args.trials*100:>6.0f}%")

    print("\nInterpretation: Die schwerste Masse mit hoher Rate = Grenze der Fingerkraft.")
    print("Zu schwach -> motor_force in der Config erhoehen (--motor zum Ausprobieren).")


if __name__ == "__main__":
    main()
