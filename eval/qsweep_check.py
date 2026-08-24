# q-Normierungs-Check: bedeutet q auf der echten Hand dasselbe wie in der Sim?
#
# Faehrt die Hand OHNE Objekt auf feste q-Werte, haelt jeden Punkt, bis die
# Bewegung ausgelaufen ist, und liest dann q_measured. Im Freilauf muss
# q_measured == q_target sein; bleibt eine Differenz stehen, ist das Kommando
# ueber den physikalischen Anschlag hinaus gefahren (Maestro clippt still) oder
# die Sensor-Kalibrierung des Kanals stimmt nicht.
#
# WICHTIG (Korrektur 2026-08-24): der Restfehler-Check sieht NUR Clipping und
# Kalibrierfehler. Die Hub->Winkel-Nichtlinearitaet der 4-Stab-Mechanik kann er
# prinzipiell NICHT sehen — Puls (Kommando) und Poti (Messung) sitzen beide auf
# dem Lead Screw und sind sich im Freilauf immer einig, egal wie krumm die
# Mechanik dazwischen ist. Die Nichtlinearitaet entscheidet ausschliesslich der
# FOTO-Vergleich: an JEDEM q-Punkt (vor allem 0.25/0.5/0.75 — die Endpunkte
# stimmen per Konstruktion ueberein!) ein Foto aus der Referenz-Perspektive,
# gegen sim_q*.png (linear) UND sim_q*_nonlin.png (Hersteller-Kennlinie,
# assets/stroke_angle_curves.yaml) halten. Welche Serie besser passt, gewinnt.
#
# Usage:
#   python -m eval.qsweep_check --port COM4                 # echte Hand
#   python -m eval.qsweep_check --sim-render                # nur Sim-Referenzbilder
from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

import yaml

from sim.hand import CONTROL_JOINTS, SERVO0_INIT

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "artifacts" / "analysis"
_STEPS     = [0.0, 0.25, 0.5, 0.75, 1.0]


def _targets_for(q: float, joints: list[str], hold_servo0: bool) -> list[float]:
    out = [0.0] * len(CONTROL_JOINTS)
    if hold_servo0:
        out[0] = SERVO0_INIT
    for j in joints:
        out[CONTROL_JOINTS.index(j)] = q
    return out


def run_hand(port: str, joints: list[str], settle_s: float) -> list[dict]:
    from hardware.ar10 import AR10Interface
    ar10 = AR10Interface(com_port=port)
    rows: list[dict] = []
    try:
        ar10.send_q_target(_targets_for(0.0, joints, True))
        time.sleep(2.0)
        for q in _STEPS:
            ar10.send_q_target(_targets_for(q, joints, True))
            time.sleep(settle_s)
            meas = [sum(v) / 5 for v in zip(*[ar10.read_q_measured() for _ in range(5)])]
            row = {"q_target": q}
            for j in joints:
                m = meas[CONTROL_JOINTS.index(j)]
                row[f"{j}_meas"] = round(m, 4)
                row[f"{j}_resid"] = round(q - m, 4)
            rows.append(row)
            resid = "  ".join(f"{j}:{row[f'{j}_resid']:+.3f}" for j in joints)
            print(f"  q={q:.2f}  Restfehler nach Settle:  {resid}")
    finally:
        ar10.send_q_target([0.0] * len(CONTROL_JOINTS))
        time.sleep(1.0)
        ar10.close()
    return rows


def render_sim(joints: list[str], out_dir: Path,
               kin_cfg: dict | None = None, suffix: str = "") -> None:
    import pybullet as p
    import pybullet_data
    from sim.hand import HandModel

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setAdditionalSearchPath(str(_REPO_ROOT / "assets"), physicsClientId=cid)
    hid = p.loadURDF(str(_REPO_ROOT / "assets" / "ar10_description" / "urdf" / "ar10.urdf"),
                     [0, 0, 0], useFixedBase=True, physicsClientId=cid)
    hand = HandModel(hid, {"motor_force": {"min": 0.5, "max": 0.5},
                           "fingertip_friction": {"min": 2.0, "max": 2.0},
                           "joint_damping": 0.1, "max_velocity": 0.5},
                     __import__("numpy").random.default_rng(0), client_id=cid,
                     kin_cfg=kin_cfg)
    # Seitenansicht: Daumen unten rechts, Finger darueber — in dieser Perspektive
    # ist der Spalt Fingerkuppe<->Daumen direkt ablesbar. Fotos moeglichst aus
    # demselben Winkel aufnehmen.
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.02, 0.08], distance=0.34,
        yaw=90, pitch=-10, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=2.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for q in _STEPS:
        hand.teleport_to(_targets_for(q, joints, True))
        img = p.getCameraImage(700, 700, view, proj,
                               renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
        name = f"sim_q{q:.2f}{suffix}.png"
        try:
            from PIL import Image
            Image.fromarray(__import__("numpy").reshape(img[2], (700, 700, 4))[:, :, :3]
                            .astype("uint8")).save(out_dir / name)
        except ImportError:
            print("  (Pillow fehlt — kein PNG geschrieben)")
            break
        print(f"  Sim-Referenz q={q:.2f} -> {out_dir / name}")
    p.disconnect(cid)


def main() -> None:
    ap = argparse.ArgumentParser(description="q-Normierungs-Check Sim vs. echte Hand.")
    ap.add_argument("--config", default="configs/power.yaml",
                    help="Bestimmt nur, welche Joints gefahren werden (finger_joints).")
    ap.add_argument("--port", default=None, help="COM-Port; ohne = nur Sim-Render.")
    ap.add_argument("--settle", type=float, default=2.0, help="Haltezeit je q-Punkt (s).")
    ap.add_argument("--sim-render", action="store_true", help="Sim-Referenzbilder schreiben.")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    joints = [j for jl in cfg["finger_joints"].values() for j in jl]
    joints = list(dict.fromkeys(joints))
    print(f"[qsweep] Joints: {joints}")

    if args.sim_render or args.port is None:
        print("\n-- Sim-Referenz (linear) --")
        render_sim(joints, _OUT_DIR / "qsweep")
        curves = _REPO_ROOT / "assets" / "stroke_angle_curves.yaml"
        if curves.exists():
            print("-- Sim-Referenz (nichtlinear, Hersteller-Kennlinie) --")
            render_sim(joints, _OUT_DIR / "qsweep",
                       kin_cfg={"stroke_angle_map": str(curves)}, suffix="_nonlin")

    if args.port is None:
        print("\n[qsweep] Kein --port: echte Hand uebersprungen.")
        return

    print("\n-- Echte Hand (OHNE Objekt, nichts im Arbeitsraum!) --")
    input("Enter -> Start ...")
    rows = run_hand(args.port, joints, args.settle)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _OUT_DIR / f"qsweep_{Path(args.config).stem}_{ts}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[qsweep] -> {path}")
    print("Deutung: Restfehler deutlich > 0 -> Kommando faehrt ueber den Anschlag "
          "(servo_limits.yaml / Maestro clippt) oder Input-Kalibrierung falsch. "
          "Die Hub-zu-Winkel-Nichtlinearitaet sieht dieser Check NICHT (Puls und "
          "Poti sind beide Hub-Koordinaten) — dafuer an JEDEM q-Punkt ein Foto "
          "aus der Referenz-Perspektive machen und gegen sim_q*.png (linear) und "
          "sim_q*_nonlin.png (Kennlinie) halten; entscheidend sind 0.25-0.75.")


if __name__ == "__main__":
    main()
