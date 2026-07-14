# Offline-Validierung des ContactDetector auf den aufgezeichneten CSVs vom
# 2026-07-08 — OHNE Hardware. Nutzt exakt den Produktionscode
# (hardware/contact_detector.py: build_baseline + ContactDetector).
#
# Prueft zwei Dinge:
#   1. Freilauf (10 Zyklen, leave-one-out): Detektor darf NIE ansprechen.
#      Jeder Zyklus wird gegen eine Baseline aus den 9 ANDEREN Zyklen
#      replayed — so wird nicht auf den eigenen Kalibrierdaten getestet.
#   2. Kugel (5 Zyklen): Detektor soll pro Zyklus anschlagen, deutlich
#      frueher als der alte 0.05-Threshold. Baseline aus allen 10
#      Freilauf-Zyklen (entspricht dem Session-Workflow: erst kalibrieren,
#      dann greifen — die Runs lagen 13 min auseinander).
#
# KALT-/WARMSTART-CONFOUND (wichtig fuer die Interpretation!):
# In der Freilauf-CSV startet nur Zyklus 0 "kalt" (Hand komplett offen,
# q_measured=0); die Zyklen 1-9 starten "warm" (q_measured ~0.02-0.05, Hand
# nach OPEN nicht ganz zurueckgefahren). Die Kugel-Zyklen starten alle KALT
# (1 s Settle + Operator-Pause). Kalte Starts haben in Steps ~25-45 systematisch
# hoeheres q_delta als warme -> gegen eine warm-dominierte Baseline entsteht dort
# ein positives Residuum, das wie Kontakt aussieht. Deshalb wird Zyklus 0 im
# FP-Test separat ausgewiesen, und fruehe Kugel-Detektionen (Steps < ~60) sind
# NICHT als Erstkontakt belastbar. eval/baseline_calibration.py startet jeden
# Zyklus kalt (wie die echten Episoden) — auf einer frischen Kalibrierung
# verschwindet der Confound.
#
# Erwartung: spaete Blockierung ab q_target ~0.55-0.8 auf beiden MCP-Joints;
# alter 0.05-Threshold triggert erst ~0.86-0.89 oder faelschlich im Startup.
#
# Usage:  python -m eval.test_detector_offline [--config configs/precision.yaml]
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hardware.contact_detector import ContactDetector, build_baseline
from sim.hand                  import CONTROL_JOINTS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANALYSIS  = _REPO_ROOT / "artifacts" / "analysis"

_FREE_CSV   = _ANALYSIS / "servo_analysis_precision_20260708_122810.csv"
_SPHERE_CSV = _ANALYSIS / "contact_latency_precision_20260708_124132.csv"


def _free_cycles(joints: list[str]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """CLOSE-Phase der Freilauf-CSV als build_baseline-Input."""
    df = pd.read_csv(_FREE_CSV)
    df = df[df["phase"] == "CLOSE"].copy()
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {j: [] for j in joints}
    for _cyc, grp in df.groupby("cycle"):
        grp = grp.sort_values("step")
        for j in joints:
            out[j].append((grp[f"{j}_q_target"].to_numpy(),
                           grp[f"{j}_q_delta"].to_numpy()))
    return out


def _replay(detector: ContactDetector, cycle_df: pd.DataFrame,
            joints: list[str], cfg: dict) -> dict:
    """Spielt einen aufgezeichneten Zyklus durch den Detektor + Trigger-Machine."""
    detector.reset()
    trigger_n     = cfg["trigger_n"]
    confirm_steps = cfg["trigger_confirmation_steps"]

    fingers     = detector.fingers
    first_bit   = {f: None for f in fingers}
    consecutive = 0
    trigger_step = None

    for _, row in cycle_df.sort_values("step").iterrows():
        q_target = [0.0] * len(CONTROL_JOINTS)
        q_meas   = [0.0] * len(CONTROL_JOINTS)
        for j in joints:
            i = CONTROL_JOINTS.index(j)
            q_target[i] = row[f"{j}_q_target"]
            q_meas[i]   = row[f"{j}_q_measured"]

        bits = detector.update(q_target, q_meas)
        k = int(row["step"])
        for fi, f in enumerate(fingers):
            if bits[fi] and first_bit[f] is None:
                first_bit[f] = k

        if int(bits.sum()) >= trigger_n:
            consecutive += 1
        else:
            consecutive = 0
        if trigger_step is None and consecutive >= confirm_steps:
            trigger_step = k

    return {"first_bit": first_bit, "trigger_step": trigger_step}


def _old_detector_trigger(cycle_df: pd.DataFrame, joints: list[str],
                          cfg: dict, threshold: float) -> dict:
    """Alte Logik (rohes q_delta > threshold) zum Vergleich, gleiche Trigger-Machine."""
    finger_joints = cfg["finger_joints"]
    fingers       = list(finger_joints.keys())
    consecutive   = 0
    trigger_step  = None
    first_bit     = {f: None for f in fingers}
    for _, row in cycle_df.sort_values("step").iterrows():
        k = int(row["step"])
        bits = 0
        for f in fingers:
            hit = any(row[f"{j}_q_delta"] > threshold for j in finger_joints[f])
            if hit and first_bit[f] is None:
                first_bit[f] = k
            bits += int(hit)
        consecutive = consecutive + 1 if bits >= cfg["trigger_n"] else 0
        if trigger_step is None and consecutive >= cfg["trigger_confirmation_steps"]:
            trigger_step = k
    return {"first_bit": first_bit, "trigger_step": trigger_step}


def main() -> None:
    parser = argparse.ArgumentParser(description="ContactDetector Offline-Validierung.")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "precision.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    det_cfg = cfg["contact_detector"]

    joints = []
    for jlist in cfg["finger_joints"].values():
        for j in jlist:
            if j not in joints:
                joints.append(j)

    bl_kwargs = dict(startup_mask_q=float(det_cfg["startup_mask_q"]),
                     settle_steps=int(det_cfg["settle_steps"]),
                     cusum_drift=float(det_cfg["cusum_drift"]),
                     cusum_margin=float(det_cfg["cusum_margin"]))

    free = _free_cycles(joints)
    n_free = len(free[joints[0]])

    # ── 1. Freilauf leave-one-out: 0 False Positives gefordert ───────────────
    print("=" * 76)
    print(f"1. FREILAUF ({n_free} Zyklen, leave-one-out) — Detektor darf NIE ansprechen")
    print("=" * 76)
    free_df = pd.read_csv(_FREE_CSV)
    free_df = free_df[free_df["phase"] == "CLOSE"].copy()
    free_df["step"] = free_df.groupby("cycle").cumcount()  # relativer Step-Index
    fp_warm = 0
    for held_out in range(n_free):
        rest = {j: [c for i, c in enumerate(free[j]) if i != held_out]
                for j in joints}
        detector = ContactDetector(det_cfg, cfg["finger_joints"],
                                   build_baseline(rest, **bl_kwargs))
        cyc_df = free_df[free_df["cycle"] == held_out]
        res = _replay(detector, cyc_df, joints, cfg)
        hits = {f: s for f, s in res["first_bit"].items() if s is not None}
        tag = "(KALTSTART, Confound erwartet)" if held_out == 0 else ""
        if hits or res["trigger_step"] is not None:
            if held_out != 0:
                fp_warm += 1
            print(f"  Zyklus {held_out}: FALSE POSITIVE  bits={hits}  "
                  f"trigger={res['trigger_step']}  {tag}")
        else:
            print(f"  Zyklus {held_out}: still (kein Bit, kein Trigger)  OK  {tag}")
    print(f"\n  -> {fp_warm}/{n_free - 1} WARME Zyklen mit False Positive"
          + ("  *** NICHT BESTANDEN ***" if fp_warm else "   BESTANDEN")
          + "\n     (Zyklus 0 zaehlt nicht: kalter Start vs. warme Baseline ist der"
          + "\n      bekannte Confound; frische Kalibrierung startet alle Zyklen kalt)")

    # ── 2. Kugel: Detektion + Vergleich mit altem 0.05-Threshold ─────────────
    print()
    print("=" * 76)
    print("2. KUGEL (5 Zyklen) — neuer Detektor vs. alter 0.05-Threshold")
    print("=" * 76)
    detector = ContactDetector(det_cfg, cfg["finger_joints"],
                               build_baseline(free, **bl_kwargs))
    sphere = pd.read_csv(_SPHERE_CSV)
    threshold_old = float(cfg["observation"]["threshold"])

    def fmt_q(cyc_df: pd.DataFrame, step: int | None) -> str:
        if step is None:
            return "—"
        qt = cyc_df.loc[cyc_df["step"] == step, "servo6_q_target"].iloc[0]
        return f"step {step:3d} (q={qt:.2f})"

    for cyc, grp in sphere.groupby("cycle"):
        new = _replay(detector, grp, joints, cfg)
        old = _old_detector_trigger(grp, joints, cfg, threshold_old)
        print(f"\n  Zyklus {cyc}:")
        for f in detector.fingers:
            print(f"    {f:7s}: neu erstes Bit {fmt_q(grp, new['first_bit'][f]):22s} "
                  f"| alt {fmt_q(grp, old['first_bit'][f])}")
        print(f"    TRIGGER: neu {fmt_q(grp, new['trigger_step']):22s} "
              f"| alt {fmt_q(grp, old['trigger_step'])}")


if __name__ == "__main__":
    main()
