# Baseline-korrigierte Kontakterkennung fuer die echte AR10-Hand.
#
# Hintergrund (Messdaten + Herleitung aller Schwellwerte:
# artifacts/analysis/SENSOR_ANALYSIS_FINDINGS.md):
# Der q_delta-Tracking-Fehler der Servos ist kein Rauschen, sondern eine
# reproduzierbare Funktion der Position (Zyklus-zu-Zyklus-Std nur ~0.003).
# Nach Abzug einer pro Session kalibrierten Baseline (eval/baseline_calibration.py)
# ist Kontakt schon ab Residuum ~0.012 vom Freilauf trennbar — statt Threshold
# 0.05 auf rohem q_delta, der die Kugel erst bei fast geschlossener Hand sieht.
#
# Drei Mechanismen:
#   1. Startup-Maske:  q_target < startup_mask_q -> Bit immer 0. Die Servos
#      haben eine Anlauf-Totzone (~11-15 Steps), in der q_delta systematisch
#      bis 0.072 spikt — dort ist keine Detektion moeglich.
#   2. Primaerdetektor: residuum = q_delta - baseline_mean(q_target) muss fuer
#      `persistence` Steps in Folge ueber max(residual_min, sigma_k*baseline_std)
#      liegen -> Bit setzt. Hysterese: Bit faellt erst, wenn das Residuum unter
#      release_factor*Threshold sinkt (kein Flackern in der Policy-Observation).
#      Faengt den transienten Erstkontakt-Buckel (~10-15 Steps, +0.010-0.015).
#   3. CUSUM (latcht):  S = max(0, S + residuum - drift). Alarm bei S ueber der
#      kalibrierten Freilauf-Schwelle. Faengt langsam wachsende Blockierung
#      (Objekt geometrisch eingeklemmt), die der Primaerdetektor unterschreitet.
#
# Die Baseline gilt fuer die Schliessrate, mit der kalibriert wurde (delta_norm).
# Faehrt die Policy langsamer oder pausiert sie, sinkt q_delta unter die
# Baseline -> Residuum negativ -> keine False Positives (Baseline ist eine
# obere Huellkurve). Nur schnelleres Fahren als kalibriert waere unsicher.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from sim.hand import CONTROL_JOINTS


class QDeltaBaseline:
    """Pro Joint: q_delta-Freilauf-Baseline (mean/std) ueber ein q_target-Grid,
    plus kalibrierte CUSUM-Alarmschwelle. Interpolation zwischen Grid-Punkten."""

    def __init__(self, joints: Dict[str, dict], meta: dict):
        self.meta = dict(meta)
        self._grid:  Dict[str, np.ndarray] = {}
        self._mean:  Dict[str, np.ndarray] = {}
        self._std:   Dict[str, np.ndarray] = {}
        self._alarm: Dict[str, float] = {}
        for j, d in joints.items():
            q = np.asarray(d["q"], dtype=float)
            order = np.argsort(q)
            self._grid[j] = q[order]
            self._mean[j] = np.asarray(d["mean"], dtype=float)[order]
            self._std[j]  = np.asarray(d["std"],  dtype=float)[order]
            self._alarm[j] = float(d["cusum_alarm"])

    @property
    def joints(self) -> List[str]:
        return list(self._grid.keys())

    def mean(self, joint: str, q_target: float) -> float:
        return float(np.interp(q_target, self._grid[joint], self._mean[joint]))

    def std(self, joint: str, q_target: float) -> float:
        return float(np.interp(q_target, self._grid[joint], self._std[joint]))

    def cusum_alarm(self, joint: str) -> float:
        return self._alarm[joint]

    # ── Persistenz ────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        out = {"meta": dict(self.meta), "joints": {}}
        for j in self.joints:
            out["joints"][j] = {
                "q":    [round(float(v), 5) for v in self._grid[j]],
                "mean": [round(float(v), 5) for v in self._mean[j]],
                "std":  [round(float(v), 5) for v in self._std[j]],
                "cusum_alarm": round(self._alarm[j], 5),
            }
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "QDeltaBaseline":
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data["joints"], data.get("meta", {}))


def _steps_since_move(qt: np.ndarray) -> np.ndarray:
    """Pro Sample: Steps seit der letzten q_target-Aenderung (0 = bewegt sich)."""
    out = np.zeros(len(qt), dtype=int)
    for k in range(1, len(qt)):
        out[k] = 0 if abs(qt[k] - qt[k - 1]) > 1e-9 else out[k - 1] + 1
    return out


def build_baseline(per_joint: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                   *, startup_mask_q: float, settle_steps: int,
                   cusum_drift: float, cusum_margin: float,
                   meta: Optional[dict] = None) -> QDeltaBaseline:
    """Baut die Baseline aus Freilauf-CLOSE-Zyklen.

    per_joint[joint] = Liste pro Zyklus von (q_target-Array, q_delta-Array),
    in Step-Reihenfolge der Rampe (alle Zyklen fahren dieselbe deterministische
    Rampe -> gleicher Step = gleiches q_target).

    Grid = die exakten q_target-Werte der Rampe; mean/std pro Grid-Punkt ueber
    alle Zyklen. Samples im SETTLE-TRANSIENT (Target hat gerade aufgehoert sich
    zu bewegen, < settle_steps her — passiert am pip_cap) werden ausgeschlossen:
    dort faellt q_delta ueber ~10-15 Steps vom Tracking- auf den Settle-Wert,
    und ein gemischter Bin wuerde am Cap systematische False Positives erzeugen
    (offline gemessen: Bits in fast jedem Freilauf-Zyklus bei Cap-Ankunft).
    Der Detektor maskiert dieselben Steps (siehe ContactDetector.update).

    CUSUM-Alarmschwelle: leave-one-out — jeder Zyklus wird als Residuum gegen
    den Mittelwert der ANDEREN Zyklen gefahren; Schwelle = cusum_margin * max
    aller so erreichten CUSUM-Werte (Untergrenze 0.02).
    """
    joints_out: Dict[str, dict] = {}
    for j, cycles in per_joint.items():
        if len(cycles) < 3:
            raise ValueError(f"{j}: mindestens 3 Zyklen noetig, {len(cycles)} gegeben.")

        # mean/std pro Grid-Punkt; Settle-Transient-Samples ausgeschlossen
        bins: Dict[float, List[float]] = {}
        for qt, qd in cycles:
            ssm = _steps_since_move(qt)
            for q, d, s in zip(qt, qd, ssm):
                if 0 < s < settle_steps:
                    continue
                bins.setdefault(round(float(q), 5), []).append(float(d))
        grid = sorted(bins.keys())
        mean = [float(np.mean(bins[q])) for q in grid]
        std  = [float(np.std(bins[q], ddof=1)) if len(bins[q]) > 1 else 0.0
                for q in grid]

        # CUSUM leave-one-out: Alignment ueber Step-Index (deterministische Rampe).
        # Nur Steps mit BEWEGTEM Target (ssm == 0): im Stand driftet der
        # Settle-Wert pro Zyklus um bis zu +-0.02 (servo7) — ein CUSUM wuerde
        # diesen Offset endlos aufsummieren (offline gemessen: Fehlalarm).
        n_steps = min(len(qt) for qt, _ in cycles)
        qd_mat = np.stack([qd[:n_steps] for _, qd in cycles])   # (Zyklen, Steps)
        qt_ref = cycles[0][0][:n_steps]
        ssm    = _steps_since_move(qt_ref)
        active = (qt_ref >= startup_mask_q) & (ssm == 0)
        max_cusum = 0.0
        for c in range(len(cycles)):
            others = np.delete(qd_mat, c, axis=0).mean(axis=0)
            resid  = qd_mat[c] - others
            s = 0.0
            for r, a in zip(resid, active):
                if not a:
                    continue
                s = max(0.0, s + r - cusum_drift)
                max_cusum = max(max_cusum, s)
        alarm = max(0.02, cusum_margin * max_cusum)

        joints_out[j] = {"q": grid, "mean": mean, "std": std, "cusum_alarm": alarm}

    return QDeltaBaseline(joints_out, meta or {})


class ContactDetector:
    """Erzeugt die binaeren Kontakt-Bits pro Finger fuer die Policy-Observation.

    Drop-in-Ersatz fuer die alte Logik `q_delta > threshold` in
    eval/policy_runner._binary_obs — gleiche Bit-Semantik (live, kein Latch im
    Primaerdetektor), nur empfindlicher und ohne Startup-False-Positives.
    Der CUSUM-Kanal latcht bewusst: kumulierte Blockierungs-Evidenz ist
    monoton, ein einmal sicherer Kontakt bleibt Kontakt.
    """

    def __init__(self, det_cfg: dict, finger_joints: Dict[str, List[str]],
                 baseline: QDeltaBaseline):
        self.baseline       = baseline
        self.finger_joints  = {f: list(js) for f, js in finger_joints.items()}
        self.fingers        = list(finger_joints.keys())
        self.residual_min   = float(det_cfg["residual_min"])
        self.sigma_k        = float(det_cfg["sigma_k"])
        self.persistence    = int(det_cfg["persistence"])
        self.release_factor = float(det_cfg["release_factor"])
        self.startup_mask_q = float(det_cfg["startup_mask_q"])
        self.settle_steps   = int(det_cfg["settle_steps"])
        self.cusum_drift    = float(det_cfg["cusum_drift"])

        watched = [j for js in self.finger_joints.values() for j in js]
        missing = [j for j in watched if j not in baseline.joints]
        if missing:
            raise ValueError(
                f"Baseline deckt Joints {missing} nicht ab — "
                f"eval/baseline_calibration.py mit passender Config neu fahren.")
        self._watched = watched
        self._idx = {j: CONTROL_JOINTS.index(j) for j in watched}
        self.reset()

    def reset(self) -> None:
        self._over  = {j: 0     for j in self._watched}
        self._bit   = {j: False for j in self._watched}
        self._cusum = {j: 0.0   for j in self._watched}
        self._cusum_hit = {j: False for j in self._watched}
        self._prev_qt: Dict[str, Optional[float]] = {j: None for j in self._watched}
        self._since_move = {j: 0 for j in self._watched}
        self._last: Dict[str, dict] = {}

    def update(self, q_target: Sequence[float],
               q_measured: Sequence[float]) -> np.ndarray:
        """Ein Policy-Step. Liefert Bits in der Reihenfolge von finger_joints."""
        for j in self._watched:
            i  = self._idx[j]
            qt = float(q_target[i])
            qd = qt - float(q_measured[i])

            prev = self._prev_qt[j]
            if prev is None or abs(qt - prev) > 1e-9:
                self._since_move[j] = 0
            else:
                self._since_move[j] += 1
            self._prev_qt[j] = qt

            if qt < self.startup_mask_q:
                # Anlauf-Totzone: keine Detektion moeglich, Zustand sauber halten.
                self._over[j] = 0
                self._bit[j] = False
                self._last[j] = {"resid": 0.0, "thr": None, "masked": "startup"}
                continue

            if 0 < self._since_move[j] < self.settle_steps:
                # Settle-Transient: Target steht (Cap erreicht / Policy pausiert),
                # q_delta faellt gerade vom Tracking- auf den Settle-Wert. Die
                # Baseline hat hier keine gueltigen Daten -> Joint-Zustand
                # einfrieren (Bit haelt, aber kein Auf-/Abbau von Evidenz).
                self._last[j] = {"resid": None, "thr": None, "masked": "settle",
                                 "cusum": self._cusum[j]}
                continue

            resid = qd - self.baseline.mean(j, qt)
            thr   = max(self.residual_min, self.sigma_k * self.baseline.std(j, qt))

            if resid > thr:
                self._over[j] += 1
                if self._over[j] >= self.persistence:
                    self._bit[j] = True
            elif self._bit[j] and resid > self.release_factor * thr:
                pass  # Hysterese: Bit haelt
            else:
                self._over[j] = 0
                self._bit[j] = False

            # CUSUM nur bei BEWEGTEM Target (siehe build_baseline): im Stand
            # driftet der Settle-Wert pro Zyklus — Aufsummieren waere Fehlalarm.
            if self._since_move[j] == 0:
                self._cusum[j] = max(0.0, self._cusum[j] + resid - self.cusum_drift)
                if self._cusum[j] > self.baseline.cusum_alarm(j):
                    self._cusum_hit[j] = True

            self._last[j] = {"resid": resid, "thr": thr, "masked": False,
                             "cusum": self._cusum[j]}

        out = np.zeros(len(self.fingers), dtype=np.float32)
        for fi, f in enumerate(self.fingers):
            if any(self._bit[j] or self._cusum_hit[j] for j in self.finger_joints[f]):
                out[fi] = 1.0
        return out

    def diagnostics(self) -> Dict[str, dict]:
        """Zustand des letzten update()-Aufrufs pro Joint (fuer Logging)."""
        out = {}
        for j in self._watched:
            d = dict(self._last.get(j, {}))
            d.update({"bit": self._bit[j], "over": self._over[j],
                      "cusum_hit": self._cusum_hit[j]})
            out[j] = d
        return out
