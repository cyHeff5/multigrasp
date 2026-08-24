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


class QDeltaBaseline:
    """Pro Joint: q_delta-Freilauf-Baseline (mean/std) ueber ein q_target-Grid,
    plus kalibrierte CUSUM-Alarmschwelle. Interpolation zwischen Grid-Punkten."""

    # Fallback-Std der statischen Komponente, wenn eine (alte) Baseline keine
    # Statik-Daten enthaelt (gemessen: Settle-Drift servo7 bis ~0.005 Std).
    _STATIC_STD_FALLBACK = 0.004

    def __init__(self, joints: Dict[str, dict], meta: dict):
        self.meta = dict(meta)
        self._grid:  Dict[str, np.ndarray] = {}
        self._mean:  Dict[str, np.ndarray] = {}
        self._std:   Dict[str, np.ndarray] = {}
        self._alarm: Dict[str, float] = {}
        # Statik: q_delta bei STEHENDEM Target (Settle-Offset der Sensorik,
        # NICHT ratenabhaengig). Getrennt von der Bewegungs-Baseline, damit die
        # Raten-Skalierung nur den dynamischen Anteil skaliert.
        self._sgrid: Dict[str, np.ndarray] = {}
        self._smean: Dict[str, np.ndarray] = {}
        self._sstd:  Dict[str, np.ndarray] = {}
        for j, d in joints.items():
            q = np.asarray(d["q"], dtype=float)
            order = np.argsort(q)
            self._grid[j] = q[order]
            self._mean[j] = np.asarray(d["mean"], dtype=float)[order]
            self._std[j]  = np.asarray(d["std"],  dtype=float)[order]
            self._alarm[j] = float(d["cusum_alarm"])
            st = d.get("static")
            if st and len(st.get("q", [])) > 0:
                sq = np.asarray(st["q"], dtype=float)
                so = np.argsort(sq)
                self._sgrid[j] = sq[so]
                self._smean[j] = np.asarray(st["mean"], dtype=float)[so]
                self._sstd[j]  = np.asarray(st["std"],  dtype=float)[so]

    def has_static(self, joint: str) -> bool:
        return joint in self._sgrid

    @property
    def joints(self) -> List[str]:
        return list(self._grid.keys())

    @property
    def calib_rate(self) -> float:
        """delta_norm, mit dem die Baseline gefahren wurde (0.0 wenn unbekannt)."""
        try:
            return float(self.meta.get("delta_norm") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def mean(self, joint: str, q_target: float) -> float:
        return float(np.interp(q_target, self._grid[joint], self._mean[joint]))

    def std(self, joint: str, q_target: float) -> float:
        return float(np.interp(q_target, self._grid[joint], self._std[joint]))

    def cusum_alarm(self, joint: str) -> float:
        return self._alarm[joint]

    def static_mean(self, joint: str, q_target: float) -> float:
        if joint not in self._sgrid:
            return 0.0
        return float(np.interp(q_target, self._sgrid[joint], self._smean[joint]))

    def static_std(self, joint: str, q_target: float) -> float:
        if joint not in self._sgrid:
            return self._STATIC_STD_FALLBACK
        return float(np.interp(q_target, self._sgrid[joint], self._sstd[joint]))

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
            if j in self._sgrid:
                out["joints"][j]["static"] = {
                    "q":    [round(float(v), 5) for v in self._sgrid[j]],
                    "mean": [round(float(v), 5) for v in self._smean[j]],
                    "std":  [round(float(v), 5) for v in self._sstd[j]],
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

        # Bewegungs-Bins (ssm == 0) und Statik-Bins (ssm >= settle_steps)
        # GETRENNT: der Settle-Offset im Stand ist ein statischer Sensorfehler
        # und nicht ratenabhaengig — die Raten-Skalierung des Detektors darf
        # nur den dynamischen Anteil (Bewegungs-Baseline minus Statik)
        # skalieren. Settle-Transient (0 < ssm < settle_steps) bleibt wie
        # gehabt ausgeschlossen.
        bins:  Dict[float, List[float]] = {}
        sbins: Dict[float, List[float]] = {}
        for qt, qd in cycles:
            ssm = _steps_since_move(qt)
            for q, d, s in zip(qt, qd, ssm):
                if s == 0:
                    bins.setdefault(round(float(q), 5), []).append(float(d))
                elif s >= settle_steps:
                    sbins.setdefault(round(float(q), 5), []).append(float(d))
        grid = sorted(bins.keys())
        mean = [float(np.mean(bins[q])) for q in grid]
        std  = [float(np.std(bins[q], ddof=1)) if len(bins[q]) > 1 else 0.0
                for q in grid]
        sgrid = sorted(q for q, v in sbins.items() if len(v) >= 5)
        smean = [float(np.mean(sbins[q])) for q in sgrid]
        sstd  = [float(np.std(sbins[q], ddof=1)) for q in sgrid]

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
        if sgrid:
            # Statik als Gerade durch die vorhandenen Stuetzpunkte extrapolieren
            # (position_noise-Daten vom 08.07.: Settle-Offset ~linear in q).
            # Mit nur einem Stuetzpunkt: Gerade durch den Ursprung.
            if len(sgrid) == 1:
                q0, m0 = sgrid[0], smean[0]
                slope, icept = (m0 / q0 if q0 > 1e-6 else 0.0), 0.0
            else:
                slope, icept = np.polyfit(sgrid, smean, 1)
            full_sq = [0.0] + sgrid + [1.0]
            full_sm = [float(slope * q + icept) if q not in sgrid
                       else smean[sgrid.index(q)] for q in full_sq]
            smax = max(sstd)
            full_ss = [smax if q not in sgrid else sstd[sgrid.index(q)]
                       for q in full_sq]
            joints_out[j]["static"] = {"q": full_sq, "mean": full_sm, "std": full_ss}

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

        # Raten-Skalierung: q_delta ist im Freilauf ~proportional zur kommandierten
        # Rate (Zeitkonstante ~6 Steps, gemessen an den fast/slow-Rampen vom
        # 2026-07-08; Auswertung 2026-08-24). Die Baseline wurde mit konstanter
        # Rate delta_norm gefahren — faehrt die Policy langsamer/stop-and-go,
        # wird die Erwartung mit v_ema/v_calib skaliert. Ohne Skalierung liegt
        # ein Joint mit halber Rate dauerhaft ~ -0.013 unter der Baseline
        # (= komplettes Detektionsbudget) und kann nie triggern.
        self.rate_scaling      = bool(det_cfg.get("rate_scaling", True))
        self.rate_tau          = float(det_cfg.get("rate_tau_steps", 6.0))
        self.std_floor         = float(det_cfg.get("std_floor", 0.001))
        # Wiederanlauf nach Pause: Anlauftransient des Servos (Groesse bis zur
        # Pause-Rampen-Messung unbekannt) -> Evidenz konservativ maskieren,
        # symmetrisch zur Settle-Maske beim Anhalten.
        self.restart_steps     = int(det_cfg.get("restart_steps", 12))
        self.restart_pause_min = int(det_cfg.get("restart_pause_min", 5))

        self._v_calib = baseline.calib_rate
        if self.rate_scaling and self._v_calib <= 0.0:
            raise ValueError(
                "contact_detector.rate_scaling braucht meta.delta_norm in der "
                "Baseline — eval/baseline_calibration.py neu fahren oder "
                "rate_scaling: false setzen.")

        watched = [j for js in self.finger_joints.values() for j in js]
        missing = [j for j in watched if j not in baseline.joints]
        if missing:
            raise ValueError(
                f"Baseline deckt Joints {missing} nicht ab — "
                f"eval/baseline_calibration.py mit passender Config neu fahren.")
        # Lazy-Import: sim/__init__ zieht sim.env, das wiederum dieses Modul
        # importiert — ein Modul-Level-Import von sim.hand waere zirkulaer.
        from sim.hand import CONTROL_JOINTS
        self._watched = watched
        self._idx = {j: CONTROL_JOINTS.index(j) for j in watched}
        self.reset()

    # Ab dieser Bewegung von q_measured gilt der Servo als angelaufen.
    _START_MOVE_Q = 0.01

    def reset(self) -> None:
        self._over  = {j: 0     for j in self._watched}
        self._bit   = {j: False for j in self._watched}
        self._cusum = {j: 0.0   for j in self._watched}
        self._cusum_hit = {j: False for j in self._watched}
        self._prev_qt: Dict[str, Optional[float]] = {j: None for j in self._watched}
        self._since_move   = {j: 0   for j in self._watched}
        self._v_ema        = {j: 0.0 for j in self._watched}
        self._restart_left = {j: 0   for j in self._watched}
        self._qm0:     Dict[str, Optional[float]] = {j: None  for j in self._watched}
        self._started = {j: False for j in self._watched}
        self._last: Dict[str, dict] = {}

    def update(self, q_target: Sequence[float],
               q_measured: Sequence[float]) -> np.ndarray:
        """Ein Policy-Step. Liefert Bits in der Reihenfolge von finger_joints."""
        for j in self._watched:
            i  = self._idx[j]
            qt = float(q_target[i])
            qd = qt - float(q_measured[i])

            prev  = self._prev_qt[j]
            dq    = 0.0 if prev is None else max(0.0, qt - prev)
            moved = prev is None or abs(qt - prev) > 1e-9
            if moved:
                if prev is not None and self._since_move[j] >= self.restart_pause_min:
                    self._restart_left[j] = self.restart_steps
                self._since_move[j] = 0
            else:
                self._since_move[j] += 1
            self._prev_qt[j] = qt

            # Kommandierte Rate als EMA mit der Servo-Zeitkonstante: dieselbe
            # Filterung, mit der der echte q_delta der Rate folgt -> die
            # skalierte Baseline-Erwartung bleibt auch bei Stop-and-go und
            # waehrend einer Probe (Target steht, Erwartung klingt ab) richtig.
            self._v_ema[j] += (dq - self._v_ema[j]) / self.rate_tau

            # Anlauf-Gate: solange der Servo nicht messbar losgefahren ist,
            # ist keine Detektion moeglich — die Totzone streut ueber Zyklen
            # und laeuft bei warmem Start ueber die q_target-Maske hinaus
            # (Freilauf-Zyklus 7 vom 08.07. triggerte sonst bei Step 27).
            # Sobald er anlaeuft, maskiert die Wiederanlauf-Maske den
            # Aufhol-Transienten.
            qm = float(q_measured[i])
            if self._qm0[j] is None:
                self._qm0[j] = qm
            if not self._started[j]:
                if qm > self._qm0[j] + self._START_MOVE_Q:
                    self._started[j] = True
                    self._restart_left[j] = max(self._restart_left[j],
                                                self.restart_steps)
                else:
                    self._over[j] = 0
                    self._bit[j] = False
                    self._last[j] = {"resid": 0.0, "thr": None,
                                     "masked": "not_started"}
                    continue

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

            if self._restart_left[j] > 0 and self._since_move[j] == 0:
                # Wiederanlauf-Transient: Evidenz einfrieren (wie Settle-Maske).
                self._restart_left[j] -= 1
                self._last[j] = {"resid": None, "thr": None, "masked": "restart",
                                 "cusum": self._cusum[j]}
                continue

            scale = 1.0
            if self.rate_scaling:
                scale = min(1.5, self._v_ema[j] / self._v_calib)
            # Erwartung = Statik (Settle-Offset, NICHT ratenabhaengig)
            #           + scale * dynamischer Anteil (Tracking-Lag ~ Rate).
            # Std analog: bewegter Anteil skaliert, stehender Anteil aus den
            # Statik-Bins (Settle-Drift, z.B. servo7 am Cap bis ~0.02).
            stat  = self.baseline.static_mean(j, qt)
            mov   = self.baseline.mean(j, qt)
            resid = qd - (stat + scale * (mov - stat))
            w_st  = 1.0 - min(scale, 1.0)
            sd    = max(self.std_floor,
                        scale * self.baseline.std(j, qt),
                        w_st * self.baseline.static_std(j, qt))
            thr   = max(self.residual_min, self.sigma_k * sd)

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
                             "scale": scale, "cusum": self._cusum[j]}

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


def synthetic_baseline(joints: Sequence[str], *, v_calib: float,
                       tau_steps: Dict[str, float], sigma: float,
                       deadband_q: Dict[str, float],
                       mean_err_q: Optional[Dict[str, float]] = None,
                       phys_lag_q: Optional[Dict[str, tuple]] = None,
                       static_mean_q: Optional[Dict[str, tuple]] = None,
                       static_sigma: float = 0.002,
                       cusum_alarm: float = 0.03, n_grid: int = 101,
                       meta: Optional[dict] = None) -> QDeltaBaseline:
    """Analytische Freilauf-Baseline fuer die SIMULIERTE Hand (Training).

    Bildet nach, was eval/baseline_calibration.py auf der echten Hand messen
    wuerde, wenn der Servo ein Verzoegerungsglied 1. Ordnung (Zeitkonstante
    tau_steps, in Policy-Steps) mit Anlauf-Totzone deadband_q ist: in der
    Totzone waechst q_delta mit dem Target (qd = q), danach faellt es
    exponentiell auf den Gleichgewichts-Lag v_calib * tau. mean_err_q
    modelliert den Kalibrierfehler einer Session (Baseline != wahre Hand);
    im Training wird er pro Episode randomisiert, damit die Policy nicht von
    einer perfekten Kalibrierung abhaengt.
    """
    joints_out: Dict[str, dict] = {}
    for j in joints:
        tau = float(tau_steps[j])
        d   = float(deadband_q[j])
        err = float((mean_err_q or {}).get(j, 0.0))
        eq  = v_calib * tau                                  # Gleichgewichts-Lag
        q   = np.linspace(0.0, 1.0, n_grid)
        # Sensor-Statik (Gain/Offset der Input-Kalibrierung) steckt auf der
        # echten Hand MIT in der gemessenen Baseline — hier genauso: die
        # Bewegungs-Baseline ist Dynamik + Statik-Gerade.
        sl, ic = (static_mean_q or {}).get(j, (0.0, 0.0))
        mean = (np.where(q <= d, q,
                         eq + (d - eq) * np.exp(-(q - d) / max(eq, 1e-6)))
                + sl * q + ic + err)
        # Nativer Physik-Tracking-Lag der Sim (PyBullet-Positionsregler,
        # einmal pro Env im Freilauf vermessen): steckt auf der echten Hand
        # automatisch in der gemessenen Baseline, hier additiv.
        if phys_lag_q and j in phys_lag_q:
            pq, pv = phys_lag_q[j]
            mean = mean + np.interp(q, pq, pv)
        joints_out[j] = {"q": q.tolist(), "mean": mean.tolist(),
                         "std": [float(sigma)] * n_grid,
                         "cusum_alarm": float(cusum_alarm)}
        # Statik: qd im Stand = -(gain*q + offset) des Sensor-Modells; die
        # Session-Kalibrierung wuerde das direkt messen.
        joints_out[j]["static"] = {
            "q":    [0.0, 1.0],
            "mean": [float(ic), float(sl + ic)],
            "std":  [float(static_sigma)] * 2,
        }
    m = {"delta_norm": float(v_calib), "synthetic": True}
    m.update(meta or {})
    return QDeltaBaseline(joints_out, m)
