from __future__ import annotations

import numpy as np
import pybullet as p


# Reihenfolge der gesteuerten Gelenke - definiert die Indexierung aller q-Vektoren.
# 0.0 = vollständig offen, 1.0 = vollständig geschlossen (normalisiert).
CONTROL_JOINTS = [
    "servo0", "servo1",   # Daumen
    "servo2", "servo3",   # Pinky
    "servo4", "servo5",   # Ring
    "servo6", "servo7",   # Mittel
    "servo8", "servo9",   # Zeige
]

# PIP-zu-DIP Kopplung: welches DIP-Gelenk welchem PIP folgt.
DIP_MIMIC_MAP = {
    "servo3": "tip1",
    "servo5": "tip2",
    "servo7": "tip3",
    "servo9": "tip4",
}

# Geometrieparameter der 4-Stab-Mechanik aus den URDF <mimic> Tags.
DIP_MULTIPLIER = 0.49
DIP_OFFSET     = 0.16   # rad

# Fingertip-Links an denen Kontakte gemessen werden.
FINGERTIP_EE_MAP = {
    "servo1": "ee5",   # Daumen
    "servo3": "ee1",   # Pinky
    "servo5": "ee2",   # Ring
    "servo7": "ee3",   # Mittel
    "servo9": "ee4",   # Zeige
}

# Startposition Daumen-Abduktion.
SERVO0_INIT = 0.5

# Gear-Constraint sehr stark dass DIP rigid an PIP gekoppelt ist (4-Stab-Mechanik der echten Hand).
# Damit zieht das PIP-Ratchet die Fingerspitze automatisch mit, ohne separate Behandlung.
_GEAR_FORCE = 1.0e9


class HandModel:
    # AR10-Hand in PyBullet, positions-geregelt über normalisierte Zielwerte.
    # Motor-Kraft und Fingerkuppen-Reibung werden pro Episode zufällig gezogen (Domain Randomization).

    def __init__(
        self,
        hand_id:    int,
        physics_cfg: dict,
        rng:        np.random.Generator,
        client_id:  int = 0,
        servo_cfg:  dict | None = None,
        kin_cfg:    dict | None = None,
    ) -> None:
        self.hand_id = int(hand_id)
        self._cid    = int(client_id)
        self._rng    = rng

        # Hub->Winkel-Nichtlinearitaet der 4-Stab-Mechanik (2026-08-24):
        # reales q_norm ist eine HUB-Koordinate (Puls/Poti auf dem Lead Screw),
        # der Gelenkwinkel haengt davon nichtlinear ab. kin_cfg.stroke_angle_map
        # zeigt auf assets/stroke_angle_curves.yaml -> _norm_to_angle nutzt
        # winkel_norm = f(q), q_measured invertiert f. q_delta bleibt davon
        # unberuehrt (beide Seiten Hub-Koordinate). None = exakt altes
        # (lineares) Verhalten, checkpoint-kompatibel.
        self._kin_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        map_file = (kin_cfg or {}).get("stroke_angle_map")
        if map_file:
            self._kin_maps = self._load_stroke_angle_map(map_file)

        # Servo-Verhaltensmodell der echten AR10 (Messdaten 2026-07-08,
        # Auswertung artifacts/analysis/SENSOR_ANALYSIS_FINDINGS.md + Update
        # 2026-08-24): Verzoegerungsglied 1. Ordnung (tau ~125-145 ms beim
        # Schliessen, Oeffnen ~2x langsamer), Anlauf-Totzone beim Kaltstart
        # (q_delta-Peak 0.05-0.075), Sensor-Gain/-Offset-Fehler, ADC-
        # Quantisierung (1 Count ~ 0.0017 q) und statisches Rauschen (~0.001).
        # enabled: false -> exakt das alte Verhalten (Checkpoint-Kompatibilitaet).
        self._servo_cfg     = servo_cfg or {}
        self._servo_enabled = bool(self._servo_cfg.get("enabled", False))
        self._control_dt    = 5.0 / 240.0   # wird von env pro Episode gesetzt
        n = len(CONTROL_JOINTS)
        self._q_servo       = [0.0] * n
        self._servo_started = [False] * n
        self._tau_close     = [0.13] * n
        self._tau_open_f    = [2.0] * n
        self._deadband      = [0.05] * n
        self._sens_gain     = [0.0] * n
        self._sens_offset   = [0.0] * n

        # Kontaktnachgiebigkeit als Feder (SENSOR_ANALYSIS_FINDINGS.md 9.9,
        # gemessen 2026-08-25 mit visueller Ground Truth). Real gilt am Sensor:
        #   - zwischen physischem Erstkontakt und messbarem Signal liegen
        #     0.10-0.25 q Ueberfahrweg, in denen NICHTS messbar ist (onset),
        #   - danach waechst das statische Restresiduum linear mit dem
        #     Ueberfahrweg, Steigung k ~ 0.2-0.3 (starre Blockierung waere k=1).
        # Die Feder sitzt physisch verteilt (Fingerglied, Getriebe, gefederter
        # Daumen, Objekt) und ist im URDF nicht modelliert - sie muss es auch
        # nicht sein: die Policy sieht ausschliesslich q_delta, also wird die
        # gemessene Kennlinie direkt im Sensormodell reproduziert. Ohne das
        # lernt rate_probe einen Erstkontakt-Detektor, den es real nicht gibt.
        self._spring_cfg    = self._servo_cfg.get("contact_spring") or {}
        self._spring_on     = self._servo_enabled and bool(self._spring_cfg)
        self._spring_k      = [0.0] * n
        self._spring_onset  = [0.0] * n
        self._contact_q:  list[float | None] = [None] * n
        self._free_steps    = [0] * n

        # Pro Episode randomisiert. Bounds werden gespeichert, damit randomize_dynamics()
        # die Werte bei jedem Reset neu ziehen kann ohne die Hand neu zu laden.
        self._physics_cfg     = physics_cfg
        self._motor_force_cfg = physics_cfg["motor_force"]
        self._friction_cfg    = physics_cfg["fingertip_friction"]
        self._motor_force = _uniform(self._motor_force_cfg, rng)
        self._friction    = _uniform(self._friction_cfg, rng)

        # Joint-Damping + max_velocity. positionGain/velocityGain werden bewusst NICHT genutzt
        # weil PyBullet sie ignoriert sobald maxVelocity in setJointMotorControl2 gesetzt ist
        # (rate-limitierter Tracker statt PD).
        self._damping  = float(physics_cfg["joint_damping"])
        self._max_vel  = float(physics_cfg["max_velocity"])

        self.joint_index  = self._build_joint_index()
        self.joint_limits = self._load_joint_limits()
        self._q_target: list[float] = [0.0] * len(CONTROL_JOINTS)

        self._link_chain  = self._build_link_chain() if self._spring_on else {}

        self._init_dynamics()
        self._setup_dip_constraints()
        self.apply_q_target(self._q_target)

    # Setup
    def _build_joint_index(self) -> dict[str, int]:
        # Liest alle Joint-Namen aus dem URDF und erstellt name → index Mapping.
        idx: dict[str, int] = {}
        for i in range(p.getNumJoints(self.hand_id, physicsClientId=self._cid)):
            name = p.getJointInfo(self.hand_id, i, physicsClientId=self._cid)[1].decode()
            idx[name] = i
        return idx

    def _load_joint_limits(self) -> dict[str, tuple[float, float]]:
        # Liest physikalische Gelenkgrenzen aus dem URDF (in rad).
        # Fallback auf gemessene AR10-Grenzen falls URDF keine gültigen Limits enthält.
        # CONTROL_JOINTS und DIP-Joints, weil non-backdrivable Enforcement beide braucht.
        limits: dict[str, tuple[float, float]] = {}
        for name in CONTROL_JOINTS + list(DIP_MIMIC_MAP.values()):
            info = p.getJointInfo(self.hand_id, self.joint_index[name], physicsClientId=self._cid)
            lo, hi = float(info[8]), float(info[9])
            if lo >= hi:
                lo, hi = 0.17, 1.57
            limits[name] = (lo, hi)
        return limits

    def _init_dynamics(self) -> None:
        # Dämpfung auf alle gesteuerten Gelenke + DIP-Gelenke setzen.
        # Reibung nur auf Fingertip-Links (Gummi-Kappen).
        all_joints = CONTROL_JOINTS + list(DIP_MIMIC_MAP.values())
        for name in all_joints:
            p.changeDynamics(self.hand_id, self.joint_index[name],
                              jointDamping=self._damping, physicsClientId=self._cid)
        for ee_name in FINGERTIP_EE_MAP.values():
            p.changeDynamics(self.hand_id, self.joint_index[ee_name],
                              lateralFriction=self._friction, physicsClientId=self._cid)
        # Bullet legt um jede konvexe MESH-Kollision einen Sicherheitsabstand von
        # 1.0 mm (an einer perfekten Box nachgemessen, 25.08.). Jeder Hand-Link
        # meldet damit Kontakt einen Millimeter zu frueh — mehr als der
        # Geometriefehler der Huellen selbst (p95 0.2-0.8 mm). 0.1 mm statt 0
        # laesst Bullet numerischen Spielraum.
        margin = float(self._physics_cfg.get("collision_margin_m", 0.0001))
        for i in range(-1, p.getNumJoints(self.hand_id, physicsClientId=self._cid)):
            p.changeDynamics(self.hand_id, i, collisionMargin=margin,
                              physicsClientId=self._cid)

    def randomize_dynamics(self, rng: np.random.Generator) -> None:
        # Domain Randomization pro Episode auf der persistenten Hand: Motorkraft +
        # Fingerkuppen-Reibung neu ziehen. Motorkraft wirkt ueber force= in
        # apply_q_target, die Reibung wird hier direkt per changeDynamics gesetzt.
        self._motor_force = _uniform(self._motor_force_cfg, rng)
        self._friction    = _uniform(self._friction_cfg, rng)
        for ee_name in FINGERTIP_EE_MAP.values():
            p.changeDynamics(self.hand_id, self.joint_index[ee_name],
                              lateralFriction=self._friction, physicsClientId=self._cid)
        self._rng = rng
        if self._servo_enabled:
            sc = self._servo_cfg
            n  = len(CONTROL_JOINTS)
            self._tau_close   = [_uniform(sc["tau_close_s"], rng)      for _ in range(n)]
            self._tau_open_f  = [_uniform(sc["tau_open_factor"], rng)  for _ in range(n)]
            self._deadband    = [_uniform(sc["startup_deadband_q"], rng) for _ in range(n)]
            g  = float(sc.get("sensor_gain_err", 0.02))
            b  = float(sc.get("sensor_offset",   0.008))
            self._sens_gain   = [float(rng.uniform(-g, g)) for _ in range(n)]
            self._sens_offset = [float(rng.uniform(-b, b)) for _ in range(n)]
        if self._spring_on:
            n  = len(CONTROL_JOINTS)
            sp = self._spring_cfg
            # k und onset sind objekt-/positionsabhaengig (9.9, n=1 Zyklus) und
            # werden deshalb pro Episode UND pro Joint gezogen, nicht fixiert.
            self._spring_k     = [_uniform(sp["k"], rng)       for _ in range(n)]
            self._spring_onset = [_uniform(sp["onset_q"], rng) for _ in range(n)]
            self._contact_q    = [None] * n
            self._free_steps   = [0] * n

    def _setup_dip_constraints(self) -> None:
        # PyBullet ignoriert URDF <mimic> Tags, deshalb wird die PIP-DIP Kopplung
        # manuell über JOINT_GEAR Constraints nachgebaut (starre 4-Stab-Mechanik).
        # Gear-Ratio negativ weil beide Gelenkachsen im URDF in -x Richtung zeigen
        # (gleiche physikalische Richtung -> negativer Ratio für gleichsinnige Kopplung).
        for pip_name, dip_name in DIP_MIMIC_MAP.items():
            # DIP-Motor deaktivieren, sonst kämpft er gegen den Constraint.
            p.setJointMotorControl2(
                self.hand_id, self.joint_index[dip_name],
                controlMode=p.VELOCITY_CONTROL, force=0,
                physicsClientId=self._cid,
            )
            c = p.createConstraint(
                parentBodyUniqueId=self.hand_id,
                parentLinkIndex=self.joint_index[pip_name],
                childBodyUniqueId=self.hand_id,
                childLinkIndex=self.joint_index[dip_name],
                jointType=p.JOINT_GEAR,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                physicsClientId=self._cid,
            )
            p.changeConstraint(c, gearRatio=-1.0 / DIP_MULTIPLIER,
                                maxForce=_GEAR_FORCE, physicsClientId=self._cid)

    # Control
    def apply_q_target(self, q_target: list[float]) -> None:
        # Sendet PD-Positionsregelung an alle CONTROL_JOINTS.
        # DIP-Gelenke folgen über den Gear-Constraint automatisch.
        if len(q_target) != len(CONTROL_JOINTS):
            raise ValueError(f"Expected {len(CONTROL_JOINTS)} values, got {len(q_target)}.")

        self._q_target = [max(0.0, min(1.0, float(v))) for v in q_target]
        if self._servo_enabled:
            self._advance_servo()
        for idx, name in enumerate(CONTROL_JOINTS):
            # Mit Servo-Modell folgt der Motor dem internen Servo-Zustand; die
            # Dynamik kommt dann aus dem Modell, nicht aus maxVelocity (das
            # sonst als zweiter, konkurrierender Raten-Limiter wirken wuerde).
            motor_q = self._q_servo[idx] if self._servo_enabled else self._q_target[idx]
            angle = self._norm_to_angle(name, motor_q)
            p.setJointMotorControl2(
                self.hand_id, self.joint_index[name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                maxVelocity=10.0 if self._servo_enabled else self._max_vel,
                force=self._motor_force,
                physicsClientId=self._cid,
            )

    def set_control_dt(self, dt: float) -> None:
        # Dauer eines Policy-Steps (substeps/sim_hz) — env setzt das pro
        # Episode, weil substeps randomisiert wird.
        self._control_dt = float(dt)

    def _advance_servo(self) -> None:
        # Ein Policy-Step Servo-Elektronik: Anlauf-Totzone, dann exponentielles
        # Nachfahren mit tau (Oeffnen langsamer als Schliessen).
        if self._spring_on:
            self._update_contact_state()
        dt = self._control_dt
        for idx in range(len(CONTROL_JOINTS)):
            qt = self._q_target[idx]
            qs = self._q_servo[idx]
            if not self._servo_started[idx]:
                if abs(qt - qs) < self._deadband[idx]:
                    continue
                self._servo_started[idx] = True
            tau = self._tau_close[idx]
            if qt < qs:
                tau *= self._tau_open_f[idx]
            self._q_servo[idx] = qs + (qt - qs) * (1.0 - float(np.exp(-dt / tau)))

    def _build_link_chain(self) -> dict[int, list[int]]:
        # Fuer jeden Link: welche CONTROL_JOINTS liegen auf dem Pfad von der
        # Handbasis zu diesem Link. Ein Kontakt an einem Fingerglied blockiert
        # genau die Gelenke proximal davon - damit laesst sich ein
        # getContactPoints-Treffer ohne Namensheuristik auf Joints abbilden.
        ctrl = {self.joint_index[n]: i for i, n in enumerate(CONTROL_JOINTS)}
        chain: dict[int, list[int]] = {-1: []}
        n_j = p.getNumJoints(self.hand_id, physicsClientId=self._cid)
        for j in range(n_j):
            parent = int(p.getJointInfo(self.hand_id, j, physicsClientId=self._cid)[16])
            chain[j] = chain.get(parent, []) + ([ctrl[j]] if j in ctrl else [])
        return chain

    def _update_contact_state(self) -> None:
        # Merkt je Joint, an welcher Position der Finger das Objekt zuerst
        # beruehrt hat. Ground Truth ist getContactPoints - der Rueckstand des
        # Gelenks gegenueber dem Servo-Kommando taugt NICHT als Kriterium, weil
        # schon das normale Fahren rund 0.005 Rueckstand erzeugt.
        # Freigabe erst nach release_steps kontaktfreien Schritten, damit ein
        # kurz weggerutschter Kontaktpunkt den Ueberfahrweg nicht zuruecksetzt.
        rel_max = int(self._spring_cfg.get("release_steps", 3))
        touched: set[int] = set()
        for c in p.getContactPoints(bodyA=self.hand_id, physicsClientId=self._cid):
            if c[2] == self.hand_id:      # Selbstkontakt ist keine Objektlast
                continue
            touched.update(self._link_chain.get(c[3], ()))
        for idx, name in enumerate(CONTROL_JOINTS):
            if idx in touched:
                self._free_steps[idx] = 0
                if self._contact_q[idx] is None:
                    pos = float(p.getJointState(self.hand_id, self.joint_index[name],
                                                 physicsClientId=self._cid)[0])
                    self._contact_q[idx] = self._angle_to_norm(name, pos)
            elif self._contact_q[idx] is not None:
                self._free_steps[idx] += 1
                if self._free_steps[idx] >= rel_max:
                    self._contact_q[idx] = None
                    self._free_steps[idx] = 0

    def teleport_to(self, q_target: list[float]) -> None:
        # Setzt Gelenkwinkel direkt ohne Physik, nur für Episode-Reset verwenden.
        # DIP-Gelenke werden manuell auf die korrekte Startposition gesetzt.
        q = [max(0.0, min(1.0, float(v))) for v in q_target]
        self._q_target = q
        self._q_servo       = list(q)
        self._servo_started = [False] * len(CONTROL_JOINTS)
        self._contact_q     = [None] * len(CONTROL_JOINTS)
        self._free_steps    = [0] * len(CONTROL_JOINTS)
        for idx, name in enumerate(CONTROL_JOINTS):
            p.resetJointState(self.hand_id, self.joint_index[name],
                               self._norm_to_angle(name, q[idx]),
                               physicsClientId=self._cid)
        for pip_name, dip_name in DIP_MIMIC_MAP.items():
            pip_angle = float(p.getJointState(self.hand_id, self.joint_index[pip_name],
                                               physicsClientId=self._cid)[0])
            p.resetJointState(self.hand_id, self.joint_index[dip_name],
                               DIP_MULTIPLIER * pip_angle + DIP_OFFSET,
                               physicsClientId=self._cid)
        self.apply_q_target(q)

    # Non-Backdrivability wird physikalisch durch die POSITION_CONTROL-Haltekraft abgebildet:
    # der Motor haelt die Zielposition gegen externe Kraefte bis motor_force. Frueher wurde das
    # per resetJointState-Teleport pro Substep erzwungen — das ignorierte Kollisionen und trieb
    # die Finger bis zu 14 mm durch die Objekte (unphysikalisch). Ohne den Teleport bleibt die
    # Durchdringung bei ~2 mm. Die PIP->DIP-Kopplung uebernimmt der JOINT_GEAR-Constraint
    # (siehe _setup_dip_constraints), ebenfalls physikalisch statt per Teleport.

    def reset_open_pose(self) -> None:
        self.teleport_to([0.0] * len(CONTROL_JOINTS))

    # Readout
    def q_target(self) -> list[float]:
        return list(self._q_target)

    def q_measured(self) -> list[float]:
        # Liest aktuelle Gelenkwinkel aus PyBullet und normalisiert auf [0, 1].
        # Mit Servo-Modell zusaetzlich das Sensor-Modell der echten Hand:
        # Gain-/Offset-Fehler der Input-Kalibrierung (pro Episode randomisiert),
        # ADC-Quantisierung und statisches Rauschen.
        out: list[float] = []
        sc    = self._servo_cfg
        quant = float(sc.get("adc_quant_q", 0.0017)) if self._servo_enabled else 0.0
        noise = float(sc.get("sensor_noise_std", 0.001)) if self._servo_enabled else 0.0
        for idx, name in enumerate(CONTROL_JOINTS):
            pos = float(p.getJointState(self.hand_id, self.joint_index[name],
                                         physicsClientId=self._cid)[0])
            q = self._angle_to_norm(name, pos)
            if self._spring_on and self._contact_q[idx] is not None:
                # Feder-Ersatzmodell statt starrer Blockierung (9.9):
                # gemeldete Position = Servo-Kommando minus k mal Ueberfahrweg,
                # gemessen ab Kontakt plus Onset-Totbereich. Daraus wird in
                # q_delta_normalized automatisch q_target - q_servo (Tracking-Lag)
                # PLUS k mal Ueberfahrweg (statisches Restresiduum) - genau die
                # beiden Anteile, die 9.9 real getrennt vermessen hat.
                over = self._q_servo[idx] - self._contact_q[idx] - self._spring_onset[idx]
                q    = self._q_servo[idx] - self._spring_k[idx] * max(0.0, over)
            if self._servo_enabled:
                q = (1.0 + self._sens_gain[idx]) * q + self._sens_offset[idx]
                if noise > 0:
                    q += float(self._rng.normal(0.0, noise))
                if quant > 0:
                    q = round(q / quant) * quant
            out.append(max(0.0, min(1.0, q)))
        return out

    def q_delta_normalized(self) -> list[float]:
        # q_target - q_measured pro Gelenk, geclipt auf [0, 1].
        # Positiver Wert = Finger schließt noch, d.h. Kontakt verhindert Bewegung.
        qt = self.q_target()
        qm = self.q_measured()
        return [max(0.0, min(1.0, t - m)) for t, m in zip(qt, qm)]

    def servo_params(self) -> dict:
        # Wahre Episoden-Parameter des Servo-Modells (fuer die synthetische
        # Detektor-Baseline im env; dort mit Kalibrierfehler beaufschlagt).
        return {
            "tau_close_s": {n: self._tau_close[i]   for i, n in enumerate(CONTROL_JOINTS)},
            "deadband_q":  {n: self._deadband[i]    for i, n in enumerate(CONTROL_JOINTS)},
            "sens_gain":   {n: self._sens_gain[i]   for i, n in enumerate(CONTROL_JOINTS)},
            "sens_offset": {n: self._sens_offset[i] for i, n in enumerate(CONTROL_JOINTS)},
        }

    @property
    def servo_enabled(self) -> bool:
        return self._servo_enabled

    # Helpers
    @staticmethod
    def _load_stroke_angle_map(path: str) -> dict:
        import yaml
        from pathlib import Path
        p_ = Path(path)
        if not p_.is_absolute():
            p_ = Path(__file__).resolve().parent.parent / p_
        with open(p_, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        curves = {k: (np.asarray(v["q"], dtype=float), np.asarray(v["f"], dtype=float))
                  for k, v in data["curves"].items()}
        return {joint: curves[cname] for joint, cname in data["joints"].items()}

    def _norm_to_angle(self, name: str, norm: float) -> float:
        # Normalisierter Wert [0, 1] -> Winkel in rad. Mit stroke_angle_map wird
        # die Hub-Koordinate erst durch die 4-Stab-Kennlinie geschickt.
        lo, hi = self.joint_limits[name]
        if name in self._kin_maps:
            qs, fs = self._kin_maps[name]
            norm = float(np.interp(norm, qs, fs))
        return lo + norm * (hi - lo)

    def _angle_to_norm(self, name: str, angle: float) -> float:
        # Winkel in rad -> normalisierte HUB-Koordinate (Inverse von _norm_to_angle).
        lo, hi = self.joint_limits[name]
        norm = (angle - lo) / (hi - lo)
        if name in self._kin_maps:
            qs, fs = self._kin_maps[name]
            norm = float(np.interp(norm, fs, qs))
        return norm


def _uniform(bounds: dict, rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds["min"], bounds["max"]))
