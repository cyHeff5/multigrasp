from __future__ import annotations

import math
import time
from pathlib import Path

import gymnasium
import numpy as np
import pybullet as p
import pybullet_data

from sim.hand     import CONTROL_JOINTS, SERVO0_INIT, HandModel
from sim.object   import GraspObject
from sim.pregrasp import compute_pregrasp
from sim.reward   import step_reward, terminal_reward
from sim.sampler  import sample_object_spec, sample_mass_kg


_SPAWN_XY = [0.0, 0.0]

# Default-Radius des Sim-Podests (Hardware-Setup: 2 cm Magnet-Plattform).
# Kann pro Config über episode.pedestal_radius überschrieben werden.
_DEFAULT_PEDESTAL_RADIUS = 0.02

# Schwellwert für "Objekt heruntergefallen": wenn der Schwerpunkt unter 30% der
# Podest-Höhe gesunken ist, ist das Objekt eindeutig vom Podest geschoben worden.
# 0.3 gewählt damit kleine Verschiebungen beim ersten Kontakt nicht als Drop zählen.
_DROP_HEIGHT_FRACTION = 0.3

_HAND_URDF = str(
    Path(__file__).resolve().parent.parent / "assets" / "ar10_description" / "urdf" / "ar10.urdf"
)


class GraspEnv(gymnasium.Env):
    # Config-getriebenes Gymnasium-Environment für AR10 Greiftraining.
    # Observation: [contact_bits_pro_finger, q_target_pro_aktivem_joint] - binäres Kontakt-Signal
    # plus Propriozeption damit der Markov-State erhalten bleibt.
    # Action: MultiDiscrete - servo0 hat {0: open, 1: stay, 2: close}, alle anderen {0: stay, 1: close}.

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg         = config
        self.grasp_type  = config["grasp_type"]
        self.render_mode = render_mode

        # Discount fuer das Potential-Based Reward Shaping. Muss mit dem PPO-gamma
        # uebereinstimmen, damit die Policy-Invarianz (Ng et al. 1999) exakt gilt.
        self._gamma = float(config["reward"]["gamma"])

        # Jede Action-Gruppe ist EIN Action-Signal, das alle Joints der Gruppe
        # identisch treibt (gekoppelte Finger schliessen symmetrisch).
        self._groups        = config["action_groups"]
        self._finger_joints = config["finger_joints"]
        self._fingers       = list(self._finger_joints.keys())

        # Sim2Real-Randomisierung: Kontakt-Threshold und Kontrollrate werden pro
        # Episode gezogen (siehe reset), damit die Policy weder von einem exakten
        # Threshold-Wert noch von einer exakten Loop-Rate abhaengt — beides ist
        # auf der echten Hand nur ungenau bekannt. Ohne Range in der Config
        # bleiben die festen Werte erhalten (altes Verhalten).
        obs_cfg = config["observation"]
        self._threshold_range = obs_cfg.get("threshold_range")
        self._threshold       = float(obs_cfg["threshold"])
        ep_cfg = config["episode"]
        self._substeps_range  = ep_cfg.get("substeps_range")
        self._substeps        = int(ep_cfg["substeps"])

        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(len(self._fingers) + len(self._groups),),
            dtype=np.float32,
        )
        # Discrete close-or-stay pro Gruppe; eine Gruppe mit Daumen-Abduktion (servo0)
        # bekommt 3 Optionen (offen/stay/zu). Halbiert die "verschwendete" Action-Mass
        # die Box-Gauss bei max(0, a) hatte und macht den Random-Walk produktiver.
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [3 if "servo0" in g else 2 for g in self._groups]
        )

        self._rng = np.random.default_rng()
        self._cid: int | None = None

        # Persistente Welt: einmal gebaut (Connection, Plane, Podest, Hand, Anchor),
        # danach pro Episode nur noch Objekt respawnen + Hand zuruecksetzen.
        self._world_built  = False
        self._pedestal_id: int | None = None
        self._hand_id:     int | None = None
        self._anchor_id:   int | None = None
        self._constraint:  int | None = None
        self._obj                      = None

    # Welt einmalig bauen
    def _build_world(self) -> None:
        # Connection, Plane, Podest, Hand und Anchor werden EINMAL erzeugt und ueber
        # alle Episoden hinweg wiederverwendet. Das spart pro Reset den teuren
        # Reconnect + das Neuladen der Hand-URDF (Mesh-Parsing). Pro Episode wird
        # in reset() nur noch das Objekt respawnt und die Hand zurueckgesetzt.
        mode      = p.GUI if self.render_mode == "human" else p.DIRECT
        self._cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._cid)
        p.setAdditionalSearchPath(
            str(Path(__file__).resolve().parent.parent / "assets"),
            physicsClientId=self._cid,
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(1.0 / self.cfg["episode"]["sim_hz"], physicsClientId=self._cid)
        # Hohe Solver-Iterations für rigide Kontakte mit mehreren simultanen Berührungen
        # (Finger + Objekt + Pedestal). PyBullet-Default 50 ist für Greifsim zu wenig.
        p.setPhysicsEngineParameter(numSolverIterations=200, physicsClientId=self._cid)
        p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf", physicsClientId=self._cid)

        # Podest - Maße aus der Config, ansonsten Hardware-Defaults. Statisch (mass=0),
        # bewegt sich nie -> einmal anlegen reicht.
        ped_h = self.cfg["episode"].get("pedestal_height", 0.04)
        ped_r = self.cfg["episode"].get("pedestal_radius", _DEFAULT_PEDESTAL_RADIUS)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=ped_r,
                                      height=ped_h, physicsClientId=self._cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ped_r,
                                   length=ped_h, rgbaColor=[0.5, 0.5, 0.5, 1.0],
                                   physicsClientId=self._cid)
        self._pedestal_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[_SPAWN_XY[0], _SPAWN_XY[1], ped_h / 2],
            physicsClientId=self._cid,
        )
        self._pedestal_h = ped_h

        # Hand EINMAL laden (an Platzhalter-Pose; reset() positioniert sie pro Episode).
        self._hand_id = p.loadURDF(
            _HAND_URDF, basePosition=[0.0, 0.0, 1.0], baseOrientation=[0, 0, 0, 1],
            physicsClientId=self._cid,
        )
        self._hand = HandModel(
            self._hand_id, self.cfg["physics"], self._rng, client_id=self._cid,
        )

        # Anchor-Body + JOINT_FIXED Constraint einmal anlegen. Anchor und Hand starten
        # an derselben Pose -> Relativtransform = Identitaet. In reset() werden BEIDE
        # gemeinsam auf die neue Pregrasp-Pose gesetzt, dadurch bleibt der Constraint
        # in Ruhelage und muss nie neu erzeugt werden.
        self._anchor_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 1.0], baseOrientation=[0, 0, 0, 1],
            physicsClientId=self._cid,
        )
        self._constraint = p.createConstraint(
            self._anchor_id, -1, self._hand_id, -1, p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0], [0, 0, 0], physicsClientId=self._cid,
        )
        p.changeConstraint(self._constraint, maxForce=500, erp=0.95, physicsClientId=self._cid)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5, cameraYaw=-48.0, cameraPitch=-14.8,
                cameraTargetPosition=[0.0, 0.0, 0.1], physicsClientId=self._cid,
            )

        self._world_built = True

    # Reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._world_built:
            self._build_world()

        # Objekt: zufällig sampeln oder aus options übernehmen (Benchmark-Eval).
        # Masse und Reibung werden immer neu gezogen — auch wenn obj_spec übergeben wurde.
        opts = options or {}
        self._obj_spec = dict(
            opts.get("obj_spec") or sample_object_spec(self.cfg["sampler"], self._rng)
        )
        sampler = self.cfg["sampler"]
        # Masse dichtebasiert (Dichte × Volumen) — sonst bekommen kleine/dünne Objekte
        # bei fester kg-Range absurde Dichten (dünner Zylinder -> 37 g/cm³).
        self._obj_spec["mass_kg"] = sample_mass_kg(self._obj_spec, sampler, self._rng)
        self._obj_spec["lateral_friction"] = float(self._rng.uniform(
            sampler["lateral_friction"]["min"], sampler["lateral_friction"]["max"]
        ))

        # Altes Objekt entfernen, neues spawnen (billig — primitive Shape bzw. URDF).
        if self._obj is not None:
            p.removeBody(self._obj.object_id, physicsClientId=self._cid)
        self._obj = GraspObject.spawn(
            self._obj_spec, self._pedestal_h, _SPAWN_XY, self._cid,
        )

        # Pregrasp-Pose: berechnen oder aus options übernehmen (Benchmark-Eval).
        pre_cfg  = self.cfg["pregrasp"]
        pregrasp = opts.get("pregrasp")
        if pregrasp is not None:
            # Feste Pregrasp-Pose (Benchmark): Orientierung 1:1 übernehmen,
            # aber denselben XY/Z-Positions-Jitter wie compute_pregrasp addieren,
            # damit der Benchmark die reale Positionierungsunsicherheit des
            # Sawyer-Arms mit abbildet.
            hand_pos  = list(pregrasp[0])
            hand_quat = list(pregrasp[1])
            hand_pos[0] += self._rng.uniform(-pre_cfg["jitter_xy_m"], pre_cfg["jitter_xy_m"])
            hand_pos[1] += self._rng.uniform(-pre_cfg["jitter_xy_m"], pre_cfg["jitter_xy_m"])
            hand_pos[2] += self._rng.uniform(-pre_cfg["jitter_z_m"],  pre_cfg["jitter_z_m"])
        else:
            hand_pos, hand_quat, _ = compute_pregrasp(
                self.grasp_type, self._obj.position(), self._obj_spec, self._rng,
                distance_m  = pre_cfg["distance_m"],
                jitter_xy_m = pre_cfg["jitter_xy_m"],
                jitter_z_m  = pre_cfg["jitter_z_m"],
            )

        # Domain Randomization der Hand pro Episode (Motorkraft + Fingerkuppen-Reibung).
        self._hand.randomize_dynamics(self._rng)

        # Kontakt-Threshold und Kontrollrate pro Episode ziehen. Der Threshold-Jitter
        # macht die Policy robust gegen die Bit-Latenz der echten Hand (kalibrierter
        # real_threshold weicht vom Sim-Wert ab); der Substep-Jitter gegen die
        # unbekannte echte Loop-Rate (step_dt + serielle I/O-Zeit). Untergrenze 0.02
        # ist ~2x der gemessene Sim-Freilauf-q_delta (0.0095) -> keine Fehltrigger.
        if self._threshold_range is not None:
            self._threshold = float(self._rng.uniform(*self._threshold_range))
        if self._substeps_range is not None:
            self._substeps = int(self._rng.integers(
                self._substeps_range[0], self._substeps_range[1] + 1))

        # Hand-Gelenke auf offene Pregrasp-Startpose zuruecksetzen (Velocity = 0).
        start_q = [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]
        self._hand.teleport_to(start_q)

        # Hand-Basis + Anchor gemeinsam auf die neue Pregrasp-Pose setzen. Beide werden
        # auf dieselbe Pose gesetzt, daher bleibt der Fixed-Constraint in Ruhelage.
        # Velocities explizit nullen, damit kein Impuls aus der letzten Episode bleibt.
        p.resetBasePositionAndOrientation(self._hand_id, hand_pos, hand_quat,
                                           physicsClientId=self._cid)
        p.resetBaseVelocity(self._hand_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._cid)
        p.resetBasePositionAndOrientation(self._anchor_id, hand_pos, hand_quat,
                                           physicsClientId=self._cid)
        p.resetBaseVelocity(self._anchor_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._cid)

        # Settle-Phase ohne Kollision: Hand kann sich in Startposition einpendeln
        # ohne das Objekt zu berühren. Danach Kollision wieder aktivieren.
        p.setCollisionFilterPair(self._hand_id, self._obj.object_id, -1, -1,
                                  enableCollision=0, physicsClientId=self._cid)
        for _ in range(self.cfg["episode"]["settle_steps"]):
            p.stepSimulation(physicsClientId=self._cid)
        p.setCollisionFilterPair(self._hand_id, self._obj.object_id, -1, -1,
                                  enableCollision=1, physicsClientId=self._cid)

        self._step_count             = 0
        self._consecutive_contact    = 0
        self._lift_triggered         = False
        self._stabilization_left     = 0
        self._pedestal_hit_ever      = False
        self._n_contact_prev         = 0

        # Referenz-Orientierung für den Drop-Check: die Objektachse, die zum
        # Episodenstart nach oben zeigt. Kippung wird relativ dazu gemessen,
        # damit bewusst gekippt gespawnte Benchmark-Teile (Seiten-/Über-Kopf-
        # Greifpose) nicht sofort als Drop zählen.
        _, q0 = p.getBasePositionAndOrientation(
            self._obj.object_id, physicsClientId=self._cid,
        )
        m0 = p.getMatrixFromQuaternion(q0)
        self._obj_up_local = (m0[6], m0[7], m0[8])

        return self._observation(), {
            "grasp_type": self.grasp_type,
            "shape":      self._obj_spec["shape"],
        }

    # Step
    def step(self, action: np.ndarray):
        n_contact_prev = self._n_contact_prev

        delta  = self._action_to_delta(action)
        next_q = [max(0.0, min(1.0, q + d)) for q, d in zip(self._hand.q_target(), delta)]
        self._apply_pip_caps(next_q)
        self._hand.apply_q_target(next_q)

        for _ in range(self._substeps):
            p.stepSimulation(physicsClientId=self._cid)
            if self.render_mode == "human":
                time.sleep(1.0 / self.cfg["episode"]["sim_hz"])

        self._step_count += 1
        obs       = self._observation()
        n_contact = int(obs[:len(self._fingers)].sum())
        self._n_contact_prev = n_contact

        # Trigger State Machine (Westling & Johansson 1984):
        # trigger_confirmation_steps consecutive frames mit >= trigger_n Kontakten -> Trigger.
        if n_contact >= self.cfg["trigger_n"]:
            self._consecutive_contact += 1
        else:
            self._consecutive_contact = 0

        if (not self._lift_triggered
                and self._consecutive_contact >= self.cfg["trigger_confirmation_steps"]):
            self._lift_triggered     = True
            self._stabilization_left = self.cfg["stabilization_steps"]

        pedestal_now = self._check_pedestal_contact()
        if pedestal_now:
            self._pedestal_hit_ever = True

        # Termination ZUERST bestimmen, damit das PBRS-Shaping den absorbierenden
        # Endzustand mit Phi=0 behandeln kann (terminal=...). Reihenfolge wie zuvor:
        # Drop hat Vorrang, dann Lift nach Stabilisierung, dann Timeout.
        dropped = self._object_dropped()
        do_lift = False
        if not dropped and self._lift_triggered:
            self._stabilization_left -= 1
            do_lift = self._stabilization_left <= 0
        timeout = (not dropped and not do_lift
                   and self._step_count >= self.cfg["episode"]["max_steps"])
        terminated = dropped or do_lift or timeout

        reward = step_reward(n_contact_prev, n_contact, self.cfg["n_target"],
                             pedestal_now, self._gamma, self.cfg["reward"],
                             terminal=terminated)

        # Drop: Objekt zu stark gekippt oder unter Podest-Niveau gefallen.
        if dropped:
            reward += terminal_reward(lifted=False, cfg=self.cfg["reward"])
            return obs, reward, True, False, self._info(n_contact, lifted=False, dropped=True)

        # Nach Stabilisierung: Lift-Test.
        if do_lift:
            lifted = self._run_lift_test()
            reward += terminal_reward(lifted, cfg=self.cfg["reward"])
            return obs, reward, True, False, self._info(n_contact, lifted=lifted)

        # Kein Trigger bis max_steps -> trotzdem Lift-Test.
        if timeout:
            lifted = self._run_lift_test()
            reward += terminal_reward(lifted, cfg=self.cfg["reward"])
            return obs, reward, True, False, self._info(n_contact, lifted=lifted)

        return obs, reward, False, False, self._info(n_contact, lifted=False)

    # Observation 
    def _observation(self) -> np.ndarray:
        # Kontakt-Bits: 1 wenn mindestens ein Joint des Fingers q_delta > threshold.
        # threshold ist der pro Episode gezogene Wert (siehe reset).
        threshold = self._threshold
        delta_all = self._hand.q_delta_normalized()
        contact = np.zeros(len(self._fingers), dtype=np.float32)
        for i, finger in enumerate(self._fingers):
            for joint_name in self._finger_joints[finger]:
                joint_idx = CONTROL_JOINTS.index(joint_name)
                if delta_all[joint_idx] > threshold:
                    contact[i] = 1.0
                    break

        # Propriozeption: normalisierte q_target pro Gruppe ([0, 1]). Gekoppelte Joints
        # haben denselben Wert, daher genuegt der erste Joint jeder Gruppe.
        q_all = self._hand.q_target()
        q_groups = np.array(
            [q_all[CONTROL_JOINTS.index(g[0])] for g in self._groups],
            dtype=np.float32,
        )
        return np.concatenate([contact, q_groups])

    # Action mapping
    def _action_to_delta(self, action: np.ndarray) -> list[float]:
        # Ein Action-Wert pro Gruppe; das Delta wird auf ALLE Joints der Gruppe
        # identisch angewendet (Kopplung). servo0-Gruppe: {0: open, 1: stay, 2: close},
        # alle anderen {0: stay, 1: close}.
        delta = [0.0] * len(CONTROL_JOINTS)
        for i, group in enumerate(self._groups):
            a = int(action[i])
            if "servo0" in group:
                step = self.cfg["action"]["thumb_abduction_delta"]
                d = (-1.0 if a == 0 else 1.0 if a == 2 else 0.0) * step
            else:
                d = a * self.cfg["action"]["delta_norm"]
            for joint_name in group:
                delta[CONTROL_JOINTS.index(joint_name)] = d
        return delta

    def _apply_pip_caps(self, q_target: list[float]) -> None:
        # Daumen-Abduktion auf erlaubten Bereich begrenzen.
        # PIP-Caps verhindern dass Finger das Objekt überfahren (nur Precision).
        rng_ = self.cfg["action"]["thumb_abduction_range"]
        idx0 = CONTROL_JOINTS.index("servo0")
        q_target[idx0] = max(rng_[0], min(rng_[1], q_target[idx0]))
        for joint_name, cap in self.cfg["action"]["pip_caps"].items():
            idx = CONTROL_JOINTS.index(joint_name)
            q_target[idx] = min(cap, q_target[idx])

    # Termination checks 
    def _check_pedestal_contact(self) -> bool:
        from sim.hand import FINGERTIP_EE_MAP
        for ee in FINGERTIP_EE_MAP.values():
            contacts = p.getContactPoints(
                self._hand_id, self._pedestal_id,
                linkIndexA=self._hand.joint_index[ee],
                physicsClientId=self._cid,
            )
            if contacts:
                return True
        return False

    def _object_dropped(self) -> bool:
        # Drop wenn das Objekt relativ zur Spawn-Lage über drop_tilt_rad gekippt
        # oder unter _DROP_HEIGHT_FRACTION der Podest-Höhe gefallen ist.
        # Kippung wird relativ zur Start-Orientierung gemessen, weil Benchmark-
        # Teile bewusst gekippt spawnen können — absoluter Winkel würde die
        # Episode sofort beenden. Yaw bleibt wie bisher unberücksichtigt.
        _, quat = p.getBasePositionAndOrientation(
            self._obj.object_id, physicsClientId=self._cid,
        )
        m = p.getMatrixFromQuaternion(quat)
        ux, uy, uz = self._obj_up_local
        # z-Komponente der Weltrichtung der ursprünglichen Hochachse = cos(Kippwinkel).
        cos_tilt = m[6] * ux + m[7] * uy + m[8] * uz
        tilt = math.acos(max(-1.0, min(1.0, cos_tilt)))
        return (tilt > self.cfg["lift"]["drop_tilt_rad"]
                or self._obj.height() < self._pedestal_h * _DROP_HEIGHT_FRACTION)

    def _run_lift_test(self) -> bool:
        # Bewegt den Anchor-Body (nicht die Hand direkt) um lift_height nach oben.
        # Die Hand folgt über den JOINT_FIXED Constraint.
        # Erfolg wenn das Objekt >= success_m mitgehoben wird.
        pos, quat = p.getBasePositionAndOrientation(
            self._hand_id, physicsClientId=self._cid,
        )
        z0       = self._obj.height()
        lift_cfg = self.cfg["lift"]

        for step in range(lift_cfg["lift_steps"] + lift_cfg["hold_steps"]):
            if step < lift_cfg["lift_steps"]:
                frac    = (step + 1) / lift_cfg["lift_steps"]
                new_pos = [pos[0], pos[1], pos[2] + frac * lift_cfg["height_m"]]
                p.resetBasePositionAndOrientation(
                    self._anchor_id, new_pos, quat, physicsClientId=self._cid,
                )
            p.stepSimulation(physicsClientId=self._cid)
            if self.render_mode == "human":
                time.sleep(1.0 / self.cfg["episode"]["sim_hz"])

        return (self._obj.height() - z0) >= lift_cfg["success_m"]

    # Info dict 
    def _info(self, n_contact: int, lifted: bool, dropped: bool = False) -> dict:
        return {
            "n_contact":       n_contact,
            "lifted":          lifted,
            "dropped":         dropped,
            "lift_triggered":  self._lift_triggered,
            "pedestal_hit":    self._pedestal_hit_ever,
            "step_count":      self._step_count,
        }

    # Cleanup 
    def close(self) -> None:
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None
        # Welt muss bei erneutem reset() neu gebaut werden.
        self._world_built = False
        self._obj         = None
