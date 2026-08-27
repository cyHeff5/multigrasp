# IO-Adapter für die echte AR10-Hand über den Pololu Maestro Servo-Controller.
# Alle q-Werte sind normalisiert: 0.0 = vollständig offen, 1.0 = vollständig geschlossen.
# com_port=None -> Mock-Mode (kein Hardware nötig, read_q_measured gibt q_target zurück).

import json
import os
import time
from typing import List, Optional

try:
    import serial
except ImportError:
    serial = None 


# Sim-Joint-Index -> Maestro-Kanal, per Hardware verifiziert.
# Weicht von der originalen Active8-Nummerierung ab weil Sim und Hardware
# die Finger in unterschiedlicher Reihenfolge nummerieren.
#   sim j0/1  (Daumen)  -> ch18/19
#   sim j2/3  (Kleiner) -> ch16/17
#   sim j4/5  (Ring)    -> ch14/15
#   sim j6/7  (Mittel)  -> ch12/13
#   sim j8/9  (Zeige)   -> ch10/11
_CHANNELS = [18, 19, 16, 17, 14, 15, 12, 13, 10, 11]

# Originale Active8-Nummerierung, wird nur gebraucht um joint_input_calibration.json auf die Sim-Indizes umzurechnen.
_OLD_CHANNELS = [10, 11, 18, 19, 16, 17, 14, 15, 12, 13]

_DEFAULT_SERVO_MIN = [4200] * 10  # vollständig geschlossen (Pulse-Untergrenze)
_DEFAULT_SERVO_MAX = [7700] * 10  # vollständig offen (Pulse-Obergrenze)

# Optionale per-Kanal-Limits, abgelesen aus dem Maestro Control Center.
# Der Maestro clippt Targets STILL auf seine gespeicherten Kanal-Limits — wenn
# unsere Limits weiter sind als seine, laufen q_target und echte Position
# auseinander und erzeugen Phantom-q_delta (falsche Kontakt-Bits).
_SERVO_LIMITS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "calibration", "servo_limits.yaml"
)


class AR10Interface:

    def __init__(
        self,
        com_port: Optional[str] = None,
        servo_min: Optional[List[int]] = None,
        servo_max: Optional[List[int]] = None,
        speed: int = 100,
        acceleration: int = 0,
        input_calibration_file: Optional[str] = None,
        adc_reads: int = 1,
        ema_alpha: Optional[float] = None,
    ):
        # Prioritaet: explizite Parameter > servo_limits.yaml > Defaults.
        file_min, file_max = self._load_servo_limits()
        self._servo_min = list(servo_min or file_min or _DEFAULT_SERVO_MIN)
        self._servo_max = list(servo_max or file_max or _DEFAULT_SERVO_MAX)
        self._q_target: List[float] = [0.0] * 10
        self._usb: Optional[serial.Serial] = None
        self._input_cal: dict = self._load_input_calibration(input_calibration_file)
        self._adc_reads: int = max(1, adc_reads)
        # EMA-Filter: q_ema = alpha * q_raw + (1-alpha) * q_ema_prev
        # alpha=1.0 = kein Filter, alpha=0.3 = starke Glaettung.
        self._ema_alpha: Optional[float] = ema_alpha
        self._ema_state: Optional[List[float]] = None

        if com_port is not None:
            if serial is None:
                raise ImportError("pyserial is required for hardware mode: pip install pyserial")
            if not self._input_cal:
                raise FileNotFoundError(
                    "joint_input_calibration.json not found — required for hardware mode."
                )
            self._usb = serial.Serial(com_port, baudrate=9600)
            for ch in _CHANNELS:
                self._set_channel_speed(ch, speed)
                time.sleep(0.05)
                self._set_channel_acceleration(ch, acceleration)
                time.sleep(0.05)

    # Pololu Maestro Protokoll
    def _send_command(self, *args: str) -> None:
        # Pololu Compact Protocol: 0xAA = Start, 0x0C = Gerätenummer, dann Befehlsbytes.
        if self._usb is None:
            return
        msg = chr(0xAA) + chr(0x0C) + "".join(args)
        self._usb.write(msg.encode("latin-1"))

    def _set_channel_speed(self, channel: int, speed: int) -> None:
        lsb = speed & 0x7F
        msb = (speed >> 7) & 0x7F
        self._send_command(chr(0x07), chr(channel), chr(lsb), chr(msb))

    def _set_channel_acceleration(self, channel: int, accel: int) -> None:
        accel = max(0, min(255, accel))
        lsb = accel & 0x7F
        msb = (accel >> 7) & 0x7F
        self._send_command(chr(0x09), chr(channel), chr(lsb), chr(msb))

    def _limits_by_channel(self) -> tuple:
        # _servo_min/_servo_max sind in SIM-JOINT-Reihenfolge indiziert, die
        # Target-Liste des 0x1F-Befehls dagegen in KANAL-Reihenfolge (ch10..ch19).
        # Diese Umsortierung haelt beide auseinander.
        lo = [0] * 10
        hi = [0] * 10
        for joint_idx, ch in enumerate(_CHANNELS):
            lo[ch - 10] = self._servo_min[joint_idx]
            hi[ch - 10] = self._servo_max[joint_idx]
        return lo, hi

    def _set_all_channel_targets(self, targets: List[int]) -> None:
        # Alle 10 Servo-Targets in einem einzigen Maestro-Befehl senden (0x1F).
        # targets ist nach KANAL indiziert (targets[0] = ch10), die Limits nach
        # Sim-Joint — ohne die Umsortierung wuerde ch10/ch11 (sim servo8/9)
        # gegen die Daumen-Limits geclippt. Solange alle Limits gleich waren
        # (Default 4200/7700) blieb das unsichtbar; mit per-Kanal-Limits aus
        # servo_limits.yaml verschiebt es servo8/9 still um bis zu 900 Puls
        # (gefunden 2026-08-25, Laborsession 1).
        lo, hi = self._limits_by_channel()
        args = [chr(0x1F), chr(10), chr(10)]
        for i, t in enumerate(targets):
            t = max(lo[i], min(hi[i], t))
            args.append(chr(t & 0x7F))
            args.append(chr((t >> 7) & 0x7F))
        self._send_command(*args)

    @staticmethod
    def _load_servo_limits() -> tuple:
        # Liest artifacts/calibration/servo_limits.yaml (per-Kanal Min/Max aus dem
        # Maestro Control Center, Sim-Joint-Reihenfolge). Fehlt die Datei -> (None, None).
        try:
            import yaml
            with open(_SERVO_LIMITS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return None, None
        lo, hi = data.get("servo_min"), data.get("servo_max")
        if lo is not None and hi is not None:
            if len(lo) != 10 or len(hi) != 10:
                raise ValueError("servo_limits.yaml: servo_min/servo_max brauchen je 10 Werte.")
            print(f"[ar10] Per-Kanal Servo-Limits aus servo_limits.yaml geladen.")
        return lo, hi

    def assert_input_calibration(self, joint_indices: List[int]) -> None:
        # Hart failen wenn ein beobachtetes Gelenk keinen Kalibrier-Eintrag hat —
        # sonst liefert read_q_measured stumm 0.0 und q_delta = q_target erzeugt
        # ein Dauer-Kontakt-Bit (stiller Sim2Real-Killer).
        if self._usb is None:
            return
        missing = [j for j in joint_indices if j not in self._input_cal]
        if missing:
            raise RuntimeError(
                f"joint_input_calibration.json: Eintraege fuer Sim-Joints {missing} fehlen — "
                "diese Gelenke wuerden q_measured=0.0 melden (Phantom-Kontakt)."
            )

    # Sensor-Kalibrierung
    @staticmethod
    def _load_input_calibration(path: Optional[str]) -> dict:
        # Liest joint_input_calibration.json und rechnet die alten Active8-Joint-Indizes auf die aktuellen Sim-Indizes um.
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "joint_input_calibration.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        cal = {}
        for joint_str, jd in data.get("joints", {}).items():
            old_j = int(joint_str)
            old_ch = _OLD_CHANNELS[old_j]
            try:
                new_j = _CHANNELS.index(old_ch)
            except ValueError:
                continue
            cal[new_j] = {
                "input_channel": int(jd["input_channel"]),
                "open_real":     float(jd["opened"]["mapped_input"]),
                "closed_real":   float(jd["closed"]["mapped_input"]),
            }
        return cal

    def _read_input_channel(self, channel: int) -> int:
        # Liest analogen Sensorwert vom Maestro-Eingangskanal.
        if self._usb is None:
            return 0
        self._send_command(chr(0x10), chr(channel))
        lsb = ord(self._usb.read())
        msb = ord(self._usb.read())
        return (msb << 8) + lsb

    def _normalize_input(self, value: float, open_val: float, closed_val: float) -> float:
        denom = closed_val - open_val
        if denom == 0.0:
            return 0.0
        return max(0.0, min(1.0, (value - open_val) / denom))

    # Normalisierung
    def _to_servo(self, q_norm: float, joint_idx: int) -> int:
        # q_norm=0 (offen) -> hi=7700, q_norm=1 (geschlossen) -> lo=4200
        lo = self._servo_min[joint_idx]
        hi = self._servo_max[joint_idx]
        return int(round(hi - max(0.0, min(1.0, q_norm)) * (hi - lo)))

    def _to_norm(self, servo_val: int, joint_idx: int) -> float:
        lo = self._servo_min[joint_idx]
        hi = self._servo_max[joint_idx]
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (servo_val - lo) / (hi - lo)))

    # Public Interface
    def send_q_target(self, q_target: List[float]) -> None:
        # Sendet 10 normalisierte Zielwerte [0, 1] an die echte Hand.
        if len(q_target) != 10:
            raise ValueError(f"q_target must have 10 values, got {len(q_target)}.")
        self._q_target = [max(0.0, min(1.0, v)) for v in q_target]
        channel_targets = [0] * 10
        for joint_idx, v in enumerate(self._q_target):
            ch = _CHANNELS[joint_idx]
            channel_targets[ch - 10] = self._to_servo(v, joint_idx)
        self._set_all_channel_targets(channel_targets)

    def read_q_measured(self) -> List[float]:
        # Liest aktuelle Gelenkpositionen von den analogen Positionssensoren, normalisiert auf [0, 1].
        # Bei adc_reads > 1 werden mehrere ADC-Samples pro Joint gemittelt (Rauschreduktion).
        # Bei ema_alpha: zusaetzlich EMA-Glaettung ueber aufeinanderfolgende Aufrufe.
        # Im Mock-Mode wird q_target zurückgegeben.
        if self._usb is None:
            return list(self._q_target)
        n = self._adc_reads
        if n <= 1:
            raw_norm = []
            for i in range(10):
                if i in self._input_cal:
                    cal = self._input_cal[i]
                    raw = self._read_input_channel(cal["input_channel"])
                    raw_norm.append(self._normalize_input(raw, cal["open_real"], cal["closed_real"]))
                else:
                    raw_norm.append(0.0)
        else:
            # Multi-read: alle Joints N-mal lesen und mitteln.
            accum = [0.0] * 10
            cal_mask = [i in self._input_cal for i in range(10)]
            for _ in range(n):
                for i in range(10):
                    if cal_mask[i]:
                        accum[i] += self._read_input_channel(self._input_cal[i]["input_channel"])
            raw_norm = []
            for i in range(10):
                if cal_mask[i]:
                    cal = self._input_cal[i]
                    raw_norm.append(self._normalize_input(accum[i] / n, cal["open_real"], cal["closed_real"]))
                else:
                    raw_norm.append(0.0)
        return self._apply_ema(raw_norm)

    def _apply_ema(self, raw: List[float]) -> List[float]:
        # EMA-Filter auf die normalisierten Messwerte anwenden.
        if self._ema_alpha is None:
            return raw
        a = self._ema_alpha
        if self._ema_state is None:
            self._ema_state = list(raw)
            return list(raw)
        for i in range(10):
            self._ema_state[i] = a * raw[i] + (1.0 - a) * self._ema_state[i]
        return list(self._ema_state)

    def reset_ema(self) -> None:
        # EMA-Zustand zuruecksetzen (z.B. nach Pregrasp-Wechsel).
        self._ema_state = None

    def position_error_norm(self) -> float:
        # Mittlerer absoluter Fehler zwischen q_target und q_measured.
        measured = self.read_q_measured()
        errors = [abs(t - m) for t, m in zip(self._q_target, measured)]
        return sum(errors) / len(errors)

    def close(self) -> None:
        if self._usb is not None:
            self._usb.close()
            self._usb = None
