# Probe-Kontakt-Messung mit visueller Ground Truth (Laborsession 1b).
#
# Frage (SENSOR_ANALYSIS_FINDINGS.md §9.8): die Hand ist nachgiebig, q_delta
# sieht den Erstkontakt nicht. Der rate_probe-Aktionsraum baut aber auf §8.6:
# "blockierter Finger haelt q_delta im Stand, freier kollabiert in 5-15 Steps".
# Nie direkt gemessen — und unklar, ab welchem Ueberfahrweg das ueberhaupt gilt.
#
# Ablauf Kontaktmodus (default):
#   1. INTERAKTIV: Der Operator schliesst die Finger per Taste, bis er den
#      physischen Kontakt SIEHT, und markiert ihn (Taste c). Das ist die
#      Ground Truth, die dem Wuerfeltest §9.7 gefehlt hat.
#   2. AUTO (Taste g): 40 Steps halten (Statik am Kontaktpunkt), dann Rampe
#      im Runner-Timing (senden -> Step-Slot abwarten -> lesen) weiter bis
#      alle Joints am Cap sind — mit einer Probe-Pause an jedem festen
#      q-Rasterpunkt (default alle 0.05). Pro Pause sieht man: kollabiert
#      q_delta wie im Freilauf oder haelt es? Und danach: Wiederanlauf
#      UNTER LAST (restart_steps wurde bisher nur im Freilauf gemessen).
#   3. Hand oeffnet, naechster Zyklus (andere Grifftiefe/Position) oder q.
#
# Referenzmodus --freerun: gleiche Rampe, gleiches Pausenraster, OHNE Objekt.
# Weil die Pausen auf einem festen q-Gitter liegen (nicht relativ zum Start),
# sind Kontakt- und Freilauflauf per q_target matchbar (§9.7: NICHT per step!).
#
# Der echte ContactDetector laeuft in der AUTO-Phase mit (Session-Baseline,
# rate_scaling, restart-Maske) — seine Bits stehen als Spur in der CSV.
# Die Auswertung am Ende ist bewusst NUR deskriptiv (Lehre aus
# eval/contact_latency.py: keine eingebauten Urteile mit kaputter Referenz).
#
# Usage (Labor, Windows-Laptop):
#   python -u -m eval.probe_contact --config configs/precision.yaml --port COM4
#   python -u -m eval.probe_contact --config configs/precision.yaml --port COM4 --freerun
# Mock-Smoke (ohne Hardware, Kommandos zeilenweise ueber stdin):
#   printf "wwwww\nc\ng\nq\n" | python -m eval.probe_contact --config configs/precision.yaml
#
# Tasten (Kontaktmodus):
#   w/s = Finger zu/auf (grob, default 0.01)   W/S = fein (default 0.002)
#   c   = Kontakt-Marker setzen (mehrfach erlaubt, alle werden gespeichert)
#   g   = AUTO-Phase starten    o = Hand oeffnen (Zyklus verwerfen)
#   q   = Ende (Hand oeffnet)   x = NOT-AUS in der AUTO-Phase (oeffnet sofort)
from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path

import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, SERVO0_INIT

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR   = _REPO_ROOT / "artifacts" / "analysis"

_HOLD0_STEPS = 40   # Statik-Fenster direkt am visuellen Kontaktpunkt
_HOLD_END    = 60   # Haltephase wenn alle Joints am Cap sind
_MAX_STEPS   = 1500


class KeySource:
    """Einzeltasten ohne Enter (Windows/msvcrt) UND parallel Zeilen von stdin
    (Reader-Thread): jedes Zeichen einer Zeile zaehlt als Taste. Damit ist das
    Skript auch bedienbar, wenn die Konsole nicht direkt am Prozess haengt
    (schtasks-Fenster, SSH, Mock per Pipe) — dann Kommandos + ENTER tippen."""

    def __init__(self):
        try:
            import msvcrt  # noqa: F401
            self._msvcrt = True
        except ImportError:
            self._msvcrt = False
        self._queue: list[str] = []
        self._eof = False
        self._thread = None

    def start_stdin_thread(self) -> None:
        # Erst NACH dem letzten normalen input() starten (liest sonst dessen
        # Enter weg).
        import threading
        if self._thread is not None:
            return
        def _reader():
            while True:
                try:
                    line = input()
                except EOFError:
                    self._eof = True
                    return
                self._queue.extend(line.strip())
        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def poll(self) -> str | None:
        # Nicht-blockierend: naechste Taste oder None.
        if self._msvcrt:
            import msvcrt
            try:
                while msvcrt.kbhit():
                    ch = msvcrt.getch()
                    self._queue.append(ch.decode("latin-1", errors="replace"))
            except OSError:
                self._msvcrt = False
                print("\n[probe-contact] Konsolentasten nicht verfuegbar — "
                      "Kommandos tippen und mit ENTER bestaetigen (z.B. wwww, c, g).")
        return self._queue.pop(0) if self._queue else None

    def eof(self) -> bool:
        return self._eof and not self._queue


def _pregrasp_q() -> list[float]:
    return [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]


def _watched(cfg: dict) -> list[str]:
    joints: list[str] = []
    for jl in cfg["finger_joints"].values():
        for j in jl:
            if j not in joints:
                joints.append(j)
    return joints


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe-Pausen-Rampe MIT Objekt und visueller Kontakt-Ground-Truth.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", default=None)
    ap.add_argument("--freerun", action="store_true",
                    help="Referenzlauf OHNE Objekt (gleiches Pausenraster, keine Interaktion).")
    ap.add_argument("--cycles", type=int, default=2, help="Zyklen im Freilaufmodus.")
    ap.add_argument("--rate", type=float, default=None,
                    help="Rampenrate der AUTO-Phase (default: action.delta_norm).")
    ap.add_argument("--grid", type=float, default=0.05,
                    help="q-Raster der Probe-Pausen (fest, damit Freilauf matchbar ist).")
    ap.add_argument("--pause-steps", type=int, default=40)
    ap.add_argument("--jog", type=float, default=0.01, help="Schrittweite Taste w/s.")
    ap.add_argument("--jog-fine", type=float, default=0.002, help="Schrittweite Taste W/S.")
    ap.add_argument("--qmax", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dt      = float(cfg["episode"]["substeps"]) / float(cfg["episode"]["sim_hz"])
    rate    = args.rate if args.rate is not None else float(cfg["action"]["delta_norm"])
    caps    = cfg["action"].get("pip_caps", {})
    joints  = _watched(cfg)
    j_idxs  = [CONTROL_JOINTS.index(j) for j in joints]
    fingers = list(cfg["finger_joints"].keys())
    cap_of  = {j: min(caps.get(j, 1.0), args.qmax) for j in joints}
    # Leitjoint fuers Pausenraster: erster Joint ohne Cap (MCP laeuft bis qmax).
    lead_j, lead_idx = next(((j, i) for j, i in zip(joints, j_idxs)
                             if caps.get(j, 1.0) >= 1.0), (joints[0], j_idxs[0]))

    ar10 = AR10Interface(com_port=args.port)
    mock = args.port is None
    if not mock:
        from eval.policy_runner import watched_joint_indices
        ar10.assert_input_calibration(watched_joint_indices(cfg))

    # Der echte ContactDetector (Session-Baseline). Optional — bei fehlender
    # Baseline laeuft die Messung trotzdem, das Rohsignal ist der Kern.
    detector = None
    try:
        from eval.policy_runner import load_contact_detector
        detector = load_contact_detector(cfg)
    except Exception as e:
        print(f"[probe-contact] WARNUNG: ContactDetector nicht geladen ({e}) — "
              f"nur Rohsignal.")

    print(f"[probe-contact] joints={joints}  lead={lead_j}  rate={rate}  "
          f"dt={dt*1000:.1f} ms  grid={args.grid}  pause={args.pause_steps} Steps  "
          f"mock={mock}  detector={'an' if detector else 'AUS'}")

    fieldnames = ["cycle", "phase", "step", "t_ms", "pause_no"]
    for j in joints:
        fieldnames += [f"{j}_q_target", f"{j}_q_measured", f"{j}_q_delta"]
    fieldnames += [f"{f}_det_bit" for f in fingers]
    rows: list[dict] = []
    markers: list[dict] = []

    def log_row(cycle: int, phase: str, step: int, t_ms: float, pause_no: int,
                q_target: list[float], q_meas: list[float],
                det_bits=None) -> dict:
        row = {"cycle": cycle, "phase": phase, "step": step,
               "t_ms": round(t_ms, 1), "pause_no": pause_no}
        for j, idx in zip(joints, j_idxs):
            row[f"{j}_q_target"]   = round(q_target[idx], 5)
            row[f"{j}_q_measured"] = round(q_meas[idx], 5)
            row[f"{j}_q_delta"]    = round(q_target[idx] - q_meas[idx], 5)
        for fi, fname in enumerate(fingers):
            row[f"{fname}_det_bit"] = int(det_bits[fi]) if det_bits is not None else -1
        rows.append(row)
        return row

    def status_line(row: dict, extra: str = "") -> None:
        deltas = "  ".join(f"{j}:{row[f'{j}_q_delta']:+.4f}" for j in joints)
        bits = "".join(str(row[f"{f}_det_bit"]) if row[f"{f}_det_bit"] >= 0 else "-"
                       for f in fingers)
        print(f"\r  [{row['phase']:11s}] step={row['step']:4d} "
              f"{lead_j}@{row[f'{lead_j}_q_target']:.3f}  {deltas}  bits={bits} {extra}   ",
              end="", flush=True)

    keys = KeySource()
    aborted = False

    def auto_phase(cycle: int, q_target: list[float]) -> None:
        # HOLD0 + Rampe mit Rasterpausen, Runner-Timing. Veraendert q_target in place.
        nonlocal aborted
        if detector is not None:
            detector.reset()
        # Naechster Rasterpunkt OBERHALB der aktuellen Leitposition.
        import math
        next_pause = (math.floor(q_target[lead_idx] / args.grid) + 1) * args.grid
        t0 = time.perf_counter()
        step = 0
        pause_no = -1
        pause_left = 0
        hold0_left = _HOLD0_STEPS
        hold_end_left = -1
        print()
        while step < _MAX_STEPS:
            ch = keys.poll()
            if ch == "x":
                print("\n  NOT-AUS — Hand oeffnet.")
                aborted = True
                return
            if hold0_left > 0:
                phase = "HOLD0"
                hold0_left -= 1
            elif pause_left > 0:
                phase = "PAUSE"
                pause_left -= 1
            elif hold_end_left > 0:
                phase = "HOLD_END"
                hold_end_left -= 1
            elif hold_end_left == 0:
                break
            else:
                for j, idx in zip(joints, j_idxs):
                    q_target[idx] = min(cap_of[j], q_target[idx] + rate)
                phase = "MOVE"
                if q_target[lead_idx] >= next_pause - 1e-6:
                    pause_no += 1
                    pause_left = args.pause_steps
                    next_pause += args.grid
                if all(q_target[idx] >= cap_of[j] - 1e-9
                       for j, idx in zip(joints, j_idxs)):
                    hold_end_left = _HOLD_END
            ar10.send_q_target(list(q_target))
            # Runner-Timing: Slot abwarten, dann lesen (§8.3).
            slot = t0 + (step + 1) * dt - time.perf_counter()
            if slot > 0:
                time.sleep(slot)
            q_meas = ar10.read_q_measured()
            det_bits = (detector.update(list(q_target), list(q_meas))
                        if detector is not None else None)
            row = log_row(cycle, phase, step,
                          (time.perf_counter() - t0) * 1000,
                          pause_no if phase == "PAUSE" else -1,
                          q_target, q_meas, det_bits)
            status_line(row)
            step += 1
        print()

    def open_hand() -> None:
        ar10.send_q_target(_pregrasp_q())
        if not mock:
            time.sleep(1.5)

    if args.freerun:
        # ── Referenz ohne Objekt ─────────────────────────────────────────────
        if not mock:
            input("\nHand ist FREI (kein Objekt)? Enter -> Start ...")
        for cycle in range(args.cycles):
            print(f"\n  Freilauf-Zyklus {cycle}")
            open_hand()   # kalt starten wie echte Episoden
            q_target = _pregrasp_q()
            auto_phase(cycle, q_target)
            if aborted:
                break
        mode_stem = "freerun"
    else:
        # ── Kontaktmodus: interaktiv -> AUTO, beliebig viele Zyklen ─────────
        print("\nTasten: w/s zu/auf (grob)  W/S fein  c=Kontakt-Marker  "
              "g=Messung starten  o=oeffnen  q=Ende")
        print("(Tasten wirken direkt; reagiert nichts, Kommando tippen und ENTER.)")
        if not mock:
            input("Objekt bereitlegen. Enter -> Hand faehrt auf Pregrasp ...")
        cycle = 0
        open_hand()
        q_target = _pregrasp_q()
        keys.start_stdin_thread()
        t0 = time.perf_counter()
        last_read = 0.0
        running = True
        while running:
            ch = keys.poll()
            if ch is None:
                if keys.eof():
                    running = False
                    continue
                # Live-Anzeige/-Log ~5 Hz, auch ohne Tastendruck.
                now = time.perf_counter()
                if now - last_read >= 0.2:
                    last_read = now
                    q_meas = ar10.read_q_measured()
                    row = log_row(cycle, "INTERACTIVE", -1,
                                  (now - t0) * 1000, -1, q_target, q_meas)
                    status_line(row, extra=f"jog={args.jog}")
                else:
                    time.sleep(0.02)
                continue

            if ch in ("w", "W", "s", "S"):
                d = args.jog if ch in ("w", "s") else args.jog_fine
                if ch in ("s", "S"):
                    d = -d
                for j, idx in zip(joints, j_idxs):
                    q_target[idx] = max(0.0, min(cap_of[j], q_target[idx] + d))
                ar10.send_q_target(list(q_target))
            elif ch == "c":
                q_meas = ar10.read_q_measured()
                row = log_row(cycle, "MARKER", -1,
                              (time.perf_counter() - t0) * 1000, -1, q_target, q_meas)
                m = {"cycle": cycle}
                for j, idx in zip(joints, j_idxs):
                    m[f"{j}_q_target"]   = round(q_target[idx], 5)
                    m[f"{j}_q_measured"] = round(q_meas[idx], 5)
                markers.append(m)
                print(f"\n  MARKER #{len(markers)} (Zyklus {cycle}): " +
                      "  ".join(f"{j}={m[f'{j}_q_target']:.3f}" for j in joints))
            elif ch == "g":
                print(f"\n  Zyklus {cycle}: AUTO-Phase ab "
                      f"{lead_j}={q_target[lead_idx]:.3f} (x = Not-Aus)")
                auto_phase(cycle, q_target)
                open_hand()
                q_target = _pregrasp_q()
                cycle += 1
                if aborted:
                    running = False
                else:
                    print(f"\n  Hand offen. Naechster Zyklus {cycle}: Objekt "
                          f"umsetzen, dann w/c/g — oder q = Ende.")
            elif ch == "o":
                open_hand()
                q_target = _pregrasp_q()
                print("\n  Hand geoeffnet (Zyklus laeuft weiter).")
            elif ch == "q":
                running = False
        mode_stem = "contact"

    open_hand()
    ar10.close()

    # ── Speichern ────────────────────────────────────────────────────────────
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.config).stem
    out  = _OUT_DIR / f"probe_contact_{mode_stem}_{stem}_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "mode": mode_stem, "config": args.config, "rate": rate,
        "grid": args.grid, "pause_steps": args.pause_steps,
        "hold0_steps": _HOLD0_STEPS, "dt_ms": round(dt * 1000, 2),
        "joints": joints, "lead_joint": lead_j,
        "detector": detector is not None, "markers": markers,
        "read_timing": "pre_next_send",
    }
    meta_out = out.with_name(out.stem + "_meta.yaml")
    with meta_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    print(f"\n[probe-contact] {len(rows)} Samples -> {out}")
    print(f"[probe-contact] Meta + {len(markers)} Marker -> {meta_out}")

    # ── Deskriptive Kurzsicht (KEINE Urteile — Auswertung passiert offline) ──
    for m in markers:
        print("  Marker Z{cycle}: ".format(**m) +
              "  ".join(f"{j}={m[f'{j}_q_target']:.3f}" for j in joints))
    cycles_seen = sorted({r["cycle"] for r in rows})
    for cyc in cycles_seen:
        pauses = sorted({r["pause_no"] for r in rows
                         if r["cycle"] == cyc and r["phase"] == "PAUSE"
                         and r["pause_no"] >= 0})
        if not pauses:
            continue
        print(f"\n  Zyklus {cyc} — q_delta je Probe-Pause (Start / +15 / Ende):")
        for pn in pauses:
            pr = [r for r in rows if r["cycle"] == cyc
                  and r["phase"] == "PAUSE" and r["pause_no"] == pn]
            qlead = pr[0][f"{lead_j}_q_target"]
            parts = []
            for j in joints:
                v0 = pr[0][f"{j}_q_delta"]
                v15 = pr[min(15, len(pr) - 1)][f"{j}_q_delta"]
                ve = pr[-1][f"{j}_q_delta"]
                parts.append(f"{j} {v0:+.4f}/{v15:+.4f}/{ve:+.4f}")
            print(f"    q={qlead:.2f}: " + "   ".join(parts))
    bits_first = {}
    for r in rows:
        for fname in fingers:
            if r[f"{fname}_det_bit"] == 1 and (r["cycle"], fname) not in bits_first:
                bits_first[(r["cycle"], fname)] = r[f"{lead_j}_q_target"]
    if bits_first:
        print("\n  Detektor-Bit zuerst gesetzt bei:")
        for (cyc, fname), q in sorted(bits_first.items()):
            print(f"    Zyklus {cyc} {fname}: q={q:.3f}")


if __name__ == "__main__":
    main()
