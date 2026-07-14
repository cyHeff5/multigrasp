# Fuehrt alle Diagnose-Tests nacheinander aus und zeichnet alles auf.
#
# Tests:
#   1. Servo-Analysis (Freilauf, 5 Zyklen) — Baseline q_delta
#   2. Servo-Analysis mit langsamerem delta_norm=0.003 — Geschwindigkeitsabhaengigkeit
#   3. Positions-abhaengiger Noise-Floor — q_delta bei verschiedenen Handpositionen
#   4. Raw-ADC-Aufzeichnung — echte Sensor-Aufloesung
#   5. Contact-Latency mit Kugel — Latenz zwischen Kontakt und Threshold
#   6. Contact-Latency mit Wuerfel — Vergleich
#
# Usage:
#   python -m eval.run_all_diagnostics --config configs/precision.yaml --port COM4
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import yaml

from sim.hand import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alle Diagnose-Tests nacheinander ausfuehren.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port",   required=True)
    parser.add_argument("--skip",   nargs="*", type=int, default=[],
                        help="Test-Nummern ueberspringen (z.B. --skip 5 6).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config

    skip = set(args.skip)

    print("=" * 60)
    print("  AR10 Diagnose-Suite")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Port:   {args.port}")
    print(f"  Tests:  1-6 (skip: {args.skip or 'keine'})")
    print()

    baseline_csv = None

    # ── Test 1: Servo-Analysis (normal speed) ────────────────────────
    if 1 not in skip:
        print("\n" + "=" * 60)
        print("  TEST 1/6: Servo-Analysis (Freilauf, delta_norm=0.005)")
        print("=" * 60)
        input("  Hand frei (KEIN Objekt)? Enter -> Start ...")

        from eval.servo_analysis import main as servo_main
        import sys
        old_argv = sys.argv
        sys.argv = ["servo_analysis",
                    "--config", args.config,
                    "--port", args.port,
                    "--cycles", "5"]
        try:
            servo_main()
        except SystemExit:
            pass
        sys.argv = old_argv

        # Neueste servo_analysis CSV finden (fuer contact_latency Baseline)
        from glob import glob
        csvs = sorted(glob(str(_REPO_ROOT / "artifacts" / "analysis" / "servo_analysis_*.csv")))
        if csvs:
            baseline_csv = csvs[-1]
            print(f"\n  -> Baseline CSV: {baseline_csv}")

        time.sleep(1.0)
    else:
        print("\n  [SKIP] Test 1: Servo-Analysis (normal)")

    # ── Test 2: Servo-Analysis (langsam) ─────────────────────────────
    if 2 not in skip:
        print("\n" + "=" * 60)
        print("  TEST 2/6: Servo-Analysis (Freilauf, delta_norm=0.003)")
        print("=" * 60)
        input("  Hand frei (KEIN Objekt)? Enter -> Start ...")

        # Temporaer Config aendern
        cfg_slow = copy.deepcopy(cfg)
        cfg_slow["action"]["delta_norm"] = 0.003
        slow_config = _REPO_ROOT / "configs" / "_temp_slow.yaml"
        with slow_config.open("w", encoding="utf-8") as f:
            yaml.dump(cfg_slow, f, default_flow_style=False)

        import sys
        old_argv = sys.argv
        sys.argv = ["servo_analysis",
                    "--config", str(slow_config),
                    "--port", args.port,
                    "--cycles", "3"]
        try:
            from eval.servo_analysis import main as servo_main
            servo_main()
        except SystemExit:
            pass
        sys.argv = old_argv

        # Temp-Config aufraeumen
        try:
            slow_config.unlink()
        except OSError:
            pass

        time.sleep(1.0)
    else:
        print("\n  [SKIP] Test 2: Servo-Analysis (langsam)")

    # ── Test 3: Positions-abhaengiger Noise-Floor ────────────────────
    if 3 not in skip:
        print("\n" + "=" * 60)
        print("  TEST 3/6: Positions-abhaengiger Noise-Floor")
        print("=" * 60)
        input("  Hand frei (KEIN Objekt)? Enter -> Start ...")

        from eval.position_noise_test import run as position_run
        position_run(cfg, args.port)

        time.sleep(1.0)
    else:
        print("\n  [SKIP] Test 3: Position Noise")

    # ── Test 4: Raw ADC ──────────────────────────────────────────────
    if 4 not in skip:
        print("\n" + "=" * 60)
        print("  TEST 4/6: Raw-ADC-Aufzeichnung")
        print("=" * 60)
        input("  Hand frei (KEIN Objekt)? Enter -> Start ...")

        from eval.raw_adc_test import run as adc_run
        adc_run(cfg, args.port)

        time.sleep(1.0)
    else:
        print("\n  [SKIP] Test 4: Raw ADC")

    # ── Test 5: Contact-Latency mit Kugel ────────────────────────────
    if 5 not in skip:
        print("\n" + "=" * 60)
        print("  TEST 5/6: Contact-Latency mit KUGEL")
        print("=" * 60)

        if baseline_csv is None:
            from glob import glob
            csvs = sorted(glob(str(_REPO_ROOT / "artifacts" / "analysis" / "servo_analysis_*.csv")))
            if csvs:
                baseline_csv = csvs[-1]

        if baseline_csv is None:
            print("  [FEHLER] Keine Baseline-CSV gefunden. Test 1 zuerst ausfuehren!")
        else:
            print(f"  Baseline: {baseline_csv}")
            print(f"\n  KUGEL aufs Podest legen!")

            import sys
            old_argv = sys.argv
            sys.argv = ["contact_latency",
                        "--config", args.config,
                        "--port", args.port,
                        "--baseline", baseline_csv,
                        "--cycles", "5"]
            try:
                from eval.contact_latency import main as cl_main
                cl_main()
            except SystemExit:
                pass
            sys.argv = old_argv

            time.sleep(1.0)
    else:
        print("\n  [SKIP] Test 5: Contact-Latency Kugel")

    # ── Test 6: Contact-Latency mit Wuerfel ──────────────────────────
    if 6 not in skip:
        print("\n" + "=" * 60)
        print("  TEST 6/6: Contact-Latency mit WUERFEL")
        print("=" * 60)

        if baseline_csv is None:
            from glob import glob
            csvs = sorted(glob(str(_REPO_ROOT / "artifacts" / "analysis" / "servo_analysis_*.csv")))
            if csvs:
                baseline_csv = csvs[-1]

        if baseline_csv is None:
            print("  [FEHLER] Keine Baseline-CSV gefunden. Test 1 zuerst ausfuehren!")
        else:
            print(f"  Baseline: {baseline_csv}")
            print(f"\n  WUERFEL aufs Podest legen!")

            import sys
            old_argv = sys.argv
            sys.argv = ["contact_latency",
                        "--config", args.config,
                        "--port", args.port,
                        "--baseline", baseline_csv,
                        "--cycles", "5"]
            try:
                from eval.contact_latency import main as cl_main
                cl_main()
            except SystemExit:
                pass
            sys.argv = old_argv
    else:
        print("\n  [SKIP] Test 6: Contact-Latency Wuerfel")

    print("\n" + "=" * 60)
    print("  ALLE TESTS ABGESCHLOSSEN")
    print("=" * 60)
    print(f"  Ergebnisse in: artifacts/analysis/")
    print()


if __name__ == "__main__":
    main()
