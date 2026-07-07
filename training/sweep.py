# Multi-Seed-Sweep: trainiert dieselbe Config mehrfach mit verschiedenen Seeds.
#
# Jeder Seed laeuft als eigener Subprozess (-m training.train --seed N), damit
# PyBullet/SB3-State zwischen den Laeufen garantiert sauber getrennt ist. Die
# Modelle landen seed-getrennt unter artifacts/models/<grasp_type>/seed_<N>/.
#
# Usage:
#   python -m training.sweep --config configs/power.yaml --n-seeds 5
#   python -m training.sweep --config configs/precision.yaml --seeds 0 1 2 3 4
#   python -m training.sweep --config configs/power.yaml --seeds 5 6 7   # weitere Seeds nachlegen
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training sweep.")
    parser.add_argument("--config",     required=True,
                        help="Grasp config (configs/power.yaml oder precision.yaml).")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml")
    parser.add_argument("--seeds",      nargs="+", type=int, default=None,
                        help="Explizite Seed-Liste, z.B. --seeds 0 1 2 3 4.")
    parser.add_argument("--n-seeds",    type=int, default=5,
                        help="Falls --seeds fehlt: Seeds 0..n-1. Default 5.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Bei Fehler in einem Seed weitermachen statt abbrechen.")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    print(f"[sweep] config = {args.config}")
    print(f"[sweep] seeds  = {seeds}")
    print(f"[sweep] {len(seeds)} Laeufe nacheinander.\n")

    results: dict[int, str] = {}
    t_start = time.time()

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"[sweep] Lauf {i}/{len(seeds)}  —  seed = {seed}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable, "-m", "training.train",
            "--config",     args.config,
            "--ppo-config", args.ppo_config,
            "--seed",       str(seed),
        ]
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))

        if proc.returncode == 0:
            results[seed] = "OK"
        else:
            results[seed] = f"FAILED (exit {proc.returncode})"
            if not args.keep_going:
                print(f"\n[sweep] Seed {seed} fehlgeschlagen — Abbruch "
                      f"(--keep-going zum Weitermachen).")
                break

    dt = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"[sweep] Fertig in {dt/60:.1f} min")
    for seed in seeds:
        print(f"  seed {seed}: {results.get(seed, 'uebersprungen')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
