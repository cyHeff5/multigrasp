"""Aggregiert die Sim-Evaluation ueber mehrere Trainings-Seeds.

Laedt fuer jeden Seed das Modell (best/best_model oder final), faehrt dieselbe
Evaluation (shapes-Grid oder benchmark) und berichtet pro Objekt-Konfiguration
Mittelwert +/- Standardabweichung der Erfolgsrate UEBER die Seeds — plus eine
Headline-Zahl (mittlere Gesamt-Erfolgsrate +/- Std ueber die Seeds).

Das ist die Zahl, die ins Paper gehoert: nicht "der beste Lauf", sondern die
Verteilung ueber unabhaengige Seeds.

Usage:
    python -m eval.aggregate_seeds --config configs/power.yaml \\
                                   --mode shapes --trials 50
    python -m eval.aggregate_seeds --config configs/precision.yaml \\
                                   --mode benchmark --trials 50 --which final
    # Explizite Modelle statt Auto-Discovery der seed_*-Ordner:
    python -m eval.aggregate_seeds --config configs/power.yaml --mode shapes \\
                                   --models path/a/best_model path/b/best_model
"""
from __future__ import annotations

import argparse
import csv
import datetime
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

import yaml

from eval.eval_sim     import run_benchmark_mode, run_shapes_mode
from eval.object_grid  import spec_label
from eval.policy_runner import load_policy
from sim               import GraspEnv


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS    = _REPO_ROOT / "artifacts" / "models"
_PEDESTAL_H_DEFAULT = 0.04


# ── Modell-Discovery ────────────────────────────────────────────────────────────

def _discover_models(grasp_type: str, which: str) -> "OrderedDict[int, Path]":
    # Findet artifacts/models/<grasp_type>/seed_*/{best/best_model | final}.zip
    base   = _MODELS / grasp_type
    leaf   = "best/best_model" if which == "best" else "final"
    found: dict[int, Path] = {}
    for d in sorted(base.glob("seed_*")):
        try:
            seed = int(d.name.split("_")[-1])
        except ValueError:
            continue
        zip_path = d / f"{leaf}.zip"
        if zip_path.exists():
            found[seed] = zip_path.with_suffix("")
        else:
            print(f"[aggregate] WARN: {zip_path} fehlt — Seed {seed} uebersprungen")
    return OrderedDict(sorted(found.items()))


# ── Aggregations-Keys ───────────────────────────────────────────────────────────

def _row_key(mode: str, row: dict):
    if mode == "benchmark":
        return (row["part_id"], row["gp_id"])
    return (row.get("shape"), row.get("size_cm"), row.get("thickness_cm"),
            row.get("width_cm"), row.get("height_cm"))


def _row_label(mode: str, row: dict) -> str:
    if mode == "benchmark":
        return f"P{row['part_id']:02d}/{row['gp_id']}"
    return spec_label(row)


# ── Eval pro Seed ───────────────────────────────────────────────────────────────

def _eval_one(model_path: Path, cfg: dict, mode: str, args) -> list[dict]:
    policy = load_policy(str(model_path))
    env    = GraspEnv(cfg, render_mode=None)
    try:
        if mode == "shapes":
            return run_shapes_mode(env, policy, cfg,
                                   n_trials=args.trials, step_cm=args.step,
                                   base_seed=args.seed)
        ped_h = cfg["episode"].get("pedestal_height", _PEDESTAL_H_DEFAULT)
        return run_benchmark_mode(env, policy, cfg["grasp_type"],
                                  args.parts, args.trials, args.seed, ped_h)
    finally:
        env.close()


# ── Aggregation ─────────────────────────────────────────────────────────────────

def _aggregate(rows_by_seed: "OrderedDict[int, list[dict]]", mode: str) -> list[dict]:
    # Pro Config-Key die Erfolgsraten ueber die Seeds einsammeln.
    rates: dict = defaultdict(dict)          # key -> {seed: rate}
    label: dict = {}
    order: list = []
    for seed, rows in rows_by_seed.items():
        for row in rows:
            k = _row_key(mode, row)
            if k not in label:
                label[k] = _row_label(mode, row)
                order.append(k)
            rates[k][seed] = float(row["success_rate"])

    seeds = list(rows_by_seed.keys())
    out: list[dict] = []
    for k in order:
        per_seed = [rates[k].get(s) for s in seeds]
        vals     = [v for v in per_seed if v is not None]
        mean     = statistics.mean(vals) if vals else 0.0
        std      = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rec = {
            "label":     label[k],
            "n_seeds":   len(vals),
            "mean_rate": round(mean, 4),
            "std_rate":  round(std, 4),
            "min_rate":  round(min(vals), 4) if vals else 0.0,
            "max_rate":  round(max(vals), 4) if vals else 0.0,
        }
        for s, v in zip(seeds, per_seed):
            rec[f"seed_{s}"] = (round(v, 4) if v is not None else None)
        out.append(rec)
    return out


def _headline(rows_by_seed: "OrderedDict[int, list[dict]]") -> tuple[float, float, list[float]]:
    # Pro Seed die makro-gemittelte Gesamt-Erfolgsrate, dann Mittel+/-Std ueber Seeds.
    per_seed_overall: list[float] = []
    for rows in rows_by_seed.values():
        if rows:
            per_seed_overall.append(statistics.mean(float(r["success_rate"]) for r in rows))
    if not per_seed_overall:
        return 0.0, 0.0, []
    mean = statistics.mean(per_seed_overall)
    std  = statistics.stdev(per_seed_overall) if len(per_seed_overall) > 1 else 0.0
    return mean, std, per_seed_overall


# ── Output ──────────────────────────────────────────────────────────────────────

def _save(agg: list[dict], headline, mode: str, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stem}_{mode}_seedagg_{ts}"
    mean, std, per_seed = headline

    with (out_dir / f"{base}.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({
            "mode": mode,
            "headline": {"mean": round(mean, 4), "std": round(std, 4),
                         "per_seed_overall": [round(v, 4) for v in per_seed]},
            "per_config": agg,
        }, f, default_flow_style=False, sort_keys=False)

    if agg:
        with (out_dir / f"{base}.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(dict.fromkeys(k for row in agg for k in row))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(agg)
    print(f"\nSaved: {out_dir / base}.{{yaml,csv}}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregiert Sim-Eval ueber Seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode",   choices=["shapes", "benchmark"], required=True)
    parser.add_argument("--which",  choices=["best", "final"], default="best",
                        help="Welches Modell pro Seed (default: best).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explizite Modellpfade statt Auto-Discovery der seed_*-Ordner.")
    parser.add_argument("--trials", type=int,   default=50)
    parser.add_argument("--step",   type=float, default=1.0, help="(shapes) Grid-Schritt in cm")
    parser.add_argument("--parts",  nargs="+",  type=int, default=list(range(1, 15)),
                        help="(benchmark) Part-IDs 1-14")
    parser.add_argument("--seed",   type=int,   default=1000, help="Eval-Basis-Seed")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    grasp_type  = cfg["grasp_type"]
    config_stem = Path(args.config).stem
    out_dir     = Path(args.output) if args.output else _REPO_ROOT / "artifacts" / "eval_results"

    # Modelle bestimmen.
    if args.models:
        models = OrderedDict((i, Path(m)) for i, m in enumerate(args.models))
    else:
        models = _discover_models(grasp_type, args.which)
    if not models:
        raise SystemExit(f"[aggregate] Keine Modelle gefunden unter "
                         f"{_MODELS / grasp_type}/seed_*/ — erst trainieren (training.sweep).")

    print(f"[aggregate] grasp_type = {grasp_type}")
    print(f"[aggregate] mode       = {args.mode}")
    print(f"[aggregate] seeds      = {list(models.keys())}  ({len(models)} Modelle)")
    print(f"[aggregate] trials/cfg = {args.trials}\n")

    rows_by_seed: "OrderedDict[int, list[dict]]" = OrderedDict()
    for seed, model_path in models.items():
        print(f"\n--- Seed {seed}: {model_path} ---")
        rows_by_seed[seed] = _eval_one(model_path, cfg, args.mode, args)

    agg      = _aggregate(rows_by_seed, args.mode)
    headline = _headline(rows_by_seed)
    mean, std, per_seed = headline

    # Tabelle.
    print("\n" + "=" * 78)
    print(f"{'Konfiguration':<46} {'mean':>7} {'std':>7} {'min':>6} {'max':>6}")
    print("-" * 78)
    for r in agg:
        print(f"{r['label']:<46} "
              f"{r['mean_rate']*100:6.1f}% {r['std_rate']*100:6.1f}% "
              f"{r['min_rate']*100:5.0f}% {r['max_rate']*100:5.0f}%")
    print("=" * 78)
    print(f"\nHEADLINE ({len(per_seed)} Seeds): "
          f"{mean*100:.1f}% +/- {std*100:.1f}%   "
          f"(per-seed overall: {[round(v*100,1) for v in per_seed]})")

    _save(agg, headline, args.mode, out_dir, config_stem)


if __name__ == "__main__":
    main()
