# Erzeugt die zwei Ergebnis-Plots aus den eval_sim-CSVs:
#   shapes    -> gruppiertes Balkendiagramm, mittlere Erfolgsrate pro Form
#   benchmark -> Punkt-Plot, eine Erfolgsrate pro Greifpunkt
# Farben wie in plot_learning_curves: Precision blau, Power rot.
#
# Usage:
#   python -m eval.plot_eval_results \
#       --precision-shapes    artifacts/eval_results/precision_shapes_20260522_012148.csv \
#       --power-shapes        artifacts/eval_results/power_shapes_20260522_014534.csv \
#       --precision-benchmark artifacts/eval_results/precision_benchmark_20260522_015745.csv \
#       --power-benchmark     artifacts/eval_results/power_benchmark_20260522_015757.csv
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_PRECISION = "#1f77b4"
_POWER     = "#d62728"

_SHAPE_ORDER = ["sphere", "cube", "cylinder", "rect_cylinder"]
_SHAPE_LABEL = {"sphere": "Sphere", "cube": "Cube",
                "cylinder": "Cylinder", "rect_cylinder": "Rect. cylinder"}

# Eigene Farben fuer die 1D-Formen (Blau/Rot bleiben den Policies vorbehalten).
_LINE_COLOR = {"sphere": "#2ca02c", "cube": "#ff7f0e", "cylinder": "#9467bd"}


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _degenerate(row: dict) -> bool:
    # rect_cylinder mit thickness > width verstoesst gegen die Sampler-
    # Konvention (thickness = kuerzere Seite) und kam im Training nie vor.
    return (row["shape"] == "rect_cylinder"
            and float(row["thickness_cm"]) > float(row["width_cm"]))


def _shape_stats(path: str) -> dict[str, dict]:
    # Pro Form: Mittel + min/max der Erfolgsrate ueber alle Groessen-Konfigs.
    by_shape: dict[str, list[float]] = defaultdict(list)
    for row in _read(path):
        if _degenerate(row):
            continue
        by_shape[row["shape"]].append(float(row["success_rate"]))
    return {
        s: {"mean": np.mean(r), "min": np.min(r), "max": np.max(r)}
        for s, r in ((s, np.array(v)) for s, v in by_shape.items())
    }


def plot_shapes(precision_csv: str, power_csv: str, out: Path) -> None:
    p = _shape_stats(precision_csv)
    w = _shape_stats(power_csv)
    shapes = [s for s in _SHAPE_ORDER if s in p or s in w]
    y = np.arange(len(shapes))
    h = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for off, stats, color, label in [
        (+h / 2, p, _PRECISION, "Precision"),
        (-h / 2, w, _POWER,     "Power"),
    ]:
        mean = np.array([stats.get(s, {}).get("mean", np.nan) for s in shapes]) * 100
        lo   = np.array([stats.get(s, {}).get("min",  np.nan) for s in shapes]) * 100
        hi   = np.array([stats.get(s, {}).get("max",  np.nan) for s in shapes]) * 100
        err  = np.vstack([mean - lo, hi - mean])
        ax.barh(y + off, mean, height=h, color=color, label=label, zorder=3,
                xerr=err, error_kw={"ecolor": "0.35", "capsize": 3, "lw": 1.0})
        for yi, m, h_ in zip(y + off, mean, hi):
            ax.text(h_ + 2.5, yi, f"{m:.0f}%", va="center", fontsize=8.5, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels([_SHAPE_LABEL[s] for s in shapes])
    ax.invert_yaxis()
    ax.set_xlabel("Erfolgsrate (%)")
    ax.set_xlim(0, 112)
    ax.set_title("Erfolgsrate pro Objektform (Simulation, shapes-Grid)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="x", alpha=0.25, zorder=0)
    _save(fig, out)


def _benchmark_points(path: str) -> list[tuple[str, float]]:
    # (Label, Erfolgsrate%) je Greifpunkt, absteigend sortiert.
    pts = []
    for r in _read(path):
        gp_num = int(r["gp_id"].split("_")[-1])
        pts.append((f"P{r['part_id']}/gp{gp_num}", float(r["success_rate"]) * 100))
    pts.sort(key=lambda t: t[1], reverse=True)
    return pts


def plot_benchmark(precision_csv: str, power_csv: str, out: Path) -> None:
    p_pts = _benchmark_points(precision_csv)
    w_pts = _benchmark_points(power_csv)
    n_max = max(len(p_pts), len(w_pts))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    for ax, pts, color, name in [
        (axes[0], p_pts, _PRECISION, "Precision"),
        (axes[1], w_pts, _POWER,     "Power"),
    ]:
        rates = np.array([r for _, r in pts])
        y = np.arange(len(pts))
        ax.barh(y, rates, height=0.7, color=color, zorder=3)
        for yi, r in zip(y, rates):
            ax.text(r + 2, yi, f"{r:.0f}", va="center", fontsize=8, color="0.25")

        mean = rates.mean()
        ax.axvline(mean, color=color, ls="--", lw=1.2, alpha=0.7, zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels([l for l, _ in pts], fontsize=8)
        # gleiche Balkendicke in beiden Panels: identische y-Spanne, 0 oben.
        ax.set_ylim(n_max - 0.4, -0.6)
        ax.set_xlim(0, 113)
        ax.set_xlabel("Erfolgsrate (%)")
        ax.set_title(f"{name}   (Ø {mean:.0f}%, {len(pts)} Greifpunkte)",
                     fontsize=10.5, fontweight="bold", color=color)
        ax.grid(axis="x", alpha=0.25, zorder=0)

    fig.suptitle("Benchmark-Teile — Erfolg pro Greifpunkt",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved -> {out}")
    print(f"saved -> {out.with_suffix('.pdf')}")
    plt.close(fig)


def _line_series(path: str) -> dict[str, tuple[list[float], list[float]]]:
    # Pro 1D-Form: (Dimension, Erfolgsrate%) sortiert.
    # sphere/cube -> size_cm, cylinder -> thickness_cm.
    rows = _read(path)
    out = {}
    for shape, dim in [("sphere", "size_cm"), ("cube", "size_cm"),
                       ("cylinder", "thickness_cm")]:
        pts = sorted((float(r[dim]), float(r["success_rate"]) * 100)
                     for r in rows if r["shape"] == shape)
        if pts:
            out[shape] = ([p[0] for p in pts], [p[1] for p in pts])
    return out


def plot_shapes_1d(precision_csv: str, power_csv: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, csv_path, name, pcolor in [
        (axes[0], precision_csv, "Precision", _PRECISION),
        (axes[1], power_csv,     "Power",     _POWER),
    ]:
        series = _line_series(csv_path)
        for shape in ("sphere", "cube", "cylinder"):
            if shape not in series:
                continue
            x, y = series[shape]
            ax.plot(x, y, "o-", color=_LINE_COLOR[shape], lw=2.2, ms=7,
                    label=_SHAPE_LABEL[shape])
        ax.set_title(name, fontsize=11, fontweight="bold", color=pcolor)
        ax.set_xlabel("Objektdimension (cm)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Erfolgsrate (%)")
    fig.suptitle("Erfolgsrate vs. Objektdimension (sphere / cube / cylinder)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out)


def _rect_grid(path: str):
    # rect_cylinder -> (Dicken, Breiten, Erfolgsraten-Matrix in %).
    rows = [r for r in _read(path) if r["shape"] == "rect_cylinder"]
    thick = sorted({float(r["thickness_cm"]) for r in rows})
    width = sorted({float(r["width_cm"]) for r in rows})
    grid = np.full((len(thick), len(width)), np.nan)
    for r in rows:
        if _degenerate(r):
            continue
        i = thick.index(float(r["thickness_cm"]))
        j = width.index(float(r["width_cm"]))
        grid[i, j] = float(r["success_rate"]) * 100
    return thick, width, grid


def plot_rect_heatmap(precision_csv: str, power_csv: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    fig.subplots_adjust(left=0.07, right=0.88, top=0.86, bottom=0.14, wspace=0.30)
    im = None
    for ax, csv_path, name, pcolor in [
        (axes[0], precision_csv, "Precision", _PRECISION),
        (axes[1], power_csv,     "Power",     _POWER),
    ]:
        thick, width, grid = _rect_grid(csv_path)
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(width)),  [f"{w:g}" for w in width])
        ax.set_yticks(range(len(thick)),  [f"{t:g}" for t in thick])
        ax.set_xlabel("Breite (cm)")
        ax.set_ylabel("Dicke (cm)")
        ax.set_title(name, fontsize=11, fontweight="bold", color=pcolor)
        for i in range(len(thick)):
            for j in range(len(width)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                # heller Text auf dunklen Zellen (dunkelrot / dunkelgruen).
                tc = "white" if (v < 22 or v > 78) else "0.12"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color=tc)
    cax = fig.add_axes((0.905, 0.16, 0.018, 0.62))
    fig.colorbar(im, cax=cax, label="Erfolgsrate (%)")
    fig.suptitle("rect_cylinder: Erfolgsrate nach Dicke × Breite",
                 fontsize=12, fontweight="bold")
    _save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision-shapes",    required=True)
    ap.add_argument("--power-shapes",        required=True)
    ap.add_argument("--precision-benchmark", required=True)
    ap.add_argument("--power-benchmark",     required=True)
    ap.add_argument("--out-dir", default="artifacts/eval_results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plot_shapes(args.precision_shapes, args.power_shapes,
                out_dir / "shapes_results.png")
    plot_shapes_1d(args.precision_shapes, args.power_shapes,
                   out_dir / "shapes_1d.png")
    plot_rect_heatmap(args.precision_shapes, args.power_shapes,
                      out_dir / "shapes_rect_heatmap.png")
    plot_benchmark(args.precision_benchmark, args.power_benchmark,
                   out_dir / "benchmark_results.png")


if __name__ == "__main__":
    main()
