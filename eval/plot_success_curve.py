# Lernkurve als Erfolgsrate statt als Reward.
#
# Warum nicht ep_rew_mean: die Reward-Verteilung ist bimodal. Eine Episode endet
# mit r_lift_success oder mit r_lift_fail, dazwischen liegen 15 Punkte. Der
# Mittelwert beider Moden hat eine Streuung von rund +/-6 um sich selbst und
# wackelt deshalb staerker als er sich bewegt. Der Anteil erfolgreicher Episoden
# ist die Groesse, an der man ablesen kann, wann die Policy den Griff kann.
#
# Datenquelle ist evaluations.npz, das EvalCallback ohnehin schreibt. Laeufe ab
# dem is_success-Patch enthalten ein successes-Array und werden direkt gelesen.
# Fuer aeltere Laeufe wird die Rate aus den Einzel-Rewards rekonstruiert, und
# zwar exakt statt per Schwellwert-Raten:
#
#   Episodenreward = r_lift +- 0 - |r_step| * L - w_pedestal * P + PBRS-Rest
#
# mit L = Episodenlaenge (steht in ep_lengths), P >= 0 = Schritte mit
# Sockelkontakt und |PBRS-Rest| <= w_contact. Eine gescheiterte Episode kann
# damit hoechstens r_lift_fail - |r_step| * L + w_contact erreichen. Alles
# darueber ist zwingend ein Erfolg. Die Klassifikation ist also eindeutig, ohne
# P zu kennen; das Skript meldet den knappsten Abstand zu dieser Grenze mit.
#
# Usage:
#   python -m eval.plot_success_curve \
#       artifacts/models/precision/seed_0_probetrig \
#       artifacts/models/power/seed_0_probetrig \
#       --out artifacts/eval_results/success_curve.png
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_REWARD = {"r_lift_fail": -5.0, "r_step": -0.01, "w_contact": 0.1}


def _reward_cfg(run_dir: Path) -> dict:
    # Die Reward-Parameter stehen in der Kopie der Grasp-Config, die train.py
    # als run_meta.yaml neben die Checkpoints legt.
    meta = run_dir / "run_meta.yaml"
    if not meta.exists():
        return dict(_DEFAULT_REWARD)
    content = yaml.safe_load(meta.read_text()) or {}
    rw = (content.get("grasp_config") or {}).get("content", {}).get("reward", {})
    return {k: float(rw.get(k, v)) for k, v in _DEFAULT_REWARD.items()}


def _load(run_dir: Path):
    npz = run_dir / "eval_logs" / "evaluations.npz"
    if not npz.exists():
        raise SystemExit(f"keine evaluations.npz unter {npz}")
    d = np.load(npz)
    steps = d["timesteps"]
    if "successes" in d:
        succ = np.asarray(d["successes"], dtype=float)
        return steps, succ.mean(axis=1), succ.shape[1], None
    r, lengths = d["results"], d["ep_lengths"]
    rw = _reward_cfg(run_dir)
    # Obergrenze dessen, was eine gescheiterte Episode erreichen kann.
    ceil_fail = rw["r_lift_fail"] - abs(rw["r_step"]) * lengths + rw["w_contact"]
    succ = r > ceil_fail
    return steps, succ.mean(axis=1), r.shape[1], float(np.abs(r - ceil_fail).min())


def _load_tb(log_dir: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # SB3 legt die Events nicht direkt in tensorboard_log ab, sondern in einem
    # Unterordner <tb_log_name>_<n>. Beide Varianten werden akzeptiert.
    root = Path(log_dir)
    candidates = [root] if any(root.glob("events.out.tfevents*")) else \
        sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("events.out.tfevents*")))
    if not candidates:
        raise KeyError(f"keine TensorBoard-Events unter {root}")
    ea = EventAccumulator(str(candidates[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(tag)
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def _wilson(p: np.ndarray, n: int, z: float = 1.96):
    # Wilson-Intervall statt normaler Naeherung, weil die Raten nahe 1.0 liegen
    # und das symmetrische Intervall dort ueber 100 Prozent hinauslaufen wuerde.
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Modellordner mit eval_logs/evaluations.npz")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--tb", nargs="*", default=None, metavar="LOGDIR",
                    help="TensorBoard-Logverzeichnisse in der Reihenfolge der "
                         "Laeufe. Zeichnet rollout/success_rate dazu, die "
                         "dichte Trainingskurve (ein Punkt je Rollout statt je "
                         "Eval). Nur fuer Laeufe ab dem is_success-Patch.")
    ap.add_argument("--baseline", nargs="*", type=float, default=None,
                    metavar="RATE",
                    help="waagerechte Vergleichslinien, je Wert eine Erfolgsrate 0..1")
    ap.add_argument("--baseline-labels", nargs="*", default=None)
    ap.add_argument("--out", default="artifacts/eval_results/success_curve.png")
    args = ap.parse_args()

    labels = args.labels or [Path(r).parent.name + "/" + Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("Anzahl --labels passt nicht zur Anzahl der Laeufe")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (run, label) in enumerate(zip(args.runs, labels)):
        steps, rate, n_ep, margin = _load(Path(run))
        if margin is not None:
            print(f"{label}: aus Rewards rekonstruiert, knappster Abstand zur "
                  f"Entscheidungsgrenze {margin:.2f}")
        lo, hi = _wilson(rate, n_ep)
        c = colours[i % len(colours)]
        ax.plot(steps / 1e6, rate * 100, color=c, lw=2, marker="o", ms=3.5,
                label=f"{label}  (n={n_ep}/Punkt)")
        ax.fill_between(steps / 1e6, lo * 100, hi * 100, color=c, alpha=0.15)
        if args.tb and i < len(args.tb):
            try:
                tb_s, tb_v = _load_tb(args.tb[i], "rollout/success_rate")
                ax.plot(tb_s / 1e6, tb_v * 100, color=c, lw=1, alpha=0.45,
                        zorder=1, label=f"{label}, Training (je Rollout)")
            except KeyError:
                print(f"WARNUNG {label}: rollout/success_rate fehlt im "
                      f"TensorBoard-Log, Lauf ist aelter als der is_success-Patch")
        print(f"{label}: Start {rate[0]*100:.0f} %, Ende {rate[-1]*100:.0f} %, "
              f"Bestwert {rate.max()*100:.0f} % bei {steps[rate.argmax()]/1e6:.2f} M")

    for j, b in enumerate(args.baseline or []):
        bl = (args.baseline_labels or [])[j] if args.baseline_labels and j < len(args.baseline_labels) else f"Baseline {b*100:.0f} %"
        ax.axhline(b * 100, color="0.45", ls="--", lw=1.2)
        ax.text(ax.get_xlim()[1], b * 100 + 1, bl, ha="right", va="bottom",
                fontsize=8, color="0.35")

    ax.set_xlabel("Environment steps (Millionen)")
    ax.set_ylabel("Erfolgsrate (%)")
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"geschrieben: {out}")


if __name__ == "__main__":
    main()
