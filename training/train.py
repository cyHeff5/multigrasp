#PPO training for AR10 grasp policies.

#Usage:
#    python -m training.train --config configs/power.yaml
#    python -m training.train --config configs/precision.yaml --gui
#    python -m training.train --config configs/power.yaml --resume artifacts/models/power/ppo_500000_steps

#Outputs:
#    artifacts/models/<grasp_type>/         # checkpoints + best model
#    artifacts/models/<grasp_type>/final    # final saved policy
#    artifacts/logs/<grasp_type>_<ts>/      # tensorboard logs
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env  import SubprocVecEnv, VecMonitor

from sim import GraspEnv


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "artifacts" / "models"
_LOGS_DIR   = _REPO_ROOT / "artifacts" / "logs"


def make_env(grasp_cfg: dict, seed: int, render: bool = False):
    # SubprocVecEnv erwartet Callables, die die Env erst im Subprocess erzeugen.
    def _init():
        env = GraspEnv(grasp_cfg, render_mode="human" if render else None)
        env.reset(seed=seed)
        return env
    return _init


def make_learning_rate(schedule: str, base_lr: float):
    # constant -> konstanter float; linear -> Callable, das von base_lr linear auf 0 faellt.
    # SB3 ruft das Callable mit progress_remaining auf (1.0 am Start -> 0.0 am Ende),
    # also LR = progress_remaining * base_lr. Linearer Decay daempft Late-Stage-Collapse.
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return lambda progress_remaining: progress_remaining * base_lr
    raise ValueError(f"Unknown lr_schedule: {schedule!r} (constant|linear).")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training for AR10 grasp.")
    parser.add_argument("--config",     required=True,
                        help="Path to grasp config (e.g. configs/power.yaml).")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml")
    parser.add_argument("--gui",        action="store_true",
                        help="Render one env with GUI (forces n_envs=1).")
    parser.add_argument("--resume",     default=None,
                        help="Checkpoint path (without .zip) to resume from.")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Ueberschreibt total_timesteps aus der ppo-config "
                             "(z.B. fuer kurze Smoke-Tests).")
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default=None,
                        help="Ueberschreibt lr_schedule aus der ppo-config (constant|linear).")
    parser.add_argument("--ent-coef",   type=float, default=None,
                        help="Ueberschreibt ent_coef (z.B. 0.02 fuer mehr Exploration).")
    parser.add_argument("--tag",        default="",
                        help="Suffix fuer den Modell-Ordner (seed_<N>_<tag>), haelt "
                             "Ablationslaeufe getrennt. Getaggte Laeufe ignoriert der "
                             "Sweep-Aggregator automatisch.")
    parser.add_argument("--n-envs",     type=int, default=None,
                        help="Anzahl paralleler Envs (Default: cpu_count). Auf dem "
                             "Heimserver 3 setzen, damit ein Kern frei bleibt.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        grasp_cfg = yaml.safe_load(f)
    with open(args.ppo_config, encoding="utf-8") as f:
        ppo_cfg = yaml.safe_load(f)

    # CLI-Overrides fuer schnelle Smoke-Tests/Ablationen (None -> Wert aus ppo-config).
    total_timesteps = args.timesteps if args.timesteps is not None else ppo_cfg["total_timesteps"]
    lr_schedule     = args.lr_schedule if args.lr_schedule is not None else ppo_cfg.get("lr_schedule", "constant")
    ent_coef        = args.ent_coef    if args.ent_coef    is not None else ppo_cfg["ent_coef"]

    grasp_type = grasp_cfg["grasp_type"]
    n_envs     = 1 if args.gui else (args.n_envs or os.cpu_count())

    # Optionaler Tag haelt Ablationslaeufe (gleicher Seed, andere Hyperparams) getrennt.
    seed_dir  = f"seed_{args.seed}" + (f"_{args.tag}" if args.tag else "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{grasp_type}_{seed_dir}_{timestamp}"
    ckpt_dir  = _MODELS_DIR / grasp_type / seed_dir
    log_dir   = _LOGS_DIR   / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] grasp_type = {grasp_type}")
    print(f"[train] seed       = {args.seed}")
    print(f"[train] n_envs     = {n_envs}")
    print(f"[train] timesteps  = {total_timesteps}"
          + ("  (CLI-Override)" if args.timesteps is not None else ""))
    print(f"[train] lr_schedule = {lr_schedule} (base {ppo_cfg['learning_rate']})")
    print(f"[train] ent_coef   = {ent_coef}")
    print(f"[train] checkpoints -> {ckpt_dir}")
    print(f"[train] logs       -> {log_dir}")

    # Reproduzierbarkeit (RL_VERIFICATION.md §2.2): exaktes Kommando, alle
    # Hyperparameter und beide Config-Inhalte neben die Modelle legen. Wird am
    # Ende des Laufs um finished/total_timesteps ergaenzt.
    def _git_state() -> dict:
        try:
            rev   = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                                   capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO_ROOT,
                                   capture_output=True, text=True).stdout.strip()
            return {"commit": rev, "dirty": bool(dirty)}
        except Exception:
            return {"commit": None, "dirty": None}

    run_meta = {
        "command":         " ".join(sys.argv),
        "started_at":      timestamp,
        "seed":            args.seed,
        "n_envs":          n_envs,
        "total_timesteps": total_timesteps,
        "lr_schedule":     lr_schedule,
        "ent_coef":        ent_coef,
        "resume":          args.resume,
        "git":             _git_state(),
        "grasp_config":    {"path": args.config,     "content": grasp_cfg},
        "ppo_config":      {"path": args.ppo_config, "content": ppo_cfg},
        "finished":        False,
    }
    with open(ckpt_dir / "run_meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump(run_meta, f, default_flow_style=False, sort_keys=False)

    # Pro Run werden die Env-Seeds um args.seed * 1000 verschoben, damit verschiedene
    # Trainings-Seeds garantiert disjunkte Env-Seed-Bereiche haben (sonst würde z.B.
    # seed=0/env=1 dieselbe Episodenfolge ziehen wie seed=1/env=0).
    env_seed_base = args.seed * 1000

    # VecMonitor zeichnet Episode-Rewards und -Längen auf -> TensorBoard-Kurven.
    train_env = VecMonitor(SubprocVecEnv([
        make_env(grasp_cfg, env_seed_base + i, render=(args.gui and i == 0))
        for i in range(n_envs)
    ]))
    # Abstand zu den Training-Env-Seeds damit Eval-Episoden sich nicht überschneiden.
    eval_env = VecMonitor(SubprocVecEnv([
        make_env(grasp_cfg, env_seed_base + 900)
    ]))

    # // n_envs weil VecEnv den Callback einmal pro vectorized step aufruft,
    # aber n_envs echte Schritte parallel gemacht werden.
    callbacks = [
        CheckpointCallback(
            save_freq=max(ppo_cfg["checkpoint_freq"] // n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval_logs"),
            eval_freq=max(ppo_cfg["eval_freq"] // n_envs, 1),
            n_eval_episodes=ppo_cfg["n_eval_episodes"],
            verbose=1,
        ),
    ]

    # Nur PPO-Hyperparameter extrahieren.
    ppo_kwargs = {k: ppo_cfg[k] for k in (
        "n_steps", "batch_size", "n_epochs", "learning_rate",
        "gamma", "gae_lambda", "clip_range",
        "ent_coef", "vf_coef", "max_grad_norm",
    )}
    # LR-Schedule (constant/linear) und ent_coef-Override anwenden.
    ppo_kwargs["learning_rate"] = make_learning_rate(lr_schedule, float(ppo_cfg["learning_rate"]))
    ppo_kwargs["ent_coef"]      = ent_coef

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, tensorboard_log=str(log_dir))
        # Checkpoint-Dateiname: ppo_<steps>_steps.zip -> [-2] = Schrittzahl
        trained_so_far = int(Path(args.resume).stem.split("_")[-2])
        remaining = max(0, total_timesteps - trained_so_far)
        # reset_num_timesteps=False: TensorBoard-Achse läuft weiter statt von 0.
        model.learn(total_timesteps=remaining, callback=callbacks,
                     reset_num_timesteps=False, tb_log_name=run_name,
                     progress_bar=True)
    else:
        # seed= sorgt dafür, dass auch Netz-Initialisierung und Action-Sampling
        # vom Seed abhängen (nicht nur die Env). Erst damit sind die Läufe
        # wirklich unabhängige Stichproben für Multi-Seed-Reporting.
        model = PPO("MlpPolicy", train_env, verbose=1, seed=args.seed,
                     tensorboard_log=str(log_dir), **ppo_kwargs)
        model.learn(total_timesteps=total_timesteps,
                     callback=callbacks, tb_log_name=run_name, progress_bar=True)

    final_path = ckpt_dir / "final"
    model.save(str(final_path))
    print(f"\n[train] Done. Final model saved to {final_path}.zip")

    run_meta["finished"]        = True
    run_meta["finished_at"]     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta["model_timesteps"] = int(model.num_timesteps)
    with open(ckpt_dir / "run_meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump(run_meta, f, default_flow_style=False, sort_keys=False)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
