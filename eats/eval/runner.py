"""Multi-seed experiment harness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eats.agents.baselines import RuleBasedPolicy, StaticPolicy
from eats.agents.dqn import run_manifest, train_dqn
from eats.eval.rollout import episode_metrics, rollout


def run_all(env_factory, seeds, out_dir, n_eval_episodes=200, eval_seed=12345,
            monitor_dir=None, model_dir=None, verbose=True):
    """Train and evaluate every condition across every seed.

    All conditions are evaluated on the same episodes (fixed `eval_seed`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_steps, all_episodes = [], []

    for seed in seeds:
        conditions = {
            "static": StaticPolicy(),
            "rule_based": RuleBasedPolicy(),
        }
        # Baselines do not learn, but they are re-evaluated per seed so that
        # every condition is exposed to the same cohorts and the same variance.
        model, _ = train_dqn(env_factory, seed=seed, monitor_dir=monitor_dir)
        conditions["dqn"] = model

        # Save the model so the explanations and the dashboard can use it later.
        if model_dir is not None:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / f"dqn_seed{seed}"))

        for name, policy in conditions.items():
            env = env_factory(seed=eval_seed)
            steps = rollout(policy, env, n_episodes=n_eval_episodes, seed=eval_seed + seed)
            steps["condition"] = name
            steps["seed"] = seed
            all_steps.append(steps)

            eps = episode_metrics(steps)
            eps["condition"] = name
            eps["seed"] = seed
            all_episodes.append(eps)

            (out_dir / f"manifest_{name}_seed{seed}.json").write_text(
                json.dumps(run_manifest(name, seed), indent=2)
            )
        if verbose:
            print(f"  seed {seed} done", flush=True)

    steps = pd.concat(all_steps, ignore_index=True)
    episodes = pd.concat(all_episodes, ignore_index=True)
    steps.to_parquet(out_dir / "steps.parquet", index=False)
    episodes.to_parquet(out_dir / "episodes.parquet", index=False)
    return steps, episodes


def seed_level(episodes, metrics):
    """Aggregate within seed; the statistics use these seed-level values."""
    return episodes.groupby(["condition", "seed"])[metrics].mean().reset_index()