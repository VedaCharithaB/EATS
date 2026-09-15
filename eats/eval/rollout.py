"""Run a policy through the environment and log every step."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rollout(policy, env, n_episodes=100, seed=0):
    """Evaluate any policy exposing `predict(obs) -> (action, state)`.

    Episodes are seeded from `seed`, so every condition gets the same sequence
    of simulated students.

    Returns one row per step.
    """
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 10_000 + ep)
        mastery_start = info["mastery"]
        step = 0
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, i = env.step(int(action))
            step += 1
            rows.append({
                "episode": ep,
                "step": step,
                "skill": i["skill"],
                "action": int(action),
                "action_name": i["action_name"],
                "masked": i["masked"],
                "difficulty": i["difficulty"],
                "correct": i["correct"],
                "hint_used": float(i["hint_count"] > 0),
                "cluster": i["cluster"],
                "mastery": i["mastery"],
                "mastery_start": mastery_start,
                "reward": reward,
                "r_learn": i["r_learn"],
                "r_hint": i["r_hint"],
                "r_misc": i["r_misc"],
                "obs_accuracy": float(obs[0]),
                "obs_hint_rate": float(obs[2]),
            })
            if terminated or truncated:
                break
    return pd.DataFrame(rows)


def episode_metrics(steps):
    """Collapse the step log into one row per episode.

    `learning_gain` is the normalised gain from the proposal,
    (post - pre) / (1 - pre), computed on the simulator's mastery estimate rather
    than observed accuracy.
    """
    out = []
    for ep, g in steps.groupby("episode"):
        pre = g["mastery_start"].iloc[0]
        post = g["mastery"].iloc[-1]
        n = len(g)
        early = g[g["step"] <= max(1, int(0.15 * n))]
        late = g[g["step"] > n - max(1, int(0.15 * n))]

        # Slope of observed accuracy across the episode - the proposal's
        # "accuracy improvement rate".
        y = g["correct"].to_numpy()
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0

        out.append({
            "episode": ep,
            "skill": g["skill"].iloc[0],
            "learning_gain": (post - pre) / (1 - pre) if pre < 1 else 0.0,
            "mastery_gain": post - pre,
            "final_mastery": post,
            "accuracy_slope": slope,
            "observed_accuracy": float(g["correct"].mean()),
            "hint_rate": float(g["hint_used"].mean()),
            "hint_rate_early": float(early["hint_used"].mean()),
            "hint_rate_late": float(late["hint_used"].mean()),
            "hint_reduction": float(early["hint_used"].mean() - late["hint_used"].mean()),
            "return": float(g["reward"].sum()),
            "masked_frac": float(g["masked"].mean()),
        })
    return pd.DataFrame(out)


def action_distribution(steps):
    """Share of each action.

    If one action takes most of the share regardless of state, the policy has
    probably collapsed onto it.
    """
    counts = steps["action_name"].value_counts(normalize=True)
    return counts.sort_values(ascending=False)
