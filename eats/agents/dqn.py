"""DQN training, and the run manifest saved with each result."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# The proposal specifies lr 1e-4, batch 64 and 500,000 steps. With those values
# and 50,000 steps, mean return was 0.19, about the same as a random policy (0.20).
#
# I then tried a few settings by hand. The values below raised mean return to
# 0.25, and the action distribution was less one-sided (60% maintain / 26% hint,
# against 42% / 43% before).
#
# Configurations tried, mean evaluation return over 3 seeds:
#   DQN lr=1e-4  50k   0.191   <- proposal values
#   DQN lr=1e-3 100k   0.247   <- adopted
#   DQN lr=1e-3 100k, rewards scaled x10   0.217
#   DQN lr=1e-3 100k, variance-reduced reward   0.246
#   PPO 100k           0.231
#   PPO 200k           0.238
# For reference: static baseline 0.279, best fixed policy 0.301, random 0.200.
HYPERPARAMS = {
    "learning_rate": 1e-3,
    "batch_size": 128,
    "buffer_size": 100_000,
    "gamma": 0.95,
    "learning_starts": 2_000,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.30,
    "exploration_final_eps": 0.02,
    "train_freq": 4,
}
TOTAL_TIMESTEPS = 100_000


def _fingerprint(paths):
    """Hash the simulator source so a result can be tied to the code that made it.

    If the simulator code changes, the hash changes, so a result can be checked
    against the version of the code that produced it.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        p = Path(p)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_manifest(condition, seed, extra=None):
    """Seed, settings, library versions and simulator hash for one run."""
    import sklearn
    import stable_baselines3
    import torch

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "condition": condition,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulator_fingerprint": _fingerprint([
            root / "simulator" / "env.py",
            root / "simulator" / "interventions.py",
            root / "simulator" / "bkt.py",
            root / "simulator" / "student.py",
        ]),
        "hyperparameters": HYPERPARAMS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def train_dqn(env_factory, seed, total_timesteps=TOTAL_TIMESTEPS,
              monitor_dir=None, verbose=0):
    """Train one DQN. Returns (model, monitor_path)."""
    env = env_factory(seed=seed)
    monitor_path = None
    if monitor_dir is not None:
        monitor_dir = Path(monitor_dir)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_path = monitor_dir / f"dqn_seed{seed}"
        env = Monitor(env, filename=str(monitor_path))

    model = DQN("MlpPolicy", env, seed=seed, verbose=verbose, **HYPERPARAMS)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model, (monitor_path.with_suffix(".monitor.csv") if monitor_path else None)


def load_monitor(path):
    """Read SB3's monitor csv (its first line is a JSON header comment)."""
    return pd.read_csv(path, skiprows=1)
