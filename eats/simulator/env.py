"""Gymnasium environment: an RL agent choosing teaching interventions."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from eats.simulator import interventions as iv
from eats.simulator.student import BKTStudent

WINDOW = 10          # must match notebook 04
EPISODE_LENGTH = 20  # matches the mean sequence length of 20.0 observed
                     # in the real data (notebook 03). Longer episodes let the
                     # mastery ESTIMATE saturate: true knowledge is monotone
                     # (no forgetting), and while the belief can fall after an
                     # incorrect answer it trends to ~0.95-0.99 in practice, so
                     # every policy reaches ceiling and the conditions become
                     # indistinguishable.


class StateBuilder:
    """Recomputes the four state features from simulated history.

    Uses the notebook 04 definitions exactly, including the conditional hint rate
    (hints per INCORRECT attempt, not per attempt) and the fitted k-means model.
    """

    def __init__(self, cluster_model, rt_stats, window=WINDOW):
        self.km = cluster_model["kmeans"]
        self.scaler = cluster_model["scaler"]
        self.remap = cluster_model["remap"]
        self.n_clusters = len(self.remap)
        # Precomputed for the hot path: assigning a cluster happens once per
        # environment step, and sklearn's predict/transform validation overhead
        # dominates the arithmetic at that scale.
        self._mean = self.scaler.mean_
        self._scale = self.scaler.scale_
        self._centroids = self.km.cluster_centers_
        self.rt_stats = rt_stats
        self.window = window
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window)

    def push(self, record):
        self.history.append(record)

    def _response_time(self, skill):
        if not self.history:
            return 0.5
        st = self.rt_stats[skill]
        ms = min(self.history[-1]["ms_first_response"], st["clip"])
        z = (np.log1p(ms / 1000) - st["mean"]) / (st["std"] or 1.0)
        return float((np.clip(z, -3, 3) + 3) / 6)

    def _error_cluster(self):
        if not self.history:
            return 0
        errs = np.array([1.0 - h["correct"] for h in self.history])
        hints = np.array([h["hint_count"] for h in self.history])
        att = np.array([h["attempt_count"] for h in self.history])

        best = run = 0
        for e in errs:
            run = run + 1 if e > 0.5 else 0
            best = max(best, run)

        profile = np.array([errs.mean(), best / self.window, att.mean(),
                            float((hints * errs).mean())])
        # Equivalent to scaler.transform followed by kmeans.predict, without the
        # per-call validation. Standard k-means assigns by squared Euclidean
        # distance to the nearest centroid.
        z = (profile - self._mean) / self._scale
        raw = int(((self._centroids - z) ** 2).sum(axis=1).argmin())
        return int(self.remap[raw])

    def observe(self, skill):
        """The 7-dimensional observation the agent sees."""
        if self.history:
            correct = np.array([h["correct"] for h in self.history])
            accuracy = float(correct.mean())
            errors = float((1 - correct).sum())
            hints_on_err = float(
                sum((1 - h["correct"]) * (h["hint_count"] > 0) for h in self.history)
            )
            hint_rate = hints_on_err / errors if errors > 0 else 0.0
        else:
            accuracy, hint_rate = 0.5, 0.0

        cluster = self._error_cluster()
        onehot = np.zeros(self.n_clusters, dtype=np.float32)
        onehot[cluster] = 1.0

        obs = np.concatenate([
            [accuracy, self._response_time(skill), hint_rate], onehot
        ]).astype(np.float32)
        return obs, cluster


class TutoringEnv(gym.Env):
    """One simulated student working a target skill for EPISODE_LENGTH steps.

    The agent observes only the four student-state features. The reward uses the
    simulator's mastery estimate, which the agent never sees, so the agent can't
    score well just by making the work easier and raising observed accuracy.
    """

    metadata = {"render_modes": []}

    def __init__(self, params, observables, cluster_model, rt_stats,
                 skills=None, seed=None):
        super().__init__()
        self.params = params
        self.observables = observables
        self.rt_stats = rt_stats
        self.skills = skills or list(params)
        self.builder = StateBuilder(cluster_model, rt_stats)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(3 + self.builder.n_clusters,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(iv.N_ACTIONS)
        self._rng = np.random.default_rng(seed)

    # -- episode lifecycle --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.skill = (options or {}).get("skill") or str(self._rng.choice(self.skills))
        self.prereq = iv.PREREQUISITES.get(self.skill)

        # Copy the fitted parameters: `step` changes G and T for the intervention
        # effects, and changing the shared objects would alter the fitted values
        # for later episodes.
        self.student = BKTStudent(
            replace(self.params[self.skill]), self.observables, self._rng
        )
        self.prereq_student = (
            BKTStudent(replace(self.params[self.prereq]), self.observables, self._rng)
            if self.prereq in self.params else None
        )

        self.difficulty = 1
        self.consecutive_lower = 0
        self.revision_pending = False
        self.step_count = 0
        self.builder.reset()

        obs, self.cluster = self.builder.observe(self.skill)
        return obs, {"skill": self.skill, "mastery": self.student.mastery}

    # -- masking ------------------------------------------------------------
    def valid_actions(self):
        """Structural limits on the actions, used alongside the reward (see
        Olukola & Rahimi, 2026)."""
        valid = np.ones(iv.N_ACTIONS, dtype=bool)
        if self.difficulty == 0 or self.consecutive_lower >= iv.MAX_CONSECUTIVE_LOWER:
            valid[0] = False                                   # lower_difficulty
        if self.difficulty == 2 or self.student.mastery < iv.RAISE_REQUIRES_MASTERY:
            valid[2] = False                                   # raise_difficulty
        if self.prereq_student is None:
            valid[4] = False                                   # recommend_revision
        valid[1] = True                                        # maintain always available
        return valid

    # -- one interaction ----------------------------------------------------
    def step(self, action):
        action = int(action)
        valid = self.valid_actions()
        masked = not valid[action]
        if masked:
            action = 1  # fall back to maintain_difficulty

        name = iv.ACTIONS[action]
        mastery_before = self.student.mastery
        cluster_before = self.cluster

        if name == "lower_difficulty":
            self.difficulty = max(0, self.difficulty - 1)
            self.consecutive_lower += 1
        elif name == "raise_difficulty":
            self.difficulty = min(2, self.difficulty + 1)
            self.consecutive_lower = 0
        else:
            self.consecutive_lower = 0

        hint = name == "provide_hint"
        revision = name == "recommend_revision"

        if revision:
            # A step spent on the prerequisite: no progress on the target skill
            # now, but a faster learn rate next step once the prerequisite is
            # reasonably secure.
            g, t = iv.effective_params(self.params[self.prereq], self.difficulty)
            self.prereq_student.params.G, self.prereq_student.params.T = g, t
            record = self.prereq_student.answer()
            self.revision_pending = (
                self.prereq_student.mastery >= iv.REVISION_PREREQ_THRESHOLD
            )
        else:
            g, t = iv.effective_params(self.params[self.skill], self.difficulty,
                                       hint=hint, revision_boost=self.revision_pending)
            self.student.params.G, self.student.params.T = g, t
            record = self.student.answer()
            self.revision_pending = False
            if hint:
                # A hint forces the item to be recorded incorrect (notebook 03), even
                # though the latent learn rate was raised.
                record["correct"] = 0.0
                record["hint_count"] = max(record["hint_count"], 1.0)

        self.builder.push(record)
        obs, self.cluster = self.builder.observe(self.skill)

        # -- reward ---------------------------------------------------------
        w = iv.REWARD_WEIGHTS
        r_learn = w["w_learning"] * (self.student.mastery - mastery_before)
        r_hint = -w["w_hint"] if hint else 0.0
        r_misc = w["w_misconception"] * (cluster_before - self.cluster) / 3.0
        reward = r_learn + r_hint + r_misc

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= EPISODE_LENGTH

        info = {
            "action_name": name,
            "masked": masked,
            "difficulty": self.difficulty,
            "mastery": self.student.mastery,
            "knows": self.student.knows,
            "cluster": self.cluster,
            "correct": record["correct"],
            "hint_count": record["hint_count"],
            "r_learn": r_learn,
            "r_hint": r_hint,
            "r_misc": r_misc,
            "skill": self.skill,
        }
        return obs, float(reward), terminated, truncated, info


def make_env(params, observables, cluster_model, rt_stats, seed=None):
    """Factory used by the training scripts and the baselines."""
    return TutoringEnv(params, observables, cluster_model, rt_stats, seed=seed)