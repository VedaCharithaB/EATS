"""Non-learning comparison policies.

Both expose SB3's `predict(obs) -> (action, state)` interface so a single
evaluation harness can run every condition without special cases.
"""

from __future__ import annotations

import numpy as np

# Observation layout (see notebook 04):
#   0 accuracy | 1 response_time | 2 hint_rate | 3..6 one-hot error cluster
IDX_ACCURACY = 0
IDX_RESPONSE_TIME = 1
IDX_HINT_RATE = 2
CLUSTER_OFFSET = 3

LOWER, MAINTAIN, RAISE, HINT, REVISE = 0, 1, 2, 3, 4


class StaticPolicy:
    """A fixed curriculum: the same sequence for every student.

    Always `maintain_difficulty`, so difficulty never changes and the state is
    never used. It is the simplest reference point.
    """

    name = "static"

    def predict(self, obs, deterministic=True):
        return MAINTAIN, None


class RuleBasedPolicy:
    """Hand-written thresholds, similar to a conventional rule-based tutor.

    Rules are checked in order of urgency:

      1. The most severe error cluster (systematic misconception): recommend
         revising the prerequisite.
      2. Very low accuracy: give a hint.
      3. Low accuracy: lower difficulty.
      4. High accuracy: raise difficulty.
      5. Otherwise: maintain.

    Thresholds 3 and 4 come from the research proposal. The policy only reacts
    to the current state; it doesn't plan ahead.
    """

    name = "rule_based"

    def __init__(self, low=0.60, high=0.85, stuck=0.40, severe_cluster=3):
        self.low = low
        self.high = high
        self.stuck = stuck
        self.severe_cluster = severe_cluster

    def predict(self, obs, deterministic=True):
        accuracy = float(obs[IDX_ACCURACY])
        cluster = int(np.argmax(obs[CLUSTER_OFFSET:]))

        if cluster >= self.severe_cluster:
            return REVISE, None
        if accuracy < self.stuck:
            return HINT, None
        if accuracy < self.low:
            return LOWER, None
        if accuracy > self.high:
            return RAISE, None
        return MAINTAIN, None


class RandomPolicy:
    """Uniform random choice, used only as a sanity check."""

    name = "random"

    def __init__(self, n_actions=5, seed=0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return int(self.rng.integers(self.n_actions)), None
