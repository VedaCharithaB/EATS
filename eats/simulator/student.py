"""A simulated student: BKT dynamics plus empirically sampled observables."""

from __future__ import annotations

import numpy as np


class EmpiricalObservables:
    """Samples response time, hints and attempts from the real data.

    Conditioned on (skill, correct). Observed values are sampled directly, so no
    distribution shape has to be assumed.
    """

    def __init__(self, frame, skills):
        self.pools = {}
        for skill in skills:
            for correct in (0, 1):
                sub = frame[(frame["skill"] == skill) & (frame["correct"] == correct)]
                self.pools[(str(skill), correct)] = {
                    "ms": sub["ms_first_response"].to_numpy(dtype=np.float64),
                    "hints": sub["hint_count"].to_numpy(dtype=np.float64),
                    "attempts": sub["attempt_count"].to_numpy(dtype=np.float64),
                }

    def sample(self, skill, correct, rng):
        pool = self.pools[(str(skill), int(correct))]
        i = rng.integers(len(pool["ms"]))
        hints = 0.0 if correct else float(pool["hints"][i])
        return {
            "ms_first_response": float(pool["ms"][i]),
            "hint_count": hints,          # a hint forces incorrectness - never on a correct item
            "attempt_count": float(pool["attempts"][i]),
        }


class BKTStudent:
    """One simulated learner working through a single skill.

    `mastery` is the estimated probability that the student knows the skill.
    The agent never sees it; it only sees the observable features, as it would in
    a real system. The reward uses this value rather than observed accuracy.
    """

    def __init__(self, params, observables, rng=None):
        self.params = params
        self.observables = observables
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        # Draw whether this student already knows the skill, then track belief.
        self.knows = self.rng.random() < self.params.L0
        self.mastery = self.params.L0
        self.opportunity = 0
        return self.mastery

    def answer(self):
        """Attempt one problem. Returns the observable record."""
        p = self.params
        if self.knows:
            correct = self.rng.random() > p.S
        else:
            correct = self.rng.random() < p.G

        obs = self.observables.sample(p.skill, int(correct), self.rng)
        self.opportunity += 1

        # True latent transition: an unknown skill may become known.
        if not self.knows and self.rng.random() < p.T:
            self.knows = True

        # Observer's belief, updated the same way a tutoring system would.
        self.mastery = p.update(self.mastery, correct)

        obs.update({"correct": float(correct), "opportunity": self.opportunity,
                    "mastery": self.mastery, "knows": bool(self.knows)})
        return obs
