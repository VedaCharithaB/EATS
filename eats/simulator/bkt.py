"""Bayesian Knowledge Tracing: fitting a student model to real interaction data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Bounds. Guess and slip capped at 0.3 to prevent degenerate fits where knowing
# the skill makes a correct answer LESS likely (Baker, Corbett & Aleven, 2008).
BOUNDS = [(0.01, 0.99),   # L0
          (0.01, 0.99),   # T
          (0.01, 0.30),   # G
          (0.01, 0.30)]   # S

MAX_OPPORTUNITIES = 50
N_RESTARTS = 5


@dataclass
class BKTParams:
    skill: str
    L0: float
    T: float
    G: float
    S: float
    n_students: int
    n_interactions: int
    log_likelihood: float

    def p_correct(self, mastery):
        """Probability of an unaided correct answer at this mastery level."""
        return mastery * (1 - self.S) + (1 - mastery) * self.G

    def update(self, mastery, correct):
        """Bayesian posterior given the observation, then the learning transition."""
        if correct:
            num = mastery * (1 - self.S)
            den = num + (1 - mastery) * self.G
        else:
            num = mastery * self.S
            den = num + (1 - mastery) * (1 - self.G)
        posterior = num / den if den > 0 else mastery
        return posterior + (1 - posterior) * self.T

    def learning_curve(self, n=25):
        """Closed-form marginal learning curve, unconditioned on any student.

        Each opportunity gives an independent chance T of learning, so
        L_t = 1 - (1-L0)(1-T)^(t-1), and P(correct) follows from guess and slip.
        """
        t = np.arange(1, n + 1)
        mastery = 1 - (1 - self.L0) * (1 - self.T) ** (t - 1)
        return t, mastery * (1 - self.S) + (1 - mastery) * self.G

    def as_dict(self):
        return asdict(self)


def build_sequence_matrix(frame, max_len=MAX_OPPORTUNITIES):
    """Pad per-student practice sequences into an [n_students, max_len] matrix.

    The forward recursion is sequential in time but independent across students,
    so stepping through time once and updating every student with numpy is much
    faster than looping in Python. The optimiser calls this many times.
    """
    frame = frame.sort_values(["user_id", "opportunity"])
    seqs = [
        g["correct"].to_numpy(dtype=np.float64)[:max_len]
        for _, g in frame.groupby("user_id", sort=False)
    ]
    seqs = [s for s in seqs if len(s) > 0]
    width = max(len(s) for s in seqs)

    obs = np.zeros((len(seqs), width))
    mask = np.zeros((len(seqs), width), dtype=bool)
    for i, s in enumerate(seqs):
        obs[i, : len(s)] = s
        mask[i, : len(s)] = True
    return obs, mask


def negative_log_likelihood(params, obs, mask):
    """How badly these four parameters explain the observed answers."""
    L0, T, G, S = params
    eps = 1e-10
    mastery = np.full(obs.shape[0], L0)
    total = 0.0

    for t in range(obs.shape[1]):
        active = mask[:, t]
        if not active.any():
            break

        p = np.clip(mastery * (1 - S) + (1 - mastery) * G, eps, 1 - eps)
        c = obs[:, t]
        total += (c * np.log(p) + (1 - c) * np.log(1 - p))[active].sum()

        num_c, num_i = mastery * (1 - S), mastery * S
        den_c = num_c + (1 - mastery) * G
        den_i = num_i + (1 - mastery) * (1 - G)
        post = np.where(c > 0.5, num_c / np.maximum(den_c, eps),
                        num_i / np.maximum(den_i, eps))
        mastery = np.where(active, post + (1 - post) * T, mastery)

    return -total


def fit_skill(frame, skill, seed=42):
    """Fit one skill's four parameters by bounded maximum likelihood."""
    obs, mask = build_sequence_matrix(frame)
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(N_RESTARTS):
        # Random restarts, because the likelihood has local optima and a single start
        # may not find the best fit.
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
        res = minimize(negative_log_likelihood, x0, args=(obs, mask),
                       bounds=BOUNDS, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res

    L0, T, G, S = best.x
    return BKTParams(str(skill), float(L0), float(T), float(G), float(S),
                     int(obs.shape[0]), int(mask.sum()), float(-best.fun))


def simulate_sequences(params, n_students, n_opportunities, seed=0):
    """Generate correct/incorrect sequences from known parameters.

    Used for parameter recovery: generate from known values, refit, and check
    the values come back.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for student in range(n_students):
        knows = rng.random() < params.L0
        for opp in range(1, n_opportunities + 1):
            correct = (rng.random() > params.S) if knows else (rng.random() < params.G)
            rows.append({"user_id": student, "opportunity": opp, "correct": int(correct)})
            if not knows and rng.random() < params.T:
                knows = True
    return pd.DataFrame(rows)
