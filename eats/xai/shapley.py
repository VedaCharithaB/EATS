"""Exact Shapley values by full coalition enumeration.

With four conceptual features the complete set of 2^4 = 16 coalitions is small
enough to enumerate, so exact values are used instead of KernelExplainer's
sampling approximation.

A Shapley value is the average marginal contribution of a feature across every
possible ordering in which features could be added. The weighting below,
|S|!(n-|S|-1)!/n!, is exactly that average.
"""

from __future__ import annotations

import itertools
from math import factorial

import numpy as np

# Conceptual features, and the observation columns each occupies.
FEATURE_GROUPS = {
    "accuracy": [0],
    "response_time": [1],
    "hint_rate": [2],
    "error_cluster": [3, 4, 5, 6],   # one-hot: one unit, not four features
}
FEATURE_NAMES = list(FEATURE_GROUPS)

READABLE = {
    "accuracy": "quiz accuracy",
    "response_time": "response time",
    "hint_rate": "hint dependency",
    "error_cluster": "error pattern",
}


def _weights(n):
    """Shapley weight for a coalition of size s, over n features."""
    return {s: factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)}


def exact_shapley(value_fn, state, background, groups=None):
    """Exact Shapley values for one state.

    value_fn : callable taking [m, d] states and returning [m] scalars
    state    : the observation to explain, shape [d]
    background : reference states, shape [b, d]. Absent features are replaced by
                 values drawn from these rows, so an attribution is always
                 relative to this reference population. A different background
                 gives different attributions.

    Returns (values dict, base_value, prediction).
    """
    groups = groups or FEATURE_GROUPS
    names = list(groups)
    n = len(names)
    state = np.asarray(state, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    b = len(background)
    weights = _weights(n)

    # Evaluate every coalition once: v(S) = mean over background of f(x_S).
    coalitions = [frozenset(c) for r in range(n + 1)
                  for c in itertools.combinations(range(n), r)]
    batch = np.empty((len(coalitions) * b, state.shape[0]), dtype=np.float32)
    for k, coalition in enumerate(coalitions):
        block = background.copy()                       # absent features: background
        for i in coalition:                             # present features: the instance
            for col in groups[names[i]]:
                block[:, col] = state[col]
        batch[k * b:(k + 1) * b] = block

    values = value_fn(batch).reshape(len(coalitions), b).mean(axis=1)
    v = dict(zip(coalitions, values))

    shapley = {}
    for i, name in enumerate(names):
        total = 0.0
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                total += weights[r] * (v[S | {i}] - v[S])
        shapley[name] = float(total)

    return shapley, float(v[frozenset()]), float(v[frozenset(range(n))])


def check_additivity(shapley, base_value, prediction, tol=1e-4):
    """Shapley values must sum to prediction - base_value (efficiency axiom).

    A check on the implementation: if the values don't sum, the coalition
    enumeration or the weighting is wrong.
    """
    total = sum(shapley.values())
    gap = abs(total - (prediction - base_value))
    return gap < tol, gap
