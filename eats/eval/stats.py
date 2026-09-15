"""Statistical comparison of the three conditions.

The unit of analysis is the seed. Episodes within a seed share a trained policy
and are not independent observations of the method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def seed_level(episodes, metrics):
    """Collapse episodes to one row per (condition, seed)."""
    return episodes.groupby(["condition", "seed"])[metrics].mean().reset_index()


def cohens_d(a, b):
    """Standardised mean difference, pooled SD."""
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(a, b, n=5000, seed=0):
    """Percentile CI for the difference in means."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = [rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
             for _ in range(n)]
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def normality(by_seed, metric, conditions):
    """Shapiro-Wilk per condition. Reported, not relied upon: with n=10 a
    non-significant result is weak evidence of normality, not proof of it."""
    out = []
    for c in conditions:
        v = by_seed.loc[by_seed["condition"] == c, metric]
        w, p = stats.shapiro(v)
        out.append({"condition": c, "W": w, "p": p, "normal_at_0.05": p > 0.05})
    return pd.DataFrame(out)


def compare(by_seed, metric, conditions, reference="dqn"):
    """Kruskal-Wallis across conditions, then Mann-Whitney against `reference`."""
    groups = [by_seed.loc[by_seed["condition"] == c, metric].to_numpy() for c in conditions]
    H, p_kw = stats.kruskal(*groups)

    ref = by_seed.loc[by_seed["condition"] == reference, metric].to_numpy()
    rows = []
    for c in conditions:
        if c == reference:
            continue
        other = by_seed.loc[by_seed["condition"] == c, metric].to_numpy()
        U, p = stats.mannwhitneyu(ref, other, alternative="two-sided")
        lo, hi = bootstrap_ci(ref, other)
        rows.append({
            "metric": metric,
            "comparison": f"{reference} vs {c}",
            "mean_ref": ref.mean(),
            "mean_other": other.mean(),
            "difference": ref.mean() - other.mean(),
            "ci_low": lo,
            "ci_high": hi,
            "cohens_d": cohens_d(ref, other),
            "U": U,
            "p": p,
            "significant_at_0.05": p < 0.05,
        })
    return {"H": H, "p_kruskal": p_kw, "n_per_group": len(ref),
            "pairwise": pd.DataFrame(rows)}


def interpret_d(d):
    """Cohen's conventional labels (a rough guide)."""
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"
