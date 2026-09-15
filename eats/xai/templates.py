"""Turn Shapley values into a short sentence for a teacher."""

from __future__ import annotations

import numpy as np

from eats.simulator import interventions as iv
from eats.xai.shapley import FEATURE_GROUPS, READABLE

# Cluster names from the notebook 04 profiles, ordered by severity.
CLUSTER_NAMES = {
    0: "on track",
    1: "occasional difficulty",
    2: "unproductive persistence",
    3: "systematic misconception",
}

ACTION_PHRASES = {
    "lower_difficulty": "easier material",
    "maintain_difficulty": "staying at this level",
    "raise_difficulty": "harder material",
    "provide_hint": "a hint",
    "recommend_revision": "revision of the earlier topic",
}

# Words instead of raw values, which are hard to interpret without context.
def _emphasis(rank):
    return ["The main reason is", "This is reinforced by", "A smaller factor is"][min(rank, 2)]


def describe_feature(name, state):
    """Plain-language description of one feature's current value."""
    if name == "accuracy":
        return f"recent accuracy of {state[0]:.0%}"
    if name == "response_time":
        pace = "slower" if state[1] > 0.55 else "faster" if state[1] < 0.45 else "typical"
        return f"a {pace} working pace" if pace != "typical" else "a typical working pace"
    if name == "hint_rate":
        return f"help being sought on {state[2]:.0%} of the problems answered wrongly"
    if name == "error_cluster":
        cluster = int(np.argmax(state[3:7]))
        return f"an error pattern of {CLUSTER_NAMES.get(cluster, cluster)}"
    return name


def explain_in_words(shapley, state, action, runner_up, top_k=2):
    """One contrastive sentence, plus supporting clauses.

    Contrastive ("why this rather than that"), following Miller (2019).

    Features are ranked by the absolute size of their Shapley value. The sign is
    not used, so the sentence doesn't say which way a feature pushed the decision.
    """
    ranked = sorted(shapley.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ranked = [(k, v) for k, v in ranked if abs(v) > 1e-6][:top_k]

    lead = (f"{iv.ACTIONS[action].replace('_', ' ').capitalize()} was recommended "
            f"rather than {ACTION_PHRASES.get(iv.ACTIONS[runner_up], 'the alternative')}")
    if not ranked:
        return lead + ", though no single factor stood out."

    parts = [f"{lead}. {_emphasis(0)} {describe_feature(ranked[0][0], state)}."]
    for rank, (name, _) in enumerate(ranked[1:], start=1):
        parts.append(f"{_emphasis(rank)} {describe_feature(name, state)}.")
    return " ".join(parts)


def technical_view(shapley, base_value, prediction, action, runner_up):
    """The raw numbers, for analysis."""
    rows = sorted(shapley.items(), key=lambda kv: abs(kv[1]), reverse=True)
    lines = [f"action={iv.ACTIONS[action]}  vs  {iv.ACTIONS[runner_up]}",
             f"margin={prediction:.4f}  base={base_value:.4f}"]
    lines += [f"  {READABLE[k]:<18} {v:+.4f}" for k, v in rows]
    lines.append(f"  {'sum':<18} {sum(shapley.values()):+.4f}")
    return "\n".join(lines)
