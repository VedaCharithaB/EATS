"""The scalar function that SHAP explains: a contrastive decision margin."""

from __future__ import annotations

import numpy as np
import torch


def q_values(model, states):
    """Q-values for a batch of states. [n, n_actions]."""
    states = np.atleast_2d(np.asarray(states, dtype=np.float32))
    with torch.no_grad():
        t = torch.as_tensor(states, device=model.device)
        return model.policy.q_net(t).cpu().numpy()


def chosen_action(model, state):
    """The action the policy would take - the a* an explanation is about."""
    return int(np.argmax(q_values(model, state)[0]))


def margin(model, states, reference_action, runner_up=None):
    """Q(s, a*) - max_{a != a*} Q(s, a), for a FIXED reference action.

    Fixing a* matters. If the reference were recomputed for each perturbed state
    the explained function would change identity partway through, and the
    resulting attributions would not describe any single decision.

    Passing `runner_up` fixes the comparison action too, which gives a strictly
    pairwise "why this rather than that" contrast. Leaving it None compares
    against whichever alternative is best in each perturbed state - the more
    general "why this rather than anything else".
    """
    q = q_values(model, states)
    chosen = q[:, reference_action]
    if runner_up is not None:
        return chosen - q[:, runner_up]
    others = np.delete(q, reference_action, axis=1)
    return chosen - others.max(axis=1)


def decision_context(model, state):
    """Everything needed to describe one decision."""
    q = q_values(model, state)[0]
    order = np.argsort(q)[::-1]
    return {
        "q_values": q,
        "action": int(order[0]),
        "runner_up": int(order[1]),
        "margin": float(q[order[0]] - q[order[1]]),
    }
