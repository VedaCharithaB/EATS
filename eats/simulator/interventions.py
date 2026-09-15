"""What each intervention does to the BKT dynamics.

All assumed intervention effects are kept in this file, so they can be printed as
a table and changed in one place.

BKT itself is fitted from data. These multipliers are not; they are based on the
literature cited below and are assumptions.
"""

from __future__ import annotations

ACTIONS = [
    "lower_difficulty",
    "maintain_difficulty",
    "raise_difficulty",
    "provide_hint",
    "recommend_revision",
]
N_ACTIONS = len(ACTIONS)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# --- Difficulty ------------------------------------------------------------
# Desirable difficulties (Bjork & Bjork, 2011): easier material raises immediate
# success but slows real learning; harder material does the opposite.
#
#   guess_mult : multiplies G - chance of success without knowing the skill
#   learn_mult : multiplies T - chance of learning at this opportunity
DIFFICULTY_EFFECTS = {
    0: {"guess_mult": 1.60, "learn_mult": 0.70, "label": "easy"},
    1: {"guess_mult": 1.00, "learn_mult": 1.00, "label": "medium"},
    2: {"guess_mult": 0.60, "learn_mult": 1.30, "label": "hard"},
}

# --- Hint ------------------------------------------------------------------
# Scaffolding (Wood, Bruner & Ross, 1976) raises learning at this opportunity.
# In ASSISTments a hint also forces the item to be recorded as incorrect, so the
# observed signal worsens while the latent state improves.
HINT_LEARN_MULT = 1.45

# --- Revision --------------------------------------------------------------
# Practising the prerequisite instead of the target skill. Costs one opportunity
# on the target, but raises the learn rate afterwards once the prerequisite is
# reasonably secure.
REVISION_LEARN_MULT = 1.50
REVISION_PREREQ_THRESHOLD = 0.50   # prerequisite mastery needed for transfer

# --- Reward weights --------------------------------------------------------
# w_learning : on the change in the mastery estimate, not observed accuracy,
#              because observed accuracy can be raised by lowering difficulty.
# w_hint     : a cost charged when the agent chooses to hint, not a penalty on the
#              observed hint rate (hinted items are always incorrect, so that would
#              overlap with accuracy; see notebook 03). Set below the typical
#              per-step learning gain (T_eff * (1 - mastery), roughly 0.035); with
#              a larger cost, hinting was never chosen.
# w_misconception : reward for moving to a lower-severity error cluster (clusters
#              are ordered by severity in notebook 04).
REWARD_WEIGHTS = {
    "w_learning": 1.00,
    "w_hint": 0.01,
    "w_misconception": 0.10,
}

# --- Structural constraints (Olukola & Rahimi, 2026) -----------------------
# Reward weighting alone did not prevent degenerate behaviour in their study, so
# two limits on actions are applied as well:
MAX_CONSECUTIVE_LOWER = 2      # cannot keep making the work easier indefinitely.
                               # NOTE: with three difficulty levels the floor is
                               # reached first and `lower` is masked anyway, so
                               # this limit rarely applies.
RAISE_REQUIRES_MASTERY = 0.40  # cannot raise difficulty on an unmastered skill

# Prerequisite graph, from the skill selection in notebook 02.
PREREQUISITES = {
    "Multiplication Fractions": "Addition and Subtraction Fractions",
    "Division Fractions": "Multiplication Fractions",
    "Equation Solving More Than Two Steps": "Equation Solving Two or Fewer Steps",
}


def effective_params(params, difficulty, hint=False, revision_boost=False):
    """Apply intervention effects to a skill's fitted BKT parameters.

    Returns (guess, learn_rate) after modulation. Slip and the prior are left
    untouched: I found nothing in the literature suggesting an intervention changes
    how often a student who knows a skill slips.
    """
    d = DIFFICULTY_EFFECTS[difficulty]
    g = min(params.G * d["guess_mult"], 0.60)
    t = params.T * d["learn_mult"]
    if hint:
        t *= HINT_LEARN_MULT
    if revision_boost:
        t *= REVISION_LEARN_MULT
    return g, min(t, 0.95)


def assumptions_table():
    """Every assumption, as a dataframe."""
    import pandas as pd

    rows = []
    for level, e in DIFFICULTY_EFFECTS.items():
        rows.append((f"difficulty = {e['label']}", "guess multiplier",
                     e["guess_mult"], "Bjork & Bjork (2011)"))
        rows.append((f"difficulty = {e['label']}", "learn-rate multiplier",
                     e["learn_mult"], "Bjork & Bjork (2011)"))
    rows.append(("provide_hint", "learn-rate multiplier", HINT_LEARN_MULT,
                 "Wood, Bruner & Ross (1976)"))
    rows.append(("recommend_revision", "learn-rate multiplier", REVISION_LEARN_MULT,
                 "prerequisite transfer"))
    rows.append(("recommend_revision", "prerequisite threshold",
                 REVISION_PREREQ_THRESHOLD, "design choice"))
    for k, v in REWARD_WEIGHTS.items():
        rows.append(("reward", k, v, "design choice"))
    rows.append(("constraint", "max consecutive lower_difficulty",
                 MAX_CONSECUTIVE_LOWER, "Olukola & Rahimi (2026)"))
    rows.append(("constraint", "mastery needed to raise difficulty",
                 RAISE_REQUIRES_MASTERY, "Olukola & Rahimi (2026)"))
    return pd.DataFrame(rows, columns=["applies to", "parameter", "value", "grounding"])
