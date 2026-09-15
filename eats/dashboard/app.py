"""EATS teacher dashboard.

Run from the project root:

    .venv\\Scripts\\python.exe -m streamlit run eats/dashboard/app.py

WHY THIS EXISTS
===============
In notebook 07 the agent reached a higher learning gain than either baseline, but
its students answered roughly 33% of questions correctly against 68% under a fixed
curriculum, because the agent hints and a hint forces the item to be recorded
incorrect.

So a teacher looking only at accuracy would think the agent was doing worse. The
dashboard shows the explanation next to each recommendation, and the "what the
system knows" column shows the difference between visible accuracy and estimated
mastery.

DESIGN NOTES
============
The class is simulated and explained when the page loads (one explanation takes
about 0.38 ms), and the results are cached.

The main panel shows explanation sentences rather than SHAP values, which are hard
to read without context. The numbers are available behind a toggle.
"""

from __future__ import annotations

import sys
from pathlib import Path

EATS = Path(__file__).resolve().parents[2]
if str(EATS) not in sys.path:
    sys.path.insert(0, str(EATS))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import DQN

from eats.simulator import interventions as iv
from eats.simulator.bkt import BKTParams
from eats.simulator.env import EPISODE_LENGTH, TutoringEnv
from eats.simulator.student import EmpiricalObservables
from eats.xai.contrastive import decision_context, margin
from eats.xai.shapley import FEATURE_NAMES, READABLE, exact_shapley
from eats.xai.templates import CLUSTER_NAMES, explain_in_words

PROCESSED = EATS / "data" / "processed"
MODELS = EATS / "models"

BLUE, RED, AMBER, GREEN, GREY = "#4C72B0", "#C44E52", "#DD8452", "#55A868", "#7F7F7F"
# Pale tints with an explicit text colour, so the cell stays legible whatever
# theme the browser requests.
STATUS_STYLE = {
    "on track": "background-color:#e8f5e9;color:#1b5e20",
    "needs attention": "background-color:#fff4e5;color:#8a5200",
    "at risk": "background-color:#fdecea;color:#8e1b16",
}

st.set_page_config(page_title="EATS teacher dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Loading. Cached so the model and reference data load only once.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and student simulator...")
def load_system():
    raw = pd.read_parquet(PROCESSED / "interactions_clean.parquet")
    bundle = joblib.load(MODELS / "bkt_params.joblib")
    params = {d["skill"]: BKTParams(**d) for d in bundle["params"]}
    skills = bundle["skills"]
    cluster_model = joblib.load(MODELS / "error_clusters.joblib")

    rt_stats = {}
    for s in skills:
        v = raw.loc[raw["skill"] == s, "ms_first_response"]
        clip = np.percentile(v, cluster_model["clip_percentile"])
        lg = np.log1p(np.minimum(v, clip) / 1000)
        rt_stats[s] = {"clip": float(clip), "mean": float(lg.mean()),
                       "std": float(lg.std())}

    observables = EmpiricalObservables(raw, skills)
    model = DQN.load(MODELS / "dqn_seed0")
    return params, skills, cluster_model, rt_stats, observables, model


def make_env(params, observables, cluster_model, rt_stats, seed=None):
    return TutoringEnv(params, observables, cluster_model, rt_stats, seed=seed)


@st.cache_data(show_spinner="Simulating the class...")
def run_cohort(n_students: int, seed: int):
    """Run the agent over a class of simulated students, explaining every decision.

    Returns one row per interaction. Every decision carries its Shapley
    attribution and a plain-language explanation, computed live.
    """
    params, skills, cluster_model, rt_stats, observables, model = load_system()
    env = make_env(params, observables, cluster_model, rt_stats, seed=seed)

    # Background for the Shapley attributions: states the agent actually meets.
    # Attributions are always relative to this reference population.
    scout = make_env(params, observables, cluster_model, rt_stats, seed=seed + 1)
    bg = []
    for ep in range(20):
        obs, _ = scout.reset(seed=seed * 97 + ep)
        while True:
            a, _ = model.predict(obs, deterministic=True)
            bg.append(obs.copy())
            obs, _, term, trunc, _ = scout.step(int(a))
            if term or trunc:
                break
    rng = np.random.default_rng(seed)
    background = np.stack(bg)[rng.choice(len(bg), min(100, len(bg)), replace=False)]

    rows = []
    for student in range(n_students):
        obs, info = env.reset(seed=seed * 1000 + student)
        skill = info["skill"]
        mastery_start = info["mastery"]
        step = 0
        while True:
            ctx = decision_context(model, obs)
            f = lambda batch, a=ctx["action"]: margin(model, batch, a)
            sv, base, pred = exact_shapley(f, obs, background)
            text = explain_in_words(sv, obs, ctx["action"], ctx["runner_up"])

            obs_next, reward, term, trunc, i = env.step(ctx["action"])
            step += 1
            rows.append({
                "student": f"S{student + 1:02d}", "step": step, "skill": skill,
                "action": i["action_name"],
                "explanation": text,
                "margin": ctx["margin"],
                "accuracy": float(obs[0]), "response_time": float(obs[1]),
                "hint_rate": float(obs[2]),
                "cluster": int(np.argmax(obs[3:7])),
                "correct": i["correct"], "hint_used": float(i["hint_count"] > 0),
                "mastery": i["mastery"], "mastery_start": mastery_start,
                "masked": i["masked"],
                **{f"shap_{k}": v for k, v in sv.items()},
            })
            obs = obs_next
            if term or trunc:
                break
    return pd.DataFrame(rows)


def class_summary(steps: pd.DataFrame) -> pd.DataFrame:
    """One row per student: their latest state, plus how they are trending."""
    out = []
    for student, g in steps.groupby("student"):
        last = g.iloc[-1]
        recent = g.tail(10)
        observed = float(recent["correct"].mean())

        # Status is judged on what a teacher can see, not on latent mastery.
        if observed < 0.45 or (last["hint_rate"] > 0.7 and observed < 0.65):
            status = "at risk"
        elif observed < 0.68:
            status = "needs attention"
        else:
            status = "on track"

        out.append({
            "student": student,
            "topic": last["skill"],
            "status": status,
            "recent accuracy": observed,
            "hint dependency": float(last["hint_rate"]),
            "error pattern": CLUSTER_NAMES[int(last["cluster"])],
            "last intervention": last["action"].replace("_", " "),
            "system estimate of mastery": float(last["mastery"]),
            "mastery gain": float(last["mastery"] - last["mastery_start"]),
        })
    return pd.DataFrame(out).sort_values(
        ["status", "recent accuracy"],
        key=lambda s: s.map({"at risk": 0, "needs attention": 1, "on track": 2})
        if s.name == "status" else s,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("EATS")
st.sidebar.caption("Explainable Adaptive Tutoring System")

n_students = st.sidebar.slider("Class size", 10, 60, 30, step=5)
seed = st.sidebar.number_input("Simulation seed", 0, 9999, 42, step=1)
show_numbers = st.sidebar.toggle(
    "Show SHAP values", value=False,
    help="Off by default. A teacher can act on 'the main reason'; a number like "
         "-0.31 has no scale to calibrate against.")

steps = run_cohort(n_students, int(seed))
summary = class_summary(steps)

st.sidebar.divider()
counts = summary["status"].value_counts()
st.sidebar.metric("At risk", int(counts.get("at risk", 0)))
st.sidebar.metric("Needs attention", int(counts.get("needs attention", 0)))
st.sidebar.metric("On track", int(counts.get("on track", 0)))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Class overview")
st.caption(
    f"{n_students} students &middot; {EPISODE_LENGTH} practice opportunities each "
    f"&middot; every recommendation explained")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Student state", "Intervention log", "Progress", "What drives decisions"])

# ---------------------------------------------------------------------------
# Panel 1 - student state
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Who needs attention")
    st.write(
        "Sorted by concern. **Recent accuracy** is what you would see in a mark book. "
        "**System estimate of mastery** is what the tutoring model believes the student "
        "actually knows &mdash; the two can disagree, and where they do is worth a look."
    )

    view = summary.copy()
    for c in ["recent accuracy", "hint dependency", "system estimate of mastery",
              "mastery gain"]:
        view[c] = view[c].map(lambda v: f"{v:.0%}" if "gain" not in c else f"{v:+.0%}")

    st.dataframe(
        view.style.map(lambda v: STATUS_STYLE.get(v, ""), subset=["status"]),
        use_container_width=True, hide_index=True, height=min(620, 60 + 35 * len(view)))

    diverging = summary[(summary["recent accuracy"] < 0.5) &
                        (summary["system estimate of mastery"] > 0.8)]
    if len(diverging):
        st.info(
            f"**{len(diverging)} student(s) look worse than they are.** Their visible "
            "accuracy is low because the system has been giving hints, and a hinted "
            "problem is always recorded as incorrect &mdash; but the model estimates their "
            "underlying understanding is strong. This is exactly the case where the "
            "explanation matters more than the mark."
        )

# ---------------------------------------------------------------------------
# Panel 2 - intervention log
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Why each recommendation was made")
    student = st.selectbox("Student", summary["student"].tolist())
    g = steps[steps["student"] == student].reset_index(drop=True)

    row = summary[summary["student"] == student].iloc[0]
    st.markdown(f"**Topic:** {row['topic']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recent accuracy", f"{row['recent accuracy']:.0%}")
    c2.metric("Hint dependency", f"{row['hint dependency']:.0%}")
    c3.metric("Status", row["status"])

    # Confidence is RELATIVE to this class, not an absolute scale.
    #
    # An earlier version used a fixed threshold on the decision margin, which marked
    # 97% of decisions as "marginal". The margins are small (median about 0.003),
    # so decisions are labelled using terciles of this class's margins instead.
    lo, hi = g["margin"].quantile([1 / 3, 2 / 3])

    st.caption(
        "**clear / moderate / close** compares each decision against the others "
        "made for this class. Most are close calls - the agent is frequently "
        "choosing between options of near-equal value, which is worth knowing "
        "when reading a recommendation.")

    st.divider()
    for _, r in g.iloc[::-1].iterrows():
        label = ("clear" if r["margin"] > hi
                 else "close" if r["margin"] < lo else "moderate")
        outcome = "correct" if r["correct"] > 0.5 else "incorrect"
        hint = " (hint given)" if r["hint_used"] > 0.5 else ""

        with st.container(border=True):
            top, right = st.columns([5, 1])
            top.markdown(f"**Opportunity {int(r['step'])} &mdash; "
                         f"{r['action'].replace('_', ' ')}**")
            right.markdown(f"<div style='text-align:right;color:{GREY};font-size:0.85em'>"
                           f"{label} call</div>", unsafe_allow_html=True)
            st.write(r["explanation"])
            st.caption(f"Outcome: {outcome}{hint}")

            if show_numbers:
                vals = {READABLE[k]: r[f"shap_{k}"] for k in FEATURE_NAMES}
                st.dataframe(
                    pd.DataFrame({"contribution": vals}).sort_values(
                        "contribution", key=abs, ascending=False).T,
                    use_container_width=True)

# ---------------------------------------------------------------------------
# Panel 3 - progress
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Progress over the practice period")
    who = st.multiselect("Students", summary["student"].tolist(),
                         default=summary["student"].tolist()[:5])

    fig = go.Figure()
    for s in who:
        g = steps[steps["student"] == s]
        rolling = g["correct"].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=g["step"], y=rolling, mode="lines+markers",
                                 name=s, line=dict(width=2)))
    if who:
        baseline = steps[steps["student"].isin(who)]["mastery_start"].mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color=GREY,
                      annotation_text="starting level", annotation_position="right")
    fig.update_layout(height=420, xaxis_title="practice opportunity",
                      yaxis_title="rolling accuracy (5-problem window)",
                      yaxis_range=[0, 1], margin=dict(t=20, b=40),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Dips often follow a hint rather than a decline in understanding: a hinted "
        "problem is recorded as incorrect regardless of whether the student then "
        "answered it. Read this chart alongside the intervention log.")

# ---------------------------------------------------------------------------
# Panel 4 - what drives decisions
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("What the system is paying attention to")
    st.write(
        "Averaged across every recommendation made to this class. Larger bars mean "
        "the feature moved more decisions."
    )

    imp = (steps[[f"shap_{k}" for k in FEATURE_NAMES]].abs().mean()
           .rename(lambda c: READABLE[c.replace("shap_", "")])
           .sort_values())

    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color=BLUE))
    fig.update_layout(height=300, xaxis_title="average influence on the decision",
                      margin=dict(t=20, b=40, l=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.write("**Which feature drives which intervention**")
    per_action = (steps.groupby("action")[[f"shap_{k}" for k in FEATURE_NAMES]]
                  .apply(lambda g: g.abs().mean())
                  .rename(columns=lambda c: READABLE[c.replace("shap_", "")]))
    per_action.index = [a.replace("_", " ") for a in per_action.index]
    st.dataframe(per_action.style.background_gradient(cmap="Blues", axis=None)
                 .format("{:.4f}"), use_container_width=True)

    st.caption(
        "If different interventions were driven by the same feature, the system would "
        "not be adapting - it would be applying one rule. Variation across rows is "
        "what adaptive behaviour looks like.")

st.divider()
st.caption(
    "Simulated students, generated from Bayesian Knowledge Tracing parameters fitted "
    "to 474,011 real ASSISTments interactions. Explanations are exact Shapley values "
    "over the four state features, computed live in under a millisecond each."
)
