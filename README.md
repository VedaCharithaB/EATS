# EATS — Explainable Adaptive Tutoring System

MSc project code (Liverpool John Moores University). A Deep Q-Network chooses a teaching
intervention for a simulated student, and each decision is explained in plain language for a
teacher.

Author: Veda Charitha Bandikallu

---

## What it does

- The student state has four features: quiz accuracy, response time, hint rate and error pattern.
- The agent chooses one of five interventions: lower, maintain or raise difficulty, give a hint,
  or recommend revision of a prerequisite skill.
- It is trained against a student simulator whose learning dynamics are fitted to real data
  (Bayesian Knowledge Tracing), and compared with a fixed curriculum and a rule-based policy.
- Each decision is explained with exact Shapley values and turned into a short sentence.
- A Streamlit dashboard shows the class, the recommendations and the explanations.

## Dataset

**ASSISTments 2012-13 School Data with Affect** (public)

- Source: https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect
- File used: `2012-2013-data-with-predictions-4-final.csv` (about 3 GB, not included here)

After cleaning and restricting to five skills: 474,011 interactions from 12,119 students.

Citation:

> Feng, M., Heffernan, N.T. and Koedinger, K.R. (2009). Addressing the assessment challenge
> with an online system that tutors as it assesses. *User Modeling and User-Adapted
> Interaction*, 19(3), pp.243–266.

## Setup

Python 3.12.

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name eats --display-name "Python (EATS)"
```

**Paths:** the notebooks were run on Windows with the project at `C:\Users\vedac\Desktop\EATS`.
Before running them elsewhere, change the `EATS = Path(...)` line at the top of each notebook,
and the CSV path in `01_ingestion.ipynb`.

## Run order

The notebooks run in order; each one reads what the previous one saved. The saved outputs are
kept, so the results can be read without running anything.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_ingestion.ipynb` | reads the raw CSV, keeps 12 columns |
| 02 | `02_cleaning.ipynb` | cleaning, practice counter, skill selection |
| 03 | `03_eda.ipynb` | exploratory figures |
| 04 | `04_features.ipynb` | state features and error-pattern clusters |
| 05 | `05_simulator.ipynb` | BKT fitting and validation, simulated student |
| 06 | `06_environment.ipynb` | intervention effects and the Gymnasium environment |
| 07 | `07_agent.ipynb` | baselines, DQN training, 10-seed comparison |
| 08 | `08_explanations.ipynb` | Shapley explanations, faithfulness check, statistical tests |

Notebooks 05–08 write the modules in `eats/` with `%%writefile`, so the notebook cells and the
`.py` files contain the same code.

A full run takes roughly 40 minutes, most of it the ten training seeds in notebook 07.

## Dashboard

```
.venv\Scripts\python.exe -m streamlit run eats\dashboard\app.py
```

It needs the model and data files created by the notebooks, so run the notebooks first.

## Layout

```
eats/
  simulator/   bkt.py            BKT fitting
               student.py        simulated student
               interventions.py  assumed intervention effects, in one place
               env.py            Gymnasium environment
  agents/      baselines.py      static and rule-based policies
               dqn.py            DQN training and run manifests
  eval/        rollout.py        step-level logging and episode metrics
               runner.py         multi-seed runs
               stats.py          statistical tests
  xai/         contrastive.py    decision margin being explained
               shapley.py        exact Shapley values
               templates.py      explanation sentences
  dashboard/   app.py            teacher dashboard
notebooks/     the analysis, in run order
```

## Notes

- Results are averaged over ten random seeds; the seed is the unit of analysis.
- Each training run writes a manifest to `results/runs/` with the seed, hyperparameters, library
  versions and a hash of the simulator source.
- The effects of the five interventions are assumptions (listed in `interventions.py`), not
  fitted from data.
- BKT has no forgetting, and the simulator has no disengagement.
- Training is unstable across seeds: four of ten seeds collapsed onto near-constant hinting.
