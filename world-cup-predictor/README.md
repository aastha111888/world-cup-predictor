# World Cup Predictor

A probabilistic football tournament forecaster that combines a match-outcome
model with Monte Carlo simulation. It predicts individual matches as
calibrated probabilities, then simulates a knockout bracket thousands of times
to estimate each team's chance of reaching every round — and of winning it all.

## What it does

Given the teams remaining in a tournament's knockout stage, the simulator plays
out the rest of the tournament 10,000 times and reports each team's probability
of advancing through each round and lifting the trophy. Because it works from
any single-elimination stage, you can re-run it as results come in and teams are
eliminated.

Example output (from a quarterfinal bracket):

```
             Quarterfinal  Semifinal  Final  Champion
Argentina          100.0%      55.7%  34.7%     23.5%
Spain              100.0%      44.3%  25.9%     16.7%
France             100.0%      51.2%  20.4%     11.7%
Portugal           100.0%      55.1%  28.0%     11.2%
England            100.0%      52.3%  27.0%     10.5%
Brazil             100.0%      48.8%  19.0%     10.5%
Germany            100.0%      47.7%  24.6%      9.1%
Netherlands        100.0%      44.9%  20.4%      7.0%
```

## How it works

The project is a pipeline, each stage a separate module that feeds the next:

1. **Data** (`load_data.py`) — loads international match results (1990–present),
   parses dates, and sorts chronologically. Processing matches in strict time
   order is essential to everything downstream.

2. **Elo ratings** (`elo.py`) — assigns every team a dynamic strength rating
   that updates after each match, with the size of the update scaled by how
   surprising the result was. Each match records both teams' ratings *as they
   stood before kickoff*.

3. **Features** (`features.py`) — for each match, engineers each team's recent
   form (points over the last 5 games), rolling average goals scored, and
   rolling average goals conceded — all computed from prior games only.

4. **Model** (`model.py`) — a logistic regression classifier predicting the
   3-way outcome (home win / draw / away win) as probabilities, from the Elo,
   form, goals, and neutral-venue features.

5. **Simulation** (`simulation.py`) — a Monte Carlo knockout simulator that uses
   the model's match probabilities to play out a bracket thousands of times.

## Results

Evaluated on held-out matches from 2018 onward (the model never saw them during
training):

| Metric | Value | Notes |
|---|---|---|
| Accuracy | 0.599 | vs. ~0.48 for always predicting "home win" |
| Log loss | 0.874 | lower is better |
| Log loss (baseline) | 1.051 | predicting class frequencies only |

The model's log loss of **0.874** clearly beats the **1.051** no-information
baseline, confirming the engineered features carry real predictive signal.

**Calibration** — because the probabilities feed directly into the simulation,
it matters that they mean what they claim. The reliability curve below shows the
predicted home-win probability against the observed frequency; it closely tracks
the diagonal, with mild overconfidence in the mid-to-high range.

![Calibration curve](calibration.png)

## Design decisions

- **Time-based train/test split (no random shuffling).** The data is split by
  date — train before 2018, test 2018+ — rather than randomly. A random split
  would leak future matches into training, letting the model "learn from the
  future" and inflating its scores. A chronological split mirrors real use: the
  model only ever sees the past to predict the future, so the reported metrics
  are honest.

- **Proper scoring over accuracy.** Log loss and calibration judge the quality
  of the *probabilities*, not just whether the top pick was right — which is what
  matters for a simulator that consumes those probabilities directly.

- **Neutral-site symmetrization.** Knockout matches are on neutral ground, so
  each matchup is predicted twice with the teams swapped between the home/away
  slots and averaged, removing any arbitrary home-advantage bias.

- **Draws in knockouts.** A drawn knockout match goes to extra time and
  penalties, modeled as a coin flip: each side receives half of the predicted
  draw probability.

## Running it

Requires the `results.csv` international-results dataset in a `data/` folder.

```bash
pip install -r requirements.txt

# Reproduce the model evaluation and calibration plot:
python model.py

# Run the tournament simulation:
python simulation.py
```

To simulate a specific bracket, edit the `remaining` list at the bottom of
`simulation.py` — the teams still in, in bracket order (`[0]` plays `[1]`, `[2]`
plays `[3]`, ...), with a power-of-two length. If a team name isn't recognized,
`find_team('...')` lists the dataset's exact spellings.

## Limitations & future work

- Team strength reflects each team's most recent pre-tournament form; it does not
  update from results *within* the tournament being simulated.
- The match model is a logistic regression baseline; a gradient-boosted model
  (XGBoost / LightGBM) would likely improve log loss and is a natural next step.
- No live data feed — ratings come from a static dataset. Wiring in a live
  football results API would make the forecaster fully current.
- The simulator handles single-elimination knockouts; a full group-stage
  simulation (round-robin standings and tiebreakers) would extend it to model a
  tournament from the very start.