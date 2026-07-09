"""
Backtest: how well did the match model predict the 2026 World Cup?

The model was trained only on matches before 2018, so every 2026 match is
genuinely out-of-sample (no leakage). Each 2026 World Cup match already
carries pre-match features (Elo, form, goals) computed from prior games
only, so we can feed those rows straight to the trained model and compare
its predicted probabilities to the actual results.

Note: knockout matches decided on penalties are recorded here as draws
(the regulation result), since shootouts are stored separately. The model
is therefore scored on the regulation outcome (home win / draw / away win).

Run with:   python backtest.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

from model import df, model, scaler, feature_cols


# ---------------------------------------------------------------------------
# 1. Pull the 2026 World Cup finals matches out of the dataset
# ---------------------------------------------------------------------------
is_2026 = df['date'].dt.year == 2026
is_world_cup = df['tournament'].str.contains('World Cup', case=False, na=False)
is_qualifier = df['tournament'].str.contains('qualif', case=False, na=False)

wc = df[is_2026 & is_world_cup & ~is_qualifier].copy()

if len(wc) == 0:
    print("No 2026 World Cup finals matches found in the dataset.")
    print("Your results.csv may predate the tournament. Most recent rows:")
    print(df[['date', 'home_team', 'away_team', 'tournament']].tail(10).to_string(index=False))
    raise SystemExit

print(f"Backtesting on {len(wc)} 2026 World Cup matches "
      f"({wc['date'].min().date()} to {wc['date'].max().date()}).\n")


# ---------------------------------------------------------------------------
# 2. Run the trained model on those matches
# ---------------------------------------------------------------------------
X = wc[feature_cols]
X_scaled = scaler.transform(X)

probs = model.predict_proba(X_scaled)   # columns follow model.classes_ (A, D, H)
preds = model.predict(X_scaled)
y_true = wc['result']

H_i = list(model.classes_).index('H')
D_i = list(model.classes_).index('D')
A_i = list(model.classes_).index('A')


# ---------------------------------------------------------------------------
# 3. Score it
# ---------------------------------------------------------------------------
acc = accuracy_score(y_true, preds)
ll = log_loss(y_true, probs, labels=model.classes_)

# Baseline: predict the pre-2018 training-era class frequencies every time
train = df[df['date'] < '2018-01-01']
class_freqs = train['result'].value_counts(normalize=True)
baseline_probs = np.tile([class_freqs[c] for c in model.classes_], (len(wc), 1))
baseline_ll = log_loss(y_true, baseline_probs, labels=model.classes_)

print("=" * 52)
print(f"  Matches evaluated : {len(wc)}")
print(f"  Accuracy          : {acc:.3f}")
print(f"  Log loss (model)  : {ll:.3f}")
print(f"  Log loss (baseline): {baseline_ll:.3f}")
print("=" * 52)
print("Lower log loss is better; beating the baseline means the model's")
print("probabilities carried real information about these matches.\n")


# ---------------------------------------------------------------------------
# 4. Per-match breakdown
# ---------------------------------------------------------------------------
table = wc[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
table['P(home)'] = probs[:, H_i]
table['P(draw)'] = probs[:, D_i]
table['P(away)'] = probs[:, A_i]
table['predicted'] = preds
table['actual'] = y_true.values
table['correct'] = (preds == y_true.values)
table['date'] = table['date'].dt.date

# Scores as integers; only the probability columns as percentages.
table['home_score'] = table['home_score'].astype(int)
table['away_score'] = table['away_score'].astype(int)
prob_cols = ['P(home)', 'P(draw)', 'P(away)']
for c in prob_cols:
    table[c] = table[c].map(lambda v: f'{v:.1%}')

pd.set_option('display.width', 160)
pd.set_option('display.max_rows', None)
print(table.to_string(index=False))