from features import features
df = features()

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'

df['result'] = df.apply(get_result, axis=1)
print(df['result'].value_counts())

feature_cols = ['home_elo', 'away_elo', 'home_form', 'away_form',
                'home_gs', 'away_gs', 'home_gc', 'away_gc', 'neutral']

X = df[feature_cols]
y = df['result']

print(df[feature_cols].head())
print(df[feature_cols].isnull().sum())

train_mask = df['date'] < '2018-01-01'
test_mask = df['date'] >= '2018-01-01'

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
accuracy = model.score(X_test_scaled, y_test)
print("Test accuracy:", accuracy)

from sklearn.metrics import log_loss

probs = model.predict_proba(X_test_scaled)
ll = log_loss(y_test, probs)
print("Log loss:", ll)

import numpy as np

# Naive baseline: predict the training-set class frequencies for every test match
class_freqs = y_train.value_counts(normalize=True)   # e.g. H: 0.48, A: 0.29, D: 0.23
# Build the probability array in the SAME class order the model uses
baseline_probs = np.tile(
    [class_freqs[c] for c in model.classes_],
    (len(y_test), 1)
)
baseline_ll = log_loss(y_test, baseline_probs)
print("Baseline log loss:", baseline_ll)
print("Model log loss:", ll)

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# model.classes_ is ['A','D','H'] — grab the column index for 'H'
h_index = list(model.classes_).index('H')
prob_home = probs[:, h_index]              # predicted P(home win) for each test match
actual_home = (y_test == 'H').astype(int)  # 1 if home actually won, else 0

frac_pos, mean_pred = calibration_curve(actual_home, prob_home, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Predicted probability of home win')
plt.ylabel('Actual frequency of home win')
plt.title('Calibration curve (home win)')
plt.legend()
plt.savefig('calibration.png', dpi=120, bbox_inches='tight')
plt.show()