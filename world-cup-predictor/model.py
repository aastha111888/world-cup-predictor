from features import features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

df = features()


def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'


df['result'] = df.apply(get_result, axis=1)

feature_cols = ['home_elo', 'away_elo', 'home_form', 'away_form',
                'home_gs', 'away_gs', 'home_gc', 'away_gc', 'neutral']
X = df[feature_cols]
y = df['result']

train_mask = df['date'] < '2018-01-01'
test_mask = df['date'] >= '2018-01-01'
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# Everything below only runs when you execute `python model.py` directly.
# It is skipped when another file (e.g. simulation.py) imports this module,
# so importing the trained model no longer triggers prints or a plot window.
if __name__ == "__main__":
    from sklearn.metrics import log_loss
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    print(df['result'].value_counts())
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))

    accuracy = model.score(X_test_scaled, y_test)
    print("Test accuracy:", accuracy)

    probs = model.predict_proba(X_test_scaled)
    ll = log_loss(y_test, probs)
    print("Log loss:", ll)

    # Naive baseline: predict the training-set class frequencies every time
    class_freqs = y_train.value_counts(normalize=True)
    baseline_probs = np.tile(
        [class_freqs[c] for c in model.classes_],
        (len(y_test), 1)
    )
    baseline_ll = log_loss(y_test, baseline_probs)
    print("Baseline log loss:", baseline_ll)
    print("Model log loss:", ll)

    # Calibration curve for the home-win probability
    h_index = list(model.classes_).index('H')
    prob_home = probs[:, h_index]
    actual_home = (y_test == 'H').astype(int)
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