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