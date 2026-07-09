import pandas as pd

def load_data():
    df = pd.read_csv('data/results.csv')


    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    df = df[df['date'] >= '1990-01-01']

    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.reset_index(drop=True)
    return df
