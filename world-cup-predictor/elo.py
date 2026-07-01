from load_data import load_data
import pandas as pd

df = load_data()

rankings = {}
home_elo = []
away_elo = []
k = 20
start_rating = 1500

for row in df.itertuples(index=True):
    home_team = row.home_team
    away_team = row.away_team
    home_rating = rankings.get(home_team, start_rating)
    away_rating = rankings.get(away_team, start_rating)
    home_elo.append(home_rating)
    away_elo.append(away_rating)
    E_home = 1 / (1 + 10 **((away_rating - home_rating) / 400))
    E_away = 1 - E_home
    home_score = row.home_score
    away_score = row.away_score
    if home_score > away_score:
        S_home = 1
        S_away = 0
    if home_score < away_score:
        S_home = 0
        S_away = 1
    if home_score == away_score:
        S_home = 0.5
        S_away = 0.5
    home_rating = home_rating + k * (S_home - E_home)
    away_rating = away_rating + k * (S_away - E_away)
    rankings[home_team] = home_rating
    rankings[away_team] = away_rating

df['home_elo'] = home_elo
df['away_elo'] = away_elo

top = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:20]
for team, rating in top:
    print(team, round(rating))