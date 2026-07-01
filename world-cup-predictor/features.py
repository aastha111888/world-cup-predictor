from load_data import load_data
from elo import elo   

def features():
    df = elo()
    history = {}
    home_form = []
    away_form = []
    home_gs, away_gs = [], []      
    home_gc, away_gc = [], []      
    window = 5

    for row in df.itertuples(index=True):
        home_team = row.home_team
        away_team = row.away_team
        home_score = row.home_score
        away_score = row.away_score

        home_games = history.get(home_team, [])
        away_games = history.get(away_team, [])

        home_form.append(sum(g[0] for g in home_games))   
        away_form.append(sum(g[0] for g in away_games))


        home_gs.append(sum(g[1] for g in home_games) / len(home_games) if home_games else 0)
        away_gs.append(sum(g[1] for g in away_games) / len(away_games) if away_games else 0)

        home_gc.append(sum(g[2] for g in home_games) / len(home_games) if home_games else 0)
        away_gc.append(sum(g[2] for g in away_games) / len(away_games) if away_games else 0)


        if home_score > away_score:
            home_points, away_points = 3, 0
        elif away_score > home_score:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        home_entry = (home_points, home_score, away_score)
        away_entry = (away_points, away_score, home_score)
        history[home_team] = (home_games + [home_entry])[-window:]
        history[away_team] = (away_games + [away_entry])[-window:]

    df['home_form'] = home_form
    df['away_form'] = away_form
    df['home_gs'] = home_gs
    df['away_gs'] = away_gs
    df['home_gc'] = home_gc
    df['away_gc'] = away_gc
    

    return df
features()
