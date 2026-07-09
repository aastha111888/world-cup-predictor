from load_data import load_data
from elo import elo


def weighted_avg(values, alpha=0.7):
    """Average that gives more weight to recent games.

    `values` is ordered oldest -> newest (the last element is the most
    recent game). Each step back in time multiplies the weight by `alpha`,
    so with alpha=0.7 the most recent game counts most and older games
    fade off exponentially. Returns 0 for an empty list.
    """
    if not values:
        return 0
    n = len(values)
    # newest game (last index) gets exponent 0 -> weight 1;
    # each older game gets a smaller weight alpha, alpha^2, ...
    weights = [alpha ** (n - 1 - i) for i in range(n)]
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


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

        # Form: total points over the recent window (unchanged).
        home_form.append(sum(g[0] for g in home_games))
        away_form.append(sum(g[0] for g in away_games))

        # Goals scored / conceded: recency-weighted averages.
        home_gs.append(weighted_avg([g[1] for g in home_games]))
        away_gs.append(weighted_avg([g[1] for g in away_games]))
        home_gc.append(weighted_avg([g[2] for g in home_games]))
        away_gc.append(weighted_avg([g[2] for g in away_games]))

        if home_score > away_score:
            home_points, away_points = 3, 0
        elif away_score > home_score:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        # Each history entry: (points, goals_scored, goals_conceded).
        # Appended to the end, so the list stays oldest -> newest.
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