"""
Flexible knockout-stage Monte Carlo tournament simulator.

Point it at the teams remaining in a tournament's knockout bracket (in
bracket order) and it simulates the rest of the tournament many times to
estimate each team's chance of reaching each round and of winning it all.

Works from ANY single-elimination stage (Round of 32 / 16, quarterfinals,
semifinals, ...) as long as the number of teams is a power of two. As
results come in and teams are eliminated, edit the team list and re-run.

Run it with:   python simulation.py
"""

from collections import defaultdict

import numpy as np
import pandas as pd

# Import the trained model, scaler, feature list, and match dataframe.
# (Because model.py now guards its evaluation under __main__, this import
#  trains the model but does NOT print metrics or open a plot.)
from model import model, scaler, feature_cols, df


# ---------------------------------------------------------------------------
# 1. Build a lookup of each team's most recent strength stats
# ---------------------------------------------------------------------------
# The model needs each team's Elo, recent form, goals scored and goals
# conceded. We stack every home appearance and every away appearance into
# one long table, then keep each team's most recent row.

_home = df[['date', 'home_team', 'home_elo', 'home_form', 'home_gs', 'home_gc']].rename(
    columns={'home_team': 'team', 'home_elo': 'elo', 'home_form': 'form',
             'home_gs': 'gs', 'home_gc': 'gc'})
_away = df[['date', 'away_team', 'away_elo', 'away_form', 'away_gs', 'away_gc']].rename(
    columns={'away_team': 'team', 'away_elo': 'elo', 'away_form': 'form',
             'away_gs': 'gs', 'away_gc': 'gc'})
_long = pd.concat([_home, _away]).sort_values('date')
team_stats = _long.groupby('team').last()   # index = team name


# Column order the model outputs probabilities in (sklearn sorts them: A, D, H)
_A = list(model.classes_).index('A')   # away win
_D = list(model.classes_).index('D')   # draw
_H = list(model.classes_).index('H')   # home win


def _feature_row(home, away):
    """One-row feature frame for a neutral-site match: `home` vs `away`."""
    h = team_stats.loc[home]
    a = team_stats.loc[away]
    data = {
        'home_elo': h['elo'], 'away_elo': a['elo'],
        'home_form': h['form'], 'away_form': a['form'],
        'home_gs': h['gs'], 'away_gs': a['gs'],
        'home_gc': h['gc'], 'away_gc': a['gc'],
        'neutral': True,   # knockout matches are played on neutral ground
    }
    return pd.DataFrame([data], columns=feature_cols)


def advance_probability(team_a, team_b):
    """
    Probability that team_a beats team_b and advances in a knockout match.

    We predict the match twice, swapping which team occupies the 'home'
    slot, and average the two — so the arbitrary home/away assignment does
    not bias a neutral-site match. A draw in a knockout goes to extra time
    and penalties, which we treat as a coin flip, so each side gets half of
    the draw probability.
    """
    # team_a as home, team_b as away
    p1 = model.predict_proba(scaler.transform(_feature_row(team_a, team_b)))[0]
    a_win_1, b_win_1, draw_1 = p1[_H], p1[_A], p1[_D]

    # team_b as home, team_a as away (symmetrise)
    p2 = model.predict_proba(scaler.transform(_feature_row(team_b, team_a)))[0]
    a_win_2, b_win_2, draw_2 = p2[_A], p2[_H], p2[_D]

    a_win = (a_win_1 + a_win_2) / 2
    draw = (draw_1 + draw_2) / 2
    return a_win + 0.5 * draw


def _round_label(n_teams):
    return {1: 'Champion', 2: 'Final', 4: 'Semifinal', 8: 'Quarterfinal',
            16: 'Round of 16', 32: 'Round of 32', 64: 'Round of 64'}.get(
        n_teams, f'{n_teams}-team round')


def _simulate_once(teams, rng, prob_cache):
    """
    Play one full knockout from `teams` (bracket order) down to a champion.
    Returns a list whose entries are the participants at each round; the
    final entry is [champion].
    """
    current = list(teams)
    participants = [list(current)]
    while len(current) > 1:
        winners = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            if (a, b) not in prob_cache:
                prob_cache[(a, b)] = advance_probability(a, b)
            winner = a if rng.random() < prob_cache[(a, b)] else b
            winners.append(winner)
        current = winners
        participants.append(list(current))
    return participants


def run(teams, n=10000, seed=42):
    """
    Monte-Carlo the knockout bracket `teams` (in bracket order) `n` times.

    teams[0] plays teams[1], teams[2] plays teams[3], and the winners meet,
    and so on. Returns a DataFrame giving each team's probability of reaching
    each round and of winning, sorted by championship probability.
    """
    if len(teams) < 2 or (len(teams) & (len(teams) - 1)) != 0:
        raise ValueError(f"Number of teams must be a power of 2 (2, 4, 8, 16, ...); got {len(teams)}")

    missing = [t for t in teams if t not in team_stats.index]
    if missing:
        raise ValueError(
            f"These team names are not in the data: {missing}\n"
            f"Check spelling against the dataset (see find_team() below)."
        )

    rng = np.random.default_rng(seed)
    prob_cache = {}
    tally = defaultdict(lambda: defaultdict(int))

    for _ in range(n):
        for round_teams in _simulate_once(teams, rng, prob_cache):
            label = _round_label(len(round_teams))
            for t in round_teams:
                tally[label][t] += 1

    # Column order: from the starting round down to Champion
    labels = [_round_label(len(teams) >> k) for k in range(len(teams).bit_length())]

    rows = []
    for t in teams:
        row = {'team': t}
        for label in labels:
            row[label] = tally[label][t] / n
        rows.append(row)

    out = pd.DataFrame(rows).set_index('team')
    return out.sort_values('Champion', ascending=False)


def find_team(substring):
    """Helper: list dataset team names containing `substring` (case-insensitive).
    Use this if run() complains a team name isn't found."""
    s = substring.lower()
    return [t for t in team_stats.index if s in t.lower()]


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # EDIT THIS LIST as the tournament progresses.
    # List the teams still in the knockout bracket, in bracket order:
    #   remaining[0] vs remaining[1], remaining[2] vs remaining[3], ...
    # The length must be a power of two (8 = quarterfinals, 4 = semis, etc.).
    #
    # If a name isn't found, run find_team('...') to see the exact spelling,
    # e.g.  python -c "from simulation import find_team; print(find_team('kor'))"
    # -----------------------------------------------------------------------
    remaining = [
        'Brazil', 'France',
        'Argentina', 'Spain',
        'Netherlands', 'Portugal',
        'England', 'Germany',
    ]

    results = run(remaining, n=10000)

    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda v: f'{v:6.1%}')
    print("\nSimulated 10,000 tournaments from the current bracket:\n")
    print(results)
    print(f"\nMost likely champion: {results.index[0]} "
          f"({results['Champion'].iloc[0]:.1%})")