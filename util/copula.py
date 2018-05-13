"""
Parse an empirical copula from historical data
"""
import pandas as pd
import numpy as np


def parse_copula(df_batting, df_pitching):
    # Calculate draftkings points that were scored by each player
    df_batting['s'] = df_batting['h'] - (df_batting['d'] + df_batting['t'] + df_batting['hr'])
    df_batting['pts'] = 3 * df_batting['s'] + \
                        5 * df_batting['d'] + \
                        8 * df_batting['t'] + \
                        10 * df_batting['hr'] + \
                        2 * df_batting['bi'] + \
                        2 * df_batting['bb'] + \
                        2 * df_batting['hp'] + \
                        5 * df_batting['sb']

    df_pitching['ip'] = df_pitching['outs'] / 3
    df_pitching['nono'] = (df_pitching['h'] == 0) & (df_pitching['cg'] > 0)
    df_pitching['pts'] = 2.25 * df_pitching['ip'] + \
                         2 * df_pitching['so'] + \
                         4 * df_pitching['wp'] + \
                         -2 * df_pitching['er'] + \
                         -0.6 * df_pitching['h'] + \
                         -0.6 * df_pitching['bb'] + \
                         -0.6 * df_pitching['hb'] + \
                         2.5 * df_pitching['cg'] + \
                         2.5 * df_pitching['sho'] + \
                         5 * df_pitching['nono']

    # Look at how different starting batting order slots correlate with each other and with the opposing pitcher
    df_batting['is_home'] = df_batting['team'] == df_batting['home']
    df_pitching['is_home'] = df_pitching['team'] == df_pitching['home']

    df_b = df_batting[df_batting['seq'] == 1][['game_id', 'is_home', 'slot', 'pts']]  # seq == 1 indicates the starter

    df_p = df_pitching[df_pitching['gs'] == 1][['game_id', 'is_home', 'pts']]

    # We want to join the batting lineup against the other team's pitcher, so flip is_home
    df_p['is_home'] = ~df_p['is_home']
    df_p['slot'] = 10

    # Merge the two data sets
    df = pd.concat([df_b, df_p], ignore_index=True)
    df = df.set_index(['game_id', 'is_home', 'slot'])
    df = df.unstack(level='slot')

    ranks = df.applymap(lambda x: x + np.random.rand() / 1000).rank(pct=True)

    return ranks.values