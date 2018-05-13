"""
Parse an empirical copula from historical data
"""
import pandas as pd
import numpy as np


def parse_copula(df_batting, df_pitching, config):
    """
    Parse the empirical copula from historical data.

    That is, calculate ranks for player performance in each game, grouped by hitters with the opposing pitcher.

    (These ranks are then sampled later in the process and fed into the inverse CDF to simulate a player's performance.)

    :param df_batting: pd.DataFrame of historical batting statistics at the player/game level
    :param df_pitching: pd.DataFrame of historical pitching statistics at the player/game level
    :param config: global configuration

    :return: np.array of shape nx10 of the empirical copula
    """
    # Calculate draftkings points that were scored by each player
    scoring = config.SCORING['batting']
    df_batting['s'] = df_batting['h'] - (df_batting['d'] + df_batting['t'] + df_batting['hr'])
    df_batting['pts'] = scoring['s'] * df_batting['s'] + \
                        scoring['d'] * df_batting['d'] + \
                        scoring['t'] * df_batting['t'] + \
                        scoring['hr'] * df_batting['hr'] + \
                        scoring['bi'] * df_batting['bi'] + \
                        scoring['bb'] * df_batting['bb'] + \
                        scoring['hp'] * df_batting['hp'] + \
                        scoring['sb'] * df_batting['sb']

    scoring = config.SCORING['pitching']
    df_pitching['ip'] = df_pitching['outs'] / 3
    df_pitching['nono'] = (df_pitching['h'] == 0) & (df_pitching['cg'] > 0)
    df_pitching['pts'] = scoring['ip'] * df_pitching['ip'] + \
                         scoring['so'] * df_pitching['so'] + \
                         scoring['wp'] * df_pitching['wp'] + \
                         scoring['er'] * df_pitching['er'] + \
                         scoring['h'] * df_pitching['h'] + \
                         scoring['bb'] * df_pitching['bb'] + \
                         scoring['hb'] * df_pitching['hb'] + \
                         scoring['cg'] * df_pitching['cg'] + \
                         scoring['sho'] * df_pitching['sho'] + \
                         scoring['nono'] * df_pitching['nono']

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