"""
Reports for after optimizing lineups
"""
import numpy as np
import pandas as pd


def lineup_dk_ids(lineups, players):
    """
    Map the lineup player indexes to their DraftKings IDs

    :param lineups: np.array of lineups, containing indexes to the players DataFrame
    :param players: pd.DataFrame of player information scraped from RotoGrinders

    :return: lineups as a DataFrame of DraftKings IDs
    """
    ids = players['id'].values
    lineups_out = np.array([l['lineup'] for l in lineups])

    ids = ids[lineups_out]

    return pd.DataFrame(ids).fillna(1).astype(np.uint32)
