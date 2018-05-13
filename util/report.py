import numpy as np
import pandas as pd


def lineup_dk_ids(lineups, players):

    ids = players['id'].values
    lineups_out = np.array([l['lineup'] for l in lineups])

    ids = ids[lineups_out]

    return pd.DataFrame(ids).fillna(1).astype(np.uint32)
