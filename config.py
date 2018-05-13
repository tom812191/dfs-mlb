import os
import numpy as np

DATA_ROOT = '/Users/tom/Projects/Portfolio/data/mlb'
DAILY_DATA_ROOT = os.path.join(DATA_ROOT, 'daily')

# External files
BATTING_FILE = os.path.join(DATA_ROOT, 'batting-raw.p')
PITCHING_FILE = os.path.join(DATA_ROOT, 'pitching-raw.p')

COPULA_FILE = os.path.join(DATA_ROOT, 'copula.npy')
EMPIRICAL_BATTING_DIST_FILE = os.path.join(DATA_ROOT, 'empirical_batting_dist.npy')

BATTING_DIST_PARAM_PCA = os.path.join(DATA_ROOT, 'batting_dist_pca.json')


# URLs
BATTING_ROTO_URL = 'https://rotogrinders.com/projected-stats/mlb-hitter?site=draftkings'
PITCHING_ROTO_URL = 'https://rotogrinders.com/projected-stats/mlb-pitcher?site=draftkings'

# Optimizers
BASIN_HOPPING_NUM_WORKERS = 2
BASIN_HOPPING_NUM_PARALLEL = 1
BASIN_HOPPING_TEMPERATURE = 0.001
BASIN_HOPPING_NITER = 50
BASIN_HOPPING_NITER_SUCCESS = 25
BASIN_HOPPING_STEP_SIZE = 3

# Lineup Settings
NUM_LINEUPS = 10
TARGET_SCORE = 160

# Misc
THETA_INIT = np.array([0.5, 0.25, 15.0, 6.0])

LINEUP_CONFIG = {
    'positions': [
        ('SP', 2),
        ('C', 1),
        ('1B', 1),
        ('2B', 1),
        ('3B', 1),
        ('SS', 1),
        ('OF', 3),
    ],
    'salary_cap': 50000,
    'min_games': 2,
    'max_hitters_one_team': 5,
}

# DraftKings
SCORING = {
    'batting': {
        's': 3,
        'd': 5,
        't': 8,
        'hr': 10,
        'bi': 2,
        'bb': 2,
        'hp': 2,
        'sb': 5,
    },
    'pitching': {
        'ip': 2.25,
        'so': 2,
        'wp': 4,
        'er': -2,
        'h': -0.6,
        'bb': -0.6,
        'hb': -0.6,
        'cg': 2.5,
        'sho': 2.5,
        'nono': 5,
    },
}
