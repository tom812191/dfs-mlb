"""
External data sources needed for daily and historical jobs
"""
import luigi

import config


class BattingData(luigi.task.ExternalTask):
    """
    Historical batting statistics parsed from the Retrosheet database.

    In CSV format at the player/game level.
    """
    def output(self):
        return luigi.LocalTarget(config.BATTING_FILE)


class PitchingData(luigi.task.ExternalTask):
    """
    Historical pitching statistics parsed from the Retrosheet database.

    In CSV format at the player/game level.
    """
    def output(self):
        return luigi.LocalTarget(config.PITCHING_FILE)


class EmpiricalBattingDistribution(luigi.task.ExternalTask):
    """
    An empirical distribution for the number of fantasy points scored by batters.

    In numpy array format, aggregated from all batters at the player/game level. The array's indexes correspond to the
    number of points scored. E.g. array[0] = 0.2 would indicate the batters scored 0 fantasy points in 20% of games.
    """
    def output(self):
        return luigi.LocalTarget(config.EMPIRICAL_BATTING_DIST_FILE)


class BattingDistributionParamPCA(luigi.task.ExternalTask):
    """
    The principal component analysis results for fitting a mixture distribution to all batters individually. That is,
    we fit a mixture distribution to each batter using MLE and recorded the parameters w, lambda, mu, sigma, and then
    performed PCA on the w, lambda, mu, sigma matrix for all players.

    In JSON format.

    See notebooks/mlb-dfs.ipynb for more details.
    """
    def output(self):
        return luigi.LocalTarget(config.BATTING_DIST_PARAM_PCA)

