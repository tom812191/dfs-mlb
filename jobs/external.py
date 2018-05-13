import luigi

import config


class BattingData(luigi.task.ExternalTask):

    def output(self):
        return luigi.LocalTarget(config.BATTING_FILE)


class PitchingData(luigi.task.ExternalTask):

    def output(self):
        return luigi.LocalTarget(config.PITCHING_FILE)


class EmpiricalBattingDistribution(luigi.task.ExternalTask):

    def output(self):
        return luigi.LocalTarget(config.EMPIRICAL_BATTING_DIST_FILE)


class BattingDistributionParamPCA(luigi.task.ExternalTask):

    def output(self):
        return luigi.LocalTarget(config.BATTING_DIST_PARAM_PCA)

