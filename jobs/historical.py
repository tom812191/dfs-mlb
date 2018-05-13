import luigi

import pandas as pd
import numpy as np

import config

import jobs.external
import util.copula


class ParseEmpiricalCopula(luigi.Task):
    def requires(self):
        return [
            jobs.external.BattingData(),
            jobs.external.PitchingData(),
        ]

    def output(self):
        return luigi.LocalTarget(config.COPULA_FILE)

    def run(self):
        df_batting = pd.read_pickle(self.input()[0].path)
        df_pitching = pd.read_pickle(self.input()[1].path)

        copula = util.copula.parse_copula(df_batting, df_pitching)

        np.save(self.output().path, copula)
