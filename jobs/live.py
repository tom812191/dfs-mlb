import luigi
import os
import datetime
import numpy as np
import pandas as pd
import json

import config

import jobs.external
import jobs.historical

import util.scrape
import util.marginal_distributions
import util.simulate
import util.optimize
import util.report


class ScrapeData(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    verbose = luigi.IntParameter(default=0)

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DAILY_DATA_ROOT,
                                              '{}_projections.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        df = util.scrape.scrape_all_data(config.BATTING_ROTO_URL, config.PITCHING_ROTO_URL)
        df.to_csv(self.output().path)


class SimulateSlate(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    verbose = luigi.IntParameter(default=0)
    num_simulations = luigi.IntParameter(default=10000)

    def requires(self):
        return [
            jobs.external.EmpiricalBattingDistribution(),
            jobs.external.BattingDistributionParamPCA(),
            ScrapeData(date=self.date),
            jobs.historical.ParseEmpiricalCopula(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DAILY_DATA_ROOT,
                                              '{}_sim_results.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        # players, principal_components, principal_component_mean, empirical_distribution, theta_init
        theta_init = config.THETA_INIT
        empirical_distribution = np.load(self.input()[0].path)

        with self.input()[1].open('r') as f:
            pcs = json.load(f)

        principal_components = np.array(pcs['principal_components'])
        principal_component_mean = np.array(pcs['principal_component_mean'])

        players = pd.read_csv(self.input()[2].path)
        copula = np.load(self.input()[3].path)

        players = util.marginal_distributions.parse_marginal_distributions(players, principal_components,
                                                                           principal_component_mean,
                                                                           empirical_distribution, theta_init,
                                                                           verbose=self.verbose)

        results = util.simulate.simulate_slate(players, copula, n=self.num_simulations,
                                               sim_result_column_name='sim_result', verbose=self.verbose)

        # extract sim results as a numpy array
        sim_results = results['sim_result'].tolist()
        sim_results = np.concatenate([r.data.reshape((1, -1)) for r in sim_results], axis=0)

        np.save(self.output().path, sim_results)


class OptimizeLineupSet(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    verbose = luigi.IntParameter(default=0)
    target = luigi.IntParameter(default=config.TARGET_SCORE)
    num_lineups = luigi.IntParameter(default=config.NUM_LINEUPS)

    def requires(self):
        return [
            ScrapeData(date=self.date, verbose=self.verbose),
            SimulateSlate(date=self.date, verbose=self.verbose),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DAILY_DATA_ROOT,
                                              '{}_lineups.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        players = pd.read_csv(self.input()[0].path)
        sim_results = np.load(self.input()[1].path)

        lineups = util.optimize.optimize_lineup_set(players, sim_results, config, self.target, self.num_lineups,
                                                    verbose=self.verbose)

        lineups = np.array(lineups)

        np.save(self.output().path, lineups)


class Report(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    verbose = luigi.IntParameter(default=0)

    def requires(self):
        return [
            ScrapeData(date=self.date, verbose=self.verbose),
            OptimizeLineupSet(date=self.date, verbose=self.verbose),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DAILY_DATA_ROOT,
                                              '{}_lineup_ids.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        players = pd.read_csv(self.input()[0].path)
        lineups = np.load(self.input()[1].path)

        lineup_ids = util.report.lineup_dk_ids(lineups, players)

        lineup_ids.to_csv(self.output().path)
