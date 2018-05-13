"""
Utility to simulate a slate of games
"""
import numpy as np


class PandasWrapper:
    def __init__(self, data):
        """
        Wrapper class to store arbitrary data in a pandas DataFrame cell

        :param data: The cell data
        """
        self._data = data

    @property
    def data(self):
        return self._data


def simulate_slate(players, copula, n=10000, sim_result_column_name='sim_result', verbose=0):
    """
    Simulate the entire slate of games

    :param players: A pandas.DataFrame with columns (name, team_grp, position, order, salary, marginal_distribution)
    :param copula: A numpy.array of the empirical copula. Columns correspond to batting order with opposing pitcher
    in the final slot
    :param n: Number of simulations to run. If n is None, use every row in the empirical copula
    :param sim_result_column_name: The DataFrame column name to store the simulation results
    :param verbose: The level of logging information to print

    :return: The same players dataframe with the added column sim_result
    """

    players_out = players.copy()
    players_out[sim_result_column_name] = None
    for team, df in players.groupby('team_grp'):
        if verbose > 0:
            print('Simulating group {}'.format(team))
        players_out.update(simulate_group(df, copula, n,
                                          sim_result_column_name=sim_result_column_name, verbose=verbose))

    return players_out


def simulate_group(players, copula, n, sim_result_column_name='sim_result', verbose=0):
    """
    Simulate a group of batters and opposing pitcher
    :param players: A pandas.DataFrame with columns (name, team_grp, position, order, marginal_distribution)
    :param copula: A numpy.array of the empirical copula. Columns correspond to batting order with opposing pitcher
    in the final slot
    :param n: Number of simulations to run. If n is None, use every row in the empirical copula
    :param sim_result_column_name: The DataFrame column name to store the simulation results
    :param verbose: The level of logging information to print

    :return: A pandas.Series of sim_results retaining the index from players
    """
    if n is not None:
        # Randomly sample n rows from the copula
        copula = copula[np.random.randint(copula.shape[0], size=n), :]

    # Simulate each player
    return players.apply(lambda row: simulate_player(row, copula, verbose=verbose), axis=1)\
        .rename(sim_result_column_name)


def simulate_player(player, copula, verbose=0):
    """
    Simulate a single player's performance by sampling from the copula

    :param player: The player to simulate as a pandas Series (a row from the players DataFrame)
    :param copula: The empirical copula as an np.array, where the first dimension gives the game and the second
    dimension gives the position (where the 10th position is the opposing pitcher)
    :param verbose: The level of logging information to print

    :return: The wrapped point data
    """
    if verbose > 0:
        print('Simulating player: {}'.format(player['name']))

    q = copula[:, player['order'] - 1].flatten()
    return PandasWrapper(player['marginal_distribution'].ppf(q))

