"""
Get marginal distributions for players
"""
import numpy as np

from scipy.stats import rv_discrete, rv_continuous, expon, norm
from scipy.optimize import root

from functools import partial


def inv_discretize(x, p, epsilon=1e-10):
    # x is a 1d array of length n of discrete values
    # p is empirical distribution from all batters, with index corresponding to the discrete value of x
    # return 2 arrays of lower and upper continuous x bounds that map to the discrete x
    p = (p + epsilon) / (p + epsilon).sum()
    p_x_lower = np.concatenate([np.array([np.inf]), p[1:] / (p[1:] + p[:-1])])
    p_x_upper = np.concatenate([p[:-1] / (p[:-1] + p[1:]), np.array([np.inf])])

    x_lower = x - p_x_lower[x.astype('int')]
    x_upper = x + p_x_upper[x.astype('int')]

    return x_lower, x_upper


class MixedExponGaussian(rv_continuous):
    def _cdf(self, x, w, lambda_, mu, sigma):
        # Two normal distributions
        # Theta is the parameter vector in the order w, loc1, scale1, loc2, scale2
        # x is the numpy array of observations

        return w * expon.cdf(x, scale=1 / lambda_) \
               + (1 - w) * norm.cdf(x, loc=mu, scale=sigma)


class CustomDisc(rv_discrete):
    def __init__(self, custom_cont, p_prior, upper_support_bound, shapes=None):
        rv_discrete.__init__(self, b=upper_support_bound, shapes=shapes)
        self.custom_cont = custom_cont
        self.p_prior = p_prior

    def _pmf(self, x, *args):
        x_lower, x_upper = inv_discretize(x, self.p_prior)

        return self.custom_cont.cdf(x_upper, *args) - self.custom_cont.cdf(x_lower, *args)


class BatterMarginal:
    def __init__(self, mean, std, principal_components, principal_component_mean, empirical_distribution, theta_init):
        self.rv = CustomDisc(MixedExponGaussian(), empirical_distribution, 64, shapes='w, lambda_, mu, sigma')
        self.mean = mean
        self.std = std
        self.principal_components = principal_components
        self.principal_component_mean = principal_component_mean
        self.theta_init = theta_init

        self.theta = self._fit_params()
        self.ppf_map = self._fit_ppf()

    def _fit_params(self):

        # Function to find zeros of
        def f_sys(proj, pca_values, rv, theta, relaxation=0):
            # proj is a 2d tuple of (mean, std)
            # pcs_values is a tuple of (components_, mean_)
            # rv is the random variable subclassed from rv_discrete
            proj_mean, proj_std = proj
            pca_components, pca_mean = pca_values

            pc3, pc4 = pca_components[:, 2].flatten(), pca_components[:, 3].flatten()

            return np.array([
                0 if relaxation > 1 else np.dot(pc3, theta) - np.dot(pc3, pca_mean),  # First PCA hyper plane
                0 if relaxation > 0 else np.dot(pc4, theta) - np.dot(pc4, pca_mean),  # Second PCA hyper plane
                rv.mean(*theta) - proj_mean,  # Distribution mean = projected mean
                0 if relaxation > 2 else rv.std(*theta) - proj_std,  # Distributuion std = projected std
            ])

        success = False
        relaxation = 0
        while not success and relaxation <= 3:
            f_bound = partial(f_sys, (self.mean, self.std), (self.principal_components, self.principal_component_mean),
                              self.rv, relaxation=relaxation)
            res = root(f_bound, self.theta_init)
            success = res.success
            relaxation += 1

        return res.x

    def _fit_ppf(self):
        return self.rv.cdf(np.arange(65), *self.theta)

    def ppf(self, q):
        return np.searchsorted(self.ppf_map, q)


class PitcherMarginal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def ppf(self, q):
        return norm.ppf(q, loc=self.mean, scale=self.std)


def parse_marginal_distributions(players, principal_components, principal_component_mean,
                                 empirical_distribution, theta_init, verbose=0):
    """
    :param players: A pandas.DataFrame with columns (name, team_grp, position, order, salary, proj_mean, proj_std)
    :return: The players DataFrame with the column marginal_distribution added
    """

    def parse_player_dist(row):
        if verbose > 0:
            print('Fitting marginal: {}'.format(row['name']))

        if row['order'] == 10:
            return PitcherMarginal(row['proj_mean'], row['proj_std'])

        return BatterMarginal(row['proj_mean'], row['proj_std'], principal_components, principal_component_mean,
                              empirical_distribution, theta_init)

    players['marginal_distribution'] = players.apply(parse_player_dist, axis=1)
    return players
