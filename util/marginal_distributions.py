"""
Get marginal distributions for players
"""
import numpy as np

from scipy.stats import rv_discrete, rv_continuous, expon, norm
from scipy.optimize import root

from functools import partial


def inv_discretize(x, p, epsilon=1e-10):
    """
    The inverse discretization function for our continuous distribution used to estimate batter scores.

    This function takes in a discrete value and outputs the corresponding range in the continuous analogue that would
    map to the discrete value.

    :param x: np.array of length n with discrete point values
    :param p: np.array containing the empirical distribution for all batters, with index corresponding to the discrete
    value of x
    :param epsilon: a small constant needed to compare ratios of buckets with zero probability

    :return: a tuple of (x_lower, x_upper), which are np.arrays with the same shape as x, that give lower and upper
    continuous bounds.
    """
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
    """
    A mixture distribution of an exponential distribution and a gaussian distribution.
    """
    def _cdf(self, x, w, lambda_, mu, sigma):
        """
        The distribution's CDF function.

        :param x: np.array of point values
        :param w: the weight parameter. The exponential is multiplied by w and the gaussian by (1 - w)
        :param lambda_: The exponential distribution's parameter
        :param mu: The gaussian's mean parameter
        :param sigma: The gaussian's standard deviation parameter

        :return: The cdf values as an np.array the same shape as x
        """

        return w * expon.cdf(x, scale=1 / lambda_) \
               + (1 - w) * norm.cdf(x, loc=mu, scale=sigma)


class CustomDisc(rv_discrete):
    def __init__(self, custom_cont, p_prior, upper_support_bound, shapes=None):
        """
        The discrete analogue to a continuous distribution estimating batter scores.

        :param custom_cont: The continuous analogues
        :param p_prior: Batter empirical distribution used for discretization
        :param upper_support_bound: The largest integer value that the discrete distribution can take
        :param shapes: shape parameters as a string for scipy.stats.rv_discrete
        """
        rv_discrete.__init__(self, b=upper_support_bound, shapes=shapes)
        self.custom_cont = custom_cont
        self.p_prior = p_prior

    def _pmf(self, x, *args):
        """
        The probability map function for our discrete distribution

        :param x: np.array of point values
        :param args: parameters for the continuous analogue

        :return: np.array of probabilities the same shape as x
        """
        x_lower, x_upper = inv_discretize(x, self.p_prior)

        return self.custom_cont.cdf(x_upper, *args) - self.custom_cont.cdf(x_lower, *args)


class BatterMarginal:
    def __init__(self, mean, std, principal_components, principal_component_mean, empirical_distribution, theta_init):
        """
        A class to represent the marginal distribution of fantasy points for a single batter.

        :param mean: The projected mean from which we fit the distribution
        :param std: The projected standard deviation from which we fit the distribution
        :param principal_components: Eigen vectors from performing PCA on historically fitted distribution parameters
        :param principal_component_mean: The corresponding mean vector for PCA
        :param empirical_distribution: The historical empirical distribution of all hitters
        :param theta_init: An initial guess of the param vector for our solver
        """
        self.rv = CustomDisc(MixedExponGaussian(), empirical_distribution, 64, shapes='w, lambda_, mu, sigma')
        self.mean = mean
        self.std = std
        self.principal_components = principal_components
        self.principal_component_mean = principal_component_mean
        self.theta_init = theta_init

        self.theta = self._fit_params()
        self.ppf_map = self._fit_ppf()

    def _fit_params(self):
        """
        Fit the parameters of the mixed distribution.

        This works by constraining the mixture distribution's mean and standard deviation to the projected mean and
        standard deviation, and then further constraining the parameters to lie on the 2d plane in the 4d parameter
        space defined by the first two principal components of the historically fit parameters.

        If no solution exists, relax the latter constraint and resolve.

        :return: the fitted params
        """

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
        """
        Create a map from discrete value to corresponding CDF value. This is much faster than calling the ppf function
        directly.
        :return: np.array mapping discrete point values to their corresponding probability. The index corresponds to the
        point value.
        """
        return self.rv.cdf(np.arange(65), *self.theta)

    def ppf(self, q):
        """
        Percent point function, i.e. the inverse cdf.
        :param float q: np.array of quantiles

        :return: np.array of value corresponding to the quantile.
        """
        return np.searchsorted(self.ppf_map, q)


class PitcherMarginal:
    def __init__(self, mean, std):
        """
        A class to represent the marginal distribution of fantasy points for a single batter.

        Pitcher scores are modeled as a simple Gaussian random variable.

        :param mean: The mean of the Gaussian distribution
        :param std: The standard deviation of the Gaussian distribution
        """
        self.mean = mean
        self.std = std

    def ppf(self, q):
        """
        The percent point function of our random variable, i.e. the inverse cdf
        :param q: np.array of quantiles
        :return: score corresponding to the quantiles
        """
        return norm.ppf(q, loc=self.mean, scale=self.std)


def parse_marginal_distributions(players, principal_components, principal_component_mean,
                                 empirical_distribution, theta_init, verbose=0):
    """
    Create marginal distributions for each player

    :param players: A pandas.DataFrame with columns (name, team_grp, position, order, salary, proj_mean, proj_std)
    :param principal_components: Eigen vectors from performing PCA on historically fitted distribution parameters
    :param principal_component_mean: The corresponding mean vector for PCA
    :param empirical_distribution: The historical empirical distribution of all hitters
    :param theta_init: An initial guess of the param vector for our solver
    :param verbose: Defines the level of printing information
    :return: The players DataFrame with the column marginal_distribution added
    """

    def parse_player_dist(row):
        """
        Create a marginal distribution for a single player

        :param row: A row in the dataframe that corresponds to a player
        :return: The marginal distribution class
        """
        if verbose > 0:
            print('Fitting marginal: {}'.format(row['name']))

        if row['order'] == 10:
            return PitcherMarginal(row['proj_mean'], row['proj_std'])

        return BatterMarginal(row['proj_mean'], row['proj_std'], principal_components, principal_component_mean,
                              empirical_distribution, theta_init)

    players['marginal_distribution'] = players.apply(parse_player_dist, axis=1)
    return players
