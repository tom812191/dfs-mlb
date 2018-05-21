This repo implements a strategy to play MLB contests on DraftKings.

External Dependencies:
* Pandas, NumPy, and SciPy are used for data manipulation, simulation, statistical modeling, etc.
* Luigi is used for workload pipelines
* Pathos is used for parallel processing


# The Strategy
Our strategy is fairly simple in principle. We construct a set of lineups such that the probability of any single lineup hitting a target winning score is maximized. That is, we count the round as a success if one lineup hits the target, even if all other lineups lose. We take this strategy because of the top-heavy payout structure, where a single first place lineup can compensate for hundreds of losing lineups.

Intuitively, we achieve this by constructing each individual lineup such that the probability distribution of points has a heavy upper tail (as opposed to a more narrow distribution that may have a higher score on average, but has a lower probability of hitting a big score). We then construct the set of lineups from the individual lineups with diversification in mind, hoping that if one lineup fails to hit big, then another one will succeed.

While this strategy is fairly simple, the challenges come from
* Modeling the dependence between different players
* Estimating individual player's fantasy point distributions
* Simulating a set of games
* Optimizing the set of lineups, as this combinatoric optimization problem is NP-hard.

## Modeling Player Dependence

### How Players Relate
Batters earn fantasy points for walks, hits, runs, RBIs, and stolen bases, while pitchers earn fantasy points for outs and strike-outs and lose fantasy points for allowing hits, walks, and runs.

So, it is immediately obvious that a batter and the opposing team's pitcher will not have independent scores, but rather will have negatively correlated scores (a hit for a batter is directly negative points for the pitcher, and an out for the pitcher implies no hit or walk for the batter, etc.).

It also stands to reason that batters on the same team will correlate positively with each other. For example, if the lead-off batter gets on base, then the next batter hits him home, they will both receive fantasy points for the run scored and the RBI respectively. Furthermore, they will both face the same pitcher, so they will on average both suffer against a great pitcher and both thrive against a poor pitcher.

Batters from opposite teams and pitchers from opposite teams could potentially correlate with each other. For example, strategy could change based on the score of the game. However, we find an effect so small that it can be safely ignored. Also, we find that players from different games do not correlate as expected.

We also find that all batters on the same team correlate with each other to some extent, with the effect being the strongest when players are close in the order.

To summarize, we are left with groups of 10 players (the 9 batters of the same team, and the pitcher they are facing) where we must model dependence within the group. We can safely assume independence between groups.

### How to Model the Dependence
As we'll find later on, individual batter distributions are complicated and non-Gaussian, so trying to fit a joint distribution for the 10 players of a group would be a nightmare.

Instead we find [Copula Theory](https://en.wikipedia.org/wiki/Copula_(probability_theory)) to be a perfect fit here. Using a copula allows us to specify the dependence structure and the marginal distributions of the players separately. Furthermore, the copula allows us to capture the [tail dependence](https://en.wikipedia.org/wiki/Tail_dependence) between players, which is vital since we're more interested in the extreme values that our distributions can exhibit rather than the typical behavior at the center of the distributions.

Instead of fitting a 10 dimensional parametric copula, we will simply build an empirical copula from our vast amount of historical data.

This process is much simpler than is sounds. We rank the performance of each batting order slot and opposing pitcher for each historical game, then we can run simulations by sampling games directly. For example, we sample a game where the lead-off hitter performed in the 51st percentile, and the second hitter in the 39th, etc. Then these two player's simulated values would be the inverse CDF of their marginal distributions at these percentiles.

## Modeling Individual Players
We'll need marginal distributions for every player which can be used with the copulas to fully simulate the set of games.

We delegate the task of projecting mean and standard deviation to [RotoGrinders](https://rotogrinders.com/). Since they are able to produce quality projections, there is not need to reinvent the wheel here.

### Pitchers
Pitcher's exhibit close to a Gaussian distribution. Technically, the actual distribution is not continuous, but it is close enough that using a Gaussian model is appropriate. Using the method of moments, we can directly use the mean and standard deviation from the RotoGrinders projections as parameters for the Guassians.

### Batters
Batter distributions are unfortunately far more complicated. We have two problems:
1. Distributions are discrete
2. Distributions do not follow the shape of any common parametric distributions

Furthermore, we will have to obtain a distribution for each player on a live basis using only the mean and a standard deviation provided from RotoGrinders.

We solve this with the following process:
1. Fit a continuous mixture distribution to each batter (with at least 100 games) historically. We use a mixture of an exponential distribution and a Gaussian distribution. This model will have 4 parameters: $\vec\theta=(w, \lambda, \mu, \sigma)$, which are the weight between the exponential and Gaussian distributions, the rate parameter for the exponential, the mean of the Gaussian, and the standard deviation of the Gaussian respectively.
2. Discretize the continuous distribution. We choose to map continuous ranges to discrete values proportional to how often the values occur in the empirical distribution for all batters.
3. Perform a principal component analysis on the parameter space $(w, \lambda, \mu, \sigma)$ using the data from the fitted distributions in step 1, and solve for the 2d plane spanned by the first two principal components. Now, we can use the projected mean and standard deviation along with the 2d plane to solve for all 4 parameters of our distribution.

The full detailed process can be viewed in [this notebook](https://github.com/tom812191/dfs-mlb/blob/master/notebooks/mlb-dfs.ipynb).

## Simulating a Set of Games
We can perform a Monte Carlo simulation for each player by sampling the copula and then using the player's marginal distribution. Concretely, we sample from the empirical copula which gives us a 10-dimensional vector of quantile values for the 10 players of a group. We then use this quantile value for each player as the argument of their inverse cumulative distribution function (CDF), which will yield the player's score. Repeat this $n$ times for each group and store the results.

## Optimizing the Set of Lineups
The final step is to optimize the set of lineups that will serve as our contest entries. We will use an iterative greedy approach to add lineups to the set, and will use Basin-Hopping to optimize individual lineups.

### Individual Lineups
For individual lineups, we maximize the probability that the lineup scores greater than or equal to a target score, subject to position and salary constraints. The objective function probability is estimated as the percentage of simulations where the lineup hits the target.

Individual lineups are optimized using the [Basin-Hopping algorithm](https://arxiv.org/pdf/cond-mat/9803344.pdf) $m$ times in parallel, then choosing the best result. The basic steps of Basin-Hopping are
1. Randomly assign the starting state
2. Perform a local optimization with a basic hill-climbing algorithm
3. Save the best result so far
4. Always accept the new local maxima as the new state if it's an improvement, and accept the new local maxima if it's worse with some probability, determined by the temperature parameter (similar to simulated annealing)
5. Mutate the current state
6. Repeat steps 2 through 5 until an ending condition is met (max number of iterations or max number of iterations without an improvement)

The key is to mutate the state enough that you can escape local maxima and enter other basins, but not so much that you're simply creating near-random lineups. We mutate by randomly swapping 3 players.

This algorithm works well for our problem as it's designed for "funnel-like but rugged" landscapes. Optimizing a lineup is "funnel-like" as you usually want many hitters from one team, so the landscape would have funnels as you add more players from the same team. The landscape is then "rugged" as a specific combination of these players might be better than others, and it may be best to have one fewer player of the team and have high value players. That is, swapping a couple players could knock you off the local maximum but still pull you down the funnel. We run many optimizations in parallel due to the multiple funnel nature of the search space.

### Sets of Lineups
Once an individual lineup is selected, we remove the simulation iterations where the individual lineup succeeded. Then, we select another lineup with the remaining simulation iterations and continue this process until we have the desired number of lineups in the set. This iterative greedy approach works well in practice and dramatically simplifies the whole optimization problem. (The search space for optimizing all individual lineups at once would simply be too large.)