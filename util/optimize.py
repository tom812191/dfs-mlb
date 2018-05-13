"""
Utility to optimize the set of lineups we will enter into a DraftKings contest.
"""
import numpy as np

import pathos.multiprocessing as mp
from functools import partial


def optimize_lineup_set(players, sim_results, config, target, num_lineups, verbose=0):
    """
    Approximate the best set of lineups such that the probability of one of the lineups hitting the target is maximized.

    We will take a greedy approach, first finding the single lineup that is most likely to hit the target, and then find
    the lineup that is most likely to hit the target where the first lineup doesn't, and so on.

    :param players: The players DataFrame giving positions, teams, salaries, etc.
    :param sim_results: 2d np.array with the first dimension indexing the player (corresponding to the index in players
    DataFrame) and the second dimension indexing the simulation number. The data is the result of the sim.
    :param config: The global configuration info
    :param target: The estimated score target needed to win the contest
    :param num_lineups: The number of lineups in the lineup set
    :param verbose: The level of logging information to print

    :return: np.array of lineups, the first dimension giving the lineup, the second giving the position, and the data
    giving the corresponding player index in the players DataFrame
    """
    lineups = []
    sr = sim_results.copy()

    for i in range(num_lineups):
        if verbose > 0:
            print('Building Lineup: {} of {}'.format(i + 1, num_lineups))

        optimize_lineup_bound = partial(optimize_lineup, players, sr, config, target, verbose=verbose)

        with mp.Pool(config.BASIN_HOPPING_NUM_WORKERS) as pool:
            results = pool.map(optimize_lineup_bound, range(config.BASIN_HOPPING_NUM_WORKERS))

        objective_values = [l[1] for l in results]
        best_idx = np.argmax(objective_values)
        lineup = results[best_idx][0]

        lineups.append({
            'lineup': lineup,
            'objective_val': objective_values[best_idx],
        })

        # Remove simulations where the lineup is successful
        lineup_sim_results = sr[lineup, :].sum(axis=0)
        is_success = lineup_sim_results >= target
        sr = sr[:, ~is_success]

    return lineups


def optimize_lineup(players, sim_results, config, target, worker_idx, verbose=0):
    """
    Optimize a single lineup using the BasinHopping algorithm

    :param players: The players DataFrame giving positions, teams, salaries, etc.
    :param sim_results: 2d np.array with the first dimension indexing the player (corresponding to the index in players
    DataFrame) and the second dimension indexing the simulation number. The data is the result of the sim.
    :param config: The global configuration info
    :param target: The estimated score target needed to win the contest
    :param worker_idx: A dummy parameter passed from the multiprocessing module
    :param verbose: The level of logging information to print

    :return: The best lineup as a tuple of (lineup, objective_value)
    """
    optimizer = BasinHopping(players, sim_results, config.LINEUP_CONFIG, target,
                             step_size=config.BASIN_HOPPING_STEP_SIZE,
                             temperature=config.BASIN_HOPPING_TEMPERATURE,
                             niter=config.BASIN_HOPPING_NITER,
                             niter_success=config.BASIN_HOPPING_NITER_SUCCESS,
                             verbose=verbose) \
        .initialize_state(config.BASIN_HOPPING_NUM_PARALLEL) \
        .run()

    return optimizer.best_lineup


class LineupOptimizer:
    def __init__(self, players, sim_results, lineup_config, target, initial_state=None, verbose=0):
        """
        A base class for lineup optimization

        :param players: The players DataFrame giving positions, teams, salaries, etc.
        :param sim_results: 2d np.array with the first dimension indexing the player (corresponding to the index in players
        DataFrame) and the second dimension indexing the simulation number. The data is the result of the sim.
        :param lineup_config: The global configuration info
        :param target: The estimated score target needed to win the contest
        :param initial_state: An optional initial state to begin optimization
        :param verbose: The level of logging information to print
        """
        self.players = players
        self.sim_results = sim_results
        self.lineup_config = lineup_config
        self.target = target
        self.verbose = verbose

        self.salary_penalty = 0.5

        self.encoded_players, self.encoded_player_indexes, self.position_player_indexes = self.encode_players()
        self.position_minimums = np.array([p[1] for p in lineup_config['positions']])

        self.state = initial_state
        self.current_objective = self.objective(initial_state) if initial_state is not None else None

        self._num_steps = None

    @property
    def num_steps(self):
        """
        The possible number of steps that can be taken from a single state

        :return: The number of steps
        """
        if self._num_steps is None:
            num_steps = []
            for idx, pos in enumerate(self.lineup_config['positions']):
                pos_name, pos_count = pos
                num_steps.append(pos_count * len(self.position_player_indexes[pos_name]))
            self._num_steps = np.array(num_steps)

        return self._num_steps

    @property
    def best_lineup(self):
        """
        The best lineup and its corresponding objective value

        :return: The best lineup and its objective value
        """
        best_idx = np.argmax(self.current_objective)
        return self.state[best_idx], self.current_objective[best_idx]

    def encode_players(self):
        """
        Encode players into a numpy array. The rows correspond to a player and
        the columns give:
            player salary
            is pos 1
            ...
            is pos 7
            is team 1
            ...
            is team n
            is game 1
            ...
            is game n/2

        :return: A tuple of (encoded, encoded_indexes, position_player_indexes)
        """

        df = self.players.copy()

        positions = [p[0] for p in self.lineup_config['positions']]
        teams = df['team'].unique().tolist()

        df['game_id'] = df.apply(lambda row: ''.join(sorted([row['team'], row['opp']])), axis=1)
        game_ids = df['game_id'].unique().tolist()

        encoded = np.zeros((len(df), 1 + len(positions) + len(teams) + len(game_ids)), np.uint16)
        encoded_indexes = {
            'salary': [0, 1],
            'position': [-1, -1],
            'team': [-1, -1],
            'game': [-1, -1]
        }
        encoded[:, 0] = df['salary'].values.astype('int')
        current_idx = 1
        encoded_indexes['position'][0] = current_idx
        for pos in positions:
            encoded[:, current_idx] = df['position'].str.contains(pos).values.astype('int')
            current_idx += 1
        encoded_indexes['position'][1] = current_idx
        encoded_indexes['team'][0] = current_idx
        for team in teams:
            encoded[:, current_idx] = (df['team'] == team).values.astype('int')
            current_idx += 1
        encoded_indexes['team'][1] = current_idx
        encoded_indexes['game'][0] = current_idx
        for game_id in game_ids:
            encoded[:, current_idx] = (df['game_id'] == game_id).values.astype('int')
            current_idx += 1
        encoded_indexes['game'][1] = current_idx

        position_player_indexes = {}
        player_indexes = np.arange(encoded.shape[0])
        idx = encoded_indexes['position'][0]
        for pos, _ in self.lineup_config['positions']:
            position_player_indexes[pos] = player_indexes[encoded[:, idx].flatten().astype('bool')]
            idx += 1
            
        return encoded, encoded_indexes, position_player_indexes

    def objective(self, lineups, hard_cap=True):
        """
        The objective function for optimization. It calculates the probability of hitting the target score

        :param lineups: The lineups to evaluate
        :param bool hard_cap: If False, then you can go over the salary cap at a penalty

        :return: np.array of objective values
        """
        lineup_sim_results = self.sim_results[lineups, :].sum(axis=1)
        pct_success = np.mean(lineup_sim_results >= self.target, axis=1)

        # Salary penalty
        if not hard_cap:
            lineups_players = self.encoded_players[lineups, :]
            total_salary = lineups_players[:, :, self.encoded_player_indexes['salary'][0]].sum(axis=1)
            pct_success[total_salary > self.lineup_config['salary_cap']] *= self.salary_penalty

        return pct_success

    def is_legal(self, lineups, hard_cap=True):
        """
        Check if the lineups are legal according to DraftKings rules

        :param lineups: The lineups to check
        :param bool hard_cap: If False, then you can go over the salary cap at a penalty

        :return: np.array of bool values indicating if each lineup is legal
        """
        # Dim 1 is lineup, dim 2 is player, dim 3 is encoding
        lineups_players = self.encoded_players[lineups, :]
        is_legal = np.ones((len(lineups,)))

        # Check salary constraint
        if hard_cap:

            total_salary = lineups_players[:, :, self.encoded_player_indexes['salary'][0]].sum(axis=1)
            is_legal = np.logical_and(is_legal, total_salary < self.lineup_config['salary_cap'])

        # Check min games constraint
        num_games = lineups_players[:, :, slice(*self.encoded_player_indexes['game'])].any(axis=1).sum(axis=1)
        is_legal = np.logical_and(is_legal, num_games >= self.lineup_config['min_games'])

        # Check max hitters single team constraint
        hitters_per_team = lineups_players[:, :, slice(*self.encoded_player_indexes['team'])].sum(axis=1)
        is_legal = np.logical_and(is_legal,
                                  (hitters_per_team <= self.lineup_config['max_hitters_one_team']).all(axis=1))

        # Check positions
        players_per_position = lineups_players[:, :, slice(*self.encoded_player_indexes['position'])].sum(axis=1)
        is_legal = np.logical_and(is_legal,
                                  (players_per_position >= self.position_minimums).all(axis=1))

        # Check total players
        is_legal = is_legal & (lineups.shape[1] == self.position_minimums.sum())

        # Check no repeat players
        is_legal = is_legal & np.array([len(l) == len(np.unique(l)) for l in lineups])

        return is_legal

    def initialize_state(self, num_lineups, hard_cap=True):
        """
        Randomly initialize the state of our lineups

        :param int num_lineups: Number of lineups to initialize
        :param bool hard_cap: If False, then you can go over the salary cap at a penalty

        :return: self
        """
        if self.state is not None:
            return

        # Randomly initialize num_lineups
        lineups = np.zeros((num_lineups, self.position_minimums.sum()), np.uint16)

        is_legal = np.zeros((num_lineups,), np.bool)

        while not np.all(is_legal):
            idx = 0
            for pos, pos_count in self.lineup_config['positions']:
                player_indexes = self.position_player_indexes[pos]
                lineups[~is_legal, idx:idx + pos_count] = player_indexes[
                    np.random.randint(len(player_indexes), size=(np.sum(~is_legal), pos_count))
                ]
                idx += pos_count
            is_legal = self.is_legal(lineups, hard_cap=hard_cap)

        self.state = lineups
        self.current_objective = self.objective(lineups)

        return self

    def legal_steps(self, lineups, hard_cap=True):
        """
        Get all legals steps from the current state.

        :param lineups: The lineups from which we calculate the legal steps
        :param bool hard_cap: If False, then you can go over the salary cap at a penalty

        :return: a list of legals steps for each lineup
        """
        # preallocate step lineups dim 1 is the lineup number, dim 2 is the step number, dim 3 is the player
        step_lineups = np.repeat(
            lineups.reshape((lineups.shape[0], 1, lineups.shape[1])), self.num_steps.sum(), axis=1)

        # calculate all steps (including illegal)
        step_idx = 0
        pos_idx = 0
        for pos, pos_count in self.lineup_config['positions']:
            players = np.tile(self.position_player_indexes[pos], pos_count)
            step_lineups[:, step_idx:step_idx + len(players), pos_idx] = players
            pos_idx += pos_count
            step_idx += len(players)

        # filter to legal only
        is_legal = self.is_legal(step_lineups.reshape((-1, self.position_minimums.sum())), hard_cap=hard_cap)\
            .reshape((lineups.shape[0], -1))

        return [step_lineups[lineup_idx, is_legal[lineup_idx], :] for lineup_idx in range(lineups.shape[0])]

    def run(self):
        """
        Abstract method to run the optimizer
        """
        raise NotImplementedError('run not implemented')


class HillClimbing(LineupOptimizer):
    """
    Implements basic hill climbing, taking the steepest step at each iteration
    """

    def hill_climb(self, lineups):
        """
        Run the hill climbing algorithm

        :param lineups: The lineups to locally optimize

        :return: The optimized lineups
        """
        done = np.zeros((lineups.shape[0]), np.bool)
        iterations = 0
        current_objective = self.objective(lineups, hard_cap=True)
        while not np.all(done):
            if self.verbose > 3:
                print('HillClimbing Iteration: {}'.format(iterations))
            # Get steps
            steps_list = self.legal_steps(lineups[~done], hard_cap=True)
            step_idx = 0
            for lineup_idx in range(lineups.shape[0]):
                if done[lineup_idx]:
                    continue

                steps = steps_list[step_idx]
                step_idx += 1

                # Evaluate steps
                obj_vals = self.objective(steps, hard_cap=True)
                best_step = np.argmax(obj_vals)

                if current_objective[lineup_idx] < obj_vals[best_step]:
                    lineups[lineup_idx] = steps[best_step]
                    current_objective[lineup_idx] = obj_vals[best_step]
                else:
                    done[lineup_idx] = True
            iterations += 1

        return lineups

    def run(self):
        """
        Run the hill climbing algorithm

        :return: self
        """
        if self.state is None:
            raise AssertionError('must first call initialize_state')

        self.state = self.hill_climb(self.state.copy())
        self.current_objective = self.objective(self.state)

        return self


class BasinHopping(HillClimbing):
    def __init__(self, *args, step_size=2, temperature=0.001, niter=250, niter_success=50, **kwargs):
        """
        An implementation of the Basin Hopping stochastic optimization algorithm

        :param args: arguments to be passed to the base class
        :param step_size: The number of positions to mutate in the Basin Hopping algorithm
        :param temperature: The temperature to determine the probability of accepting a worse solution in Basin Hopping
        :param niter: The number of iterations for Basin Hopping
        :param niter_success: The number of iterations without a global improvement to run before stopping short
        :param kwargs: arguments to be passed to the base class
        """
        super().__init__(*args, **kwargs)

        self.step_size = step_size
        self.temperature = temperature
        self.niter = niter
        self.niter_success = niter_success

        self._idx_pos_map = None

    @property
    def idx_pos_map(self):
        """
        Maps indexes in a lineup to their corresponding position

        :return: The index position map
        """
        if self._idx_pos_map is None:
            self._idx_pos_map = []
            for pos, count in self.lineup_config['positions']:
                for c in range(count):
                    self._idx_pos_map.append(pos)
        return self._idx_pos_map

    def accept_test(self, old_objective, new_objective):
        """
        Accept any improvement. Accept a worse value with probability e^((new - old) / temperature)

        :param old_objective: np.array of old objective values
        :param new_objective: np.array of new objective values

        :return: np.array of boolean values if we should accept the step
        """
        improvement = new_objective > old_objective
        accept_worse = np.exp((new_objective - old_objective) / self.temperature) >= \
                       np.random.rand(old_objective.shape[0])

        return np.logical_or(improvement, accept_worse)

    def mutate(self, lineups, hard_cap=True):
        """
        Randomly mutate each lineup

        :param lineups: the lineups to randomly mutate
        :param hard_cap: Can we go over the salary cap at a penalty

        :return: The mutated lineups
        """
        # Choose positions to mutate
        replace_positions = []
        for lineup_idx, _ in enumerate(lineups):
            replace_positions.append(np.random.choice(lineups.shape[1], size=self.step_size, replace=False))

        done = np.zeros(lineups.shape[0], dtype=np.bool)
        mutated = lineups.copy()
        while not np.all(done):
            for lineup_idx, _ in enumerate(lineups):
                if done[lineup_idx]:
                    continue

                for step_idx in range(self.step_size):
                    position_idx = replace_positions[lineup_idx][step_idx]
                    mutated[lineup_idx, position_idx] = np.random.choice(
                        self.position_player_indexes[self.idx_pos_map[position_idx]])

            done = self.is_legal(mutated, hard_cap=hard_cap)

        return mutated

    def run(self):
        """
        Run the BasinHopping algorithm

        :return: self
        """
        if self.state is None:
            raise AssertionError('must first call initialize_state')

        best_state = self.state.copy()
        best_objective = self.current_objective.copy()

        current_state = self.state.copy()
        current_objective = self.current_objective.copy()

        niter_success_count = 0
        for i in range(self.niter):
            if niter_success_count >= self.niter_success:
                break

            # Randomly mutate the state
            mutated_state = self.mutate(current_state)

            # Run local optimization
            mutated_state = self.hill_climb(mutated_state)
            mutated_objective = self.objective(mutated_state)

            # Store best objective vals
            is_improvement = mutated_objective > best_objective
            best_state[is_improvement] = mutated_state[is_improvement]
            best_objective[is_improvement] = mutated_objective[is_improvement]
            if not np.any(is_improvement):
                niter_success_count += 1
            else:
                niter_success_count = 0

            # Run accept test
            should_accept = self.accept_test(current_objective, mutated_objective)
            current_state[should_accept] = mutated_state[should_accept]
            current_objective[should_accept] = mutated_objective[should_accept]

            if self.verbose > 1:
                print('Basin Hopping Iteration: {}'.format(i))

            if self.verbose > 2:
                print('Best Objectives: {}'.format(best_objective))

        self.state = best_state
        self.current_objective = best_objective

        return self
