#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
import math
from bandits_to_rank.tools.tools_BAL import start_up, newton
from bandits_to_rank.tools.tools import swap_full
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class GRAB:
    """
    """

    def __init__(self, nb_arms, nb_positions, T=None, gamma=None, forced_initiation=False, gap_type='reward', optimism='KL'):
        """
        Parameters
        ----------
        nb_arms
        nb_positions
        T : int or None
            Either the horizon of a game, or None (unknown horizon). Used to set the confidence interval size with
            log(T) (known horizon) or log(t) + 3 log(log(t)) with t the number of time current arm has been the leader.
        gamma
            periodicity at which GRAB is forced to play the leader (default to nb_arms-1)
        forced_initiation
            if True, the nb_arms first iterations consist in recommending a permutation in order to try each item
            at each position at least once
        """
        if gamma is None or gamma < 1:
            gamma = nb_arms - 1
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.gamma = gamma
        self.forced_initiation = forced_initiation
        if T is None or T < 1:
            self.threshold = None
        else:
            self.threshold = math.log(T)

        try:
            self.get_gap = {'reward': self.optimistic_gap_reward,
                            'first': self.optimistic_gap_first,
                            'reward and first': self.optimistic_gap_rewad_and_first,
                            'second': self.optimistic_gap_second,
                            'both': self.optimistic_gap_both,
                             #'first_and_both': self.optimistic_gap_first_and_both,
                            }[gap_type]
        except KeyError:
            raise ValueError(f'unknown criterion for exploration: {gap_type}')

        self.optimistic_index, self.pessimistic_index  = {
            "KL": (self.optimistic_index_KL, self.pessimistic_index_KL),
            "UCB": (self.optimistic_index_UCB, self.pessimistic_index_UCB),
            "TS": (self.index_TS, self.index_TS)
            }[optimism]

        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.running_t = 0      # number of played iterations
        self.extended_leader = np.arange(self.nb_arms); shuffle(self.extended_leader)
        self.list_transpositions = []

        # rhos_hat[item, pos] = average feedback gathered for item 'item' at position 'pos'
        self.rhos_hat = np.zeros((self.nb_arms, self.nb_arms))
        # nb_plays[item, pos] = number of feedbacks gathered for item 'item' at position 'pos'
        self.nb_plays = np.zeros((self.nb_arms, self.nb_positions))
        if self.threshold is not None:
            # upper_bounds_rho[item, pos] = precomputed optimism value for the probability of click on item 'item' at position 'pos'
            self.upper_bounds_rho = np.ones((self.nb_arms, self.nb_positions))
        self.leader_count = defaultdict(self.empty)     # number of time each arm has been the leader

    @staticmethod
    def empty():    # to enable pickling
        return 0

    def choose_next_arm(self):
        (i, j) = (0, 0)
        nb_leader = self.leader_count[tuple(self.extended_leader[:self.nb_positions])]
        if self.forced_initiation and (self.running_t < self.nb_arms):
            proposition = np.array([(self.running_t+i) % self.nb_arms for i in range(self.nb_positions)])
            return proposition, 0

        elif nb_leader % self.gamma > 0:
            # preliminary
            leader_opt_index = np.zeros(self.nb_arms)
            for l in range(self.nb_positions):
                leader_opt_index[l] = self.optimistic_index(self.extended_leader[l], l, nb_leader)
            # argmax
            delta_upper_bound_max = 0
            for (k, l) in self.list_transpositions:
                item_k, item_l = self.extended_leader[k], self.extended_leader[l]
                value = self.get_gap(leader_opt_index, k, l, item_k, item_l, nb_leader)
                if (value > delta_upper_bound_max):
                    (i, j) = (k, l)
                    delta_upper_bound_max = value
        proposition = np.array(swap_full(self.extended_leader, (i, j), self.nb_positions))
        return proposition, 0

    def optimistic_gap_reward(self, leader_opt_index, k, l, item_k, item_l, nb_leader):
        return - leader_opt_index[k] - leader_opt_index[l] \
                + self.optimistic_index(item_l, k, nb_leader) + self.optimistic_index(item_k, l, nb_leader)

    def optimistic_gap_first(self, leader_opt_index, k, l, item_k, item_l, nb_leader):
        return min(- leader_opt_index[k] - leader_opt_index[l]
                   + self.optimistic_index(item_l, k, nb_leader) + self.optimistic_index(item_k, l, nb_leader),     # reward
                   max(self.optimistic_index(item_k, l, nb_leader) - leader_opt_index[k],   # compare both positions
                       self.optimistic_index(item_l, k, nb_leader) - leader_opt_index[k])   # compare both items
                   )

    def optimistic_gap_rewad_and_first(self, leader_opt_index, k, l, item_k, item_l, nb_leader):
        return max(self.optimistic_index(item_k, l, nb_leader) - leader_opt_index[k],   # compare both positions
                   self.optimistic_index(item_l, k, nb_leader) - leader_opt_index[k])   # compare both items

    def optimistic_gap_second(self, leader_opt_index, k, l, item_k, item_l, nb_leader):
        pessi_ll = self.pessimistic_index(item_l, l, nb_leader)
        return max(pessi_ll - self.pessimistic_index(item_l, k, nb_leader),   # compare both positions
                   pessi_ll - self.pessimistic_index(item_k, l, nb_leader))   # compare both items

    def optimistic_gap_first_and_second(self, leader_opt_index, k, l, item_k, item_l, nb_leader):
        pessi_ll = self.pessimistic_index(item_l, l, nb_leader)
        return max(min(self.optimistic_index(item_k, l, nb_leader) - leader_opt_index[k],
                       pessi_ll - self.pessimistic_index(item_l, k, nb_leader)),   # compare both positions
                   min(self.optimistic_index(item_l, k, nb_leader) - leader_opt_index[k],
                       pessi_ll - self.pessimistic_index(item_k, l, nb_leader))    # compare both items
                   )

    def optimistic_gap_both(self, leader_opt_index, k, l, item_k, item_l, nb_leader):
        pessi_ll = self.pessimistic_index(item_l, l, nb_leader)
        return max(self.optimistic_index(item_k, l, nb_leader) + pessi_ll - leader_opt_index[k] - self.pessimistic_index(item_l, k, nb_leader),   # compare both positions
                   self.optimistic_index(item_l, k, nb_leader) + pessi_ll - leader_opt_index[k] - self.pessimistic_index(item_k, l, nb_leader))   # compare both items

    def optimistic_index_KL(self, i, k, nb_total_trial):
        if k >= self.nb_positions:
            return 0
        if self.threshold is None:
            if self.nb_plays[i, k] == 0 or nb_total_trial < 3:
                return 1
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
            start = start_up(self.rhos_hat[i, k], threshold, self.nb_plays[i, k])
            return newton(self.rhos_hat[i, k], threshold, self.nb_plays[i, k], start)
        else:
            return self.upper_bounds_rho[i, k]

    def pessimistic_index_KL(self, i, k, nb_total_trial):
        if k >= self.nb_positions:
            return 0
        if self.threshold is None:
            if self.nb_plays[i, k] == 0 or nb_total_trial < 3:
                return 1
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
            start = start_up(1.-self.rhos_hat[i, k], threshold, self.nb_plays[i, k])
            return 1.-newton(1.-self.rhos_hat[i, k], threshold, self.nb_plays[i, k], start)
        else:
            return self.upper_bounds_rho[i, k]

    def optimistic_index_UCB(self, i, k, nb_total_trial):
        if k >= self.nb_positions:
            return 0
        if not self.nb_plays[i, k] or nb_total_trial < 3:
            return 1
        else:
            return self.rhos_hat[i, k] + (2 * np.log(nb_total_trial) / self.nb_plays[i, k]) ** .5

    def pessimistic_index_UCB(self, i, k, nb_total_trial):
        if k >= self.nb_positions:
            return 0
        if not self.nb_plays[i, k] or nb_total_trial < 3:
            return 1
        else:
            return self.rhos_hat[i, k] - (2 * np.log(nb_total_trial) / self.nb_plays[i, k]) ** .5

    def index_TS(self, i, k, nb_total_trial):
        if k >= self.nb_positions:
            return 0
        if not self.nb_plays[i, k]:
            return 1
        else:
            alpha = self.rhos_hat[i, k] * self.nb_plays[i, k]
            beta = self.nb_plays[i, k] - alpha
            return np.random.beta(alpha + 1, beta + 1)

    def update(self, propositions, rewards):
        self.running_t += 1
        # update statistics
        self.leader_count[tuple(self.extended_leader[:self.nb_positions])] += 1
        for k in range(self.nb_positions):
            item_k = propositions[k]
            kappa_theta, n = self.rhos_hat[item_k, k], self.nb_plays[item_k, k]
            kappa_theta, n = kappa_theta + (rewards[k] - kappa_theta) / (n + 1), n + 1
            self.rhos_hat[item_k, k], self.nb_plays[item_k, k] = kappa_theta, n
            if self.threshold:
                start = start_up(kappa_theta, self.threshold, n)
                upper_bound = newton(kappa_theta, self.threshold, n, start)
                self.upper_bounds_rho[item_k, k] = upper_bound

        # update the leader L(n) (in the neighborhood of previous leader)
        self.update_leader()
        self.update_transition()

    def update_leader(self):
        """

        Returns
        -------

        Examples
        -------
        >>> import numpy as np
        >>> player = GRAB(nb_arms=5, nb_positions=3, T=100)
        >>> mu_hats = np.array([[0.625, 0.479, 0.268, 0., 0.],
        ...        [0.352, 0.279, 0.139, 0., 0.],
        ...        [0.585, 0.434, 0.216, 0., 0.],
        ...        [0.868, 0.655, 0.335, 0., 0.],
        ...        [0.292, 0.235, 0.108, 0., 0.]])
        >>> player.rhos_hat = mu_hats
        >>> mu_hats[[2, 3, 1], np.arange(3)].sum()
        1.379
        >>> mu_hats[[3, 0, 2], np.arange(3)].sum()
        1.563
        >>> mu_hats[[3, 2, 0], np.arange(3)].sum()
        1.57

        >>> player.update_leader()
        >>> player.extended_leader
        array([3, 2, 0, 1, 4])

        """
        row_ind, col_ind = linear_sum_assignment(-self.rhos_hat)
        self.extended_leader = row_ind[np.argsort(col_ind)]

    def update_transition(self):
        pi = np.argsort(-self.rhos_hat[self.extended_leader[:self.nb_positions], np.arange(self.nb_positions)])
        self.list_transpositions = []
        for k in range(self.nb_positions - 1):
            self.list_transpositions.append((pi[k], pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            self.list_transpositions.append((pi[self.nb_positions - 1], k))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
