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

    def __init__(self, nb_arms, nb_positions, T=None, gamma=None, forced_initiation=False):
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


        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.running_t = 0      # number of played iterations

        # --- leader ---
        self.extended_leader = np.arange(self.nb_arms); shuffle(self.extended_leader)
        self.leader_count = defaultdict(self.empty)     # number of time each arm has been the leader
        self.list_transpositions = []

        # --- recommendations ---
        self.chosen_transposition = None
        self.next_recommendation = None

        # --- P(click on j at position k) ---
        self.proba_click_ik = np.zeros((self.nb_arms, self.nb_arms))
        self.nb_play_ik = np.zeros((self.nb_arms, self.nb_positions))

        # --- E(c_i - c_j | c_i != c_j) ---
        self.nb_play_diff_ij = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.average_diff_ij = np.zeros([self.nb_arms, self.nb_arms])
        self.nb_diff_ij = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        if self.threshold is not None:
            self.upper_bound_expected_diff_ij = np.ones((self.nb_arms, self.nb_arms))



    @staticmethod
    def empty():    # to enable pickling
        return 0

    def choose_next_arm(self):
        ## --- first recommendations
        if self.forced_initiation and (self.running_t < self.nb_arms):
                recommendation = np.array([(self.running_t+i) % self.nb_arms for i in range(self.nb_positions)])
                self.next_recommendation = None
                return recommendation, 0
        ## --- apply already precomputed recommendation
        if self.next_recommendation is not None:
            recommendation = self.next_recommendation
            self.next_recommendation = None
            return recommendation, 0
        ## --- choose a recommendation in the neighborhood of the leader
        nb_leader = self.leader_count[tuple(self.extended_leader[:self.nb_positions])]
        delta_upper_bound_max = 0.5
        if nb_leader % self.gamma > 0:
            for (k1, k) in self.list_transpositions:
                item_k1, item_k = self.extended_leader[k1], self.extended_leader[k]
                value = self.optimistic_index(item_k, item_k1, nb_leader)
                if (value > delta_upper_bound_max):
                    (k_best, l_best) = (k1, k)
                    delta_upper_bound_max = value
        if delta_upper_bound_max > 0.5:
            recommendation = np.array(swap_full(self.extended_leader, (k_best, l_best), self.nb_positions))
            self.chosen_transposition = (k_best, l_best)
            self.next_recommendation = np.copy(self.extended_leader[:self.nb_positions])
            return recommendation, 0
        else:
            self.chosen_transposition = None
            self.next_recommendation = None
            return np.copy(self.extended_leader[:self.nb_positions]), 0

    def optimistic_index(self, i, j, nb_total_trial):
        if self.threshold is None:
            n = self.nb_diff_ij[i][j]
            if n == 0 or nb_total_trial < 3:
                return 1
            s = 0.5 + 0.5 * self.average_diff_ij[i][j]
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
            start = start_up(s, threshold, n)
            return newton(s, threshold, n, start)
        else:
            return self.upper_bound_expected_diff_ij[i, j]

    def update(self, propositions, rewards):
        self.running_t += 1
        self.leader_count[tuple(self.extended_leader[:self.nb_positions])] += 1

        self.update_probability_of_click(propositions, rewards)
        if self.chosen_transposition is not None:
            self.update_expected_difference(propositions, rewards)
        self.update_leader()
        self.update_transition()

    def update_probability_of_click(self, propositions, rewards):
        for k in range(self.nb_positions):
            item_k = propositions[k]
            p_click, n = self.proba_click_ik[item_k, k], self.nb_play_ik[item_k, k]
            p_click, n = p_click + (rewards[k] - p_click) / (n + 1), n + 1
            self.proba_click_ik[item_k, k], self.nb_play_ik[item_k, k] = p_click, n

    def update_expected_difference(self, propositions, rewards):
        k1, k = self.chosen_transposition
        r1 = rewards[k1]
        r = rewards[k] if k < self.nb_positions else 0
        i = self.extended_leader[k1]
        j = self.extended_leader[k]
        if propositions[k1] == i:
            c_i = r1
            c_j = r
        else:
            c_i = r
            c_j = r1
        self.nb_play_diff_ij[i][j] += 1
        self.nb_play_diff_ij[j][i] = self.nb_play_diff_ij[i][j]
        if c_i != c_j:
            self.nb_diff_ij[i][j] += 1
            self.nb_diff_ij[j][i] = self.nb_diff_ij[i][j]
            self.average_diff_ij[i][j] += (c_i - c_j - self.average_diff_ij[i][j]) / self.nb_diff_ij[i][j]
            self.average_diff_ij[j][i] = - self.average_diff_ij[i][j]
            if self.threshold:
                n = self.nb_diff_ij[i][j]
                s = 0.5 + 0.5 * self.average_diff_ij[i][j]
                start = start_up(s, self.threshold, n)
                upper_bound = newton(s, self.threshold, n, start)
                self.upper_bound_expected_diff_ij[i, j] = upper_bound
                s = 0.5 + 0.5 * self.average_diff_ij[j][i]
                start = start_up(s, self.threshold, n)
                upper_bound = newton(s, self.threshold, n, start)
                self.upper_bound_expected_diff_ij[j, i] = upper_bound

    def update_leader(self):
        """

        Returns
        -------

        Examples
        -------
        >>> import numpy as np
        >>> player = GRAB(nb_arms=5, nb_positions=3, T=100)
        >>> mu_hats = np.array([[0.625, 0.268, 0.479, 0., 0.],
        ...        [0.352, 0.139, 0.279, 0., 0.],
        ...        [0.585, 0.216, 0.434, 0., 0.],
        ...        [0.868, 0.335, 0.655, 0., 0.],
        ...        [0.292, 0.108, 0.235, 0., 0.]])
        >>> player.proba_click_ik = mu_hats
        >>> mu_hats[[2, 3, 1], np.arange(3)].sum()
        1.1989999999999998
        >>> mu_hats[[3, 0, 2], np.arange(3)].sum()
        1.57
        >>> mu_hats[[3, 2, 0], np.arange(3)].sum()
        1.5630000000000002

        >>> player.update_leader()
        >>> player.extended_leader
        array([3, 0, 2, 1, 4])
        >>> player.extended_leader  # Version "iterative-greedy"
        array([3, 2, 0, 1, 4])

        """
        if False:
            row_ind, col_ind = linear_sum_assignment(-self.proba_click_ik)
            self.extended_leader = row_ind[np.argsort(col_ind)]
        else:
            arms = np.arange(self.nb_arms)
            positions = np.arange(self.nb_positions)
            for n_remaining in range(self.nb_positions):
                i, k = np.unravel_index(np.argmax(self.proba_click_ik[arms[n_remaining:], :][:, positions[n_remaining:]], axis=None)
                                        , (self.nb_arms - n_remaining, self.nb_positions - n_remaining))
                arms[n_remaining], arms[i + n_remaining] = arms[i + n_remaining], arms[n_remaining]
                positions[n_remaining], positions[k + n_remaining] = positions[k + n_remaining], positions[n_remaining]
            self.extended_leader = arms[np.concatenate((np.argsort(positions), np.arange(self.nb_positions, self.nb_arms)), axis=0)]

    def update_transition(self):
        pi = np.argsort(-self.proba_click_ik[self.extended_leader[:self.nb_positions], np.arange(self.nb_positions)])
        self.list_transpositions = []
        for k in range(self.nb_positions - 1):
            self.list_transpositions.append((pi[k], pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            self.list_transpositions.append((pi[self.nb_positions - 1], k))

    def print_info(self):
        print('leader', self.extended_leader)
        print('P(c_ik)')
        print(self.proba_click_ik)
        print(self.nb_play_ik)
        print('P(c_ik - c_jk | c_ik neq c_jk)')
        print(self.nb_play_diff_ij)
        print(self.nb_diff_ij)
        print(self.average_diff_ij)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
