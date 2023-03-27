#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import log, sqrt
from bandits_to_rank.tools.tools_BAL import start_up, newton
from bandits_to_rank.tools.tools import swap
import numpy as np
from random import randint
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class PBB_PBRank:
    """
    """

    def __init__(self, nb_arms, nb_positions, T, gamma, memory_size=np.inf):
        """
        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.list_transpositions = [(0, 0)]
        self.gamma = gamma

        self.certitude = log(T)  # = -log(p) où p est la probabilité que nos estimateurs soient faux.
        # Cette certitude induit automatiquement une précision délimitant l'intervalle de confiance
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """

        # clean the log
        self.precision = 0  # estimateur du min (kappa_theta - kappa_theta') au voisinage du leader_from
        self.running_t = 0  # compteur des observations moyennées pour estimer precision
        self.extended_leader = [i for i in range(self.nb_arms)]; shuffle(self.extended_leader)
        self.list_transpositions = [(0, 0)]
        self.transition_start = 0
        self.kappa_thetas = np.zeros((self.nb_arms, self.nb_arms))
        self.times_kappa_theta = np.zeros((self.nb_arms, self.nb_positions)) \
            # times_kappa_theta[pos, item] = nbre d'observations du couple (kappa_theta) + 1
        self.upper_bound_kappa_theta = np.ones((self.nb_arms, self.nb_arms))
        self.leader_count = defaultdict(self.empty)  # number of time each arm has been the leader

    @staticmethod
    def empty(): # to enable pickling
        return 0

    def choose_next_arm(self):
        proposition = self.extended_leader[:self.nb_positions]
        if self.leader_count[tuple(self.extended_leader[:self.nb_positions])] % self.gamma > 0:
            #print('traspositionused',self.list_transpositions)
            self.transition_start = 1 - self.transition_start
            for (k, l) in self.list_transpositions:
                proposition = np.array(swap(proposition, (k, l), self.extended_leader[self.nb_positions:]))
        return proposition, 0

    def update(self, propositions, rewards):
        # update statistics
        #print(propositions,rewards)
        self.leader_count[tuple(self.extended_leader[:self.nb_positions])] += 1
        for k in range(self.nb_positions):
            item_k = propositions[k]
            kappa_theta, n = self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k]
            kappa_theta, n = kappa_theta + (rewards[k] - kappa_theta) / (n + 1), n + 1
            start = start_up(kappa_theta, self.certitude, n)
            upper_bound = newton(kappa_theta, self.certitude, n, start)
            self.kappa_thetas[item_k, k], self.times_kappa_theta[item_k, k] = kappa_theta, n
            self.upper_bound_kappa_theta[item_k, k] = upper_bound

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
        >>> player = OSUB_PBM(nb_arms=5, nb_positions=3, T=100)
        >>> mu_hats = np.array([[0.625, 0.479, 0.268, 0., 0.],
        ...        [0.352, 0.279, 0.139, 0., 0.],
        ...        [0.585, 0.434, 0.216, 0., 0.],
        ...        [0.868, 0.655, 0.335, 0., 0.],
        ...        [0.292, 0.235, 0.108, 0., 0.]])
        >>> player.kappa_thetas = mu_hats
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
        row_ind, col_ind = linear_sum_assignment(-self.kappa_thetas)
        self.extended_leader = row_ind[np.argsort(col_ind)]

    def update_transition(self):
        pi = np.argsort(-self.kappa_thetas[self.extended_leader[:self.nb_positions], np.arange(self.nb_positions)])
        self.list_transpositions = [(0, 0)]
        pi_extended = list(pi) + ([i for i in range(self.nb_positions,self.nb_arms)])
        possible_rank_transpo = [(i, i+1) for i in range(self.transition_start, self.nb_positions-1, 2)]

        for (i,j) in possible_rank_transpo:
            pos_i, pos_j = pi_extended[i], pi_extended[j]
            item_i,item_j = self.extended_leader[pos_i],self.extended_leader[pos_j]
            value = - self.upper_bound_kappa_theta[item_i, pos_i] - self.upper_bound_kappa_theta[item_j, pos_j] \
                        + self.upper_bound_kappa_theta[item_j, pos_i] + self.upper_bound_kappa_theta[item_i, pos_j]
            if value > 0:
                self.list_transpositions.append((pos_i, pos_j))

        ######## OUT_indice
        if (self.transition_start != self.nb_positions % 2) :
                pos_Last = pi_extended[self.nb_positions-1]
                item_Last= self.extended_leader[pos_Last]
                values =[- self.upper_bound_kappa_theta[item_Last, pos_Last] +
                        self.upper_bound_kappa_theta[self.extended_leader[i], pos_Last]
                        for i in pi_extended[self.nb_positions:]]
                if self.nb_positions != self.nb_arms:
                    max_val =max(values)
                    max_index_out = self.nb_positions+values.index(max_val)
                    #rd_index_out = randint(self.nb_positions - 1, self.nb_arms - 1)
                    if max_val > 0:
                        #print('Add out item by transpo', (pos_Last, max_index_out))
                        self.list_transpositions.append((pos_Last, max_index_out))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
