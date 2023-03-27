#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
import numpy as np
from collections import defaultdict
from bandits_to_rank.tools.tools import swap, unused, bound_KL_brentq
from bandits_to_rank.tools.tools_BAL import start_up, newton

from math import log

class OSUB:
    """
    !!! Warning !!! assumes nb_arm == nb_positions
    """

    def __init__(self, nb_arms, nb_positions, memory_size):
        """
        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.list_transpositions = [(0, 0)]
        for i in range(self.nb_positions):
            for j in range(i+1, self.nb_arms):
                self.list_transpositions.append((i, j))
        self.memory_size = memory_size
        self.delta = len(self.list_transpositions)
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.leader = np.arange(self.nb_arms)
        np.random.shuffle(self.leader)
        self.leader = tuple(self.leader[:self.nb_positions])

        self.stats = defaultdict(self.empty)
        self.t = 0

    @staticmethod
    def empty(): # to enable pickling
        return {'N': 0, 'S': 0, 'R': 0, 'mu_hat': 0, 't': -np.inf, 'leader_count': 0}


    def choose_next_arm(self):
        # Change leader if required
        max_mu_hat = -np.inf
        for key in self.stats.keys():
            mu_hat = self.stats[key]['mu_hat']
            if mu_hat > max_mu_hat:
                max_mu_hat = mu_hat
                self.leader = key
        self.stats[self.leader]['leader_count'] += 1
        # choose an arm
        if self.stats[self.leader]['leader_count'] % self.delta == 1:
            propositions = self.leader
        else:
            max_theta = -np.inf
            remaining = unused(self.leader, self.nb_arms)
            neighborhood = [swap(self.leader, trans, remaining) for trans in self.list_transpositions]
            for neighbour in neighborhood:
                #print('neighbour',neighbour)
                #print('l_count',self.stats[neighbour]['leader_count'])
                n = self.stats[neighbour]['N'] + 1
                mu = (self.stats[neighbour]['S'] + 1) / n
                if self.stats[neighbour]['leader_count'] <= 1:
                    delta = log(1)

                else:
                    delta = log(self.stats[neighbour]['leader_count']-1)


                #print('mu,', mu, 'delta', delta, 'n', n)
                start = start_up(mu, delta, n)
                theta = newton(mu, delta, n,start)
                #theta = bound_KL_brentq(mu, delta, n)
                if theta > max_theta:
                    max_theta = theta
                    propositions = neighbour
        return np.array(propositions), 0

    def update(self, propositions, rewards):
        # update statistics
        self.t += 1
        nb_c = np.sum(rewards)
        p = nb_c / self.nb_positions
        r = np.random.binomial(1, p)
        self.stats[tuple(propositions)]['R'] += nb_c
        self.stats[tuple(propositions)]['N'] += 1
        self.stats[tuple(propositions)]['S'] += r
        self.stats[tuple(propositions)]['mu_hat'] = self.stats[tuple(propositions)]['R'] / self.stats[tuple(propositions)]['N']
        self.stats[tuple(propositions)]['t'] = self.t

        # keep memory small enough
        if len(self.stats) == self.memory_size:
            min_t = np.inf
            for key, val in self.stats.items():
                t = val['t']
                if t < min_t:
                    min_t = t
                    key_to_be_deleted = key
            self.stats.pop(key_to_be_deleted)



if __name__ == "__main__":
    import doctest

    doctest.testmod()
