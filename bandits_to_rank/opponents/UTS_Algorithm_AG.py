#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from typing import List, Any

import numpy as np
from math import sqrt, log
import scipy
from collections import defaultdict

from itertools import product

from bandits_to_rank.sampling.pbm_inference import SVD


### HELP FUNCTION

def est_voisin(l1, l2):
    nb_same = np.sum(l1 != l2)
    return nb_same <= 2


def transposition(l, transposition):
    i, j = transposition
    res = l.copy()
    res[i], res[j] = res[j], res[i]
    return res


def get_maximum_indices(mat, look_at=None):
    if look_at is None:
        look_at = np.array(True, shape=mat.shape) # Todo : Doesn't work, try to make it work
    max_temp = mat[0][0]
    max_temp_indices = (0, 0)
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if look_at[i][j] and mat[i][j] > max_temp:
                max_temp = mat[i][j]
                max_temp_indices = (i, j)
    return max_temp_indices


class UTS_Rank_One:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """

    def __init__(self, nb_arms, nb_place, min_N=10):
        """
        :param nb_arms:
        :param nb_place:
        """
        self.nb_arms = nb_arms
        self.nb_place = nb_place
        self.list_transpositions = [(0, 0)]
        for i in range(self.nb_arms):
            for j in range(i):
                self.list_transpositions.append((j, i))
        self.min_N = min_N
        self.delta = 10
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.leader = np.arange(self.nb_arms)
        np.random.shuffle(self.leader)
        self.clean_leader()

        self.t = 0

    def clean_leader(self):
        """ Clean log data. /To be ran when changing a leader. """
        self.leader_count = 0
        self.N = np.triu(np.zeros([self.nb_arms, self.nb_arms]), 1)
        self.S = np.triu(np.zeros([self.nb_arms, self.nb_arms]), 1)
        self.R = np.triu(np.zeros([self.nb_arms, self.nb_arms]), 1)
        self.mu_hat = np.triu(np.zeros([self.nb_arms, self.nb_arms]), 1)


    def choose_next_arm(self):  #### Attention resampling
        # Change leader if required
        i, j = get_maximum_indices(self.mu_hat, self.N > self.min_N)
        if i != 0 or j != 0:
            print(f'change of leader at iteration {self.t}.')
            print(f'new leader: {transposition(self.leader, (i,j))} by transposition {(i,j)}')
            print(f'stats: leader = {self.mu_hat[0,0]:.2f} ({int(self.N[0,0])}), new leader = {self.mu_hat[i,j]:.2f} ({int(self.N[i,j])})')
            self.leader = transposition(self.leader, (i,j))
            nol, sol, rol, mu_hat_ol = self.N[0][0], self.S[0][0], self.R[0][0], self.mu_hat[0][0]
            nl, sl, rl, mu_hat_l = self.N[i][j], self.S[i][j], self.R[i][j], self.mu_hat[i][j]
            self.clean_leader()
            self.N[0][0], self.S[0][0], self.R[0][0], self.mu_hat[0][0] = nl, sl, rl, mu_hat_l
            self.N[i][j], self.S[i][j], self.R[i][j], self.mu_hat[i][j] = nol, sol, rol, mu_hat_ol
        # choose an arm
        self.leader_count += 1
        if self.leader_count % self.delta == 0:
            transpo = (0,0)
        else:
            max_theta = -np.inf
            transpo = (-1, -1)
            for indices in self.list_transpositions:
                i, j = indices
                a = self.S[i][j] + 1
                b = self.N[i][j] - self.S[i][j] + 1
                theta = np.random.beta(a, b)
                if theta > max_theta:
                    max_theta = theta
                    transpo = indices
        propositions = transposition(self.leader, transpo)
        return propositions, 0

    def update(self, propositions, rewards):
        if not est_voisin(self.leader, propositions):
            raise ValueError(f'manage only rewards obtained while playing in the neighborhoud of the leader.\nleader: {self.leader}, \nplayed: {propositions}')
        try:
            I, J = np.where(self.leader != propositions)[0]
        except ValueError as e:
            I, J = 0, 0
        self.t += 1
        nb_c = np.sum(rewards)
        L = len(propositions)
        p = nb_c / L
        r = np.random.binomial(1, p)
        self.R[I][J] += nb_c
        self.N[I][J] += 1
        self.S[I][J] += r
        self.mu_hat[I][J] = self.R[I][J]/self.N[I][J]



class UTS_fOSUB:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """

    def __init__(self, nb_arms, nb_place):
        """
        :param nb_arms:
        :param nb_place:
        """
        self.nb_arms = nb_arms
        self.nb_place = nb_place
        self.list_transpositions = [(0, 0)]
        for i in range(self.nb_arms):
            for j in range(i):
                self.list_transpositions.append((j, i))
        self.delta = 10
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.leader = np.arange(self.nb_arms)
        np.random.shuffle(self.leader)
        self.leader_count = 0
        self.clean_leader()

        self.t = 0

    def clean_leader(self):
        """ Clean log data. /To be ran when changing a leader. """
        self.stats = defaultdict(lambda: {'N': 0, 'S': 0, 'R': 0, 'mu_hat': 0})

    def choose_next_arm(self):  #### Attention resampling
        # Change leader if required
        neighborhood = [tuple(transposition(self.leader, val)) for val in self.list_transpositions]
        for val in neighborhood:
            if val not in list(self.stats.keys()):
                self.stats[val]
        max_mu_hat = np.inf
        arr = np.arange(self.nb_arms)
        for key in self.stats.keys():
            mu_hat = self.stats[key]['mu_hat']
            if mu_hat > max_mu_hat:
                max_mu_hat = mu_hat
                arr = key
        self.leader = arr
        """
            print(f'change of leader at iteration {self.t}.')
            print(f'new leader: {transposition(self.leader, (i, j))} by transposition {(i, j)}')
            print(
                f'stats: leader = {self.mu_hat[0, 0]:.2f} ({int(self.N[0, 0])}), new leader = {self.mu_hat[i, j]:.2f} ({int(self.N[i, j])})')
        """
        # choose an arm
        self.leader_count += 1
        if self.leader_count % self.delta == 0:
            key = self.leader
        else:
            max_theta = -np.inf
            key = np.arange(self.nb_arms)
            for neighbour in neighborhood:
                a = self.stats[tuple(neighbour)]['S'] + 1
                b = self.stats[tuple(neighbour)]['N'] - self.stats[tuple(neighbour)]['S'] + 1
                theta = np.random.beta(a, b)
                if theta > max_theta:
                    max_theta = theta
                    key = neighbour
        propositions = list(key)
        return propositions, 0

    def update(self, propositions, rewards):
        self.t += 1
        nb_c = np.sum(rewards)
        L = len(propositions)
        p = nb_c / L
        r = np.random.binomial(1, p)
        self.stats[tuple(propositions)]['R'] += nb_c
        self.stats[tuple(propositions)]['N'] += 1
        self.stats[tuple(propositions)]['S'] += r
        self.stats[tuple(propositions)]['mu_hat'] = self.stats[tuple(propositions)]['R'] / self.stats[tuple(propositions)]['N']

class UTS_iOSUB:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """

    def __init__(self, nb_arms, nb_place, max_length):
        """
        :param nb_arms:
        :param nb_place:
        """
        self.nb_arms = nb_arms
        self.nb_place = nb_place
        self.list_transpositions = [(0, 0)]
        for i in range(self.nb_arms):
            for j in range(i):
                self.list_transpositions.append((j, i))
        self.max_length = max_length
        self.delta = 10
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.leader = np.arange(self.nb_arms)
        np.random.shuffle(self.leader)
        self.leader_count = 0
        self.clean_leader()

        self.t = 0

    def clean_leader(self):
        """ Clean log data. /To be ran when changing a leader. """
        self.stats = defaultdict(lambda: {'N': 0, 'S': 0, 'R': 0, 'mu_hat': 0, 't': 0})

    def choose_next_arm(self):  #### Attention resampling
        # Change leader if required
        neighborhood = [tuple(transposition(self.leader, val)) for val in self.list_transpositions]
        for val in neighborhood:
            if val not in list(self.stats.keys()):
                if len(self.stats) < self.max_length:
                    self.stats[val]
                else:
                    min_t = self.max_length
                    for key in self.stats.keys():
                        t = self.stats[key]['t']
                        if t < min_t:
                            min_t = t
                            key_to_be_deleted = key
                    self.stats.pop(key_to_be_deleted)
                    self.stats[val]
        max_mu_hat = np.inf
        arr = np.arange(self.nb_arms)
        for key in self.stats.keys():
            mu_hat = self.stats[key]['mu_hat']
            if mu_hat > max_mu_hat:
                max_mu_hat = mu_hat
                arr = key
        self.leader = arr
        """
            print(f'change of leader at iteration {self.t}.')
            print(f'new leader: {transposition(self.leader, (i, j))} by transposition {(i, j)}')
            print(
                f'stats: leader = {self.mu_hat[0, 0]:.2f} ({int(self.N[0, 0])}), new leader = {self.mu_hat[i, j]:.2f} ({int(self.N[i, j])})')
        """
        # choose an arm
        self.leader_count += 1
        if self.leader_count % self.delta == 0:
            key = self.leader
        else:
            max_theta = -np.inf
            key = np.arange(self.nb_arms)
            for neighbour in neighborhood:
                a = self.stats[tuple(neighbour)]['S'] + 1
                b = self.stats[tuple(neighbour)]['N'] - self.stats[tuple(neighbour)]['S'] + 1
                theta = np.random.beta(a, b)
                if theta > max_theta:
                    max_theta = theta
                    key = neighbour
        propositions = list(key)
        return propositions, 0

    def update(self, propositions, rewards):
        self.t += 1
        nb_c = np.sum(rewards)
        L = len(propositions)
        p = nb_c / L
        r = np.random.binomial(1, p)
        self.stats[tuple(propositions)]['R'] += nb_c
        self.stats[tuple(propositions)]['N'] += 1
        self.stats[tuple(propositions)]['S'] += r
        self.stats[tuple(propositions)]['mu_hat'] = self.stats[tuple(propositions)]['R'] / self.stats[tuple(propositions)]['N']



if __name__ == "__main__":
    import doctest

    doctest.testmod()
