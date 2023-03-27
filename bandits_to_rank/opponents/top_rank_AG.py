#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from typing import List, Any

import numpy as np
from math import sqrt, log
import scipy
from scipy.special import erf

from itertools import product

from bandits_to_rank.sampling.pbm_inference import SVD


def partition(L, G):
    L_remain = set()
    Ps = []
    while L != set():
        for val, better in G:
            if better in L_remain or better in L:
                L_remain.add(val)
                L.discard(val)
        Ps.append(list(L))
        L = L_remain
        L_remain = set()
    return Ps


class TOP_RANK:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """

    def __init__(self, nb_arms, T, L, K, nb_positions=None):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param T: number of trials
        """
        self.L = L
        self.nb_arms = nb_arms
        self.nb_trials = T  # n in TopRank
        self.delta = 1 / self.nb_trials
        self.S = np.zeros([self.nb_arms, self.nb_arms])
        self.N = np.zeros([self.nb_arms, self.nb_arms])
        self.P = [list(L)]
        self.c = (4 * np.sqrt(2 / np.pi) / erf(np.sqrt(2)))
        self.G = []
        self.K = K
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        self.S = np.zeros([self.nb_arms, self.nb_arms])
        self.N = np.zeros([self.nb_arms, self.nb_arms])
        self.G = []
        self.P = [list(self.L)]
        pass

    def choose_next_arm(self):  #### Attention resampling
        propositions = []
        K = self.K
        nb_group = len(self.P)
        P = self.P
        while K != 0:
            for i in range(nb_group):
                group = P[i]
                l = len(group)
                if K >= l:
                    K = K - l
                    np.random.shuffle(group)
                    for val in group:
                        propositions.append(val)
                else:
                    l = K
                    group = group[:K]
                    K = 0
                    np.random.shuffle(group)
                    for val in group:
                        propositions.append(val)
        propositions = np.array(propositions)
        return propositions, 0

    def update(self, propositions, rewards):
        # S, N
        # G
        # P
        # C vecteur qui contient des 0 et des 1 pour si un produit a été cliqué ou step (on va avoir besoin de rewards et propositions)
        C = np.zeros(self.nb_arms)
        C[propositions] = rewards
        C_final = []
        k = 0
        for group in self.P:
            l = len(group)
            c = C[k:k + l]
            C_final.append(c)
            k += l
        for i in range(len(C_final)):
            group = C_final[i]
            for j in range(len(group)):
                for k in range(len(group)):
                    val = group[j] - group[k]
                    self.S[j][k] += val
                    self.N[j][k] += np.abs(val)
        for i in range(self.nb_arms):
            for j in range(self.nb_arms):
                if self.N[i][j] and self.S[i][j] >= np.sqrt(2 * self.N[i][j] * np.log((self.delta / self.c) * np.sqrt(self.N[i][j]))):
                    self.G.append((j, i))
        self.P = partition(self.L, self.G)
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
