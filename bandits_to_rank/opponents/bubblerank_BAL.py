#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
import numpy as np
from math import sqrt,log
from random import shuffle, randint
from bandits_to_rank.tools.tools import unused

class BUBBLERANK:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal

    !!! assume positions ranked from the most attractive to the less attractive !!!

    """
    def __init__(self, nb_arms, nb_positions, T=10):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param T: number of trials
        """
        self.nb_arms = nb_arms #pour suivre l'article on suppose que nb_arms = nb_positions
        self.T = T
        self.nb_positions = nb_positions
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.time = 1 #variable t
        self.Rbar_t = np.arange(self.nb_arms)  # displayed items + acceptable items
        shuffle(self.Rbar_t)
        self.S_t = np.zeros([self.nb_arms, self.nb_arms])
        self.N_t = np.zeros([self.nb_arms, self.nb_arms])

    def choose_next_arm(self):#### Attention resampling
        h = self.time % 2
        proposition = self.Rbar_t.copy()
        for k in range(1, (self.nb_positions - h) // 2 + 1):
            i, j = proposition[2*k - 2 + h], proposition[2*k - 1 + h]
            if self.S_t[i][j] <= 4*sqrt(self.N_t[i][j] * log(self.T)):  # corresponds to delta = 1/T^4
                if np.random.rand() < 0.5:
                    proposition[2 * k - 2 + h], proposition[2 * k - 1 + h] = j, i
        # last item in Rbar_t may be exchanged with non-displayed ones.
        if h != self.nb_positions % 2 and len(proposition) > self.nb_positions and np.random.rand() < 0.5:
            # pick one at random
            pos = randint(self.nb_positions, len(proposition)-1)
            proposition[self.nb_positions-1] = proposition[pos]

        return proposition[: self.nb_positions], 0


    def update(self, propositions, rewards):
        h = self.time % 2
        for k in range(1, (self.nb_positions - h) // 2 + 1):
            i, j = propositions[2*k - 2 + h], propositions[2*k - 1 + h]
            delta = rewards[2*k - 2 + h] - rewards[2*k - 1 + h]
            if abs(delta) == 1:
                self.S_t[i][j] += delta
                self.N_t[i][j] += 1
                self.S_t[j][i] -= delta
                self.N_t[j][i] += 1
        if h != self.nb_positions % 2 and len(self.Rbar_t) > self.nb_positions:
            # handling last position
            i = self.Rbar_t[self.nb_positions-1]
            if propositions[-1] == i:
                j_outside = self.Rbar_t[randint(self.nb_positions, self.Rbar_t.shape[0]-1)]
                delta = rewards[-1]
            else:
                j_outside = propositions[-1]
                delta = -rewards[-1]
            if abs(delta) == 1:
                self.S_t[i][j_outside] += delta
                self.N_t[i][j_outside] += 1
                self.S_t[j_outside][i] -= delta
                self.N_t[j_outside][i] += 1
        self.time += 1
        if self.time == self.T:   #double-tricking
            self.T *= 2

        for k in range(self.nb_positions - 1):
            i, j = self.Rbar_t[k], self.Rbar_t[k + 1]
            if (self.S_t[j][i] > 4 * sqrt(self.N_t[j][i] * log(self.T))):   # corresponds to delta = 1/T^4
                self.Rbar_t[k], self.Rbar_t[k + 1] = j, i
        if h != self.nb_positions % 2 and len(self.Rbar_t) > self.nb_positions:
            # remove outside item or last item
            i = self.Rbar_t[self.nb_positions - 1]
            pos_j_outside = np.argwhere(self.Rbar_t == j_outside)[0]
            if self.S_t[j_outside][i] > 4 * sqrt(self.N_t[j_outside][i] * log(self.T)):   # corresponds to delta = 1/T^4
                self.Rbar_t[self.nb_positions-1] = j_outside
                self.Rbar_t[pos_j_outside] = self.Rbar_t[-1]
                self.Rbar_t = self.Rbar_t[:-1]
            elif (self.S_t[i][j_outside] > 4 * sqrt(self.N_t[i][j_outside] * log(self.T))):   # corresponds to delta = 1/T^4
                self.Rbar_t[pos_j_outside] = self.Rbar_t[-1]
                self.Rbar_t = self.Rbar_t[:-1]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
