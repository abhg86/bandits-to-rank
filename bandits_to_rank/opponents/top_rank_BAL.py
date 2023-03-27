#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
import numpy as np
from math import sqrt,log, erf, pi

class TOP_RANK:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, T, nb_positions=None):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.
        :param nb_arms:
        :param nb_positions:
        :param T: number of trials
        """
        self.nb_arms = nb_arms #je suppose qu'il s'agit de L
        self.T = T
        self.nb_positions = nb_positions
        self.c = 4 * sqrt(2 / pi) / erf(sqrt(2))
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.graphe_t = []  # graphe de la relation d'ordre
        self.partition_t = []
        self.full_rewards = np.zeros(self.nb_arms) #full_rewards[i] = rewards[j] si propositions[j] = i. Sinon, full_rewards[i] = 0
        self.S_t = [self.nb_arms * [0] for i in range(self.nb_arms)]
        self.N_t = [self.nb_arms * [0] for i in range(self.nb_arms)]

    def min(self, X):
        #Va calculer la fonction min_G(X) de l'article ainsi que X\min_G(X)
        a_supprimer = set()  #On recolte tous les éléments de X qui n'appartiennent step au min
        for (i, j) in self.graphe_t:
            if (i in X) and (j in X):
                a_supprimer.add(i)
        res = list(X - a_supprimer) #On met de coté le min
        return res, a_supprimer

    def choose_next_arm(self):#### Attention resampling
        a_traiter = set(range(self.nb_arms))
        self.partition_t = []
        proposition = []
        while len(a_traiter) > 0:
            partie, a_traiter = self.min(a_traiter)
            self.partition_t.append(partie)
            shuffle(partie)
            proposition += partie
        return np.array(proposition[: self.nb_positions]), 0

    def update(self, propositions, rewards):
        self.full_rewards[propositions] = rewards
        for partie in self.partition_t:
            for i in partie:
                for j in partie:
                    self.S_t[i][j] += self.full_rewards[i] - self.full_rewards[j]
                    self.N_t[i][j] += abs(self.full_rewards[i] - self.full_rewards[j])
                    if (self.N_t[i][j] > 0) and (self.S_t[i][j] >=
                                            sqrt(2 * self.N_t[i][j] * log(self.c * sqrt(self.N_t[i][j]) * self.T ))):
                        self.graphe_t.append((j, i))

        self.full_rewards[propositions] = 0




if __name__ == "__main__":
    import doctest
    doctest.testmod()
