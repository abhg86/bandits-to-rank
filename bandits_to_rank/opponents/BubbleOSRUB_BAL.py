#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import sqrt, log

class BUBBLEOSUB:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, T, nb_positions):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.
        :param nb_positions:
        :param T: number of trials
        """
        self.nb_positions = nb_positions
        self.certitude = sqrt(log(T))
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.Leader = [i for i in range(self.nb_positions)]; shuffle(self.Leader)
        self.key = str(self.Leader)
        self.proposition = [item for item in self.Leader]
        self.kappa_thetas = [self.nb_positions * [0] for j in range(self.nb_positions)] #kappa_thetas[pos, item] = kap_pos * thet_item
        self.times_kappa_theta = [self.nb_positions * [1] for j in range(self.nb_positions)] \
                    #times_kappa_theta[pos, item] = nbre d'observations du couple (kappa_theta) + 1
        self.current_kappa_thetas = self.nb_positions * [0] #current_kappa_thetas[pos] = kappa_thetas[pos][Leader[pos]]
        self.sorted_pos = [i for i in range(self.nb_positions)] # = argsort(self.current_kappa_thetas)
        self.leader = 0  # compte le nombre de step de leadership consécutif du leader actuel
        self.voisinage_reduit = [(0, 0)] + [(i, i + 1) for i in range(self.nb_positions - 1)] # contient
                         # la liste des transpositions à tester dans le sens décroissant des theta_kappa
        self.parcours = [] # contient les transposition à proposer
        self.leader_dict = {} # garde en mémoire les statistiques l_a(n)

    def choose_next_arm(self):
        for (k, l) in self.parcours:
            self.proposition[k], self.proposition[l] = self.proposition[l], self.proposition[k]
        return self.proposition, 0

    def update(self, propositions, rewards):
        self.leader += 1
        for k in range(self.nb_positions):
            kappa_theta, n = self.kappa_thetas[k][propositions[k]], self.times_kappa_theta[k][propositions[k]]
            kappa_theta, n = kappa_theta + (rewards[k] - kappa_theta) / (n + 1), n + 1
            self.kappa_thetas[k][propositions[k]], self.times_kappa_theta[k][propositions[k]] = kappa_theta, n

        #reinitialiser proposition sous la forme de leader. Le faire ici car l'objet propo et self.propo sont les mêmes
        for (k, l) in self.parcours:
            self.proposition[k], self.proposition[l] = self.proposition[l], self.proposition[k]

        # On cherche le leader L(n)
        (i, j) = (0, 0)
        for (k, l) in self.voisinage_reduit:
            value = - self.kappa_thetas[k][self.Leader[k]] - self.kappa_thetas[l][self.Leader[l]] \
                    + self.kappa_thetas[k][self.Leader[l]] + self.kappa_thetas[l][self.Leader[k]]
            if (value > 0):
                (i, j) = (k, l)
                break

        if not (i, j) == (0, 0):  # Si on a changé de leader
            self.leader_dict[self.key] = self.leader #sauvegarde de la statistique leader

            self.Leader[i], self.Leader[j] = self.Leader[j], self.Leader[i]
            self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]
            self.key = str(self.Leader)

            if self.key in self.leader_dict:
                self.leader = self.leader_dict[self.key]
            else:
                self.leader = 0

        #mise à jour de sorted_pos
        for k in range(self.nb_positions):
            self.current_kappa_thetas[k] = self.kappa_thetas[k][self.Leader[k]]
        self.sorted_pos.sort(key=self.current_kappa_thetas.__getitem__)

        #mise à jour du voisinage réduit
        for k in range(self.nb_positions - 1):
            self.voisinage_reduit[k + 1] = (self.sorted_pos[k], self.sorted_pos[k + 1])

        # on parcourt selon les kappa*theta ordonnés
        self.parcours = []
        h = self.leader % 3 - 1
        if h > - 1:
            for n in range((self.nb_positions - h) // 2):
                k, l = self.sorted_pos[2 * n + h], self.sorted_pos[2 * n + 1 + h]
                item_k, item_l = self.Leader[k], self.Leader[l]
                if (- self.kappa_thetas[k][item_k] - self.kappa_thetas[l][item_l]
                    + self.kappa_thetas[k][item_l] + self.kappa_thetas[l][item_k] + self.certitude *
                   (- 1 / sqrt(2 * self.times_kappa_theta[k][item_k]) - 1 / sqrt(2 * self.times_kappa_theta[l][item_l])
                    + 1 / sqrt(2 * self.times_kappa_theta[k][item_l]) + 1 / sqrt(2 * self.times_kappa_theta[l][item_k])) > 0):
                    self.parcours.append((k, l))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
