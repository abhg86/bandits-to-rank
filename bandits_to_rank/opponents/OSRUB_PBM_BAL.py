#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import log
from bandits_to_rank.tools.tools_BAL import start_up, newton

class OSUB_PBM:
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
        self.T = T
        self.voisinage = [(0, 0)]
        for i in range(1, nb_positions):
            for j in range(i):
                self.voisinage.append((i, j))
        self.certitude = log(T)  # = -log(p) où p est la probabilité que nos estimateurs soient faux.
        # Cette certitude induit automatiquement une précision délimitant l'intervalle de confiance
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """

        # clean the log
        self.precision = 0  # estimateur du min (kappa_theta - kappa_theta') au voisinage du leader
        self.running_t = 0  # compteur des observations moyennées pour estimer precision
        self.Leader = [i for i in range(self.nb_positions)]; shuffle(self.Leader)
        self.key = str(self.Leader)
        self.proposition = [item for item in self.Leader]
        self.proposed = 0, 0  # la transposition du leader qui vient d'être proposée
        self.kappa_thetas = [self.nb_positions * [0] for j in range(self.nb_positions)]  # kappa_thetas[pos, item] = \
        # kappa_pos * theta_item
        self.times_kappa_theta = [self.nb_positions * [0] for j in range(self.nb_positions)] \
            # times_kappa_theta[pos, item] = nbre d'observations du couple (kappa_theta) + 1
        self.upper_bound_kappa_theta = [self.nb_positions * [1] for j in range(self.nb_positions)]
        self.leader = 0  # compte le nombre de step de leadership consécutif du leader actuel
        self.memory = len(self.voisinage) * [(None, None)] #garde en mémoire les gamma + 1 derniers (key, leader)
        self.writing_cell = 0 #l'indice de la première case libre de memory

    def choose_next_arm(self):
        (i, j) = (0, 0)
        if self.leader % len(self.voisinage) > 0:
            delta_upper_bound_max = 0
            for (k, l) in self.voisinage:
                item_k, item_l = self.Leader[k], self.Leader[l]
                value = - self.upper_bound_kappa_theta[k][item_k] - self.upper_bound_kappa_theta[l][item_l] \
                        + self.upper_bound_kappa_theta[k][item_l] + self.upper_bound_kappa_theta[l][item_k]
                if (value > delta_upper_bound_max):
                    (i, j) = (k, l)
                    delta_upper_bound_max = value
        self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]
        self.proposed = i, j
        return self.proposition, 0

    def pop(self):
        #enleve le plus vieil element de la file memory
        if self.writing_cell == len(self.memory):
            for i in range(self.writing_cell - 1):
                self.memory[i] = self.memory[i + 1]
                self.memory[i + 1] = (None, None)
            self.writing_cell -= 1

    def delete(self):
        #Supprime de memory l'information (s'il en existe) concernant le Leader et fait le decalage pour combler le trou
        drapeau = False
        for i in range(self.writing_cell):
            if self.memory[i][0] == self.key:
                l = self.memory[i][1]
                drapeau, index = True, i
                break
            elif self.memory[i][0] == None:
                break
        if drapeau:
            for i in range(index, self.writing_cell - 1):
                self.memory[i] = self.memory[i + 1]
                self.memory[i + 1] = (None, None)
            self.writing_cell -= 1
            return l
        return 0

    def update(self, propositions, rewards):
        self.leader += 1
        i, j = self.proposed
        for k in range(self.nb_positions):
            kappa_theta, n = self.kappa_thetas[k][propositions[k]], self.times_kappa_theta[k][propositions[k]]
            kappa_theta, n = kappa_theta + (rewards[k] - kappa_theta) / (n + 1), n + 1
            start = start_up(kappa_theta, self.certitude, n)
            upper_bound = newton(kappa_theta, self.certitude, n, start)
            self.kappa_thetas[k][propositions[k]], self.times_kappa_theta[k][propositions[k]] = kappa_theta, n
            self.upper_bound_kappa_theta[k][propositions[k]] = upper_bound

        # reinitialiser proposition sous la forme de leader. Le faire ici car l'objet propo et self.propo sont les mêmes
        self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]

        # On cherche maintenant le leader L(n)
        (i, j) = (0, 0)
        for (k, l) in self.voisinage:
            value = - self.kappa_thetas[k][self.Leader[k]] - self.kappa_thetas[l][self.Leader[l]] \
                    + self.kappa_thetas[k][self.Leader[l]] + self.kappa_thetas[l][self.Leader[k]]
            if (value > 0):
                (i, j) = (k, l)
                break

        if not (i, j) == (0, 0):  # Si on a changé de leader
            data = (self.key, self.leader) #sauvegarde des données concernant le précédent leader
            self.Leader[i], self.Leader[j] = self.Leader[j], self.Leader[i]
            self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]
            self.key = str(self.Leader)
            self.leader = self.delete()

            self.pop()
            self.memory[self.writing_cell] = data #stocke la sauvegarde sur le précédent leader
            self.writing_cell += 1


if __name__ == "__main__":
    import doctest
    doctest.testmod()
