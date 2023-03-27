#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import log
from bandits_to_rank.tools.tools_BAL import start_up, newton
from copy import deepcopy


class RUB_UCB:
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
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.Leader = [i for i in range(self.nb_positions)]; shuffle(self.Leader)
        self.proposition = [i for i in self.Leader]
        self.proposed = (0, 0) #transposition du leader qui a été dernièrement proposée

        self.deja_vu = {} #Si deja_vu[str(a)] = (str(a'), i, j) alors a' est le leader le plus récent \
        # contenant a dans son voisinage et a' rond (i, j) = a
        self.leader_dict = {}  # leader_dict[str(a)] = l_a(n)
        self.times_dict = {} # times_dict[a][i, j] = t_{a rond (i, j)}(n)
        self.mu_dict = {} # mu_dict[a][i, j] = û_{a rond (i, j)} (n)
        self.upper_bound_dict = {}
        self.keys_ring_dict = {}  #c'est le trousseau des clés des voisins du leader
        if not self.Leader in self.leader_dict:
            self.init_data_Leader()
        self.get_info_neighbour()

    def init_data_Leader(self):
        self.leader_dict[self.Leader] = 1
        self.times_dict[self.Leader] = {}#[[0 for i in range(self.nb_positions)] for j in range(self.nb_positions)]
        self.mu_dict[self.Leader] = {}#[[0 for i in range(self.nb_positions)] for j in range(self.nb_positions)]            self.upper_bound_dict[self.Leader] = {}#[[1 for i in range(self.nb_positions)] for j in range(self.nb_positions)]
        self.keys_ring_dict[self.Leader] = {}#[['' for i in range(self.nb_positions)] for j in range(self.nb_positions)]
        for (i, j) in self.voisinage:
            self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]
            self.times_dict[self.Leader][i, j] = 0
            self.mu_dict[self.Leader][i, j] = 0
            self.upper_bound_dict[self.Leader][i, j] = 1
            self.keys_ring_dict[self.Leader][i, j] = str(self.proposition)
            self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]

    def get_info_neighbour(self):
         # on définit des pointeurs vers les données courantes au sein du dictionnaire
        self.current_mu = self.mu_dict[self.Leader]
        self.current_times = self.times_dict[self.Leader]
        self.current_upper_bound = self.upper_bound_dict[self.Leader]
        self.current_keys = self.keys_ring_dict[self.Leader]

        #collectionne toutes les données enregistrées concernant le voisinage du nouveau leader
        for (i, j) in self.voisinage:
            if self.current_keys[i, j] in self.deja_vu: #si on a déjà des données sur action
                key_source, k, l = self.deja_vu[self.current_keys[i, j]]
                self.current_times[i, j] = self.times_dict[key_source][k, l]
                self.current_mu[i, j] = self.mu_dict[key_source][k, l]
                self.current_upper_bound[i, j] = self.upper_bound_dict[key_source][k, l]
            self.deja_vu[self.current_keys[i, j]] = (self.Leader, i, j)

    def transpose(self, i, j ):
        proposition = deepcopy(self.Leader)
        proposition[i], proposition[j] = proposition[j], proposition[i]
        return proposition

    def choose_next_arm(self):
        (i, j) = (0, 0)
        if self.leader_dict[self.Leader] % len(self.voisinage) != 1:
            upper_bound_max = self.current_upper_bound[i, j]
            for (k, l) in self.voisinage:
                value = self.current_upper_bound[k, l]
                if (value > upper_bound_max):
                    (i, j) = (k, l)
                    upper_bound_max = value

        proposition = self.transpose(i,j)
        self.proposed = i, j
        return proposition, 0

    def update(self, propositions, rewards):
        self.leader_dict[self.Leader] += 1
        i, j = self.proposed
        #on réinitialise self.proposed à leader
        #self.proposition[i], self.proposition[j] = self.proposition[j], self.proposition[i]

        self.current_times[i, j] += 1
        self.current_mu[i, j] += (sum(rewards) - self.current_mu[i, j]) / self.current_times[i, j]

        # on va calculer le nouvel upper_bound (i, j)
        mu = self.current_mu[i, j] / self.nb_positions
        start = start_up(mu, log(self.leader_dict[self.Leader]), self.current_times[i, j])
        self.current_upper_bound[i, j] = newton(mu, log(self.leader_dict[self.Leader]), self.current_times[i, j], start)


        # On cherche maintenant le leader L(n)
        self.look_for_leader(i, j)

    def look_for_leader(self, i, j):

        if (i, j) == (0, 0):  # si le score du leader a été mise à jour
            mu_max = self.current_mu[0, 0]
            for (k, l) in self.voisinage:
                if (self.current_mu[k, l] > mu_max):
                    mu_max = self.current_mu[k, l]
                    i, j = k, l
        else:   #si c'est le score d'un autre bras qui a été mis à jour
            if (self.current_mu[i, j] <= self.current_mu[0, 0]):
                i, j = 0, 0

        if (i, j) != (0, 0):
            self.Leader = self.transpose(i,j)
            if not self.Leader in self.leader_dict:
                self.init_data_Leader()
            self.get_info_neighbour()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
