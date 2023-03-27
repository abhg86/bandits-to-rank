#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
from math import log
import numpy as np
import operator
from bandits_to_rank.tools.tools_BAL import start_up, newton
class OSUB_finit_memory:
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
        self.certitude = log(T)
        self.voisinage = [(0, 0)]
        for i in range(1, nb_positions):
            for j in range(i):
                self.voisinage.append((i, j))
        self.seuil = 1000
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.Leader = [i for i in range(self.nb_positions)]; shuffle(self.Leader)
        self.key = str(self.Leader) #la clé pour les dictionnaires
        self.proposed = self.key #clé du bras récemment proposé
        self.sorted_pos = [i for i in range(self.nb_positions)]  # = argsort(self.rew)
        self.voisinage_reduit = [(i + 1, i) for i in range(self.nb_positions - 1)]
        self.leader_dict = {}  # leader_dict[str(a)] = l_a(n)
        self.times_dict = {} # times_dict[a] = t_{a}(n)
        self.mu_dict = {} # mu_dict[a] = û_{a} (n)
        self.ucb_dict = {}
        self.rew_dict = {} # rew_dict[a] = somme des rewards de a
        self.keys_ring_dict = {} #c'est le trousseau des clés des voisins du leader
        self.voisinage_dict = {} #c'est les elements explicites du voisinage
        self.memory_mu_vois = {} #garde en mémoire la moyenne des gains des voisinages des bras joués
        self.init_arm(self.key, self.Leader)
        self.init_Leader()

    def init_arm(self, key, arm):
        #crée le bras et ajoute des entrées aux différents dictionnaires nécessaires
        self.times_dict[key] = 0
        self.mu_dict[key] = 0
        self.ucb_dict[key] = 1
        self.rew_dict[key] = np.zeros(self.nb_positions)
        self.keys_ring_dict[key] = {}
        self.voisinage_dict[key] = {(0, 0): arm}
        for (i, j) in self.voisinage:
            arm[i], arm[j] = arm[j], arm[i]
            self.keys_ring_dict[key][i, j] = str(arm)
            arm[i], arm[j] = arm[j], arm[i]

        #collectionne et partage les données concernant le voisinage du bras
        for (i, j) in self.voisinage:
            key_ij = self.keys_ring_dict[key][i, j]
            if key_ij in self.times_dict: #si on a déjà des données sur action
                self.voisinage_dict[key][i, j] = self.voisinage_dict[key_ij][0, 0]
                self.voisinage_dict[key_ij][i, j] = arm

    def init_Leader(self):
        if not self.key in self.leader_dict:
            self.leader_dict[self.key] = 0
        self.update_voisinage_reduit()

    def update_voisinage_reduit(self):
        self.sorted_pos.sort(key = self.rew_dict[self.key].__getitem__)
        for n in range(self.nb_positions - 1):
            i, j = max(self.sorted_pos[n], self.sorted_pos[n + 1]), min(self.sorted_pos[n], self.sorted_pos[n + 1])
            self.voisinage_reduit[n] = (i, j)

            key_ij = self.keys_ring_dict[self.key][i, j]
            if not key_ij in self.times_dict:  # si on n'a pas encore de données sur action
                arm_ij = [item for item in self.Leader]
                arm_ij[i], arm_ij[j] = arm_ij[j], arm_ij[i]
                self.init_arm(key_ij, arm_ij)

    def compute_mu_vois(self, key):
        r, t = 0, 0
        for (i, j) in self.voisinage:
            k = self.keys_ring_dict[key][i, j]
            if k in self.times_dict:
                r += self.mu_dict[k]
                t += self.times_dict[k]
        return r / t, self.times_dict[key]

    def choose_next_arm(self):
        (i, j) = (0, 0)
        if self.leader_dict[self.key] % self.nb_positions > 0:
            upper_bound_max = self.ucb_dict[self.key]
            for (k, l) in self.voisinage_reduit:
                key_kl = self.keys_ring_dict[self.key][k, l]
                value = self.ucb_dict[key_kl]
                if (value > upper_bound_max):
                    (i, j) = (k, l)
                    upper_bound_max = value
        self.proposed = self.keys_ring_dict[self.key][i, j]
        return self.voisinage_dict[self.key][i, j], 0

    def update(self, propositions, rewards):
        self.leader_dict[self.key] += 1
        self.times_dict[self.proposed] += 1
        self.mu_dict[self.proposed] += sum(rewards)
        self.rew_dict[self.proposed] += rewards

        # on va calculer le nouvel upper_bound (i, j)
        mu = self.mu_dict[self.proposed] / (self.nb_positions * self.times_dict[self.proposed])
        start = start_up(mu, self.certitude, self.times_dict[self.proposed])
        self.ucb_dict[self.proposed] = newton(mu, self.certitude, self.times_dict[self.proposed], start)

        # mettre à jour les valeurs dans memory
        if self.times_dict[self.proposed] > self.seuil:
            mu_vois = self.compute_mu_vois(self.proposed)
            self.memory_mu_vois[self.proposed] = mu_vois

        #on met à jour le voisinage reduit
        if self.proposed == self.key:
            self.update_voisinage_reduit()

        # On cherche maintenant le leader L(n)
        if len(self.memory_mu_vois) > 0:
            key_leader = max(self.memory_mu_vois.items(), key=operator.itemgetter(1))[0]
            if not key_leader == self.key:
                self.Leader = self.voisinage_dict[key_leader][0, 0]
                self.key = key_leader
                self.init_Leader()

if __name__ == "__main__":
    import doctest
    doctest.testmod()