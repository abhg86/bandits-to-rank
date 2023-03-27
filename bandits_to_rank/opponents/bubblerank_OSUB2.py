#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle,randint

import numpy as np
from math import sqrt,log
from copy import deepcopy

from bandits_to_rank.sampling.pbm_inference import SVD


def ordonne_indice_function_kappa(indices, kappas):
    """

    :param indices:
    :param kappas:
    :return:

    >>> import numpy as np
    >>> index = np.array([5, 1, 3])
    >>> kappas = np.array([1, 0.9, 0.8])
    >>> order_index_according_to_kappa(index, kappas)
    array([5, 1, 3])

    >>> index = np.array([5, 1, 3])
    >>> kappas = np.array([1, 0.8, 0.9])
    >>> order_index_according_to_kappa(index, kappas)
    array([5, 3, 1])
    """

    nb_position = len(kappas)
    indice_kappa_ordonne =  np.array(kappas).argsort()[::-1][:nb_position]
    res = np.ones(nb_position, dtype=np.int)
    nb_put_in_res = 0
    for i in indice_kappa_ordonne:
        res[i]=indices[nb_put_in_res]
        nb_put_in_res+=1
    return res



class BUBBLERANK_OSUB2:
    """
    Source : "BubbleRank: Safe Online Learning to Re-Rank via Implicit Click Feedback"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, discount_factor, R_init=None, nb_positions=None, nb_shuffles=0):
        """
        One of both `discount_factor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param discount_factor: 
        :param R:
        :param nb_shuffles: number of shuffle to do in the initial list

        """

        
        self.discount_factor = discount_factor
        nb_positions = len(discount_factor)

        if nb_arms != nb_positions:
            raise ValueError(f'BubbleRank requires the number of arms ({nb_arms}) to be equal to the number of positions ({nb_positions}).')

        self.nb_positions = nb_positions
        self.nb_arms = nb_arms

        self.R_init = R_init
        print("R_init", self.R_init)

        if nb_shuffles != 0:
            if R_init is None:
                raise ValueError("R must be defined if nb_shuffles is not 0")
            else :
                for i in range(nb_shuffles):
                    rand1 = np.random.randint(0, self.nb_arms -1)
                    rand2 = np.random.randint(0, self.nb_arms -1)
                    self.R_init[rand1], self.R_init[rand2] = self.R_init[rand2], self.R_init[rand1]

        self.clean()
     
    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        if self.R_init is None:
            self.R_bar = np.arange(self.nb_arms)
            shuffle(self.R_bar)

        else:
            if len(self.R_init)!=self.nb_positions:
                print(len(self.R_init))
                print(self.nb_positions)
                raise ValueError("List propose out of order")
            else:
                self.R_bar = np.array(self.R_init, dtype=np.uint)   # btw. get a copy of self.R_init
        self.R_propose =[]
        self.time = 1
        self.doubt = np.zeros([self.nb_arms, self.nb_arms], dtype=np.bool)
        self.s = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.n = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        
        
    def is_prob_better(self, i, j):
        return self.s[i][j] > 4*sqrt(log(self.time) * self.n[i][j])

    def choose_next_arm(self):#### Attention resampling
        propositions = deepcopy(self.R_bar[0:self.nb_positions])
        h = self.time%2
        # randomly permute R_bar
        for k in range((self.nb_positions - h)//2):
            i = propositions[2*k + h]
            j = propositions[2*k+1 + h]
            sure_ij = self.is_prob_better(i, j)
            if not sure_ij :
                self.doubt[i][j] = True
                self.doubt[j][i] = True
                if randint(0,1)==1:
                    propositions[2*k + h] = j
                    propositions[2*k+1 + h] = i
        self.R_propose = propositions
        return ordonne_indice_function_kappa(propositions, self.discount_factor), 0
        #return propositions,0

    def get_reward_arm(self,i,propositions, rewards):
        propositions_list=list(propositions)
        if i in propositions_list:
            pos = propositions_list.index(i)
            rew = rewards[pos]
        else :
            rew = 0
        return rew

    def update_matrix(self, propositions, rewards):
        h = self.time % 2
        for k in range((self.nb_positions - h) // 2):
            i = self.R_propose[2 * k + h]
            j = self.R_propose[2 * k + 1 + h]
            C_i = self.get_reward_arm(i, self.R_propose, rewards)
            C_j = self.get_reward_arm(j, self.R_propose, rewards)

            # print('clicks', C_i,C_j)
            # print(i,j)
            # print(self.doubt[i][j])

            if C_i != C_j and self.doubt[i][j]:
                self.s[i][j] += C_i - C_j
                self.n[i][j] += 1
                self.s[j][i] += C_j - C_i
                self.n[j][i] += 1
            # print("s", self.s)
            # print("n", self.n)
        self.doubt = np.zeros([self.nb_arms, self.nb_arms], dtype=np.bool)

    def update_R_bar(self):
        for k in range(self.nb_positions-1):
            i = self.R_bar[k]
            j = self.R_bar[k+1]
            if self.s[i][j] < 0:
                self.R_bar[k] = j
                self.R_bar[k+1] = i

    def update(self, propositions, rewards):
        self.update_matrix(propositions, rewards)
        self.update_R_bar()
        self.time+=1

    def get_leader(self):
        return self.R_bar


if __name__ == "__main__":
    import doctest

    doctest.testmod()
