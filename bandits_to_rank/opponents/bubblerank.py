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



class BUBBLERANK:
    """
    Source : "BubbleRank: Safe Online Learning to Re-Rank via Implicit Click Feedback"
    reject sampling with beta preposal
    """
    def __init__(self, nb_arms, R=None, delta=0.5, nb_positions=None, discount_factor=None, lag=1, prior_s=1, prior_f=1, nb_shuffles=0):
        """
        One of both `discount_factor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param discount_factor: if None, discount factors are inferred from logs every `lag` iterations
        :param lag:
        :param R:
        :param delta:
        :param prior_s:
        :param prior_f:
        :param nb_shuffles: number of shuffle to do in the basic list

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>> discount_factor = [1, 0.9, 0.7]
        >>> player = BUBBLERANK(nb_arms, discount_factor=discount_factor)

        # function to assert choices have the right form
        >>> def assert_choices(choices, nb_choices):
        ...     assert len(choices) == nb_choices, "recommmendation list %r should be of size %d" % (str(choices), nb_choices)
        ...     assert len(np.unique(choices)) == nb_choices, "there is duplicates in recommmendation list %r" % (str(choices))
        ...     for pos in range(nb_choices):
        ...          assert 0 <= choices[pos] < nb_arms, "recommendation in position %d is out of bound in recommmendation list %r" % (pos, str(choices))

        # First choices should be random uniform
        >>> n_runs = 100
        >>> counts = np.zeros(nb_arms)
        >>> # try one choice
        >>> assert_choices(player.choose_next_arm(), nb_choices)
        >>> # try several choices
        >>> for _ in range(n_runs):
        ...     choices = player.choose_next_arm()
        ...     assert_choices(choices, nb_choices)
        ...     for arm in choices:
        ...         counts[arm] += 1
        >>> # almost uniform ?
        >>> assert np.all(np.abs(counts/nb_choices/n_runs - 1./nb_arms) < 0.1), str(counts/nb_choices/n_runs)


        # Other choices have to be coherent
        >>> n_runs = 100
        >>> nb_choices = 1
        >>> discount_factor = [1]
        >>> nb_arms = 2
        >>> player = BUBBLERANK(nb_arms, discount_factor=discount_factor)
        >>> for _ in range(3):
        ...     player.update(np.array([0]), np.array([1]))
        ...     player.update(np.array([1]), np.array([0]))
        >>> # try one choice
        >>> assert_choices(player.choose_next_arm(), nb_choices)
        >>> # try several choices
        >>> counts = np.zeros(nb_arms)
        >>> for _ in range(n_runs):
        ...     choices = player.choose_next_arm()
        ...     assert_choices(choices, nb_choices)
        ...     for arm in choices:
        ...         counts[arm] += 1
        >>> # cover each arm
        >>> assert np.all(counts > 0), "%r" % str(counts)
        >>> # first arm is more drawn
        >>> assert np.all(counts[0] >= counts), "%r" % str(counts)
        >>> # second arm is less drawn
        >>> assert np.all(counts[1] <= counts), "%r" % str(counts)

        # Other choices have to be coherent
        >>> n_runs = 500
        >>> nb_choices = 1
        >>> discount_factor = [1]
        >>> nb_arms = 10
        >>> player = BUBBLERANK(nb_arms, discount_factor=discount_factor)
        >>> for i in range(nb_arms):
        ...     for _ in range(5):
        ...         player.update(np.array([i]), np.array([1]))
        ...         player.update(np.array([i]), np.array([0]))
        >>> player.last_present = np.array([0])
        >>> for _ in range(5):
        ...     player.update(np.array([0]), np.array([1]))
        ...     player.update(np.array([1]), np.array([0]))
        >>> counts = np.zeros(nb_arms)
        >>> # try one choice
        >>> assert_choices(player.choose_next_arm(), nb_choices)
        >>> # try several choices
        >>> for _ in range(n_runs):
        ...     choices = player.choose_next_arm()
        ...     assert_choices(choices, nb_choices)
        ...     for arm in choices:
        ...         counts[arm] += 1
        >>> # cover each arm
        >>> assert np.all(counts > 0), "%r" % str(counts)
        >>> # first arm is more drawn
        >>> assert np.all(counts[0] >= counts), "%r" % str(counts)
        >>> # second arm is less drawn
        >>> assert np.all(counts[1] <= counts), "%r" % str(counts)
        """
        if (discount_factor is None) == (nb_positions is None):
            raise ValueError("One of both `discount_facor` and `nb_positions` has to be defined")
        if discount_factor is not None:
            self.known_discount = True
            self.discount_factor = discount_factor
            nb_positions = len(discount_factor)
        else:
            self.known_discount = False
            self.lag = lag

        if nb_arms != nb_positions:
            raise ValueError(f'BubbleRank requires the number of arms ({nb_arms}) to be equal to the number of positions ({nb_positions}).')

        self.prior_s = prior_s
        self.prior_f = prior_f
        self.nb_positions = nb_positions
        self.nb_arms = nb_arms

        self.delta = delta
        self.R_init = R
        if nb_shuffles != 0:
            if R is None:
                raise ValueError("R must be defined if nb_shuffles is not 0")
            else :
                for i in range(nb_shuffles):
                    rand1 = np.random.randint(0, self.nb_arms -1)
                    rand2 = np.random.randint(0, self.nb_arms -1)
                    self.R_init[rand1], self.R_init[rand2] = self.R_init[rand2], self.R_init[rand1]

        self.clean()
     
    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the model
        if not self.known_discount:
            self.learner = SVD(self.nb_arms, self.nb_positions)
            self.learner.nb_views = np.ones((self.nb_arms, self.nb_positions)) * (self.prior_s+self.prior_f)
            self.learner.nb_clicks = np.ones((self.nb_arms, self.nb_positions)) * self.prior_s
            self.discount_factor = np.ones(self.nb_positions, dtype=np.float)

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
        self.time = 0
        self.s = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.n = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        
        
    def is_prob_better(self, i, j):
        return self.s[i][j] > 2*sqrt(self.n[i][j] * log(1 / self.delta))

    def choose_next_arm(self):#### Attention resampling
        propositions = deepcopy(self.R_bar[0:self.nb_positions])
        h = self.time%2
        # randomly permute R_bar
        for k in range((self.nb_positions - h)//2):
            i = propositions[2*k + h]
            j = propositions[2*k+1 + h]
            if not self.is_prob_better(i, j) and randint(0,1)==1:
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
            #i = self.R_bar[2 * k - 1 + h]
            #j = self.R_bar[2 * k + h]
            #C_i = self.get_reward_arm(i, propositions, rewards)
            #C_j = self.get_reward_arm(j, propositions, rewards)
            # print('rewards', C_i,C_j)
            if C_i != C_j:
                self.s[i][j] += C_i - C_j
                self.n[i][j] += 1
                self.s[j][i] += C_j - C_i
                self.n[j][i] += 1
            #print("t changed",self.S_list)

    def update_R_bar(self):
        for k in range(self.nb_positions-1):
            i = self.R_bar[k]
            j = self.R_bar[k+1]
            if self.is_prob_better(j, i):
                # exchange i an j for good 
                print(i,j,'permanetly changed' )
                self.R_bar[k] = j
                self.R_bar[k+1] = i

    def update(self, propositions, rewards):
        self.update_matrix(propositions, rewards)
        self.update_R_bar()
        if not self.known_discount:
            self.learner.add_session(propositions, rewards)
            if self.time < 100 or self.time % self.lag == 0:
                self.learner.learn()
                self.discount_factor = self.learner.get_kappas()
        self.time+=1

    def get_leader(self):
        return self.R_bar


if __name__ == "__main__":
    import doctest

    doctest.testmod()
