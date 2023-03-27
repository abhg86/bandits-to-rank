#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import random

import numpy as np
from math import log

from bandits_to_rank.tools.tools_BAL import start_up, newton

class CASCADEKL_UCB:
    """
    Source : "Caascadig Bandits:LEarning to Rank in the Cascade Model"
    """
    def __init__(self, nb_arms, nb_position):
        """
        One of both `discount_facor` and `nb_position` has to be defined.

        :param nb_arms:
        :param nb_position:


        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_position
        self.clean()

    def clean(self):
        """ Clean log data.
        To be ran before playing a new game.
        """
        # clean the model
        self.nb_trials = 0
        # clean the log
        self.weights_hat = np.zeros(self.nb_arms, dtype=np.float)
        self.n_try = np.zeros(self.nb_arms, dtype=np.int) # number of times a proposal has been drawn for arm i's parameter
        self.upper_bound_weights = np.ones(self.nb_arms, dtype=np.float)

    def choose_next_arm(self):
        if self.nb_trials < self.nb_arms:
            proposition = np.array([(self.nb_trials + i) % self.nb_arms for i in range(self.nb_positions)])
            return proposition, 0

        self.time_reject = 0

        return np.argsort(-self.upper_bound_weights)[:self.nb_positions], self.time_reject

    
    def update(self, propositions, rewards):
        self.nb_trials += 1
        # update statistics
        if self.nb_trials-1 < self.nb_arms:
            self.n_try[propositions[0]] += 1
            self.weights_hat[propositions[0]] = rewards[0]
        else:
            # get index of first item clic
            list_index_reward_positive = np.nonzero(rewards == 1)[0]
            if len(list_index_reward_positive) != 0:
                index_first_click = list_index_reward_positive[0]
            else :
                index_first_click = self.nb_positions

            # update statistics
            proposition_seen = propositions[:index_first_click+1]
            self.n_try[proposition_seen] += 1
            self.weights_hat[proposition_seen] += (rewards[:index_first_click+1] - self.weights_hat[proposition_seen]) / self.n_try[proposition_seen]

        # update upper-bounds
        if self.nb_trials >= self.nb_arms:
            certitude = log(self.nb_trials) + 3*log(log(self.nb_trials))
            for item_k in range(self.nb_arms):
                weight, n = self.weights_hat[item_k], self.n_try[item_k]
                start = start_up(weight, certitude, n)
                self.upper_bound_weights[item_k] = newton(weight, certitude, n, start)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
