#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle

import numpy as np
from math import sqrt, log
from itertools import product

from bandits_to_rank.sampling.pbm_inference import SVD
from bandits_to_rank.tools.tools import order_index_according_to_kappa


class TopRank:
  def __init__(self, nb_arms, nb_pos, T):
    self.nb_arms = nb_arms
    self.nb_pos = nb_pos
    self.T = T
    
    self.clean()

  def clean(self):
    self.pulls = np.ones((self.nb_arms, self.nb_arms))    # number of pulls
    self.reward = np.zeros((self.nb_arms, self.nb_arms))  # cumulative reward

    self.G = np.ones((self.nb_arms, self.nb_arms), dtype = bool)
    self.P = np.zeros(self.nb_arms)
    self.P2 = np.ones((self.nb_arms, self.nb_arms))

  def name(self):
    return "TopRank"

  def rerank(self):
    Gt = (self.reward / self.pulls - 2 * np.sqrt(np.log(self.T) / self.pulls)) > 0
    if not np.array_equal(Gt, self.G):
      self.G = np.copy(Gt)

      Pid = 0
      self.P = - np.ones(self.nb_arms, dtype=np.int16)
      while (self.P == -1).sum() > 0:
        items = np.flatnonzero(Gt.sum(axis=0) == 0)
        self.P[items] = Pid
        Gt[items, :] = 0
        Gt[:, items] = 1
        Pid += 1

      self.P2 = \
        (np.tile(self.P[np.newaxis], (self.nb_arms, 1)) == np.tile(self.P[np.newaxis].T, (1, self.nb_arms))).astype(float)

  def update(self, action, r):
    clicks = np.zeros(self.nb_arms)
    clicks[action] = r

    M = np.outer(clicks, 1 - clicks) * self.P2
    self.pulls += M + M.T
    self.reward += M - M.T

    self.rerank()

  def choose_next_arm(self):
    action = np.argsort(self.P + 1e-6 * np.random.rand(self.nb_arms))[: self.nb_pos]
    return action, 0

  def print_info(self):
      print('partition: id  ', np.sort(self.P))
      print('partition: arms', np.argsort(self.P))
      print(self.pulls)
      print(self.reward / (self.pulls + 10**-7))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
