#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
from random import shuffle
import math
from itertools import product
from copy import copy
from bandits_to_rank.tools.tools_BAL import start_up, newton
from bandits_to_rank.tools.tools import unused, swap
import numpy as np
from collections import defaultdict


class UniRankFirstPos:
    """
    """

    def __init__(self, nb_arms, nb_positions, T=None, sigma=None, neighbor_type='bubble', gamma=None, memory_size=np.inf):
        """
        T : int or None
            Either the horizon of a game, or None (unknown horizon). Used to set the confidence interval size with
            log(T) (known horizon) or log(t) + 3 log(log(t)) with t the number of time current arm has been the leader.
        sigma :
            order on positions. sigma[i] is the i-th best position
        """
        if gamma is None:
            gamma = nb_arms - 1
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.horizon = T
        self.sigma = sigma
        self.neighbor_type = neighbor_type
        self.gamma = gamma
        self.memory_size = memory_size
        self.base_neighborhood = self.get_base_neighbor()

        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.running_t = 0
        leader = [i for i in range(self.nb_arms)]
        shuffle(leader)
        self.leader = tuple(leader[:self.nb_positions])
        self.list_transpositions = []
        self.logs = defaultdict(self.empty)  # number of time each arm has been the leader
        # add current leader to logs
        leader_log = self.logs[self.leader]
        leader_log['remaining'] = unused(self.leader, self.nb_arms)
        if self.sigma is None:
            leader_log['hat_pi'] = np.arange(self.nb_positions)
            shuffle(leader_log['hat_pi'])
        else:
            leader_log['hat_pi'] = self.sigma

    def empty(self):   # to enable pickling
        return {'remaining': None,  # list of items not displayed by the leader
                'nb_leader': 0,
                'hat_rhos': np.zeros(self.nb_arms),    # rho defaults to 0 for non-existing positions
                'hat_pi': None,
                'nb_trials': 0,
                'last_play': 0
                }

    def choose_next_arm(self):
        leader_log = self.logs[self.leader]
        proposition = self.leader
        if leader_log['nb_leader'] % self.gamma != 0:
            for (k, l) in self.list_transpositions:
                neighbor = swap(self.leader, (k, l), leader_log['remaining'])
                neighbor_log = self.logs.get(neighbor, None)
                if neighbor_log is None:
                    value = 1
                else:
                    if self.sigma is None:
                        value = self.optimistic_index(max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l])
                                                      , neighbor_log['nb_trials'], leader_log['nb_leader'])\
                                - self.optimistic_index(max(leader_log['hat_rhos'][k], leader_log['hat_rhos'][l])
                                                        , leader_log['nb_trials'], leader_log['nb_leader'])
                    else:
                        value = self.optimistic_index(neighbor_log['hat_rhos'][k]
                                                      , neighbor_log['nb_trials'], leader_log['nb_leader'])\
                                - self.optimistic_index(leader_log['hat_rhos'][k]
                                                        , leader_log['nb_trials'], leader_log['nb_leader'])
                if value > 0:
                    proposition = neighbor
                    break
        return np.array(proposition), 0

    def optimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        if self.horizon is None:
            if nb_trial == 0 or nb_total_trial < 3:
                return 1
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
        else:
            threshold = math.log(self.horizon)
        start = start_up(hat_rho, threshold, nb_trial)
        return newton(hat_rho, threshold, nb_trial, start)

    def pessimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        return 1-self.optimistic_index(1-hat_rho, nb_trial, nb_total_trial)

    def get_neighbor(self, hat_pi):
        if len(hat_pi) < self.nb_arms:
            new = np.arange(self.nb_arms)
            new[:len(hat_pi)] = hat_pi
            hat_pi = new
        return [(hat_pi[k], hat_pi[l]) for k, l in self.base_neighborhood]

    def get_base_neighbor(self):
        if self.neighbor_type == 'bubble':
            return self.get_bubble_neighbor()
        elif self.neighbor_type == 'qsort':
            return self.get_qsort_neighbor()
        elif self.neighbor_type == 'shell3,1':
            return self.get_shell_neighbor(seq=[3, 1])
        elif self.neighbor_type == 'jumps3':
            return self.get_jumps_neighbor(gap=3)
        elif self.neighbor_type == 'jumps3.1':
            return self.get_jumps_neighbor(gap=3, beg=1)
        raise ValueError(f'unknown neighborhood type {self.neighbor_type}')

    def get_bubble_neighbor(self, hat_pi=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100)
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        res = []
        for k in range(self.nb_positions - 1):
            res.append((hat_pi[k], hat_pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[self.nb_positions - 1], k))
        return res

    def get_qsort_neighbor(self, hat_pi=None, beg=0, end=None):
        """

        Parameters
        ----------
        hat_pi
        beg
        end

        Returns
        -------

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=4, nb_positions=4, neighbor_type='qsort')
        >>> player.get_neighbor([0, 1, 3, 2])
        [(0, 3), (1, 3), (3, 2), (0, 1)]
        >>> player = UniRankFirstPos(nb_arms=10, nb_positions=10, neighbor_type='qsort')
        >>> player.get_neighbor([0, 1, 3, 2, 4, 5, 8, 9, 7, 6])
        [(0, 5), (1, 5), (3, 5), (2, 5), (4, 5), (5, 8), (5, 9), (5, 7), (5, 6), (0, 3), (1, 3), (3, 2), (3, 4), (0, 1), (2, 4), (8, 7), (9, 7), (7, 6), (8, 9)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if end is None:
            end = self.nb_arms
        if len(hat_pi) < end:
            raise NotImplementedError('qsort neighborhood only implemented for nb_arms == nb_positions')
        def rec(hat_pi, beg, end):
            if beg >= end-1:
                return []
            else:
                mid = (beg + end) // 2
                res = [(hat_pi[l], hat_pi[mid]) for l in range(beg, mid)]
                res += [(hat_pi[mid], hat_pi[l]) for l in range(mid+1, end)]
                res += self.get_qsort_neighbor(hat_pi, beg, mid)
                res += self.get_qsort_neighbor(hat_pi, mid+1, end)
                return res
        return rec(hat_pi, beg, end)

    def get_shell_neighbor(self, hat_pi=None, seq=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='shell3,1')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 3), (1, 2), (4, 5), (4, 6), (0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if seq is None:
            seq = [3,1]
        res = []
        for gap in seq:
            for k in range(self.nb_positions - gap):
                res.append((hat_pi[k], hat_pi[k + gap]))
            for k in range(self.nb_positions, self.nb_arms):
                res.append((hat_pi[self.nb_positions - gap], k))
        return res

    def get_jumps_neighbor(self, hat_pi=None, gap=None, beg=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='jumps3')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6), (0, 3)]
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='jumps3.1')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6), (1, 2)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if gap is None:
            gap = 3
        if beg is None:
            beg = 0
        res = []
        for k in range(self.nb_positions - 1):
            res.append((hat_pi[k], hat_pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[self.nb_positions - 1], k))
        for k in range(beg, self.nb_positions - gap, gap):
            res.append((hat_pi[k], hat_pi[k + gap]))
        """
        last = beg + (self.nb_positions - beg) // gap_type * gap_type
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[last], k))
        #"""
        return res

    def update(self, propositions, rewards):
        self.running_t += 1
        self.logs[self.leader]['nb_leader'] += 1
        # update statistics
        log = self.logs[tuple(propositions)]
        log['nb_trials'] += 1
        log['last_play'] = self.running_t
        log['hat_rhos'][:self.nb_positions] += (rewards - log['hat_rhos'][:self.nb_positions]) / log['nb_trials']
        if self.sigma is None:
            log['hat_pi'] = np.argsort(-log['hat_rhos'][:self.nb_positions])
        else:
            log['hat_pi'] = self.sigma

        # shrink memory if too big
        if len(self.logs) > self.memory_size:
            min_key = min(self.logs.items(), key=lambda k: k[1]['last_play'])[0]
            self.logs.pop(min_key)

        # update the leader
        leader_log = self.update_leader()

        # update the leader's neighborhood
        self.list_transpositions = self.get_neighbor(leader_log['hat_pi'])

    def update_leader(self):
        previous_leader_log = self.logs[self.leader]
        leader_log = previous_leader_log
        for (k, l) in self.list_transpositions:
            neighbor = swap(self.leader, (k, l), previous_leader_log['remaining'])
            neighbor_log = self.logs.get(neighbor, None)
            if neighbor_log is not None:
                if self.sigma is None:
                    value = max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l]) \
                        - max(previous_leader_log['hat_rhos'][k], previous_leader_log['hat_rhos'][l])
                else:
                    value = neighbor_log['hat_rhos'][k] - previous_leader_log['hat_rhos'][k]
                if value > 0:
                    self.leader = neighbor
                    leader_log = self.logs[self.leader]
                    if leader_log['remaining'] is None:
                        leader_log['remaining'] = unused(self.leader, self.nb_arms)
                    break
        return leader_log


class UniRankMaxGap(UniRankFirstPos):
    """
    """

    def __init__(self, nb_arms, nb_positions, T=None, sigma=None, neighbor_type='bubble', gamma=None, memory_size=np.inf, bound_l='o', bound_n='o', lead_l='o', lead_n='a'):
        """
        T : int or None
            Either the horizon of a game, or None (unknown horizon). Used to set the confidence interval size with
            log(T) (known horizon) or log(t) + 3 log(log(t)) with t the number of time current arm has been the leader.
        """
        super().__init__(nb_arms=nb_arms, nb_positions=nb_positions, T=T, sigma=sigma, neighbor_type=neighbor_type, gamma=gamma, memory_size=memory_size)
        self.bound_l = bound_l
        self.bound_n = bound_n
        self.lead_l = lead_l
        self.lead_n = lead_n

    def estimator(self, hat_rho, nb_trial, nb_total_trial, bound):
        if bound == 'o':
            return self.optimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'p':
            return self.pessimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'a':
            return hat_rho
        else:
            raise ValueError(f'unkwon estimator {bound}')

    def choose_next_arm(self):
        leader_log = self.logs[self.leader]
        proposition = self.leader
        if leader_log['nb_leader'] % self.gamma != 0:
            value_max = 0
            for (k, l) in self.list_transpositions:
                neighbor = swap(self.leader, (k, l), leader_log['remaining'])
                neighbor_log = self.logs.get(neighbor, None)
                if neighbor_log is None:
                    value = 1
                else:
                    if self.sigma is None:
                        value = self.estimator(max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l])
                                                      , neighbor_log['nb_trials'], leader_log['nb_leader'], self.bound_n)\
                                - self.estimator(max(leader_log['hat_rhos'][k], leader_log['hat_rhos'][l])
                                                        , leader_log['nb_trials'], leader_log['nb_leader'], self.bound_l)
                    else:
                        value = self.estimator(neighbor_log['hat_rhos'][k]
                                                      , neighbor_log['nb_trials'], leader_log['nb_leader'], self.bound_n)\
                                - self.estimator(leader_log['hat_rhos'][k]
                                                        , leader_log['nb_trials'], leader_log['nb_leader'], self.bound_l)
                if value > value_max:
                    value_max = value
                    proposition = neighbor
        return np.array(proposition), 0

    def update_leader(self):
        previous_leader_log = self.logs[self.leader]
        leader_log = previous_leader_log
        value_max = 0
        for (k, l) in self.list_transpositions:
            neighbor = swap(self.leader, (k, l), previous_leader_log['remaining'])
            neighbor_log = self.logs.get(neighbor, None)
            if neighbor_log is not None:
                if self.sigma is None:
                    value = self.estimator(max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l])
                                                      , neighbor_log['nb_trials'], previous_leader_log['nb_leader'], self.lead_n) \
                            - self.estimator(max(previous_leader_log['hat_rhos'][k], previous_leader_log['hat_rhos'][l])
                                                        , previous_leader_log['nb_trials'], previous_leader_log['nb_leader'], self.lead_l)
                else:
                    value = self.estimator(neighbor_log['hat_rhos'][k]
                                                      , neighbor_log['nb_trials'], previous_leader_log['nb_leader'], self.lead_n) \
                            - self.estimator(previous_leader_log['hat_rhos'][k]
                                                        , previous_leader_log['nb_trials'], previous_leader_log['nb_leader'], self.lead_l)
                if value > value_max:
                    value_max = value
                    new_leader = neighbor
        """
        if self.sigma is not None:
            accelerated_change = False
            for (k, l) in self.list_transpositions:
                value = - self.optimistic_index(previous_leader_log['hat_rhos'][k]
                                              , previous_leader_log['nb_trials'], previous_leader_log['nb_leader']) \
                        + self.pessimistic_index(previous_leader_log['hat_rhos'][l]
                                                 , previous_leader_log['nb_trials'], previous_leader_log['nb_leader'])
                if value > value_max:
                    accelerated_change = True
                    value_max = value
                    new_leader = swap(self.leader, (k, l), previous_leader_log['remaining'])
            if accelerated_change:
                print(f'!!! accelerated change from {self.leader} to {new_leader} at iteration {self.running_t}')
                log = self.logs[new_leader]
                log['hat_pi'] = self.sigma

        # """
        if value_max > 0:
            self.leader = new_leader
            leader_log = self.logs[self.leader]
            if leader_log['remaining'] is None:
                leader_log['remaining'] = unused(self.leader, self.nb_arms)
        return leader_log


class UniRankMaxRatio(UniRankFirstPos):
    """
    """

    def __init__(self, nb_arms, nb_positions, T=None, gamma=None, memory_size=np.inf):
        """
        T : int or None
            Either the horizon of a game, or None (unknown horizon). Used to set the confidence interval size with
            log(T) (known horizon) or log(t) + 3 log(log(t)) with t the number of time current arm has been the leader.
        """
        super().__init__(nb_arms=nb_arms, nb_positions=nb_positions, T=T, gamma=gamma, memory_size=memory_size)

    def choose_next_arm(self):
        leader_log = self.logs[self.leader]
        proposition = self.leader
        if leader_log['nb_leader'] % self.gamma != 0:
            value_max = 1
            for (k, l) in self.list_transpositions:
                neighbor = swap(self.leader, (k, l), leader_log['remaining'])
                neighbor_log = self.logs.get(neighbor, None)
                if neighbor_log is None:
                    value = np.inf
                else:
                    best_pos_hat = [k, l, k, l][np.argmax([neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l], leader_log['hat_rhos'][k], leader_log['hat_rhos'][l]])]
                    value = self.optimistic_index(neighbor_log['hat_rhos'][best_pos_hat]
                                                  , neighbor_log['nb_trials'], leader_log['nb_leader'])\
                            / self.optimistic_index(leader_log['hat_rhos'][best_pos_hat]
                                                    , leader_log['nb_trials'], leader_log['nb_leader'])
                if value > value_max:
                    value_max = value
                    proposition = neighbor
        return np.array(proposition), 0

    def update_leader(self):
        previous_leader_log = self.logs[self.leader]
        leader_log = previous_leader_log
        value_max = 1
        for (k, l) in self.list_transpositions:
            neighbor = swap(self.leader, (k, l), previous_leader_log['remaining'])
            neighbor_log = self.logs.get(neighbor, None)
            if neighbor_log is not None:
                best_pos_hat = [k, l, k, l][np.argmax(
                    [neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l], previous_leader_log['hat_rhos'][k],
                     previous_leader_log['hat_rhos'][l]])]
                value = neighbor_log['hat_rhos'][best_pos_hat] / previous_leader_log['hat_rhos'][best_pos_hat]
                if value > value_max:
                    value_max = value
                    new_leader = neighbor
        if value_max > 0:
            self.leader = new_leader
            leader_log = self.logs[self.leader]
            if leader_log['remaining'] is None:
                leader_log['remaining'] = unused(self.leader, self.nb_arms)
        return leader_log


class UniRankWithMemory:
    """
    """

    def __init__(self, nb_arms, nb_positions, T=None, sigma=None, neighbor_type='bubble', gamma=None, memory_size=np.inf, bound_l='o', bound_n='o', lead_l='o', lead_n='a'):
        """
        T : int or None
            Either the horizon of a game, or None (unknown horizon). Used to set the confidence interval size with
            log(T) (known horizon) or log(t) + 3 log(log(t)) with t the number of time current arm has been the leader.
        sigma :
            order on positions. sigma[i] is the i-th best position
        """
        if gamma is None:
            gamma = nb_arms - 1
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.horizon = T
        self.sigma = sigma
        self.neighbor_type = neighbor_type
        self.gamma = gamma
        self.memory_size = memory_size
        self.base_neighborhood = self.get_base_neighbor()
        self.bound_l = bound_l
        self.bound_n = bound_n
        self.lead_l = lead_l
        self.lead_n = lead_n

        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.running_t = 0
        leader = [i for i in range(self.nb_arms)]
        shuffle(leader)
        self.leader = tuple(leader[:self.nb_positions])
        self.list_transpositions = []
        self.node_logs = defaultdict(self.empty_node)  # number of time each arm has been the leader, ...
        self.shared_logs = defaultdict(self.empty_rho)  # shared_logs[set(s)][item i] = \rho_{(s+i,|s|+1}
        # add current leader to logs
        leader_log = self.node_logs[self.leader]
        leader_log['remaining'] = unused(self.leader, self.nb_arms)
        if self.sigma is None:
            raise NotImplementedError(f'sigma has to be known')

    def empty_node(self):   # to enable pickling
        return {'remaining': None,  # list of items not displayed by the leader
                'nb_leader': 0,
                'last_play': 0
                }

    def empty_rho(self):   # to enable pickling
        return {'hat_rhos': np.zeros(self.nb_arms),    # rho defaults to 0 for non-played
                'nb_trials': np.zeros(self.nb_arms, dtype=np.int)
                }

    def choose_next_arm(self):
        leader_log = self.node_logs[self.leader]
        proposition = self.leader
        if leader_log['nb_leader'] % self.gamma != 0:
            value_max = 0
            for (k, l) in self.list_transpositions:
                if self.sigma is None:
                    """
                    value = self.estimator(max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l])
                                                  , neighbor_log['nb_trials'], leader_log['nb_leader'], self.bound_n)\
                            - self.estimator(max(leader_log['hat_rhos'][k], leader_log['hat_rhos'][l])
                                                    , leader_log['nb_trials'], leader_log['nb_leader'], self.bound_l)
                    """
                else:
                    item_l = self.leader[l] if l < self.nb_positions else leader_log['remaining'][l-self.nb_positions]
                    local_rho = self.shared_logs.get(tuple(np.sort(self.leader[:k])), None)
                    if local_rho is None or local_rho['nb_trials'][item_l] == 0:
                        value = np.inf
                    else:
                        value = self.estimator(local_rho['hat_rhos'][item_l]
                                               , local_rho['nb_trials'][item_l]
                                               , leader_log['nb_leader']
                                               , self.bound_n)\
                                - self.estimator(local_rho['hat_rhos'][self.leader[k]]
                                                 , local_rho['nb_trials'][self.leader[k]]
                                                 , leader_log['nb_leader']
                                                 , self.bound_l)
                if value > value_max:
                    value_max = value
                    proposition = swap(self.leader, (k, l), leader_log['remaining'])
        return np.array(proposition), 0

    def estimator(self, hat_rho, nb_trial, nb_total_trial, bound):
        if bound == 'o':
            return self.optimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'p':
            return self.pessimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'a':
            return hat_rho
        else:
            raise ValueError(f'unkwon estimator {bound}')

    def optimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        if self.horizon is None:
            if nb_trial == 0 or nb_total_trial < 3:
                return 1
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
        else:
            threshold = math.log(self.horizon)
        start = start_up(hat_rho, threshold, nb_trial)
        return newton(hat_rho, threshold, nb_trial, start)

    def pessimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        return 1-self.optimistic_index(1-hat_rho, nb_trial, nb_total_trial)

    def get_neighbor(self):
        if self.sigma is None:
            raise NotImplementedError(f'sigma has to be known')
        else:
            hat_pi = self.sigma
        if len(hat_pi) < self.nb_arms:
            new = np.arange(self.nb_arms)
            new[:len(hat_pi)] = hat_pi
            hat_pi = new
        return [(hat_pi[k], hat_pi[l]) for k, l in self.base_neighborhood]

    def get_base_neighbor(self):
        if self.neighbor_type == 'bubble':
            return self.get_bubble_neighbor()
        elif self.neighbor_type == 'qsort':
            return self.get_qsort_neighbor()
        elif self.neighbor_type == 'shell3,1':
            return self.get_shell_neighbor(seq=[3, 1])
        elif self.neighbor_type == 'jumps3':
            return self.get_jumps_neighbor(gap=3)
        elif self.neighbor_type == 'jumps3.1':
            return self.get_jumps_neighbor(gap=3, beg=1)
        raise ValueError(f'unknown neighborhood type {self.neighbor_type}')

    def get_bubble_neighbor(self, hat_pi=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100)
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        res = []
        for k in range(self.nb_positions - 1):
            res.append((hat_pi[k], hat_pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[self.nb_positions - 1], k))
        return res

    def get_qsort_neighbor(self, hat_pi=None, beg=0, end=None):
        """

        Parameters
        ----------
        hat_pi
        beg
        end

        Returns
        -------

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=4, nb_positions=4, neighbor_type='qsort')
        >>> player.get_neighbor([0, 1, 3, 2])
        [(0, 3), (1, 3), (3, 2), (0, 1)]
        >>> player = UniRankFirstPos(nb_arms=10, nb_positions=10, neighbor_type='qsort')
        >>> player.get_neighbor([0, 1, 3, 2, 4, 5, 8, 9, 7, 6])
        [(0, 5), (1, 5), (3, 5), (2, 5), (4, 5), (5, 8), (5, 9), (5, 7), (5, 6), (0, 3), (1, 3), (3, 2), (3, 4), (0, 1), (2, 4), (8, 7), (9, 7), (7, 6), (8, 9)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if end is None:
            end = self.nb_arms
        if len(hat_pi) < end:
            raise NotImplementedError('qsort neighborhood only implemented for nb_arms == nb_positions')
        def rec(hat_pi, beg, end):
            if beg >= end-1:
                return []
            else:
                mid = (beg + end) // 2
                res = [(hat_pi[l], hat_pi[mid]) for l in range(beg, mid)]
                res += [(hat_pi[mid], hat_pi[l]) for l in range(mid+1, end)]
                res += self.get_qsort_neighbor(hat_pi, beg, mid)
                res += self.get_qsort_neighbor(hat_pi, mid+1, end)
                return res
        return rec(hat_pi, beg, end)

    def get_shell_neighbor(self, hat_pi=None, seq=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='shell3,1')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 3), (1, 2), (4, 5), (4, 6), (0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if seq is None:
            seq = [3,1]
        res = []
        for gap in seq:
            for k in range(self.nb_positions - gap):
                res.append((hat_pi[k], hat_pi[k + gap]))
            for k in range(self.nb_positions, self.nb_arms):
                res.append((hat_pi[self.nb_positions - gap], k))
        return res

    def get_jumps_neighbor(self, hat_pi=None, gap=None, beg=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='jumps3')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6), (0, 3)]
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='jumps3.1')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6), (1, 2)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if gap is None:
            gap = 3
        if beg is None:
            beg = 0
        res = []
        for k in range(self.nb_positions - 1):
            res.append((hat_pi[k], hat_pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[self.nb_positions - 1], k))
        for k in range(beg, self.nb_positions - gap, gap):
            res.append((hat_pi[k], hat_pi[k + gap]))
        """
        last = beg + (self.nb_positions - beg) // gap_type * gap_type
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[last], k))
        #"""
        return res

    def update(self, propositions, rewards):
        self.running_t += 1
        self.node_logs[self.leader]['nb_leader'] += 1
        # update statistics
        for k in range(self.nb_positions):
            log = self.shared_logs[tuple(np.sort(propositions[:k]))]
            log['nb_trials'][propositions[k]] += 1
            log['hat_rhos'][propositions[k]] += (rewards[k] - log['hat_rhos'][propositions[k]]) / log['nb_trials'][propositions[k]]

        # shrink memory if too big
        if len(self.node_logs) > self.memory_size:
            min_key = min(self.node_logs.items(), key=lambda k: k[1]['last_play'])[0]
            self.node_logs.pop(min_key)

        # update the leader
        leader_log = self.update_leader()

        # update the leader's neighborhood
        self.list_transpositions = self.get_neighbor()

    def update_leader(self):
        previous_leader_log = self.node_logs[self.leader]
        leader_log = previous_leader_log
        value_max = 0
        for (k, l) in self.list_transpositions:
            if self.sigma is None:
                """
                value = self.estimator(max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l])
                                                  , neighbor_log['nb_trials'], previous_leader_log['nb_leader'], self.lead_n) \
                        - self.estimator(max(previous_leader_log['hat_rhos'][k], previous_leader_log['hat_rhos'][l])
                                                    , previous_leader_log['nb_trials'], previous_leader_log['nb_leader'], self.lead_l)
                """
            else:
                item_l = self.leader[l] if l < self.nb_positions else leader_log['remaining'][l - self.nb_positions]
                local_rho = self.shared_logs.get(tuple(np.sort(self.leader[:k])), self.empty_rho())
                value = self.estimator(local_rho['hat_rhos'][item_l]
                                       , local_rho['nb_trials'][item_l]
                                       , leader_log['nb_leader']
                                       , self.lead_n) \
                        - self.estimator(local_rho['hat_rhos'][self.leader[k]]
                                         , local_rho['nb_trials'][self.leader[k]]
                                         , leader_log['nb_leader']
                                         , self.lead_l)
            if value > value_max:
                value_max = value
                new_leader = swap(self.leader, (k, l), previous_leader_log['remaining'])
        """
        if self.sigma is not None:
            accelerated_change = False
            for (k, l) in self.list_transpositions:
                value = - self.optimistic_index(previous_leader_log['hat_rhos'][k]
                                              , previous_leader_log['nb_trials'], previous_leader_log['nb_leader']) \
                        + self.pessimistic_index(previous_leader_log['hat_rhos'][l]
                                                 , previous_leader_log['nb_trials'], previous_leader_log['nb_leader'])
                if value > value_max:
                    accelerated_change = True
                    value_max = value
                    new_leader = swap(self.leader, (k, l), previous_leader_log['remaining'])
            if accelerated_change:
                print(f'!!! accelerated change from {self.leader} to {new_leader} at iteration {self.running_t}')
                log = self.logs[new_leader]
                log['hat_pi'] = self.sigma

        # """
        if value_max > 0:
            self.leader = new_leader
        leader_log = self.node_logs[self.leader]
        if leader_log['remaining'] is None:
            leader_log['remaining'] = unused(self.leader, self.nb_arms)
        leader_log['last_play'] = self.running_t
        return leader_log


class OSUB_TOP_RANK:
    """
    Source : "Multiple-Play  Bandits  in  the  Position-Based  Model"
    reject sampling with beta preposal
    """

    def __init__(self, nb_arms, nb_positions, T=None, sigma=None, recommended_partition_choice='as much as possible from top to bottom, greedy-specific rule for remaining', neighbor_type='bubble', slight_optimism='log log', fine_grained_partition=False, global_time_for_threshold=False):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param discount_factor: if None, discount factors are inferred from logs every `lag` iterations
        :param lag:
        :param T: number of trials
        :param prior_s:
        :param prior_f:

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>> discount_factor = [1, 0.9, 0.7]
        >>> player = TOP_RANK(nb_arms, discount_factor=discount_factor)

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
        >>> player = TOP_RANK(nb_arms, discount_factor=discount_factor)
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
        >>> player = TOP_RANK(nb_arms, discount_factor=discount_factor)
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
        #recommended_partition_choice = 'max'
        #recommended_partition_choice = 'best merge or best remaining item'
        #recommended_partition_choice = 'as much as possible from top to bottom, specific rule for remaining'
        #recommended_partition_choice = 'as much as possible from top to bottom, greedy-specific rule for remaining'
        #recommended_partition_choice = 'as much as possible from top to bottom'
        #recommended_partition_choice = 'as much individual promotions as possible'

        self.nb_positions = nb_positions
        self.nb_arms = nb_arms
        self.horizon = T
        if sigma is None:
            raise NotImplementedError(f'sigma has to be known')
        self.sigma = sigma
        self.slight_optimism = slight_optimism
        self.fine_grained_partition = fine_grained_partition
        self.recommended_partition_choice = recommended_partition_choice

        self.neighbor_type = neighbor_type
        self.base_neighborhood = self.get_base_neighbor()
        self.global_time_for_threshold = global_time_for_threshold
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game. """
        # clean the log
        self.time = 0
        self.nb_bandit_at_cutoff = np.zeros(self.nb_positions, dtype=np.int)
        self.partition = [list(range(self.nb_arms))]
        self.condensed_partition = (0,) * self.nb_arms
        self.nb_explorations = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.tau_hats = np.zeros([self.nb_arms, self.nb_arms])
        self.nb_diffs = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.G = np.ones([self.nb_arms, self.nb_arms], dtype=np.bool)
        self.tried_transpositions = []
        """
        self.start_order = np.arange(self.nb_arms)
        shuffle(self.start_order)
        """
        self.node_logs = defaultdict(self.empty_node)  # number of time each arm has been the leader, ...

    def empty_node(self):   # to enable pickling
        return {'nb_leader': 0
                }

    def choose_next_arm(self):
        """
        Returns
        -------

        """
        """
        # first recommendations are fixed
        if self.time < self.nb_arms:
            self.recommended_partition = [list(range(self.nb_arms))]
            return self.start_order[(np.arange(self.nb_positions) + self.time) % self.nb_arms], 0
        """
        # choose time for threshold
        if self.global_time_for_threshold:
            time_for_threshold = self.time
        else:
            time_for_threshold = self.node_logs[self.condensed_partition]['nb_leader']

        # choose the partition
        if self.recommended_partition_choice == 'max':
            # Apply the most promising merge of two partitions
            value_max = 0.5
            values = np.zeros(len(self.partition))
            k_beg, k_end = 0, len(self.partition[0])    # interval for current partition
            last_partition = 0
            for j_p in range(1, len(self.partition)):
                k_beg, k_end = k_end, k_end + len(self.partition[j_p])
                if k_beg < self.nb_positions:
                    i_p = j_p-1
                    last_partition = j_p
                else:
                    i_p = last_partition
                value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                            for i, j in zip(self.partition[i_p], self.partition[j_p]))
                values[j_p] = value
                if value > value_max:
                    value_max = value
                    i_p_max = i_p
                    j_p_max = j_p
            if value_max == 0.5:
                self.recommended_partition = self.partition
            else:
                self.nb_bandit_at_cutoff[min(sum(len(P) for P in self.partition[:j_p_max]), self.nb_positions)-1] += 1
                self.recommended_partition = self.partition[:i_p_max] + [self.partition[i_p_max] + self.partition[j_p_max]]
                if j_p_max == i_p_max + 1:
                    self.recommended_partition += self.partition[(j_p_max + 1):]
            if math.ceil(math.log(self.time+1, 2)) != math.ceil(math.log(self.time+2, 2)):
                print('optimism values:', self.time, value_max, [round(v, 3) for v in values])
        elif self.recommended_partition_choice == 'best merge or best remaining item':
            # Apply the most promising option amoung: (1) merge of two displayed partitions, (2) add an undisplayed item to the last displayed partition
            value_max = 0.5
            values = np.zeros(len(self.partition))
            k_beg, k_end = 0, len(self.partition[0])  # interval for current partition
            last_partition = 0
            # go through displayed partitions
            for j_p in range(1, len(self.partition)):
                k_beg, k_end = k_end, k_end + len(self.partition[j_p])
                if k_beg < self.nb_positions:
                    i_p = j_p - 1
                    last_partition = j_p
                    value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                                for i, j in zip(self.partition[i_p], self.partition[j_p]))
                    values[j_p] = value
                    if value > value_max:
                        value_max = value
                        i_p_max = i_p
                        j_p_max = j_p
                        item_max = None
                else:
                    i_p = last_partition
                    values_in_part = [max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                                          for i in self.partition[i_p])
                                      for j in self.partition[j_p]]
                    value = max(values_in_part)
                    values[j_p] = value
                    if value > value_max:
                        value_max = value
                        i_p_max = i_p
                        j_p_max = j_p
                        item_max = self.partition[j_p][np.argmax(values_in_part)]
            if value_max == 0.5:
                self.recommended_partition = self.partition
            else:
                self.nb_bandit_at_cutoff[
                    min(sum(len(P) for P in self.partition[:j_p_max]), self.nb_positions) - 1] += 1
                if item_max is None:
                    self.recommended_partition = self.partition[:i_p_max] + [self.partition[i_p_max] + self.partition[j_p_max]]
                    if j_p_max == i_p_max + 1:
                        self.recommended_partition += self.partition[(j_p_max + 1):]
                else:
                    self.recommended_partition = self.partition[:i_p_max] + [self.partition[i_p_max] + [item_max]]
            if math.ceil(math.log(self.time + 1, 2)) != math.ceil(math.log(self.time + 2, 2)):
                print('optimism values:', self.time, value_max, [round(v, 3) for v in values])
        elif self.recommended_partition_choice == 'as much as possible from top to bottom, specific rule for remaining':
            # Apply all possible merges of two partitions (starting from top)
            values = np.zeros(len(self.partition))
            k_beg, k_end = 0, len(self.partition[0])    # interval for current partition
            last_partition = 0
            self.recommended_partition = [self.partition[0]]
            i_p = 0
            k_beg_i = 0
            self.recommended_partition = []
            # - merge while in the range of positions
            while i_p < len(self.partition) and k_beg_i + len(self.partition[i_p]) < self.nb_positions:
                j_p = i_p + 1
                value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                            for i, j in zip(self.partition[i_p], self.partition[j_p]))
                values[j_p] = value
                if value > 0.5:
                    # merge
                    self.recommended_partition.append(self.partition[i_p] + self.partition[j_p])
                    self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1
                    k_beg_i = k_beg_i + len(self.partition[i_p]) + len(self.partition[j_p])
                    i_p += 2
                else:
                    # do not merge
                    self.recommended_partition.append(self.partition[i_p])
                    k_beg_i = k_beg_i + len(self.partition[i_p])
                    i_p += 1
            # - add the best remaining arm
            if i_p < len(self.partition) and k_beg_i < self.nb_positions:
                self.recommended_partition.append(copy(self.partition[i_p]))
                value_max = 0.
                for j_p in range(i_p + 1, len(self.partition)):
                    for j in self.partition[j_p]:
                        value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                                    for i in self.partition[i_p])
                        values[j_p] = max(value, values[j_p])
                        if value > value_max:
                            value_max = value
                            j_max = j
                if value_max > 0.5:
                    self.recommended_partition[-1].append(j_max)
                    self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1
            #if self.time % 500 == 0 or self.time % 500 == 1 :
            if math.ceil(math.log(self.time + 1, 2)) != math.ceil(math.log(self.time + 2, 2)):
                print('optimism values:', self.time, values.max(), [round(v, 3) for v in values])
                print(self.partition)
                print(self.recommended_partition)
        elif self.recommended_partition_choice == 'as much as possible from top to bottom, greedy-specific rule for remaining':
            # Apply all possible merges of two partitions (starting from top)
            values = np.zeros(len(self.partition))
            k_beg, k_end = 0, len(self.partition[0])  # interval for current partition
            last_partition = 0
            self.recommended_partition = [self.partition[0]]
            i_p = 0
            k_beg_i = 0
            self.recommended_partition = []
            # - merge while in the range of positions
            while i_p < len(self.partition) and k_beg_i + len(self.partition[i_p]) < self.nb_positions:
                j_p = i_p + 1
                value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                            for i, j in zip(self.partition[i_p], self.partition[j_p]))
                values[j_p] = value
                if value > 0.5:
                    # merge
                    self.recommended_partition.append(self.partition[i_p] + self.partition[j_p])
                    self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1
                    k_beg_i = k_beg_i + len(self.partition[i_p]) + len(self.partition[j_p])
                    i_p += 2
                else:
                    # do not merge
                    self.recommended_partition.append(self.partition[i_p])
                    k_beg_i = k_beg_i + len(self.partition[i_p])
                    i_p += 1
            # - add the best remaining arm
            if i_p < len(self.partition) and k_beg_i < self.nb_positions:
                self.recommended_partition.append(copy(self.partition[i_p]))
                value_max = 0.
                for j_p in range(i_p + 1, len(self.partition)):
                    for j in self.partition[j_p]:
                        value = max(
                            self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                            for i in self.partition[i_p])
                        values[j_p] = max(value, values[j_p])
                        if value > value_max:
                            value_max = value
                            j_max = j
                    if value_max > 0.5:
                        self.recommended_partition[-1].append(j_max)
                        self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1
                        break
            if self.time % 500 == 0 or self.time % 500 == 1 :
            # if math.ceil(math.log(self.time + 1, 2)) != math.ceil(math.log(self.time + 2, 2)):
                print('optimism values:', self.time, values.max(), [round(v, 3) for v in values])
                print(self.partition)
                print(self.recommended_partition)
        elif self.recommended_partition_choice == 'as much as possible from top to bottom':
            # Apply all possible merges of two partitions (starting from top)
            values = np.zeros(len(self.partition))
            merged = np.zeros(len(self.partition), dtype=np.bool)
            k_beg, k_end = 0, len(self.partition[0])  # interval for current partition
            last_partition = 0
            self.recommended_partition = [self.partition[0]]
            for j_p in range(1, len(self.partition)):
                k_beg, k_end = k_end, k_end + len(self.partition[j_p])
                if k_beg < self.nb_positions:
                    i_p = j_p - 1
                    last_partition = j_p
                else:
                    i_p = last_partition
                value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                            for i, j in zip(self.partition[i_p], self.partition[j_p]))
                values[j_p] = value
                if value > 0.5 and not merged[i_p]:
                    merged[i_p] = True
                    merged[j_p] = True
                    Pi = self.recommended_partition.pop()
                    self.recommended_partition.append(Pi + self.partition[j_p])
                    self.nb_bandit_at_cutoff[min(k_end, self.nb_positions) - 1] += 1
                    if k_beg >= self.nb_positions:
                        break
                else:
                    if k_beg < self.nb_positions:
                        self.recommended_partition.append(self.partition[j_p])
            if self.time % 500 == 0 or self.time % 500 == 1 :
            #if math.ceil(math.log(self.time + 1, 2)) != math.ceil(math.log(self.time + 2, 2)):
                print('optimism values:', self.time, values.max(), [round(v, 3) for v in values])
                print(self.partition)
                print(self.recommended_partition)
        elif self.recommended_partition_choice == 'as much individual promotions as possible':
            # Apply all possible merges of (2k, 2k+1) partitions on even turns, and of (2k-1, 2k) partitions on odd turns
            values = np.zeros(len(self.partition))
            if self.time % 2 == 0:
                i_p = 0
                k_beg_i = 0
                self.recommended_partition = []
            else:
                i_p = 1
                k_beg_i = len(self.partition[0])
                self.recommended_partition = [self.partition[0]]
            while i_p < len(self.partition) and k_beg_i + len(self.partition[i_p]) < self.nb_positions:
                j_p = i_p + 1
                Pi = copy(self.partition[i_p])
                Pj = copy(self.partition[j_p])
                for j in self.partition[j_p]:
                    value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                                for i in self.partition[i_p])
                    values[j_p] = max(value, values[j_p])
                    if value > 0.5:
                        Pi.append(j)
                        Pj.remove(j)
                        self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1/len(self.partition[j_p])
                self.recommended_partition.append(Pi)
                self.recommended_partition.append(Pj)
                i_p += 2
                k_beg_i = k_beg_i + len(Pi) + len(Pj)
            if i_p < len(self.partition) and k_beg_i < self.nb_positions:
                self.recommended_partition.append(copy(self.partition[i_p]))
                value_max = 0.5
                #"""
                for j_p in range(i_p + 1, len(self.partition)):
                    for j in self.partition[j_p]:
                        value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                                    for i in self.partition[i_p])
                        values[j_p] = max(value, values[j_p])
                        if value > value_max:
                            value_max = value
                            j_max = j
                            j_p_max = j_p
                if value_max > 0.5:
                    self.recommended_partition[-1].append(j_max)
                    self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1 / len(self.partition[j_p_max])
                """
                for j_p in range(i_p + 1, len(self.partition)):
                    for j in self.partition[j_p]:
                        value = max(self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
                                    for i in self.partition[i_p])
                        values[j_p] = max(value, values[j_p])
                        if value > 0.5:
                            self.recommended_partition[-1].append(j)
                            self.nb_bandit_at_cutoff[min(k_beg_i + len(self.partition[i_p]), self.nb_positions) - 1] += 1 / len(self.partition[j_p])
                #"""
            if self.time % 500 == 0 or self.time % 500 == 1 :
            #if math.ceil(math.log(self.time + 1, 2)) != math.ceil(math.log(self.time + 2, 2)):
                print('optimism values:', self.time, values.max(), [round(v, 3) for v in values])
                print(self.partition)
                print(self.recommended_partition)

        # pick a leader according to current partition
        recommendation = []
        for part in self.recommended_partition:
            shuffle(part)
            recommendation = recommendation + part

        #TODO: assume currently that sigma = [0, 1, 2, ...]
        return np.array(recommendation[:self.nb_positions]), 0

    def get_reward_arm(self, i, propositions, rewards):
        propositions_list = list(propositions)
        if i in propositions_list:
            pos = propositions_list.index(i)
            rew = rewards[pos]
        else:
            rew = 0
        return rew

    def update_matrix_and_graph(self, propositions, rewards):
        k_beg, k_end = -1, 0    # interval for current partition
        for Pc in self.recommended_partition:
            k_beg, k_end = k_end, k_end + len(Pc) #TODO remove 'get_reward' by using (k_beg, k_end)
            if k_beg >= self.nb_positions:
                break
            for i, j in product(Pc, Pc):
                self.nb_explorations[i][j] += 1
                # --- update matrix ---
                C_i = self.get_reward_arm(i, propositions, rewards)
                C_j = self.get_reward_arm(j, propositions, rewards)
                # print('rewards', C_i,C_j)
                if C_i != C_j:
                    self.nb_diffs[i][j] += abs(C_i - C_j)
                    self.tau_hats[i][j] += (C_i - C_j - self.tau_hats[i][j]) / self.nb_diffs[i][j]

        # --- update graph ---
        if self.slight_optimism == 'tau hat':
            self.G = self.tau_hats >= 0
        if self.slight_optimism == 'log log':
            self.G = self.tau_hats + np.sqrt(np.log(np.log(self.time + 3)) / self.nb_diffs) >= 0
        if self.slight_optimism == 'sqrt log':
            self.G = self.tau_hats + np.sqrt(np.sqrt(np.log(self.time + 3)) / self.nb_diffs) >= 0
        if self.slight_optimism == 'log0.8':
            self.G = self.tau_hats + (np.sqrt(np.log(self.time + 3)) / self.nb_diffs) ** 0.8 >= 0
        if self.slight_optimism == 'pure loglog':
            self.G = self.tau_hats + (np.sqrt(np.log(self.time + 3)) / self.nb_diffs) ** 0.8 >= 0
        self.G[np.arange(self.nb_arms), np.arange(self.nb_arms)] = True

    def partition_arm(self):
        """

        Returns
        -------

        Examples
        -------
        >>> player = OSUB_TOP_RANK(nb_arms=4, nb_positions=2, T=100, sigma=np.arange(2), fine_grained_partition=False)
        >>> tau_hats = 0.1*np.array([[ 0,  2,  1,  3],
        ...                      [-2,  0, -2,  0],
        ...                      [-1,  2,  0,  0],
        ...                      [-3,  0,  0,  0]])
        >>> nb_diffs, t = 10, 5
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True,  True,  True,  True],
               [ True,  True,  True,  True],
               [ True,  True,  True,  True],
               [False,  True,  True,  True]])
        >>> player.partition_arm()
        >>> player.partition
        [[0, 1, 2], [3]]
        >>> player.condensed_partition
        (0, 0, 0, 1)
        >>> nb_diffs, t = 100, 5
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True,  True,  True,  True],
               [False,  True, False,  True],
               [False,  True,  True,  True],
               [False,  True,  True,  True]])
        >>> player.partition_arm()
        >>> player.partition
        [[0], [2, 3], [1]]
        >>> player.condensed_partition
        (0, 2, 1, 1)
        >>> tau_hats = 0.1*np.array([[ 0,  2,  1,  3],
        ...                      [-2,  0, -2,  3],
        ...                      [-1,  2,  0, -1],
        ...                      [-3, -3,  1,  0]])
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True,  True,  True,  True],
               [False,  True, False,  True],
               [False,  True,  True, False],
               [False, False,  True,  True]])
        >>> player.partition_arm()
        >>> player.partition
        [[0], [1, 2, 3]]
        >>> player.condensed_partition
        (0, 1, 1, 1)
        >>> tau_hats = 0.1*np.array([[ 0, -2, -1, -3],
        ...                      [ 2,  0, -2,  3],
        ...                      [ 1,  2,  0, -1],
        ...                      [ 3, -3,  1,  0]])
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True, False, False, False],
               [ True,  True, False,  True],
               [ True,  True,  True, False],
               [ True, False,  True,  True]])
        >>> player.partition_arm()
        >>> player.partition
        [[0, 1, 2, 3]]
        >>> player.condensed_partition
        (0, 0, 0, 0)
        >>> player = OSUB_TOP_RANK(nb_arms=4, nb_positions=2, T=100, sigma=np.arange(2), fine_grained_partition=True)
        >>> nb_diffs, t = 100, 5
        >>> tau_hats = 0.1*np.array([[ 0,  2,  1,  3],
        ...                      [-2,  0, -2,  3],
        ...                      [-1,  2,  0, -1],
        ...                      [-3, -3,  1,  0]])
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True,  True,  True,  True],
               [False,  True, False,  True],
               [False,  True,  True, False],
               [False, False,  True,  True]])
        >>> player.partition_arm()
        >>> player.partition
        [[0], [1, 2, 3]]
        >>> player.condensed_partition
        (0, 1, 1, 1)
        >>> tau_hats = 0.1*np.array([[ 0, -2, -1, -3],
        ...                      [ 2,  0, -2,  3],
        ...                      [ 1,  2,  0, -1],
        ...                      [ 3, -3,  1,  0]])
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True, False, False, False],
               [ True,  True, False,  True],
               [ True,  True,  True, False],
               [ True, False,  True,  True]])
        >>> player.partition_arm()
        >>> player.partition
        [[1, 2, 3], [0]]
        >>> player.condensed_partition
        (1, 0, 0, 0)
        >>> nb_diffs, t = 0, 0
        >>> player.G = tau_hats + np.sqrt(np.log(np.log(t + 3)) / nb_diffs) >= 0
        >>> player.G
        array([[ True,  True,  True,  True],
               [ True,  True,  True,  True],
               [ True,  True,  True,  True],
               [ True,  True,  True,  True]])
        """
        remaining_ind = list(range(self.nb_arms))
        self.partition = []
        while remaining_ind:
            # --- create a part with good items ---
            ind = np.array(remaining_ind)
            P = ind[np.argwhere(self.G[ind, :][:, ind].all(axis=1)).reshape(-1)]
            self.partition.append(list(P))
            remaining_ind = list(set(remaining_ind) ^ set(P))
            #print(remaining_ind, P, self.partition)

            if len(P) == 0 and remaining_ind:
                self.partition.pop()
                self.partition.append(list(remaining_ind))
                break

        if self.fine_grained_partition:
            #print(self.partition)
            remaining_ind = self.partition.pop()
            last_parts = []
            while remaining_ind:
                # --- create a part with bad items ---
                ind = np.array(remaining_ind)
                P = ind[np.argwhere(self.G[ind, :][:, ind].sum(axis=1) == 1).reshape(-1)]
                last_parts.insert(0, list(P))
                remaining_ind = list(set(remaining_ind) ^ set(P))
                #print('bad', remaining_ind, P, last_parts)

                if len(P) == 0 and remaining_ind:
                    del last_parts[0]
                    self.partition.append(list(remaining_ind))
                    break
            self.partition += last_parts

        # build the condensed representation of the partition
        condensed_partition = np.zeros(self.nb_arms, dtype=np.int)
        for i_p, parts in enumerate(self.partition):
            condensed_partition[parts] = i_p
        self.condensed_partition = tuple(condensed_partition)
        self.node_logs[self.condensed_partition]['nb_leader'] += 1

    def update(self, propositions, rewards):
        self.time += 1
        self.update_matrix_and_graph(propositions, rewards)
        self.partition_arm()

    def get_param_estimation(self):
        raise NotImplementedError()

    def estimator(self, hat_rho, nb_trial, nb_total_trial, bound):
        if bound == 'o':
            return self.optimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'p':
            return self.pessimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'a':
            return hat_rho
        else:
            raise ValueError(f'unkwon estimator {bound}')

    def optimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        """

        Parameters
        ----------
        hat_rho
        nb_trial
        nb_total_trial

        Examples
        -------
        >>> player = OSUB_TOP_RANK(nb_arms=4, nb_positions=2, sigma=np.arange(2), )
        >>> player.optimistic_index(0.5 + 0.5 * 0, 100, 100)
        0.7048432465681482
        >>> player.optimistic_index(0.5 + 0.5 * 0, 0, 100)
        1
        """
        if self.horizon is None:
            if nb_trial == 0 or nb_total_trial < 3:
                return 1
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
        else:
            threshold = math.log(self.horizon)
        start = start_up(hat_rho, threshold, nb_trial)
        return newton(hat_rho, threshold, nb_trial, start)

    def pessimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        return 1-self.optimistic_index(1-hat_rho, nb_trial, nb_total_trial)

    def get_neighbor(self):
        if self.sigma is None:
            raise NotImplementedError(f'sigma has to be known')
        else:
            hat_pi = self.sigma
        if len(hat_pi) < self.nb_arms:
            new = np.arange(self.nb_arms)
            new[:len(hat_pi)] = hat_pi
            hat_pi = new
        return [(hat_pi[k], hat_pi[l]) for k, l in self.base_neighborhood]

    def get_base_neighbor(self):
        if self.neighbor_type == 'bubble':
            return self.get_bubble_neighbor()
        elif self.neighbor_type == 'qsort':
            return self.get_qsort_neighbor()
        elif self.neighbor_type == 'shell3,1':
            return self.get_shell_neighbor(seq=[3, 1])
        elif self.neighbor_type == 'jumps3':
            return self.get_jumps_neighbor(gap=3)
        elif self.neighbor_type == 'jumps3.1':
            return self.get_jumps_neighbor(gap=3, beg=1)
        raise ValueError(f'unknown neighborhood type {self.neighbor_type}')

    def get_bubble_neighbor(self, hat_pi=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100)
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        res = []
        for k in range(self.nb_positions - 1):
            res.append((hat_pi[k], hat_pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[self.nb_positions - 1], k))
        return res

    def get_qsort_neighbor(self, hat_pi=None, beg=0, end=None):
        """

        Parameters
        ----------
        hat_pi
        beg
        end

        Returns
        -------

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=4, nb_positions=4, neighbor_type='qsort')
        >>> player.get_neighbor([0, 1, 3, 2])
        [(0, 3), (1, 3), (3, 2), (0, 1)]
        >>> player = UniRankFirstPos(nb_arms=10, nb_positions=10, neighbor_type='qsort')
        >>> player.get_neighbor([0, 1, 3, 2, 4, 5, 8, 9, 7, 6])
        [(0, 5), (1, 5), (3, 5), (2, 5), (4, 5), (5, 8), (5, 9), (5, 7), (5, 6), (0, 3), (1, 3), (3, 2), (3, 4), (0, 1), (2, 4), (8, 7), (9, 7), (7, 6), (8, 9)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if end is None:
            end = self.nb_arms
        if len(hat_pi) < end:
            raise NotImplementedError('qsort neighborhood only implemented for nb_arms == nb_positions')
        def rec(hat_pi, beg, end):
            if beg >= end-1:
                return []
            else:
                mid = (beg + end) // 2
                res = [(hat_pi[l], hat_pi[mid]) for l in range(beg, mid)]
                res += [(hat_pi[mid], hat_pi[l]) for l in range(mid+1, end)]
                res += self.get_qsort_neighbor(hat_pi, beg, mid)
                res += self.get_qsort_neighbor(hat_pi, mid+1, end)
                return res
        return rec(hat_pi, beg, end)

    def get_shell_neighbor(self, hat_pi=None, seq=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='shell3,1')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 3), (1, 2), (4, 5), (4, 6), (0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if seq is None:
            seq = [3,1]
        res = []
        for gap in seq:
            for k in range(self.nb_positions - gap):
                res.append((hat_pi[k], hat_pi[k + gap]))
            for k in range(self.nb_positions, self.nb_arms):
                res.append((hat_pi[self.nb_positions - gap], k))
        return res

    def get_jumps_neighbor(self, hat_pi=None, gap=None, beg=None):
        """

        Examples
        -------
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='jumps3')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6), (0, 3)]
        >>> player = UniRankFirstPos(nb_arms=7, nb_positions=5, T=100, neighbor_type='jumps3.1')
        >>> player.get_neighbor([0, 1, 4, 3, 2])
        [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5), (2, 6), (1, 2)]
        """
        if hat_pi is None:
            hat_pi = np.arange(self.nb_positions)
        if gap is None:
            gap = 3
        if beg is None:
            beg = 0
        res = []
        for k in range(self.nb_positions - 1):
            res.append((hat_pi[k], hat_pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[self.nb_positions - 1], k))
        for k in range(beg, self.nb_positions - gap, gap):
            res.append((hat_pi[k], hat_pi[k + gap]))
        """
        last = beg + (self.nb_positions - beg) // gap_type * gap_type
        for k in range(self.nb_positions, self.nb_arms):
            res.append((hat_pi[last], k))
        #"""
        return res

    def print_info(self):
        leader = self.partition
        print('leader', leader)
        print(self.nb_explorations)
        print(self.nb_diffs)
        print(self.tau_hats)
        print(self.G)




class DCGUniRank(UniRankFirstPos):
    """
    """

    def __init__(self, nb_arms, nb_positions, T=None, gamma=None, position_weights=None, memory_size=np.inf):
        """
        T : int or None
            Either the horizon of a game, or None (unknown horizon). Used to set the confidence interval size with
            log(T) (known horizon) or log(t) + 3 log(log(t)) with t the number of time current arm has been the leader.
        """
        super().__init__(nb_arms=nb_arms, nb_positions=nb_positions, T=T, gamma=gamma, memory_size=memory_size)
        if position_weights is None:
            C = 10
            position_weights = (1-(1/C)**(nb_positions-np.arange(nb_positions))) / (1-1/C)
        self.position_weights = np.array(position_weights)

    def empty(self):   # to enable pickling
        return {'remaining': None,  # list of items not displayed by the leader
                'nb_leader': 0,
                'hat_rhos': np.zeros(self.nb_arms),    # rho defaults to 0 for non-existing positions
                'hat_pi': None,
                'nb_trials': 0,
                'last_play': 0,
                'hat_leader_index': 0
                }

    def choose_next_arm(self):
        leader_log = self.logs[self.leader]
        proposition = self.leader
        if leader_log['nb_leader'] % self.gamma != 0:
            for (k, l) in self.list_transpositions:
                neighbor = swap(self.leader, (k, l), leader_log['remaining'])
                neighbor_log = self.logs.get(neighbor, None)
                if neighbor_log is None:
                    value = 1
                else:
                    value = self.optimistic_index(max(neighbor_log['hat_rhos'][k], neighbor_log['hat_rhos'][l])
                                                  , neighbor_log['nb_trials'], leader_log['nb_leader'])\
                            - self.optimistic_index(max(leader_log['hat_rhos'][k], leader_log['hat_rhos'][l])
                                                    , leader_log['nb_trials'], leader_log['nb_leader'])
                if value > 0:
                    proposition = neighbor
                    break
        return np.array(proposition), 0

    def leader_index(self, hat_rhos, pi):
        """

        Parameters
        ----------
        hat_rhos
        pi

        Returns
        -------

        Examples
        -------
        >>> import numpy as np
        >>> player = DCGUniRank(nb_arms=7, nb_positions=5, T=100)
        >>> hat_rhos = np.array([0.9, 0.3, 0.5, 0.7, 0.6])
        >>> pi = np.argsort(-hat_rhos)
        >>> pi
        array([0, 3, 4, 2, 1])
        >>> 0.9 + 0.7/2 + 0.6/3 + 0.5/4 + 0.3/5
        1.635
        >>> player.leader_index(hat_rhos, pi)
        1.635
        >>> 0.9 + 0.3/2 + 0.5/3 + 0.7/4 + 0.6/5
        1.5116666666666667
        >>> player.leader_index(hat_rhos, np.arange(5))
        1.5116666666666667
        """
        return np.sum(hat_rhos[pi] * self.position_weights)

    def update(self, propositions, rewards):
        self.running_t += 1
        self.logs[self.leader]['nb_leader'] += 1
        # update statistics
        log = self.logs[tuple(propositions)]
        log['nb_trials'] += 1
        log['last_play'] = self.running_t
        log['hat_rhos'][:self.nb_positions] += (rewards - log['hat_rhos'][:self.nb_positions]) / log['nb_trials']
        log['hat_pi'] = np.argsort(-log['hat_rhos'][:self.nb_positions])
        log['hat_leader_index'] = self.leader_index(log['hat_rhos'], log['hat_pi'])

        # shrink memory if too big
        if len(self.logs) > self.memory_size:
            min_key = min(self.logs.items(), key=lambda k: k[1]['last_play'])[0]
            self.logs.pop(min_key)

        # update the leader
        #TODO: speed-up by having a heap on tested rankings with score corresponding to 'hat_leader_index'
        # would work as that score is updated only for the displayed recommmendation
        leader_item = max(self.logs.items(), key=lambda k: k[1]['hat_leader_index'])
        self.leader = leader_item[0]
        if leader_item[1]['remaining'] is None:
            leader_item[1]['remaining'] = unused(self.leader, self.nb_arms)


        # update the leader's neighborhood
        pi = leader_item[1]['hat_pi']
        self.list_transpositions = []
        for k in range(self.nb_positions - 1):
            self.list_transpositions.append((pi[k], pi[k + 1]))
        for k in range(self.nb_positions, self.nb_arms):
            self.list_transpositions.append((pi[self.nb_positions - 1], k))


class UniGRAB:
    """
    the leader is one of the best recommendations
    """

    def __init__(self, nb_arms, nb_positions, T=None, potential_explore_exploit='best', undisplayed_explore_exploit='best', pure_explore='all', global_time_for_threshold=False, seed=42):
        """
        One of both `discount_facor` and `nb_positions` has to be defined.

        :param nb_arms:
        :param nb_positions:
        :param discount_factor: if None, discount factors are inferred from logs every `lag` iterations
        :param lag:
        :param T: number of trials
        :param prior_s:
        :param prior_f:

        >>> import numpy as np
        >>> nb_arms = 10
        >>> nb_choices = 3
        >>> discount_factor = [1, 0.9, 0.7]
        >>> player = UniGRAB(nb_arms, discount_factor=discount_factor)

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
        >>> player = UniGRAB(nb_arms, discount_factor=discount_factor)
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
        >>> player = UniGRAB(nb_arms, discount_factor=discount_factor)
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
        #potential_explore_exploit = 'best'
        #potential_explore_exploit = 'first'
        #potential_explore_exploit = 'even-odd'
        #potential_explore_exploit = 'as_much_as_possible_from_top_to_bottom'

        #pure_explore = 'all'
        #pure_explore = 'focused'

        #undisplayed_explore_exploit = 'best'
        #undisplayed_explore_exploit = 'all_potentials'

        self.nb_positions = nb_positions
        self.nb_arms = nb_arms
        self.horizon = T
        self.potential_explore_exploit = potential_explore_exploit
        self.undisplayed_explore_exploit = undisplayed_explore_exploit
        self.pure_explore = pure_explore

        self.global_time_for_threshold = global_time_for_threshold
        self.rng = np.random.default_rng(seed)
        self.clean()

    def clean(self):
        """ Clean log data. /To be ran before playing a new game.

        Examples
        --------
        """
        # clean the log
        self.time = 0
        self.nb_pure_explore = 0
        self.nb_explorations = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.nb_explorations_at = np.zeros(self.nb_positions+1, dtype=np.int)
        self.tau_hats = np.zeros([self.nb_arms, self.nb_arms])
        self.nb_diffs = np.zeros([self.nb_arms, self.nb_arms], dtype=np.int)
        self.G = np.ones([self.nb_arms, self.nb_arms], dtype=np.bool)   # G[i, j] iff i more atttractive or equal to j (given tau_hats)
        self.node_logs = defaultdict(self.empty_node)  # number of time each arm has been the leader, ...
        # initial recommendations
        self.set_pure_explore_recommendations()

    def empty_node(self):   # to enable pickling
        return {'nb_leader': 0
                }

    def get_exploration_perm(self, exploration_inds):
        exploration_perm = np.arange(self.nb_arms)
        for k in exploration_inds:
            k1 = min(k - 1, self.nb_positions-1)
            exploration_perm[k1], exploration_perm[k] = exploration_perm[k], exploration_perm[k1]
        return exploration_perm

    def set_pure_explore_recommendations(self):
        """

        Examples
        --------

        >>> player = UniGRAB(nb_arms=4, nb_positions=2, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [2 3 0 1] [0] [1]
        [3 2 0 1] [0] [1]
        [0 1 2 3] [0] [1]
        [1 0 2 3] [0] [1]
        [3 1 0 2] [0] [1]
        [1 3 0 2] [0] [1]
        [0 3 2 1] [0] [1]
        [3 0 2 1] [0] [1]
        [1 2 0 3] [0] [1]
        [2 1 0 3] [0] [1]

        >>> player = UniGRAB(nb_arms=4, nb_positions=3, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [1 2 0 3] [0 2] [1 3]
        [2 1 3 0] [0 2] [1 3]
        [0 1 2 3] [0 2] [1 3]
        [1 0 3 2] [0 2] [1 3]
        [0 2 3 1] [0 2] [1 3]
        [2 0 1 3] [0 2] [1 3]

        >>> player = UniGRAB(nb_arms=4, nb_positions=4, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [1 2 0 3] [0 2] [1 3]
        [2 1 3 0] [0 2] [1 3]
        [0 1 2 3] [0 2] [1 3]
        [1 0 3 2] [0 2] [1 3]
        [0 2 3 1] [0 2] [1 3]
        [2 0 1 3] [0 2] [1 3]

        >>> player = UniGRAB(nb_arms=3, nb_positions=1, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [1 2 0] [0] [1]
        [2 1 0] [0] [1]
        [0 1 2] [0] [1]
        [1 0 2] [0] [1]
        [0 2 1] [0] [1]
        [2 0 1] [0] [1]

        >>> player = UniGRAB(nb_arms=3, nb_positions=2, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [1 2 0] [0] [1]
        [2 1 0] [0] [1]
        [0 1 2] [0] [1]
        [1 0 2] [0] [1]
        [0 2 1] [0] [1]
        [2 0 1] [0] [1]

        >>> player = UniGRAB(nb_arms=3, nb_positions=3, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [1 2 0] [0] [1]
        [2 1 0] [0] [1]
        [0 1 2] [0] [1]
        [1 0 2] [0] [1]
        [0 2 1] [0] [1]
        [2 0 1] [0] [1]

        >>> player = UniGRAB(nb_arms=5, nb_positions=3, seed=42)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [2 3 1 4 0] [0 2] [1 3]
        [3 2 4 1 0] [0 2] [1 3]
        [0 1 3 4 2] [0 2] [1 3]
        [1 0 4 3 2] [0 2] [1 3]
        [0 2 3 1 4] [0 2] [1 3]
        [2 0 1 3 4] [0 2] [1 3]
        [0 3 4 2 1] [0 2] [1 3]
        [3 0 2 4 1] [0 2] [1 3]
        [1 2 0 4 3] [0 2] [1 3]
        [2 1 4 0 3] [0 2] [1 3]
        """
        # TODO: ? update to consider all permutations ?
        self.nb_pure_explore += 1
        nb_pos_for_exploration = min((self.nb_positions+1)//2*2, self.nb_arms)
        self.next_full_recommendations = []
        self.next_exploration_pairs = []
        exploration_inds = np.arange(1, nb_pos_for_exploration, 2).reshape((1, -1))
        exploration_pairs = np.concatenate((exploration_inds-1, exploration_inds), axis=0)
        exploration_perm = self.get_exploration_perm(exploration_inds.reshape(-1))

        n2 = (2 * self.nb_arms - nb_pos_for_exploration + 1) // 2  # add items to couple with non-recommended items
        n = 2 * n2
        for i in range(n - 1):
            # couples
            j = i + n2 - 1
            k = j + n2
            couples = np.zeros((2, n2), dtype=np.int8)
            couples[0, 1:] = np.arange(i, j) % (n - 1) + 1
            couples[1, -1::-1] = np.arange(j, k) % (n - 1) + 1
            couples = couples[:, np.argsort((couples >= self.nb_arms).sum(axis=0))]     # non-displayed items in last positions

            # recommendation (random order on couples)
            n_true_couples = ((couples >= self.nb_arms).sum(axis=0) == 0).sum()
            random_order = self.rng.permutation(n_true_couples)
            couples[:, :n_true_couples] = couples[:, random_order]
            full_recommendation = couples.reshape(-1, order='F')
            full_recommendation = full_recommendation[full_recommendation < self.nb_arms]     # restrict to displayed items
            self.next_full_recommendations += [full_recommendation, full_recommendation[exploration_perm]]
            self.next_exploration_pairs += [exploration_pairs, exploration_pairs]

    def set_extremely_focused_pure_explore_recommendations(self, leader):
        """

        Examples
        --------

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=4, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[3][4] = False
        >>> player.G[2][4] = False
        >>> player.G[2][5] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [1] [2]
        [0 2 1 3 4 5] [1] [2]
        [0 1 4 2 3 5] [1] [2]
        [0 4 1 2 3 5] [1] [2]
        [0 2 4 1 3 5] [1] [2]
        [0 4 2 1 3 5] [1] [2]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[2][4] = False
        >>> player.G[2][5] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [1] [2]
        [0 2 1 3 4 5] [1] [2]
        [0 1 4 2 3 5] [1] [2]
        [0 4 1 2 3 5] [1] [2]
        [0 2 4 1 3 5] [1] [2]
        [0 4 2 1 3 5] [1] [2]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=2, seed=42)
        >>> player.G[1][4] = False
        >>> player.G[4][3] = False
        >>> player.G[4][5] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 4 3 5] [2] [3]
        [0 4 2 1 3 5] [2] [3]
        [0 1 2 3 4 5] [2] [3]
        [0 3 2 1 4 5] [2] [3]
        [0 4 2 3 1 5] [2] [3]
        [0 3 2 4 1 5] [2] [3]
        """
        # TODO: ? remove or update ?
        raise NotImplementedError('current implementation does not work with last test')
        self.nb_pure_explore += 1

        # first position at which previous item (denoted $i$) seems strictly less attractive than current item (denoted $j$): i < j
        k = 1 + np.argwhere(np.logical_not(
                    np.concatenate((self.G[leader[:(self.nb_positions - 1)], leader[1:self.nb_positions]],
                                    self.G[leader[self.nb_positions - 1], leader[self.nb_positions:]]))
                                   )
                    )[0, 0]
        k1 = min(k - 1, self.nb_positions - 1)
        i = leader[k1]
        j = leader[k]

        # one position such that the item (denoted $l$) seems less attractive or equal to $i$ and strictly more attractive than $j$: i >= l > j
        k2 = k1 + 1 + np.argwhere(self.G[i, leader[(k1+1):]] & np.logical_not(self.G[j, leader[(k1+1):]]))[0, 0]
        l = leader[k2]

        # to ease the definition of the recommendations when j and k are not displayed given the leader
        if k2 < k:
            leader[k], leader[k2] = leader[k2], leader[k]
            k, k2 = k2, k

        # recommendations (to update the comparisons i-j, i-l, and j-l)
        # the comparison is done in positions (k1,k)
        # the recommendation up to k-2 is not change, and we consider three basic recommmendations:
        # * [..., i, ..., j, remaining unchanged]
        # * [..., i, ..., k, j, remaining without k]
        # * [..., j, ..., k, i, remaining without k]
        exploration_perm = np.arange(self.nb_arms)
        exploration_perm[k1] = k
        exploration_perm[k] = k1
        self.next_full_recommendations = [leader, leader[exploration_perm]]
        reco = np.concatenate((leader[:k1], [i], leader[(k1+1):k], [l, j], leader[(k+1):k2], leader[(k2+1):]))
        self.next_full_recommendations += [reco, reco[exploration_perm]]
        reco = np.concatenate((leader[:k1], [j], leader[(k1+1):k], [l, i], leader[(k+1):k2], leader[(k2+1):]))
        self.next_full_recommendations += [reco, reco[exploration_perm]]
        self.next_exploration_pairs = [np.array([[k1], [k]])] * 6

    def set_focused_pure_explore_recommendations(self, leader):
        """

        Examples
        --------

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=6, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[2][3] = False
        >>> player.G[3][4] = False
        >>> player.G[3][5] = False
        >>> player.G[2][5] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [1 1 2] [2 3 3]
        [0 1 3 2 4 5] [1 1 2] [2 3 3]
        [0 2 1 3 4 5] [1 1 2] [2 3 3]
        [0 2 3 1 4 5] [1 1 2] [2 3 3]
        [0 3 1 2 4 5] [1 1 2] [2 3 3]
        [0 3 2 1 4 5] [1 1 2] [2 3 3]
        [0 1 2 5 3 4] [1 1 2] [2 3 3]
        [0 1 5 2 3 4] [1 1 2] [2 3 3]
        [0 2 1 5 3 4] [1 1 2] [2 3 3]
        [0 2 5 1 3 4] [1 1 2] [2 3 3]
        [0 5 1 2 3 4] [1 1 2] [2 3 3]
        [0 5 2 1 3 4] [1 1 2] [2 3 3]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=4, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[2][3] = False
        >>> player.G[3][4] = False
        >>> player.G[3][5] = False
        >>> player.G[2][5] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [1 1 2 3] [2 3 3 5]
        [0 1 3 2 4 5] [1 1 2 2] [2 3 3 5]
        [0 2 1 3 4 5] [1 1 2 3] [2 3 3 5]
        [0 2 3 1 4 5] [1 1 2 2] [2 3 3 5]
        [0 3 1 2 4 5] [1 1 2 1] [2 3 3 5]
        [0 3 2 1 4 5] [1 1 2 1] [2 3 3 5]
        [0 1 2 5 3 4] [1 1 2 3] [2 3 3 4]
        [0 1 5 2 3 4] [1 1 2 2] [2 3 3 4]
        [0 2 1 5 3 4] [1 1 2 3] [2 3 3 4]
        [0 2 5 1 3 4] [1 1 2 2] [2 3 3 4]
        [0 5 1 2 3 4] [1 1 2 1] [2 3 3 4]
        [0 5 2 1 3 4] [1 1 2 1] [2 3 3 4]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[2][3] = False
        >>> player.G[3][4] = False
        >>> player.G[3][5] = False
        >>> player.G[2][5] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 5 4] [1 1 1 2 2] [2 3 4 3 4]
        [0 2 1 3 5 4] [1 1 1 2 2] [2 3 4 3 4]
        [0 1 3 2 5 4] [1 1 2 2] [2 3 3 4]
        [0 3 1 2 5 4] [1 2 1 1] [2 3 3 4]
        [0 2 3 1 5 4] [1 1 2 2] [2 3 3 4]
        [0 3 2 1 5 4] [1 2 1 1] [2 3 3 4]
        [0 1 5 2 3 4] [1 1 2 2] [2 3 3 4]
        [0 5 1 2 3 4] [1 2 1 1] [2 3 3 4]
        [0 2 5 1 3 4] [1 1 2 2] [2 3 3 4]
        [0 5 2 1 3 4] [1 2 1 1] [2 3 3 4]

        >>> nb_arms = 5
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[2][3] = False
        >>> player.G[2][4] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4] [1 1 1 2 2] [2 3 4 3 4]
        [0 2 1 3 4] [1 1 1 2 2] [2 3 4 3 4]
        [0 1 3 2 4] [1 1 2 2] [2 3 3 4]
        [0 3 1 2 4] [1 2 1 1] [2 3 3 4]
        [0 2 3 1 4] [1 1 2 2] [2 3 3 4]
        [0 3 2 1 4] [1 2 1 1] [2 3 3 4]
        [0 1 4 2 3] [1 1 2 2] [2 3 3 4]
        [0 4 1 2 3] [1 2 1 1] [2 3 3 4]
        [0 2 4 1 3] [1 1 2 2] [2 3 3 4]
        [0 4 2 1 3] [1 2 1 1] [2 3 3 4]

        >>> nb_arms = 8
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=2, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[1][3] = False
        >>> player.G[1][4] = False
        >>> player.G[2][5] = False
        >>> player.G[3][5] = False
        >>> player.G[4][5] = False
        >>> player.G[2][2] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5 6 7] [1 1 1 1] [2 3 4 5]
        [0 2 1 3 4 5 6 7] [1 1 1 1] [2 3 4 5]
        [0 3 1 2 4 5 6 7] [1 1 1 1] [2 3 4 5]
        [0 4 1 2 3 5 6 7] [1 1 1 1] [2 3 4 5]
        [0 5 1 2 3 4 6 7] [1 1 1 1] [2 3 4 5]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=2, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[1][3] = False
        >>> player.G[1][4] = False
        >>> player.G[2][5] = False
        >>> player.G[3][5] = False
        >>> player.G[4][5] = False
        >>> player.G[2][2] = False
        >>> player.set_focused_pure_explore_recommendations(np.arange(nb_arms))
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [1 1 1 1] [2 3 4 5]
        [0 2 1 3 4 5] [1 1 1 1] [2 3 4 5]
        [0 3 1 2 4 5] [1 1 1 1] [2 3 4 5]
        [0 4 1 2 3 5] [1 1 1 1] [2 3 4 5]
        [0 5 1 2 3 4] [1 1 1 1] [2 3 4 5]
        """
        # TODO: upgrade: when 3 or more consecutive strictly less attractive => do simultaneous comparisons with all permutations
        self.nb_pure_explore += 1

        if not self.G[leader[:(self.nb_positions - 1)], leader[1:self.nb_positions]].all():  # --- the incoherence is between 2 displayed items ---
            # first position at which previous item (denoted $i$) seems strictly less attractive than current item (denoted $j$): i < j
            k1 = np.argwhere(np.logical_not(self.G[leader[:(self.nb_positions - 1)], leader[1:self.nb_positions]]))[0, 0]
            i = leader[k1]
            k = k1 + 1
            j = leader[k]
            # any position such that the item (denoted $l$) seems less attractive or equal to $i$ and strictly more attractive than $j$: i >= l > j
            k2s = k + 1 + np.argwhere(self.G[i, leader[(k + 1):]] & np.logical_not(self.G[j, leader[(k + 1):]]))[:, 0]
            ls = leader[k2s]

            # recommendations (to update the comparisons i-j, i-l, and j-l)
            if k < self.nb_positions - 1:
                # there is 3 positions available to compare => for each l we do comparison of i, j and l through 6 permutations
                # we consider six basic recommmendations (for each l):
                # * [..., i, j, l, remaining unchanged (without l)]
                # * [..., i, l, j, remaining unchanged (without l)]
                # * [..., j, i, l, remaining unchanged (without l)]
                # * [..., j, l, i, remaining unchanged (without l)]
                # * [..., l, j, i, remaining unchanged (without l)]
                # * [..., l, i, j, remaining unchanged (without l)]
                # this allows the comparison between positions (k1, k), (k1, k+1), and (k, k+1)
                self.next_full_recommendations = []
                self.next_exploration_pairs = []
                for i_k2, (k2, l) in enumerate(zip(k2s, ls)):
                    self.next_full_recommendations += [
                        np.concatenate((leader[:k1], [i, j, l], leader[(k + 1):k2], leader[(k2 + 1):])),
                        np.concatenate((leader[:k1], [i, l, j], leader[(k + 1):k2], leader[(k2 + 1):])),
                        np.concatenate((leader[:k1], [j, i, l], leader[(k + 1):k2], leader[(k2 + 1):])),
                        np.concatenate((leader[:k1], [j, l, i], leader[(k + 1):k2], leader[(k2 + 1):])),
                        np.concatenate((leader[:k1], [l, i, j], leader[(k + 1):k2], leader[(k2 + 1):])),
                        np.concatenate((leader[:k1], [l, j, i], leader[(k + 1):k2], leader[(k2 + 1):])),
                        ]
                    if k < self.nb_positions - 2:
                        self.next_exploration_pairs += [np.array([[k1, k1, k], [k, k+1, k+1]])] * 6
                    else:   # k+1 == self.nb_positions - 1
                        # we can also compare the item l with other items ls
                        self.next_exploration_pairs += [np.array([[k1, k1, k] + [k + 1] * (len(k2s) - 1),
                                                                  [k, k + 1, k + 1] + (k2s[:i_k2]+1).tolist()
                                                                  + k2s[(i_k2+1):].tolist()]),
                                                        np.array([[k1, k1, k] + [k] * (len(k2s) - 1),
                                                                  [k, k + 1, k + 1] + (k2s[:i_k2] + 1).tolist()
                                                                  + k2s[(i_k2 + 1):].tolist()]),
                                                        np.array([[k1, k1, k] + [k + 1] * (len(k2s) - 1),
                                                                  [k, k + 1, k + 1] + (k2s[:i_k2] + 1).tolist()
                                                                  + k2s[(i_k2 + 1):].tolist()]),
                                                        np.array([[k1, k1, k] + [k] * (len(k2s) - 1),
                                                                  [k, k + 1, k + 1] + (k2s[:i_k2] + 1).tolist()
                                                                  + k2s[(i_k2 + 1):].tolist()]),
                                                        np.array([[k1, k1, k] + [k1] * (len(k2s) - 1),
                                                                  [k, k + 1, k + 1] + (k2s[:i_k2] + 1).tolist()
                                                                  + k2s[(i_k2 + 1):].tolist()]),
                                                        np.array([[k1, k1, k] + [k1] * (len(k2s) - 1),
                                                                  [k, k + 1, k + 1] + (k2s[:i_k2] + 1).tolist()
                                                                  + k2s[(i_k2 + 1):].tolist()])
                                                        ]
            else:   # k1 == self.nb_positions - 2
                # there is 2 positions available to compare => we do 1 comparison at these positions, plus lots of comparisons with undisplayed items
                # we consider the recommendations of the form:
                # * [..., i, j, remaining] and [..., j, i, remaining]
                # * [..., i, l, remaining] and [..., l, i, remaining] for each l
                # * [..., j, l, remaining] and [..., l, j, remaining] for each l
                # this allows
                # * the comparison between positions k1 and k
                # * the comparison between the positions of i and the position of j
                # * the comparison between positions k1 and any considered undisplayed positions when i or j is in position k
                # * the comparison between positions k and any considered undisplayed positions when i or j is in position k1
                top_reco = leader[:k1]
                undisplayed = leader[self.nb_positions:]
                remaining_items = undisplayed[np.logical_not(self.G[i, undisplayed]) | self.G[j, undisplayed]]
                exploration_perm = np.arange(self.nb_arms)
                exploration_perm[k1] = k
                exploration_perm[k] = k1
                reco = np.concatenate((top_reco, np.array([i, j]), ls, remaining_items))
                self.next_full_recommendations = [reco, reco[exploration_perm]]
                self.next_exploration_pairs = [np.array([[k1]*(len(ls)+1) +
                                                         [k]*len(ls),
                                                         list(range(k, k+len(ls)+1)) +
                                                         list(range(k+1, k+len(ls)+1))
                                                         ])] * 2
                for k3, l in enumerate(ls):
                    reco = np.concatenate((top_reco, [i, l, j], ls[:k3], ls[(k3 + 1):], remaining_items))
                    self.next_full_recommendations += [reco, reco[exploration_perm]]
                    reco = np.concatenate((top_reco, [j, l, i], ls[:k3], ls[(k3 + 1):], remaining_items))
                    self.next_full_recommendations += [reco, reco[exploration_perm]]
                self.next_exploration_pairs += [np.array([[k1, k1] + [k] * len(ls),
                                                         [k, k + 1] + list(range(k + 1, k + 1 + len(ls)))
                                                         ]),
                                                np.array([[k1, k] + [k1] * len(ls),
                                                          [k, k + 1] + list(range(k + 1, k + 1 + len(ls)))
                                                          ])
                                                ] * 2 * len(ls)

        else:     # --- the incoherence is between the last displayed items and an undisplayed one---
            # the recommendations will be [unchanged first self.nb_positions-1 positions, l, ...] where l is either
            # * the item i = leader[self.nb_positions - 1]
            # * an item j in leader[self.nb_positions:] such that j is strictly more attractive than i
            # * an item l in leader[self.nb_positions:] such that l is less attractive than i, and l is strictly more attractive than one item j
            # this set of recommendations updates the loged data on any couple of items put in last displayed position
            k1 = self.nb_positions - 1
            i = leader[k1]
            undisplayed = leader[self.nb_positions:]

            # undisplayed items strictly more attractive than i
            inds_k = np.logical_not(self.G[i, undisplayed])
            js = undisplayed[inds_k]

            # undisplayed items less attractive than i, and strictly more attractive than one item in js
            inds_k2 = self.G[i, undisplayed] & (self.G[js, :][:, undisplayed].sum(axis=0) < len(js))
            ls = undisplayed[inds_k2]

            # recommendations
            top_reco = leader[:k1]
            items_to_compare = np.concatenate(([i], js, ls))
            remaining_items = undisplayed[np.logical_not(inds_k | inds_k2)]
            self.next_full_recommendations = []
            for k3, l in enumerate(items_to_compare):
                self.next_full_recommendations += [np.concatenate((top_reco, [l], items_to_compare[:k3], items_to_compare[(k3 + 1):], remaining_items))]
            self.next_exploration_pairs = [np.array([[k1] * (len(items_to_compare)-1),
                                                      list(range(k1 + 1, k1 + len(items_to_compare)))
                                                      ])
                                            ] * len(items_to_compare)

    def greedy_get_best_recommendation(self):
        """

        Examples
        --------

        >>> player = UniGRAB(nb_arms=6, nb_positions=4, seed=42)
        >>> player.G[1][2] = False
        >>> player.G[1][4] = False
        >>> player.G[3][2] = False
        >>> player.G[3][4] = False
        >>> player.G[2][1] = False
        >>> player.G[2][3] = False
        >>> for _ in range(6): print(player.greedy_get_best_recommendation())
        [0 4 5 2 3 1]
        [5 0 4 2 3 1]
        [5 0 4 1 3 2]
        [5 4 0 3 2 1]
        [0 4 5 1 2 3]
        [4 0 5 2 1 3]
        """
        # add small uniform noise to randomly break even scores
        return np.argsort(-self.G.sum(axis=1) + self.rng.uniform(high=10**-7, size=self.nb_arms))

    def iterative_greedy_get_best_recommendation(self):
        """

        Examples
        --------

        >>> player = UniGRAB(nb_arms=6, nb_positions=4, seed=42)
        >>> player.G[1][3] = False
        >>> player.G[1][4] = False
        >>> player.G[3][1] = False
        >>> player.G[3][4] = False
        >>> player.G[2][1] = False
        >>> player.G[2][3] = False
        >>> for _ in range(6): print(player.iterative_greedy_get_best_recommendation())
        [4 0 5 1 3 2]
        [0 4 5 3 1 2]
        [4 0 5 3 1 2]
        [0 4 5 1 3 2]
        [5 0 4 3 1 2]
        [5 0 4 1 3 2]
       """
        # add small uniform noise to randomly break even scores
        scores = self.G + self.rng.uniform(high=10**-7, size=(self.nb_arms, self.nb_arms))
        leader = np.arange(self.nb_arms)
        for i in range(self.nb_arms-1):
            k_best = i + np.argmax(scores[leader[i:], :][:, leader[i:]].sum(axis=1))
            leader[i], leader[k_best] = leader[k_best], leader[i]
        return leader

    def set_recommendations_from_indices(self, leader, exploration_inds):
        """
        define a set of consecutive recommendations such that the exploration_pairs are explored
        (they may not be in the displayed part of the leader)
        Parameters
        ----------
        leader:
        exploration_inds:
            indices to explore

        Examples
        --------

        >>> nb_arms = 10
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=7, seed=42)
        >>> player.set_recommendations_from_indices(np.arange(nb_arms), [])
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5 6 7 8 9] [] []

        >>> nb_arms = 10
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=7, seed=42)
        >>> player.set_recommendations_from_indices(np.arange(nb_arms), [1, 4, 8])
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5 6 7 8 9] [0 3 6] [1 4 8]
        [1 0 2 4 3 5 8 7 6 9] [0 3 6] [1 4 8]
        """
        if exploration_inds:
            exploration_perm = self.get_exploration_perm(exploration_inds)
            self.next_full_recommendations = [leader, leader[exploration_perm]]
            self.next_exploration_pairs = [np.array([list(min(i, self.nb_positions) - 1 for i in exploration_inds), exploration_inds])] * 2
        else:
            self.next_full_recommendations = [leader]
            self.next_exploration_pairs = [np.zeros((2, 0))]

    def set_recommendations_from_indices_and_undisplayed(self, leader, exploration_inds, optimistic_values):
        """
        define a set of consecutive recommendations such that
        * the exploration_pairs are explored (they are in the displayed part of the leader)
        * the undisplayed items with a "positive" optimism value are explored altogether at the last displayed position

        Parameters
        ----------
        leader:
        exploration_inds:
            indices to explore in the displayed part of the leader
        optimistic_values: np.array(self.nb_positions)
            optimism values
            * values[0] means nothing and is equal to 0.5
            * values[k] compare leader[min(k-1, self.nb_positions-1)] to leader[k]

        Examples
        --------

        >>> nb_arms = 10
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=7, seed=42)
        >>> values = 0.5 * np.ones(nb_arms)
        >>> player.set_recommendations_from_indices_and_undisplayed(np.arange(nb_arms), [], values)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5 6 7 8 9] [] []

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=5, seed=42)
        >>> values = 0.5 * np.ones(nb_arms)
        >>> player.set_recommendations_from_indices_and_undisplayed(np.arange(nb_arms), [1, 4], values)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [0 3] [1 4]
        [1 0 2 4 3 5] [0 3] [1 4]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> values = 0.5 * np.ones(nb_arms)
        >>> values[4] = 1.
        >>> player.set_recommendations_from_indices_and_undisplayed(np.arange(nb_arms), [], values)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 4 3 5] [2] [3]
        [0 1 4 2 3 5] [2] [3]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> values = 0.5 * np.ones(nb_arms)
        >>> values[3] = 1.
        >>> values[4] = 1.
        >>> player.set_recommendations_from_indices_and_undisplayed(np.arange(nb_arms), [], values)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [2 2] [3 4]
        [0 1 3 2 4 5] [2 2] [3 4]
        [0 1 4 2 3 5] [2 2] [3 4]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> values = 0.5 * np.ones(nb_arms)
        >>> values[4] = 1.
        >>> player.set_recommendations_from_indices_and_undisplayed(np.arange(nb_arms), [1], values)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 4 3 5] [0 2] [1 3]
        [1 0 4 2 3 5] [0 2] [1 3]

        >>> nb_arms = 6
        >>> player = UniGRAB(nb_arms=nb_arms, nb_positions=3, seed=42)
        >>> values = 0.5 * np.ones(nb_arms)
        >>> values[3] = 1.
        >>> values[4] = 1.
        >>> player.set_recommendations_from_indices_and_undisplayed(np.arange(nb_arms), [1], values)
        >>> for reco, inds in zip(player.next_full_recommendations, player.next_exploration_pairs): print(reco, inds[0], inds[1])
        [0 1 2 3 4 5] [0 2 2] [1 3 4]
        [1 0 3 2 4 5] [0 2 2] [1 3 4]
        [0 1 4 2 3 5] [0 2 2] [1 3 4]
        [1 0 2 3 4 5] [0] [1]
        """
        items_to_compare = np.concatenate(([leader[self.nb_positions - 1]], leader[self.nb_positions:][optimistic_values[self.nb_positions:] > 0.5]))
        remaining_items = leader[self.nb_positions:][optimistic_values[self.nb_positions:] <= 0.5]
        exploration_perm = self.get_exploration_perm(exploration_inds)
        top_recommendations = [leader[:(self.nb_positions - 1)], leader[exploration_perm][:(self.nb_positions - 1)]]
        self.next_full_recommendations = []
        for k, i in enumerate(items_to_compare):
            self.next_full_recommendations += [np.concatenate(
                (top_recommendations[k % 2], [i], items_to_compare[:k], items_to_compare[(k + 1):], remaining_items))]
        self.next_exploration_pairs = [np.array([list(i - 1 for i in exploration_inds) + [self.nb_positions - 1] * (len(items_to_compare) - 1),
                                                 exploration_inds + list(range(self.nb_positions, self.nb_positions + len(items_to_compare) - 1))
                                                 ])] * len(items_to_compare)
        if len(items_to_compare) % 2 == 1 and exploration_inds:
            self.next_full_recommendations += [leader[exploration_perm]]
            self.next_exploration_pairs += [np.array([list(i - 1 for i in exploration_inds), exploration_inds])]

    def set_explore_exploit_recommendations(self, leader):
        """

        Examples
        --------
        """
        # TODO: check if nb_leader is properly defined when doing comparison through permutations
        self.node_logs[tuple(leader[:self.nb_positions])]['nb_leader'] += 1

        # choose time for threshold
        if self.global_time_for_threshold:
            time_for_threshold = self.time
        else:
            time_for_threshold = self.node_logs[tuple(leader[:self.nb_positions])]['nb_leader']

        # choose the recommendations
        if self.potential_explore_exploit == 'best':
            # Apply the most promising exchange of two items
            values = 0.5 * np.ones(self.nb_arms)
            for k in range(1, self.nb_arms):
                i = leader[min(k - 1, self.nb_positions - 1)]
                j = leader[k]
                values[k] = self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i], time_for_threshold)
            k_max = np.argmax(values)
            if k_max == 0:
                self.set_recommendations_from_indices(leader, [])
                #self.next_full_recommendations = [leader]
                #self.next_exploration_pairs = [np.zeros((2, 0))]
            else:
                if k_max < self.nb_positions:
                    self.set_recommendations_from_indices(leader, [k_max])
                else:
                    if self.undisplayed_explore_exploit == 'best':
                        self.set_recommendations_from_indices(leader, [k_max])
                    elif self.undisplayed_explore_exploit == 'all_potentials':
                        self.set_recommendations_from_indices_and_undisplayed(leader, [], values)
                    else:
                        raise ValueError(f'unknown undisplayed_explore_exploit {self.undisplayed_explore_exploit}')
        elif self.potential_explore_exploit == 'first':
            # Apply the first promising exchange of two items
            k_max = None
            for k in range(1, self.nb_arms):
                i = leader[min(k - 1, self.nb_positions - 1)]
                j = leader[k]
                value = self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i],
                                              time_for_threshold)
                if value > 0.5:
                    k_max = k
                    break
            if k_max is None:
                self.set_recommendations_from_indices(leader, [])
                #self.next_full_recommendations = [leader]
                #self.next_exploration_pairs = [np.zeros((2, 0))]
            elif k_max < self.nb_positions:
                    self.set_recommendations_from_indices(leader, [k_max])
            else:
                if self.undisplayed_explore_exploit == 'best':
                    self.set_recommendations_from_indices(leader, [k_max])
                elif self.undisplayed_explore_exploit == 'all_potentials':
                    values = 0.5 * np.ones(self.nb_arms)
                    for k in range(k_max, self.nb_arms):
                        i = leader[min(k - 1, self.nb_positions - 1)]
                        j = leader[k]
                        values[k] = self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i],
                                                          time_for_threshold)
                    self.set_recommendations_from_indices_and_undisplayed(leader, [], values)
                else:
                    raise ValueError(f'unknown undisplayed_explore_exploit {self.undisplayed_explore_exploit}')
        elif self.potential_explore_exploit == 'even-odd':
            # Apply all possible exchanges of two items first among even positions, and then among odd positions
            values = 0.5 * np.ones(self.nb_arms)
            for k in range(1, self.nb_arms):
                i = leader[min(k - 1, self.nb_positions - 1)]
                j = leader[k]
                values[k] = self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i],
                                                  time_for_threshold)
            next_full_recommendations = []
            next_exploration_pairs = []
            for i0 in range(1, 3):
                exploration_inds = []
                # exchange among displayed positions
                for k in range(i0, self.nb_positions, 2):
                    if values[k] > 0.5:
                        exploration_inds.append(k)
                # overall exchanges
                may_exchange_with_last = (i0 + self.nb_positions) % 2 == 0
                if self.undisplayed_explore_exploit == 'best':
                    k_max = self.nb_positions + np.argmax(values[self.nb_positions:])
                    if may_exchange_with_last and values[k_max] > 0.5:
                        exploration_inds.append(k_max)
                    self.set_recommendations_from_indices(leader, exploration_inds)
                elif self.undisplayed_explore_exploit == 'all_potentials':
                    if may_exchange_with_last:
                        self.set_recommendations_from_indices_and_undisplayed(leader, exploration_inds, values)
                    else:
                        self.set_recommendations_from_indices(leader, exploration_inds)
                else:
                    raise ValueError(f'unknown undisplayed_explore_exploit {self.undisplayed_explore_exploit}')
                next_full_recommendations += self.next_full_recommendations
                next_exploration_pairs += self.next_exploration_pairs
            self.next_full_recommendations = next_full_recommendations
            self.next_exploration_pairs = next_exploration_pairs
        elif self.potential_explore_exploit == 'as_much_as_possible_from_top_to_bottom':
            # Apply all possible exchanges of two items (starting from top)
            k = 1
            exploration_inds = []
            # exchange among displayed positions
            while k < self.nb_positions:
                i = leader[k - 1]
                j = leader[k]
                value = self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i],
                                              time_for_threshold)
                if value > 0.5:
                    exploration_inds.append(k)
                    k += 2
                else:
                    k += 1
            # overall exchanges
            may_exchange_with_last = not exploration_inds or exploration_inds[-1] != self.nb_positions - 1
            values = 0.5 * np.ones(self.nb_arms)
            if may_exchange_with_last:
                i = leader[self.nb_positions - 1]
                for k in range(self.nb_positions, self.nb_arms):
                    j = leader[k]
                    values[k] = self.optimistic_index(0.5 + 0.5 * self.tau_hats[j][i], self.nb_diffs[j][i],
                                                      time_for_threshold)
            if self.undisplayed_explore_exploit == 'best':
                k_max = self.nb_positions + np.argmax(values[self.nb_positions:])
                if may_exchange_with_last and values[k_max] > 0.5:
                    exploration_inds.append(k_max)
                self.set_recommendations_from_indices(leader, exploration_inds)
            elif self.undisplayed_explore_exploit == 'all_potentials':
                if may_exchange_with_last:
                    self.set_recommendations_from_indices_and_undisplayed(leader, exploration_inds, values)
                else:
                    self.set_recommendations_from_indices(leader, exploration_inds)
            else:
                raise ValueError(f'unknown undisplayed_explore_exploit {self.undisplayed_explore_exploit}')
        else:
            raise ValueError(f'unknown exploration-exploitation strategy {self.potential_explore_exploit}')

    def choose_next_arm(self):
        """
        Returns
        -------

        """
        if not self.next_full_recommendations:
            # --- prepare next recommendations ---
            leader = self.iterative_greedy_get_best_recommendation()
            leader_score = (self.G[leader[:(self.nb_positions-1)], leader[1:self.nb_positions]].sum()
                            + self.G[leader[self.nb_positions-1], leader[self.nb_positions:]].sum()
                           )
            if leader_score == (self.nb_arms - 1):
                self.set_explore_exploit_recommendations(leader)
            else:
                if self.pure_explore == 'all':
                    self.set_pure_explore_recommendations()
                elif self.pure_explore == 'focused':
                    self.set_focused_pure_explore_recommendations(leader)
                else:
                    raise ValueError(f'unknown pure-explore strategy "{self.pure_explore}"')

        # --- recommend the first recommendation in the list ---
        self.full_recommendation = self.next_full_recommendations.pop()
        self.exploration_pairs = self.next_exploration_pairs.pop()
        return np.array(self.full_recommendation[:self.nb_positions]), 0

    def update(self, propositions, rewards):
        self.time += 1

        # --- update matrix ---
        for k1, k in zip(self.exploration_pairs[0], self.exploration_pairs[1]):
            i = self.full_recommendation[k1]
            j = self.full_recommendation[k]
            C_i = rewards[k1]
            if k < self.nb_positions:
                self.nb_explorations_at[k] += 1
                C_j = rewards[k]
            else:
                self.nb_explorations_at[self.nb_positions] += 1
                C_j = 0
            self.nb_explorations[i][j] += 1
            self.nb_explorations[j][i] = self.nb_explorations[i][j]
            if C_i != C_j:
                self.nb_diffs[i][j] += 1
                self.nb_diffs[j][i] = self.nb_diffs[i][j]
                self.tau_hats[i][j] += (C_i - C_j - self.tau_hats[i][j]) / self.nb_diffs[i][j]
                self.tau_hats[j][i] = - self.tau_hats[i][j]

        # --- update graph ---
        self.G = self.tau_hats >= 0

    def get_param_estimation(self):
        raise NotImplementedError()

    def estimator(self, hat_rho, nb_trial, nb_total_trial, bound):
        if bound == 'o':
            return self.optimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'p':
            return self.pessimistic_index(hat_rho, nb_trial, nb_total_trial)
        elif bound == 'a':
            return hat_rho
        else:
            raise ValueError(f'unkwon estimator {bound}')

    def optimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        """

        Parameters
        ----------
        hat_rho
        nb_trial
        nb_total_trial

        Examples
        -------
        >>> player = OSUB_TOP_RANK(nb_arms=4, nb_positions=2, sigma=np.arange(2), )
        >>> player.optimistic_index(0.5 + 0.5 * 0, 100, 100)
        0.7048432465681482
        >>> player.optimistic_index(0.5 + 0.5 * 0, 0, 100)
        1
        """
        if self.horizon is None:
            if nb_trial == 0 or nb_total_trial < 3:
                return 1
            threshold = math.log(nb_total_trial) + 3 * math.log(math.log(nb_total_trial))
        else:
            threshold = math.log(self.horizon)
        start = start_up(hat_rho, threshold, nb_trial)
        return newton(hat_rho, threshold, nb_trial, start)

    def pessimistic_index(self, hat_rho, nb_trial, nb_total_trial):
        return 1-self.optimistic_index(1-hat_rho, nb_trial, nb_total_trial)

    def print_info(self):
        leader = self.greedy_get_best_recommendation()
        leader_score = (self.G[leader[:(self.nb_positions - 1)], leader[1:self.nb_positions]].sum()
                        + self.G[leader[self.nb_positions - 1], leader[self.nb_positions:]].sum()
                        )
        print('leader', leader)
        print('leader score', leader_score)
        print('nb pure explorations', self.nb_pure_explore)
        for reco, inds in zip(self.next_full_recommendations, self.next_exploration_pairs): print(reco, inds[0], inds[1])
        print(self.nb_explorations_at)
        print(self.nb_explorations)
        print(self.nb_diffs)
        print(self.tau_hats)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
