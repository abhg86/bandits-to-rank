#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from bandits_to_rank.environment import Environment_PBM, Environment_Cascade

from bandits_to_rank.referee import Referee

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import json

from bandits_to_rank.bandits import *
from bandits_to_rank.environment import *
from bandits_to_rank.referee import *
from bandits_to_rank.opponents import oracle,random_player,greedy,bc_mpts
from bandits_to_rank.opponents.pbm_ts import *
from bandits_to_rank.sampling.pbm_inference import *

import os
import time
import json
import gzip
import numpy as np
import pandas as pd
import pylab
import zipfile
import matplotlib.pyplot as plt
from math import floor, log
from random import shuffle
from copy import deepcopy
from itertools import product


def barerror_value(data, type_errorbar='confidence'):
    nb_game = len(data)
    average = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if type_errorbar is None:
        return average, average, average
    elif type_errorbar == 'std':
        return average, average-std, average+std
    elif type_errorbar == 'standart_error':
        return average, average-std/np.sqrt(nb_game), average+std/np.sqrt(nb_game)
    elif type_errorbar == 'confidence':
        return average, average - std / np.sqrt(nb_game) * 4, average + std / np.sqrt(nb_game) * 4

def barerror_value(data, type_errorbar='confidence'):
    nb_game = len(data)
    average = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if type_errorbar is None:
        return average, average, average
    elif type_errorbar == 'std':
        return average, average-std, average+std
    elif type_errorbar == 'standart_error':
        return average, average-std/np.sqrt(nb_game), average+std/np.sqrt(nb_game)
    elif type_errorbar == 'confidence':
        return average, average - std / np.sqrt(nb_game) * 4, average + std / np.sqrt(nb_game) * 4


def myplot(X_val, data, color='red', linestyle='-', label='TopRank', type_errorbar='confidence'):
    average, lower, upper = barerror_value(np.array(data), type_errorbar=type_errorbar)
    plt.plot(X_val, average, color=color, linestyle=linestyle, label=label)
    plt.fill_between(X_val, lower, upper, color=color, alpha=0.3, linestyle=linestyle, label='')
    print(f'{label}: {average[-1]}')


def run_games(player, env, nb_trials, nb_records, nb_games=10, nb_relevant_positions=None):
    # --- prepare referee ---
    ref = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records, print_trial=20)


    # --- run one game ---
    for _ in range(nb_games):
        player.clean()
        ref.play_game(player, nb_relevant_positions=nb_relevant_positions)
        try:
            print('dict size', len(player.stats))
        except:
            pass

    return ref


def run_exp(thetas, kappas, is_pbm, nb_trials, nb_records, nb_games, nb_relevant_positions=None):
    if is_pbm:
        env = Environment_PBM(thetas, kappas, label="purely simulated")
    else:
        env = Environment_Cascade(thetas, np.argsort(-np.array(kappas)))
        print("best", env.get_best_index(), sum(env.get_expected_reward(env.get_best_index())), env.get_expected_reward(env.get_best_index()))
        try:
            print([0,1,2], sum(env.get_expected_reward([0,1,2])),env.get_expected_reward([0,1,2]))
            print([1,2,3], sum(env.get_expected_reward([1,2,3])),env.get_expected_reward([1,2,3]))
            print([2,1,3], sum(env.get_expected_reward([2,1,3])),env.get_expected_reward([2,1,3]))
            print([2,0,3], sum(env.get_expected_reward([2,0,3])),env.get_expected_reward([2,0,3]))
        except:
            pass
        try:
            print([2, 1, 0, 3], sum(env.get_expected_reward([2, 1, 0, 3])), env.get_expected_reward([2, 1, 0, 3]))
            print([1, 2, 3], sum(env.get_expected_reward([1, 2, 3])), env.get_expected_reward([1, 2, 3]))
            print([2, 1, 3], sum(env.get_expected_reward([2, 1, 3])), env.get_expected_reward([2, 1, 3]))
            print([2, 0, 3], sum(env.get_expected_reward([2, 0, 3])), env.get_expected_reward([2, 0, 3]))
        except:
            pass
    nb_prop, nb_positions = env.get_setting()

    # --- logs ---
    referees = {}

    # --- run Oracle ---
    if False:
        from bandits_to_rank.opponents.oracle import Oracle
        player = Oracle(best_arm=np.argsort(-thetas)[:nb_positions][np.argsort(np.argsort(-kappas))])
        referees['oracle'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK AG ---
    if False:
        from bandits_to_rank.opponents.top_rank_AG import TOP_RANK
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, L=list(range(nb_prop)), K=nb_positions)
        referees['TopRank AG'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK BAL ---
    if False:
        from bandits_to_rank.opponents.top_rank_BAL import TOP_RANK
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, nb_positions=nb_positions)
        referees['TopRank BAL'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK oracle ---
    if False:
        from bandits_to_rank.opponents.top_rank import TOP_RANK
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, discount_factor=kappas, horizon_time_known=True,doubling_trick_active=False)
        #player = TOP_RANK(nb_arms=nb_prop, T=10, discount_factor=kappas, horizon_time_known=True, doubling_trick_active=True)
        referees['TopRank, oracle'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK fast implem ---
    if False:
        from bandits_to_rank.opponents.top_rank_dev import TopRank
        player = TopRank(nb_arms=nb_prop, nb_pos=nb_positions, T=nb_trials)
        #player = TOP_RANK(nb_arms=nb_prop, T=10, discount_factor=kappas, horizon_time_known=True, doubling_trick_active=True)
        referees['TopRank (fast), oracle'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK greedy ---
    if False:
        from bandits_to_rank.opponents.top_rank import TOP_RANK
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, nb_positions=nb_positions)
        referees['TopRank, greedy'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK kappa decreases---
    if False:
        from bandits_to_rank.opponents.top_rank import TOP_RANK
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, discount_factor=np.arange(nb_positions-1, -1, -1))
        referees['TopRank, kappa decreases'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TOP_RANK kappa increases---
    if False:
        from bandits_to_rank.opponents.top_rank import TOP_RANK
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, discount_factor=np.arange(nb_positions))
        referees['TopRank, kappa increases'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run BubbleRank BAL ---
    if False:
        from bandits_to_rank.opponents.bubblerank_BAL import BUBBLERANK
        player = BUBBLERANK(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials)
        referees['BubbleRank BAL'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run BubbleRank ---
    if False:
        from bandits_to_rank.opponents.bubblerank import BUBBLERANK
        player = BUBBLERANK(nb_prop, delta=nb_trials**(-4), discount_factor=kappas)
        referees['BubbleRank'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run CASCADEKL_UCB kappa decreases ---
    if False:
        from bandits_to_rank.opponents.cascadekl_ucb import CASCADEKL_UCB
        player = CASCADEKL_UCB(nb_arms=nb_prop, nb_position=nb_positions)
        referees['CascadeKL-UCB, kappa decreases'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB (reward) ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1)
        referees['GRAB, known horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB unknown horizon---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, gamma=nb_prop - 1)
        referees['GRAB, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB TS ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1, optimism='TS')
        referees['GRAB, TS'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB first ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1, gap_type='first')
        referees['GRAB, first'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB first TS ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1, gap_type='first', optimism='TS')
        referees['GRAB, first, TS'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB reward and first TS ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1, gap_type='reward and first', optimism='TS')
        referees['GRAB, reward and first, TS'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB second ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1, gap_type='second')
        referees['GRAB, second'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB both ---
    if False:
        from bandits_to_rank.opponents.grab import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials, gamma=nb_prop - 1, gap_type='both')
        referees['GRAB, both'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run GRAB_diff unknown horizon---
    if False:
        from bandits_to_rank.opponents.grab_dev import GRAB
        player = GRAB(nb_arms=nb_prop, nb_positions=nb_positions, gamma=nb_prop - 1)
        referees['GRAB_diff, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankFirstPos ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankFirstPos
        player = UniRankFirstPos(nb_arms=nb_prop, nb_positions=nb_positions)
        referees['UniRankFirstPos, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankFirstPos known horizon ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankFirstPos
        player = UniRank(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials)
        referees['UniRankFirstPos, known horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, bound_l='o', bound_n='o', lead_l='o', lead_n='a')
        referees['UniRankMaxGap oooa, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, bound_l='o', bound_n='o', lead_l='o', lead_n='a', neighbor_type='jumps3')
        referees['UniRankMaxGap oooa, jumps3, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap known sigma---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), bound_l='o', bound_n='o', lead_l='o', lead_n='a')
        referees['UniRankMaxGap oooa, known sigma, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), bound_l='o', bound_n='o', lead_l='o', lead_n='a', neighbor_type='jumps3')
        referees['UniRankMaxGap oooa, known sigma, jumps3, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), bound_l='o', bound_n='o', lead_l='o', lead_n='a', neighbor_type='jumps3.1')
        referees['UniRankMaxGap oooa, known sigma, jumps3.1, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, bound_l='p', bound_n='o', lead_l='o', lead_n='a')
        referees['UniRankMaxGap pooa, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, bound_l='o', bound_n='o', lead_l='a', lead_n='a')
        referees['UniRankMaxGap ooaa, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, bound_l='o', bound_n='o', lead_l='p', lead_n='a')
        referees['UniRankMaxGap oopa, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxGap test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxGap
        player = UniRankMaxGap(nb_arms=nb_prop, nb_positions=nb_positions, bound_l='o', bound_n='o', lead_l='o', lead_n='p')
        referees['UniRankMaxGap ooop, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankMaxRatio ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankMaxRatio
        player = UniRankMaxRatio(nb_arms=nb_prop, nb_positions=nb_positions)
        referees['UniRankMaxRatio, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run DCGUniRank ---
    if False:
        from bandits_to_rank.opponents.uni_rank import DCGUniRank
        player = DCGUniRank(nb_arms=nb_prop, nb_positions=nb_positions)
        referees['DCGUniRank, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run DCGUniRank known horizon ---
    if False:
        from bandits_to_rank.opponents.uni_rank import DCGUniRank
        player = DCGUniRank(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials)
        referees['DCGUniRank, known horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankWithMemory known sigma---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankWithMemory
        player = UniRankWithMemory(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), bound_l='o', bound_n='o', lead_l='o', lead_n='a')
        referees['UniRankWithMemory oooa, known sigma, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankWithMemory known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankWithMemory
        player = UniRankWithMemory(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), bound_l='o', bound_n='o', lead_l='o', lead_n='a', neighbor_type='jumps3')
        referees['UniRankWithMemory oooa, known sigma, jumps3, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniRankWithMemory known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniRankWithMemory
        player = UniRankWithMemory(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), bound_l='o', bound_n='o', lead_l='o', lead_n='a', neighbor_type='jumps3.1')
        referees['UniRankWithMemory oooa, known sigma, jumps3.1, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom')
        referees['OSUB_TOP_RANK, known sigma, unknown horizon, merge from top'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top & careful remaining ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom, specific rule for remaining')
        referees['OSUB_TOP_RANK, known sigma, unknown horizon, merge from top & careful remaining'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top & greedy-careful remaining ---
    # submitted to NeurIPS'21
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom, greedy-specific rule for remaining', global_time_for_threshold=True)
        referees['OSUB_TOP_RANK, known sigma, unknown horizon, merge from top & greedy-careful remaining (with bug)'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top & greedy-careful remaining, right time used by choose_next_arm ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom, greedy-specific rule for remaining', global_time_for_threshold=False)
        referees['OSUB_TOP_RANK, known sigma, unknown horizon, merge from top & greedy-careful remaining'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, best merge or best remaining item ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='best merge or best remaining item', global_time_for_threshold=False, slight_optimism='tau hat')
        referees['OSUB_TOP_RANK, known sigma, best merge or best remaining item'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom', slight_optimism='tau hat')
        referees['OSUB_TOP_RANK, known sigma, tau hat, unknown horizon, merge from top'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top & careful remaining test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom, specific rule for remaining', slight_optimism='tau hat')
        referees['OSUB_TOP_RANK, known sigma, tau hat, unknown horizon, merge from top & careful remaining'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma, merge from top & greedy-careful remaining test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), recommended_partition_choice='as much as possible from top to bottom, greedy-specific rule for remaining', slight_optimism='tau hat')
        referees['OSUB_TOP_RANK, known sigma, tau hat, unknown horizon, merge from top & greedy-careful remaining'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), slight_optimism='sqrt log')
        referees['OSUB_TOP_RANK, known sigma, sqrt log, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), slight_optimism='log0.8')
        referees['OSUB_TOP_RANK, known sigma, log0.8, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), fine_grained_partition=True)
        referees['OSUB_TOP_RANK, known sigma, fine_grained_partition, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), slight_optimism='sqrt log', fine_grained_partition=True)
        referees['OSUB_TOP_RANK, known sigma, fine_grained_partition, sqrt log, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB_TOP_RANK known sigma test ---
    if False:
        from bandits_to_rank.opponents.uni_rank import OSUB_TOP_RANK
        player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_positions, sigma=np.argsort(-kappas), slight_optimism='log0.8', fine_grained_partition=True)
        referees['OSUB_TOP_RANK, known sigma, fine_grained_partition, log0.8, unknown horizon'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TS-OSUB-10^3 ---
    if False:
        from bandits_to_rank.opponents import OSRUB_bis
        player = OSRUB_bis.OSUB(nb_arms=nb_prop, nb_positions=nb_positions, memory_size=1000)
        referees['TS-OSUB, mem=1000'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run TS-OSUB-inf ---
    if False:
        from bandits_to_rank.opponents import OSRUB_bis
        player = OSRUB_bis.OSUB(nb_arms=nb_prop, nb_positions=nb_positions, memory_size=10000000)
        referees['TS-OSUB, mem=10^7'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run OSUB-PBM mem = inf---
    if False:
        from bandits_to_rank.opponents.OSRUB_PBM_bis import OSUB_PBM
        player = OSUB_PBM(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials)
        referees['OSUB-PBM, mem=inf'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run fast OSUB-PBM ---
    if False:
        from bandits_to_rank.opponents.OSRUB_PBM_bis import fast_OSUB_PBM
        player = fast_OSUB_PBM(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials)
        referees['fast OSUB-PBM'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run fast OSUB-PBM sqrt ---
    if False:
        from bandits_to_rank.opponents.OSRUB_PBM_bis import fast_OSUB_PBM_bis
        player = fast_OSUB_PBM_bis(nb_arms=nb_prop, nb_positions=nb_positions, T=nb_trials)
        referees['fast OSUB-PBM sqrt'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run MLMR ---
    if False:
        for exploration_factor in [0.1, 1., 2., 10., 100.]:
            from bandits_to_rank.opponents.mlmr import MLMR
            player = MLMR(nb_arms=nb_prop, nb_positions=nb_positions, exploration_factor=exploration_factor)
            referees[f'MLMR {exploration_factor}'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run KL-MLMR ---
    if False:
        from bandits_to_rank.opponents.mlmr import KL_MLMR
        player = KL_MLMR(nb_arms=nb_prop, nb_positions=nb_positions, horizon=nb_trials)
        referees['KL-MLMR'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run PB-MHB with TGRW ---
    if False:
        from bandits_to_rank.bandits import propos_trunk_GRW, TS_MH_kappa_desordonne
        proposal = propos_trunk_GRW(c=1000., vari_sigma=True)
        player = TS_MH_kappa_desordonne(nb_arms=nb_prop, nb_position=nb_positions, proposal_method=proposal, step=1, part_followed=True)
        referees['PB-MHB, TGRW, 1000'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run PB-MHB with LGRW ---
    if False:
        from bandits_to_rank.bandits import propos_logit_RW, TS_MH_kappa_desordonne
        for c in [0.1, 1.]:
            proposal = propos_logit_RW(c=c, vari_sigma=True)
            player = TS_MH_kappa_desordonne(nb_arms=nb_prop, nb_position=nb_positions, proposal_method=proposal, step=1, part_followed=True)
            referees[f'PB-MHB, LGRW, {c}'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run PB-MHB with uniform ---
    if False:
        from bandits_to_rank.bandits import propos_uniform, TS_MH_kappa_desordonne
        proposal = propos_uniform()
        player = TS_MH_kappa_desordonne(nb_arms=nb_prop, nb_position=nb_positions, proposal_method=proposal, step=1, part_followed=True)
        referees['PB-MHB, uniform'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run PB-Langevin ---
    if False:
        from bandits_to_rank.opponents.PB_GB import PB_GB
        for h_param, gamma in product([10**-3, 10**-2], [10**-2, 1., 10**2]):
            player = PB_GB(nb_arms=nb_prop, nb_position=nb_positions, h_param=h_param, N=1, L_smooth_param=1, m_strongconcav_param=1, gamma=gamma, part_followed=True)
            referees[f'PB-Langevin, h={h_param}, $\\gamma$={gamma}'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run UniGRAB all ---
    if False:
        from bandits_to_rank.opponents.uni_rank import UniGRAB
        explorations = ['best', 'first', 'even-odd', 'as_much_as_possible_from_top_to_bottom']
        pure_explorations = ['all', 'focused']
        undisplayed_explore_exploits = ['best', 'all_potentials']
        explorations = ['best']
        pure_explorations = ['focused']
        undisplayed_explore_exploits = ['best']
        for exploration, pure_exploration, undisplayed_explore_exploit in product(explorations, pure_explorations, undisplayed_explore_exploits):
            player = UniGRAB(nb_arms=nb_prop, nb_positions=nb_positions, potential_explore_exploit=exploration, pure_explore=pure_exploration, undisplayed_explore_exploit=undisplayed_explore_exploit)
            referees[f'UniGRAB, {exploration} {pure_exploration} {undisplayed_explore_exploit}'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    # --- run PMED ---
    if False:
        from bandits_to_rank.opponents.pmed import PMED
        player = PMED(nb_arms=nb_prop, nb_positions=nb_positions, alpha=1.0, gap_MLE=1, gap_q=10)
        referees['PMED'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)

    if False:
        from bandits_to_rank.opponents.pmed import PMED
        ref = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records, print_trial=20)
        # --- run one game ---
        for _ in range(nb_games):
            player = PMED(nb_arms=nb_prop, nb_positions=nb_positions, alpha=1.0, gap_MLE=1, gap_q=10)
            player.clean()
            ref.play_game(player)

        referees['PMED one by one'] = ref
    
    if True:
        from bandits_to_rank.opponents.bubblerank_OSUB2 import BUBBLERANK_OSUB2
        player = BUBBLERANK_OSUB2(nb_arms=nb_prop, discount_factor=np.linspace(1,0,nb_prop), nb_positions=nb_prop, R_init=[0,1,2,3,4])
        referees['BUBBLERANK_OSUB2'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games, nb_relevant_positions=nb_relevant_positions)


    # --- one plot ---
    type_errorbar = 'standart_error'

    plt.subplot(1, 2, 1)
    for i, key in enumerate(referees.keys()):
        if key != 'oracle':
            myplot(X_val=referees[key].get_recorded_trials(),
                   data=np.array(referees[key].record_results['expected_best_reward'])
                        - np.array(referees[key].record_results['expected_reward']),
                   color=f'C{i+1}', linestyle='-', label=key, type_errorbar=type_errorbar)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Expected Regret')
    #plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.xscale('log')
    #plt.xlim([1000, 20000000])
    #plt.ylim([70, 200000])

    plt.subplot(1, 2, 2)
    for i, key in enumerate(referees.keys()):
        myplot(X_val=referees[key].get_recorded_trials(),
               data=np.array(referees[key].record_results['time_to_play']),
               color=f'C{i+1}', linestyle='-', label=key, type_errorbar=type_errorbar)
    plt.xlabel('Time')
    plt.ylabel('Computation Time')
    #plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.xscale('log')
    #plt.xlim([1000, 20000000])
    #plt.ylim([70, 200000])

    """
    plt.subplot(2, 2, 3)
    for i, key in enumerate(referees.keys()):
        myplot(X_val=referees[key].get_recorded_trials(),
               data=np.array(referees[key].record_results['stat_norm'])[:, :, 0],
               color=f'C{i+1}', linestyle='-', label=key, type_errorbar=type_errorbar)
    plt.xlabel('Time')
    plt.ylabel('error on $\\hat\\theta$')
    #plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.xscale('log')
    #plt.xlim([1000, 20000000])
    #plt.ylim([70, 200000])

    plt.subplot(2, 2, 4)
    for i, key in enumerate(referees.keys()):
        myplot(X_val=referees[key].get_recorded_trials(),
               data=np.array(referees[key].record_results['stat_norm'])[:, :, 1],
               color=f'C{i+1}', linestyle='-', label=key, type_errorbar=type_errorbar)
    plt.xlabel('Time')
    plt.ylabel('error on $\\hat\\kappa$')
    #plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.xscale('log')
    #plt.xlim([1000, 20000000])
    #plt.ylim([70, 200000])
    """

    plt.show()

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    # --- prepare data ---

    # --- PBM Yandex-like ---
    #kappas = [1., 0.992, 0.986, 0.906, 0.892, 0.863, 0.838, 0.798, 0.786, 0.779]
    #kappas = [1., 0.992, 0.986, 0.906, 0.892]
    #kappas = [1., 0.992, 0.986]
    #thetas = [0.67, 0.59, 0.567, 0.561, 0.561, 0.526, 0.518, 0.517, 0.514, 0.513, 0.512, 0.509, 0.508, 0.507, 0.477]
    #thetas = [0.67, 0.59, 0.567, 0.561, 0.561, 0.526, 0.518, 0.517, 0.514, 0.513]
    #thetas = [0.67, 0.59, 0.567, 0.561, 0.561]

    # --- CM Yandex-like ---
    kappas = [1., 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    kappas = [1., 0.95, 0.9, 0.85, 0.8]
    #kappas = [0.95, 0.8, 0.9, 1., 0.85]
    #thetas = [0.5, 0.3333, 0.2, 0.1667, 0.1429, 0.1111, 0.1, 0.0909, 0.0769, 0.0714, 0.0667, 0.0625, 0.0588, 0.0556, 0.0526, 0.0476, 0.0455, 0.0435, 0.0417, 0.04]
    thetas = [0.5, 0.3333, 0.2, 0.1667, 0.1429, 0.1111, 0.1, 0.0909, 0.0769, 0.0714]
    thetas = [0.1111, 0.1, 0.5, 0.3333, 0.2, 0.1667, 0.1429, 0.0909, 0.0769, 0.0714]


    # --- test TGRW vs LGRW, thetas/kappas proches de 0.5 ---
    #kappas = [1., 0.6, 0.5, 0.4]
    #kappas = [1., 0.4, 0.5, 0.6]
    #thetas = [0.7, 0.65, 0.6, 0.55, 0.4, 0.35]
    #thetas = [0.55, 0.4, 0.7, 0.65, 0.6, 0.35]

    # --- removing assumption "strict" ---
    #kappas = [1., 0.6, 0.5]
    #kappas = [1., 0.4, 0.5, 0.6]
    #thetas = [0.3, 0.9, 0.9, 0.9, 0.9, 0.3]

    # --- situation "à peine strict" ---
    #kappas = [1., 0.6, 0.5]
    #kappas = [1., 0.4, 0.5, 0.6]
    #thetas = [0.5, 0.49, 0.51, 0.48, 0.52]

    #kappas = [1, 0.8, 0.6, 0.4, 0.2]
    #thetas = [0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]   # ok mais double bosse (à tester sur 10**6)
    #thetas = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]                # ok mais pas convergé (à tester sur 10**6)
    #thetas = [0.15, 0.1, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
    #thetas = [0.1, 0.095, 0.09, 0.085, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]      # bien pour CM6 (x4) mais TopRank a presque trouvé la bonne combinaison, TB(x30) pour PBM6
    #thetas = [0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]      # CM6 bof,
    #thetas = [0.1, 0.1, 0.1, 0.1, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095]    # petite virgule pour CM6, bien(x4) pour PBM6

    #kappas = [1, 0.6, 0.3]
    #thetas = [0.1, 0.095, 0.09, 0.09, 0.09, 0.09]                # ok en CM6 (de peu) et PBM6
    #thetas = [0.1, 0.09, 0.08, 0.08, 0.08, 0.08]                #

    #kappas = [1, 0.8, 0.6, 0.4, 0.2]
    #thetas = [0.1, 0.09, 0.08, 0.07, 0.06]                #

    #kappas = [1, 0.9, 0.82, 0.75, 0.72]
    #thetas = [0.1, 0.09, 0.08, 0.07, 0.06]                # PBM5 cool (résolution vers 10^4) ; CM absurde ; qui de PBM6 ?
    #thetas = [0.1, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]  # PBM5 et CM5 cool; idem PBM6/CM6
    #kappas = [1, 0.9, 0.83, 0.78, 0.75]
    #thetas = [0.1, 0.08, 0.06, 0.04, 0.02, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]  # PBM/CM5 cool; PBM/CM6 (presque) parfait
    #thetas = [0.1, 0.08, 0.06, 0.04, 0.02]  # PBM/CM5 cool; PBM/CM6 (presque) parfait


    # --- CM Yandex-like ---
    #thetas = [0.5, 0.3333, 0.2, 0.1667, 0.1429, 0.1111, 0.1, 0.0909, 0.0769, 0.0714]    # CM
    #thetas = [0.5, 0.3333, 0.2, 0.1667, 0.1429, 0.1111]
    #kappas = [1.0, 0.48574928517746124, 0.3297979789061402]  # KDD-PBM
    #thetas = [0.04998115, 0.04669623, 0.03660217, 0.03430312, 0.03062911]
    #thetas = [0.04998115, 0.04669623, 0.03660217, 0.03430312]


    # --- situation "non-strict" ---
    #kappas = [1., 0.95, 0.9, 0.85, 0.8]
    #thetas = [0.95, 0.95, 0.95, 0.95, 0.05, 0.03]

    # --- test simple ---
    #kappas = [1., 0.9, 0.8]
    kappas = [1., 0.8, 0.9]
    #kappas = [1., 0.4, 0.6]
    #thetas = [0.7, 0.65, 0.6, 0.55, 0.4, 0.35]
    #thetas = [0.9, 0.7, 0.3]
    thetas = [0.3, 0.7, 0.4, 0.1]

    # --- simul in ICML'22 ---
    kappas = [1, 0.9, 0.83, 0.78, 0.75]
    kappas = [1, 0.75, 0.5,0.25, 0.000001]
    thetas = [0.1, 0.08, 0.06, 0.04, 0.02, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

    thetas = [1, 0.75, 0.25, 0.5, 0.000001]
    # thetas = [1, 0.5, 0.5, 0.1, 0.1]
    nb_trials = 1000000
    nb_records = 10000
    nb_games = 1


    run_exp(thetas=np.array(thetas), kappas=np.array(kappas), is_pbm=True, nb_trials=nb_trials, nb_records=nb_records, nb_games=nb_games, nb_relevant_positions=None)
