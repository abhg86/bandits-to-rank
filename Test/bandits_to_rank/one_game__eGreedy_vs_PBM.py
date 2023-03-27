#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from bandits_to_rank.environment import Environment_PBM
from bandits_to_rank.opponents import greedy
from bandits_to_rank.referee import Referee

import matplotlib.pyplot as plt
import numpy as np


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

nb_prop, nb_place = env.get_setting()
player = greedy.greedy_EGreedy(c, nb_prop, nb_place, maj)

def run_games(player, env, nb_trials, nb_records, nb_games=10):
    # --- prepare referee ---
    ref = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records, print_trial=20)


    # --- run one game ---
    for _ in range(nb_games):
        player.clean()
        ref.play_game(player)

    return ref


def run_exp(thetas, kappas, nb_trials, nb_records, nb_games):
    env = Environment_PBM(thetas, kappas, label="purely simulated")
    nb_prop, nb_positions = env.get_setting()

    # --- logs ---
    referees = {}

    # --- run epsilon-greedy ---
    if True:
        for c in [1., 10.**2, 10.**4]:
            nb_prop, nb_place = env.get_setting()
            player = glouton.Glouton_EGreedy(c=c, nb_arms=nb_prop, nb_position=nb_positions, count_update=1)
            referees[f'$\\epsilon$-greedy, c={c}'] = run_games(player, env, nb_trials, nb_records, nb_games=nb_games)

    # --- one plot ---
    type_errorbar = 'standart_error'

    plt.subplot(2, 2, 1)
    for i, key in enumerate(referees.keys()):
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

    plt.show()


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    # --- prepare data ---
    kappas = [1, 0.75, 0.6, 0.3, 0.1]
    thetas = [0.3, 0.2, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.01, 0.01]

    nb_trials = 10000
    nb_records = 100
    nb_games = 10

    run_exp(thetas=thetas, kappas=kappas, nb_trials=nb_trials, nb_records=nb_records, nb_games=nb_games)