#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from bandits_to_rank.environment import Environment_PBM
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

    # --- run PMED ---
    if True:
        from bandits_to_rank.opponents.pmed import PMED
        for alpha in [0.001, 1., 1000.]:
            player = PMED(nb_arms=nb_prop, nb_positions=nb_positions, alpha=alpha)
            referees[f'PMED, $\\alpha={alpha}$'] = run_games(player=player, env=env, nb_trials=nb_trials, nb_records=nb_records, nb_games=nb_games)

    # --- run TOP_RANK ---
    if True:
        from bandits_to_rank.opponents.top_rank import TOP_RANK
        kappasbis = kappas + [0]*(len(thetas)-len(kappas))
        envbis = Environment_PBM(thetas, kappasbis, label="purely simulated")
        player = TOP_RANK(nb_arms=nb_prop, T=nb_trials, discount_factor=kappasbis)
        referees['TopRank'] = run_games(player, envbis, nb_trials, nb_records, nb_games=nb_games)

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

    plt.subplot(2, 2, 2)
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
    kappas = [1, 0.9, 0.8, 0.7, 0.5]
    thetas = [0.9, 0.6, 0.4, 0.2, 0.1]
    thetas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    kappas = [1, 0.9, 0.8]
    thetas = [0.9, 0.6, 0.4]


    nb_trials = 100
    nb_records = 20
    nb_games = 2

    run_exp(thetas=thetas, kappas=kappas, nb_trials=nb_trials, nb_records=nb_records, nb_games=nb_games)