#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from bandits_to_rank.environment import Environment_PBM
from bandits_to_rank.opponents import UTS_Algorithm_AG
from bandits_to_rank.opponents import greedy
from bandits_to_rank.referee import Referee
import matplotlib.pyplot as plt
import numpy as np
"""
# --- prepare data ---
kappas = [1, 0.75, 0.6, 0.3, 0.1]
thetas = [1, 0.75, 0.6, 0.3, 0.1]

env = Environment_PBM(thetas, kappas, label="purely simulated, std")

# --- prepare referee ---
nb_trials = 1000000
nb_records = 100

referee = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records)


# --- prepare player ---
max_length = 100
nb_prop, nb_place = env.get_setting()
player = UTS_Algorithm_AG.UTS_iOSUB(nb_prop, nb_place, max_length)



# --- run one game ---
referee.play_game(player)


# --- one plot ---
type_errorbar = 'standart_error'
color = 'red'
linestyle = '-'
label = '$\\varepsilon$-greedy'

trials = referee.get_recorded_trials()
mu, d_10, d_90 = referee.get_regret_expected()
plt.plot(trials, mu, color=color, linestyle=linestyle, label=label)
if type_errorbar is not None:
    X_val, Y_val, Yerror = referee.barerror_value(type_errorbar=type_errorbar)
    nb_trials=len(X_val)
    spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
    spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
    spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
    #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
    #fmt = 'none', capsize = 0, ecolor = color)
neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]
plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')

plt.xlabel('Time')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
#plt.xlim([1000, 20000000])
#plt.ylim([70, 200000])

plt.show()
"""

def multiple_run(nb_run):
    arrays = []
    for _ in range(nb_run):
        # --- prepare data ---
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.3, 0.2, 0.15, 0.01, 0.01]

        env = Environment_PBM(thetas, kappas, label="purely simulated, std")

        # --- prepare referee ---
        nb_trials = 10000
        nb_records = 100

        referee = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records)

        # --- prepare player ---
        c = 1000
        maj = 1
        max_length = 100
        nb_prop, nb_place = env.get_setting()
        player = greedy.greedy_EGreedy(c, nb_prop, nb_place, maj)

        # --- run one game ---
        referee.play_game(player)

        # --- one plot ---
        type_errorbar = 'standart_error'
        color = 'red'
        linestyle = '-'
        label = '$\\varepsilon$-greedy'
        mu, d_10, d_90 = referee.get_regret_expected()
        trials = referee.get_recorded_trials()
        arrays.append(mu)
        player.clean()
    nb_array = len(arrays)
    nb_element = len(arrays[0])
    arr = []
    for i in range(nb_element):
        for array in arrays:
            arr.append(array[i])
    arr = np.array(arr)
    arr = arr.reshape([nb_element, nb_array])
    mean_arr = np.array([np.mean(arr_i) for arr_i in arr])
    var_arr = np.array([np.var(arr_i) for arr_i in arr])
    inf = mean_arr - np.sqrt(var_arr)
    sup = mean_arr + np.sqrt(var_arr)
    x = trials
    plt.plot(x, mean_arr)
    plt.plot(x, inf)
    plt.plot(x, sup)
    plt.fill_between(x, mean_arr, inf, alpha=0.3)
    plt.fill_between(x, mean_arr, sup, alpha=0.3)
    plt.show()

multiple_run(10)
if __name__ == "__main__":
    import doctest

    doctest.testmod()
