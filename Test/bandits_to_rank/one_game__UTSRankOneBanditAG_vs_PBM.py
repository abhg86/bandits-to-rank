#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from bandits_to_rank.environment import Environment_PBM
from bandits_to_rank.opponents import UTS_Algorithm_AG
from bandits_to_rank.referee import Referee
import matplotlib.pyplot as plt


# --- prepare data ---
kappas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
thetas = [0.3, 0.2, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.01, 0.01]
kappas = [1, 0.9, 0.8, 0.7]
thetas = [1, 0.8, 0.6, 0.4]

env = Environment_PBM(thetas, kappas, label="purely simulated, std")

# --- prepare referee ---
nb_trials = 100000
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

if __name__ == "__main__":
    import doctest

    doctest.testmod()