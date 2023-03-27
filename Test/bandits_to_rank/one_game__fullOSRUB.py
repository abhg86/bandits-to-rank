#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from bandits_to_rank.environment import Environment_PBM
from bandits_to_rank.opponents.OSRUB_BAL import OSUB
from bandits_to_rank.opponents.OSRUB_URM_BAL import OSUB_finit_memory
from bandits_to_rank.opponents.OSRUB_PBM_BAL import OSUB_PBM
from bandits_to_rank.opponents.BubbleOSRUB_BAL import BUBBLEOSUB
from bandits_to_rank.referee import Referee
import matplotlib.pyplot as plt


# --- prepare data ---
#kappas = sorted([0.3, 0.7, 0.5, 0.2, 0.9, 0.6, 0.75, 0.8, 0.4, 0.25]) #sorted([0.7, 0.5, 0.9, 0.6, 0.8, 0.4]) #sorted([0.3, 0.55, 0.7, 0.5, 0.35, 0.2, 0.45, 0.9, 0.85, 0.6, 0.75, 0.8, 0.65, 0.4, 0.25]) #[0.5, 0.4, 0.6, 0.45, 0.55] #[0.5, 0.1, 0.9, 0.3, 0.7]
#thetas = sorted([0.4, 0.7, 0.3, 0.1, 0.9, 0.5, 0.15, 0.8, 0.6, 0.2]) #sorted([0.4, 0.7, 0.9, 0.5, 0.8, 0.6]) #sorted([0.55, 0.4, 0.7, 0.3, 0.65, 0.1, 0.85, 0.5, 0.15, 0.8, 0.45, 0.6, 0.35, 0.2, 0.75]) #[0.2, 0.3, 0.1, 0.25, 0.15] #[0.5, 0.9, 0.1, 0.7, 0.3]

kappas = [0.3, 0.7, 0.5, 0.2, 0.9, 0.6, 0.75, 0.8, 0.4, 0.25]
thetas = [0.4, 0.7, 0.3, 0.1, 0.9, 0.5, 0.15, 0.8, 0.6, 0.2]
#kappas = [1, 0.6, 0.3]
#thetas = [0.9, 0.6, 0.1]

#kappas = [1, 0.3, 0.6]
#thetas = [0.6, 0.9,  0.1]
env = Environment_PBM(thetas, kappas, label="purely simulated, std")


# --- prepare referee ---
nb_trials = 1000000
nb_records = 100

#referee_OSUB = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records)
referee_OSUB_P = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records)
#referee_OSUB_F = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records)
#referee_OSUB_B = Referee(env, nb_trials, all_time_record=False, len_record_short=nb_records)


# --- prepare player ---
nb_prop, nb_positions = env.get_setting()
#player_OSUB = OSUB(nb_prop, nb_trials, nb_positions)
player_OSUB_P = OSUB_PBM(nb_prop, nb_trials, nb_positions)
#player_OSUB_F = OSUB_finit_memory(nb_prop, nb_trials, nb_positions)
#player_OSUB_B = BUBBLEOSUB(nb_prop, nb_trials, nb_positions)


# --- run one game ---
#referee_OSUB.play_game(player_OSUB)
referee_OSUB_P.play_game(player_OSUB_P)
#referee_OSUB_F.play_game(player_OSUB_F)
#referee_OSUB_B.play_game(player_OSUB_B)

# --- one plot ---
type_errorbar = 'standart_error'
color = 'red'
linestyle = '-'

'''label = 'OSUB'
trials = referee_OSUB.get_recorded_trials()
mu_OSUB, d_10, d_90 = referee_OSUB.get_regret_expected()
plt.plot(trials, mu_OSUB, color='orange', linestyle=linestyle, label=label)
'''
label = 'OSRUB_PBM'
trials = referee_OSUB_P.get_recorded_trials()
mu_OSUB_P, d_10, d_90 = referee_OSUB_P.get_regret_expected()
plt.plot(trials, mu_OSUB_P, color='blue', linestyle=linestyle, label=label)
'''
label = 'OSRUB_finit'
trials = referee_OSUB_F.get_recorded_trials()
mu_OSUB_F, d_10, d_90 = referee_OSUB_F.get_regret_expected()
plt.plot(trials, mu_OSUB_F, color='green', linestyle=linestyle, label=label)

label = 'BUBBLEOSRUB_BAL'
trials = referee_OSUB_B.get_recorded_trials()
mu_OSUB_B, d_10, d_90 = referee_OSUB_B.get_regret_expected()
plt.plot(trials, mu_OSUB_B, color=color, linestyle=linestyle, label=label)

if type_errorbar is not None:
    X_val, Y_val, Yerror = referee_OSUB.barerror_value(type_errorbar=type_errorbar)
    nb_trials = len(X_val)
    spars_X_val = [X_val[i] for i in range(0, nb_trials, 200)]
    spars_Y_val = [Y_val[i] for i in range(0, nb_trials, 200)]
    spars_Yerror = [Yerror[i] for i in range(0, nb_trials, 200)]
    #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
    #fmt = 'none', capsize = 0, ecolor = color)
neg_Yerror = [mu_OSUB[i]-Yerror[i] for i in range(len(Yerror))]
pos_Yerror = [mu_OSUB[i]+Yerror[i] for i in range(len(Yerror))]
plt.fill_between(X_val, neg_Yerror, pos_Yerror , color=color, alpha=0.3, linestyle=linestyle, label='')
'''
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
