# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Get Graphics
"""

# %%
# To ease pairing with .py file
# %autosave 0
# %reload_ext autoreload
# %autoreload 2 

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

np.set_printoptions(precision=3)


# %%
refs = {}

def retrieve_data_from_zip(file_name, my_assert=True, accept_empty_file=True):
    if os.path.isfile(file_name):
        with gzip.GzipFile(file_name, 'r') as fin:    
            json_bytes = fin.read()

        json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
        data = json.loads(json_str)  
        referee_ob = Referee (None, -1, all_time_record=True)
        referee_ob.record_results = data
        if my_assert:
            print(file_name)
            for rec in referee_ob.record_results['env_parameters'][:4]:
                try:
                    print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['kappas'][0:3])) 
                except:
                    try:
                        print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['order_view'][:]))
                    except:
                        print(rec['label'], np.array(rec['thetas'][0:3]))


    else:
        print(f'!!! unknown file: {file_name}')
        referee_ob = None
    
    return referee_ob



# %%
def time_sec_to_HMS(sec):
    heure=sec//3600
    rest_h=sec%3600
    minute=rest_h//60
    rest_m=rest_h%60

    return(str(int(heure))+'H '+str(int(minute))+'min '+str(int(rest_m))+'sec')


# %%
def error_value(ref,stat_to_draw,type_errorbar='standart_error'):
        nb_game = len(stat_to_draw)
        xValues = ref.record_results['time_recorded'][0]
        regret_pertrial =np.mean(stat_to_draw,axis=1)
        yValues = regret_pertrial
        yErrorValues = []
        yErrorValues=np.std(stat_to_draw,axis=1)
        if type_errorbar=='std':
            return xValues,yValues,yErrorValues
        elif type_errorbar=='standart_error':
            yErrorValues/=sqrt(nb_game)
            return xValues,yValues,yErrorValues
        elif type_errorbar=='confidence':
            yErrorValues/=sqrt(nb_game)
            yErrorValues*=4
            return xValues,yValues,yErrorValues


# %% [markdown]
"""
## PBM_ desorder 
"""

# %%
resdir = "./results/real_Yandex/"

refs['PBM_desorder'] = {}
refs['PBM_desorder']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_OSUB_10000000_memory__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_desorder']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_OSUB_inf_memory__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_desorder']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_OSUB_PBM_1000000_T__games_1000000_nb_trials_1000_record_length_200_games.gz')

#refs['PBM_desorder']['TopRank_0001'] = retrieve_data_from_zip(resdir + 'Yandex_all__extended_kappas__Bandit_TopRank_greedy_0.001_delta____games_100000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_desorder']['TopRank'] = retrieve_data_from_zip(resdir +'Yandex_all__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_desorder']['PB_MHB'] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_desorder']['MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_desorder']['PMED'] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_202_games.gz')

# %%
for rec in refs['PBM_desorder']['MLMR'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['kappas'][:])) 


# %%

def myplot(ref, label, color, linestyle,type_errorbar='standart_error'):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
        if type_errorbar is not None:
            X_val,Y_val,Yerror=ref.barerror_value(type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(3, 3))


#myplot(ref = refs['PBM_desorder']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['PBM_desorder']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['PBM_desorder']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

#myplot(ref = refs['PBM_desorder']['TopRank_0001'], label='TopRank, $\delta$=0.001', color = 'C2', linestyle = '--')
myplot(ref = refs['PBM_desorder']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

myplot(ref = refs['PBM_desorder']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['PBM_desorder']['MLMR'], label='MLMR', color = 'C6', linestyle = '-')

myplot(ref = refs['PBM_desorder']['PMED'], label='PMED', color = 'C5', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 1000000])
plt.ylim([1, 200000])
plt.savefig("./results/graph/Yandex_PBM_desorder.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## PBM_ order 
"""

# %%
resdir = "./results/real_Yandex/"


refs['PBM_order'] = {}
refs['PBM_order']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_10000000_memory__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_order']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_inf_memory__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_order']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_PBM_1000000_T__games_1000000_nb_trials_1000_record_length_200_games.gz')

#refs['PBM_order']['TopRank_0001'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__extended_kappas__Bandit_TopRank_greedy_0.001_delta____games_100000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_order']['TopRank'] = retrieve_data_from_zip(resdir +'Yandex_all__sorted_kappa__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_order']['PB_MHB_100'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_PB-MHB_warm-up_start_1_step_TGRW_100.0_c_vari_sigma_proposal__games_100000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_order']['PB_MHB'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_order']['MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_order']['KL-MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_KL-MLMR_1000000_horizon__games_1000000_nb_trials_1000_record_length_200_games.gz')


refs['PBM_order']['PMED'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_210_games.gz')

# %%
for rec in refs['PBM_order']['KL-MLMR'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['kappas'][:])) 


# %%

def myplot(ref, label, color, linestyle,type_errorbar='standart_error'):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
        if type_errorbar is not None:
            X_val,Y_val,Yerror=ref.barerror_value(type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(5, 5))


myplot(ref = refs['PBM_order']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['PBM_order']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['PBM_order']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

#myplot(ref = refs['PBM_order']['TopRank_0001'], label='TopRank, $\delta$=0.001', color = 'C2', linestyle = '--')
#myplot(ref = refs['PBM_desorder']['TopRank'], label='TopRank_desorder', color = 'C2', linestyle = '--')
myplot(ref = refs['PBM_order']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

#myplot(ref = refs['PBM_order']['PB_MHB_100'], label='PB_MHB, c=100, m=1', color = 'C3', linestyle = '--')
myplot(ref = refs['PBM_order']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['PBM_order']['PMED'], label='PMED', color = 'C5', linestyle = '-')

myplot(ref = refs['PBM_order']['KL-MLMR'], label='MLMR', color = 'C6', linestyle = '-')


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 1000000])
plt.ylim([1, 200000])
plt.savefig("./results/graph/Yandex_PBM_order.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## PBM_ increase 
"""

# %%
shortresdir = "./results/"
resdir = "./results/real_Yandex/"


refs['PBM_increase'] = {}
refs['PBM_increase']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__Bandit_OSUB_inf_memory__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_increase']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__Bandit_OSUB_PBM_1000000_T__games_1000000_nb_trials_1000_record_length_200_games.gz')

#refs['PBM_order']['TopRank_0001'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__extended_kappas__Bandit_TopRank_greedy_0.001_delta____games_100000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_increase']['TopRank'] = retrieve_data_from_zip(resdir +'Yandex_all__increasing_kappa__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_increase']['PB_MHB'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_increase']['PB_MHB_except'] = retrieve_data_from_zip(shortresdir + 'Yandex_all__increasing_kappa_except_first__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_increase']['MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_increase']['KL-MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__Bandit_KL-MLMR_1000000_horizon__games_1000000_nb_trials_1000_record_length_200_games.gz')


refs['PBM_increase']['PMED'] = retrieve_data_from_zip(resdir + 'Yandex_all__increasing_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_200_games.gz')

# %%
for rec in refs['PBM_increase']['PMED'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['kappas'][:])) 


# %%

def myplot(ref, label, color, linestyle,type_errorbar='standart_error'):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
        if type_errorbar is not None:
            X_val,Y_val,Yerror=ref.barerror_value(type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(5, 5))


myplot(ref = refs['PBM_increase']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['PBM_increase']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

myplot(ref = refs['PBM_increase']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

#myplot(ref = refs['PBM_increase']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')
myplot(ref = refs['PBM_increase']['PB_MHB_except'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['PBM_increase']['PMED'], label='PMED', color = 'C5', linestyle = '-')

myplot(ref = refs['PBM_increase']['KL-MLMR'], label='MLMR', color = 'C6', linestyle = '-')


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 1000000])
plt.ylim([1, 200000])
plt.savefig("./results/graph/Yandex_PBM_increase.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## PBM altogether
"""


# %%

def myplot(ref, label, color, linestyle,type_errorbar='standart_error'):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
        if type_errorbar is not None:
            X_val,Y_val,Yerror=ref.barerror_value(type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(8, 8))

for setting, linestyle in zip(['PBM_order', 'PBM_desorder', 'PBM_increase'], ['-', '--', ':']):
    myplot(ref = refs[setting]['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = linestyle)
    myplot(ref = refs[setting]['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = linestyle)

    myplot(ref = refs[setting]['TopRank'], label='TopRank', color = 'C2', linestyle = linestyle)

    myplot(ref = refs[setting]['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = linestyle)
    try:
        myplot(ref = refs[setting]['PB_MHB_except'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = linestyle)
    except:
        pass

    myplot(ref = refs[setting]['PMED'], label='PMED', color = 'C5', linestyle = linestyle)

    myplot(ref = refs[setting]['MLMR'], label='MLMR', color = 'C6', linestyle = linestyle)


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 1000000])
plt.ylim([1, 200000])
plt.savefig("./results/graph/Yandex_PBM_increase.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## CM_ order 
"""

# %%
resdir = "./results/real_Yandex/"


refs['CM_order'] = {}
#refs['CM_order']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_10000000_memory__games_100000_nb_trials_1000_record_length_200_games.gz')
refs['CM_order']['OSUB_inf_memory_105'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_OSUB_inf_memory__games_100000_nb_trials_1000_record_length_200_games.gz')
refs['CM_order']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_OSUB_inf_memory__games_1000000_nb_trials_1000_record_length_33_games.gz')

refs['CM_order']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_OSUB_PBM_1000000_T__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_order']['TopRank_greedy'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__extended_kappas__Bandit_TopRank_greedy_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_order']['TopRank'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_order']['PB_MHB'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_order']['MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_200_games.gz')
refs['CM_order']['KL-MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_KL-MLMR_1000000_horizon__games_1000000_nb_trials_1000_record_length_200_games.gz')


refs['CM_order']['PMED'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_200_games.gz')

# %%
for rec in refs['CM_order']['OSUB_inf_memory'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['order_view'][:])) 


# %%

def myplot(ref, label, color, linestyle,type_errorbar='standart_error'):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
        if type_errorbar is not None:
            X_val,Y_val,Yerror=ref.barerror_value(type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(5, 5))


#myplot(ref = refs['CM_order']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['CM_order']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['CM_order']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

#myplot(ref = refs['PBM_order']['TopRank_0001'], label='TopRank, $\delta$=0.001', color = 'C2', linestyle = '--')
myplot(ref = refs['CM_order']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')
#myplot(ref = refs['CM_order']['TopRank_greedy'], label='TopRank greedy+old CM', color = 'C2', linestyle = ':')


myplot(ref = refs['CM_order']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['CM_order']['PMED'], label='PMED', color = 'C5', linestyle = '-')

myplot(ref = refs['CM_order']['KL-MLMR'], label='MLMR', color = 'C6', linestyle = '-')
#myplot(ref = refs['CM_order']['MLMR'], label='MLMR', color = 'C6', linestyle = '--')


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 1000000])
plt.ylim([1, 200000])
plt.savefig("./results/graph/Yandex_CM_order.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## CM_ desorder 
"""

# %%
resdir = "./results/real_Yandex/"


refs['CM_desorder'] = {}
#refs['CM_desorder']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_10000000_memory__games_100000_nb_trials_1000_record_length_200_games.gz')
#refs['CM_desorder']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_inf_memory__games_100000_nb_trials_1000_record_length_200_games.gz')
refs['CM_desorder']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all_order_view_shuffle__Bandit_OSUB_PBM_1000000_T__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_desorder']['TopRank'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all_order_view_shuffle__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_desorder']['PB_MHB'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all_order_view_shuffle__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_desorder']['MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all_order_view_shuffle__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['CM_desorder']['PMED'] = retrieve_data_from_zip(resdir + 'Yandex_CM_all_order_view_shuffle__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_200_games.gz')

# %%
for rec in refs['CM_desorder']['MLMR'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas'][0:3]), np.array(rec['order_view'][:])) 


# %%

def myplot(ref, label, color, linestyle,type_errorbar='standart_error'):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
        if type_errorbar is not None:
            X_val,Y_val,Yerror=ref.barerror_value(type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(3, 3))


#myplot(ref = refs['CM_desorder']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
#myplot(ref = refs['CM_desorder']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['CM_desorder']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

#myplot(ref = refs['PBM_order']['TopRank_0001'], label='TopRank, $\delta$=0.001', color = 'C2', linestyle = '--')
myplot(ref = refs['CM_desorder']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')
#myplot(ref = refs['CM_order']['TopRank'], label='TopRank, order', color = 'C2', linestyle = '--')

myplot(ref = refs['CM_desorder']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['CM_desorder']['PMED'], label='PMED', color = 'C5', linestyle = '-')

myplot(ref = refs['CM_desorder']['MLMR'], label='MLMR', color = 'C6', linestyle = '-')


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 1000000])
plt.ylim([1, 20000])
plt.savefig("./results/graph/Yandex_CM_desorder.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
### legend
"""

# %%
pwd

# %%
#### Get only legend 
plt.figure(figsize=(8, 8))
ax = plt.axes([0.1, 0.1, 0.67, 0.8])
def myplot(ref, label, color, linestyle):
    trials = ref.get_recorded_trials()
    mu, d_10, d_90 = ref.get_regret_expected()
    plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
    #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)

        
#myplot(ref = refs['PBM_desorder']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['PBM_desorder']['OSUB_inf_memory'] , label='DCGUniRank', color = 'C1', linestyle = '-')
myplot(ref = refs['PBM_desorder']['OSUB_PBM'], label='GRAB', color = 'C4', linestyle = '-')

myplot(ref = refs['PBM_desorder']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

myplot(ref = refs['PBM_desorder']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['PBM_desorder']['PMED'], label='PMED', color = 'C5', linestyle = '-')

myplot(ref = refs['PBM_desorder']['MLMR'], label='KL-CombUCB1', color = 'C6', linestyle = '-')


figlegend = pylab.figure(figsize=(7.5,1))
ha, lb = ax.get_legend_handles_labels()
print(lb)
figlegend.legend(ha, lb,"center right", fontsize='small', ncol=6)

figlegend.savefig('./results/graph/Yandex_legend.pdf')
      


# %% [markdown]
"""
# timings
"""


# %%
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



# %%
referees = refs['PBM_order']
type_errorbar = 'standart_error'
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

# %%
time_sec_to_HMS(np.array(refs['PBM_order']['PMED'].record_results['time_to_play'])[:,-1].mean())

# %%
