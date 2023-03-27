# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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

# %% jupyter={"outputs_hidden": true}
resdir = "./results/simul/"

refs = {}
refs['PBM_desorder'] = {}
refs['PBM_desorder']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__Bandit_OSUB_10000000_memory__games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['PBM_desorder']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__Bandit_OSUB_inf_memory__games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['PBM_desorder']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__Bandit_OSUB_PBM_100000000_T__games_100000000_nb_trials_1000_record_length_20_games.gz')

#refs['PBM_desorder']['TopRank_0001'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__extended_kappas__Bandit_TopRank_greedy_0.001_delta____games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['PBM_desorder']['TopRank'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_20_games.gz')
refs['PBM_desorder']['MLMR'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_20_games.gz')

refs['PBM_desorder']['PB_MHB'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_20_games.gz')

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

plt.figure(figsize=(8, 8))


myplot(ref = refs['PBM_desorder']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['PBM_desorder']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['PBM_desorder']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

#myplot(ref = refs['PBM_desorder']['TopRank_0001'], label='TopRank, $\delta$=0.001', color = 'C2', linestyle = '--')
myplot(ref = refs['PBM_desorder']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

myplot(ref = refs['PBM_desorder']['PB_MHB'], label='PB_MHB, c=100, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['PBM_desorder']['MLMR'], label='MLMR', color = 'C5', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 2000000])
plt.ylim([1, 20000])
#plt.savefig("./result/graph/Yandex__eGeeedy_20g_10000000t.pdf", bbox_inches = 'tight',
#    pad_inches = 0)


# %% [markdown]
"""
## PBM_ order 
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./results/simul/"


refs['PBM_order'] = {}
refs['PBM_order']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__sorted_kappa__Bandit_OSUB_10000000_memory__games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['PBM_order']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__sorted_kappa__Bandit_OSUB_inf_memory__games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['PBM_order']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__sorted_kappa__Bandit_OSUB_PBM_100000000_T__games_100000000_nb_trials_1000_record_length_20_games.gz')

refs['PBM_order']['TopRank'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__sorted_kappa__extended_kappas__Bandit_TopRank_greedy_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_20_games.gz')

refs['PBM_order']['MLMR'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__sorted_kappa__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_20_games.gz')

refs['PBM_order']['PB_MHB'] = retrieve_data_from_zip(resdir + 'purely_simulated__test__sorted_kappa__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_20_games.gz')

# %%
for rec in refs['PBM_order']['OSUB_PBM'].record_results['env_parameters'][:10]:
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

plt.figure(figsize=(8, 8))


myplot(ref = refs['PBM_order']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['PBM_order']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['PBM_order']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

myplot(ref = refs['PBM_order']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

myplot(ref = refs['PBM_order']['PB_MHB'], label='PB_MHB, c=100, m=1', color = 'C3', linestyle = '-')
myplot(ref = refs['PBM_desorder']['PB_MHB'], label='PB_MHB, c=100, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['PBM_order']['MLMR'], label='MLMR', color = 'C5', linestyle = '-')


myplot(ref = refs['PBM_desorder']['TopRank'], label='TopRank', color = 'C2', linestyle = '--')

myplot(ref = refs['PBM_desorder']['PB_MHB'], label='PB_MHB, c=100, m=1', color = 'C3', linestyle = '--')


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 2000000])
plt.ylim([1, 20000])
#plt.savefig("./result/graph/Yandex__eGeeedy_20g_10000000t.pdf", bbox_inches = 'tight',
#    pad_inches = 0)


# %% [markdown]
"""
## CM_order
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./results/simul/"

refs = {}
refs['CM_order'] = {}
refs['CM_order']['OSUB'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM__Bandit_OSUB_inf_memory__games_100000_nb_trials_1000_record_length_20_games.gz')
refs['CM_order']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM__Bandit_OSUB_PBM_100000_T__games_100000_nb_trials_1000_record_length_20_games.gz')

refs['CM_order']['TopRank'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM__Bandit_TopRank_100000.0_delta_TimeHorizonKnown___games_100000_nb_trials_1000_record_length_20_games.gz')

refs['CM_order']['MLMR'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM__Bandit_KL-MLMR_100000_horizon__games_100000_nb_trials_1000_record_length_20_games.gz')

refs['CM_order']['PB_MHB'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_100000_nb_trials_1000_record_length_20_games.gz')

refs['CM_order']['PMED'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_20_games.gz')

# %%
for rec in refs['CM_order']['PMED'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas'][0:8]), np.array(rec['order_view'][:])) 


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

plt.figure(figsize=(3.5, 3.5))


myplot(ref = refs['CM_order']['OSUB'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['CM_order']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

myplot(ref = refs['CM_order']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')

myplot(ref = refs['CM_order']['PB_MHB'], label='PB_MHB, c=1000, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['CM_order']['MLMR'], label='MLMR', color = 'C6', linestyle = '-')

myplot(ref = refs['CM_order']['PMED'], label='PMED', color = 'C5', linestyle = '-')


plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([1, 200000])
#plt.ylim([0.2, 20000])
plt.savefig("./results/graph/test_CM_order.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## CM_desorder
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./results/simul/"

#refs = {}
refs['CM_desorder'] = {}
refs['CM_desorder']['OSUB_memory_107'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM_order_view_shuffle__Bandit_OSUB_10000000_memory__games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['CM_desorder']['OSUB_inf_memory'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM_order_view_shuffle__Bandit_OSUB_inf_memory__games_100000000_nb_trials_1000_record_length_20_games.gz')
refs['CM_desorder']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM_order_view_shuffle__Bandit_OSUB_PBM_100000000_T__games_100000000_nb_trials_1000_record_length_20_games.gz')

refs['CM_desorder']['PB_MHB'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM_order_view_shuffle__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_20_games.gz')

refs['CM_desorder']['MLMR'] = retrieve_data_from_zip(resdir + 'purely_simulated__test_CM_order_view_shuffle__Bandit_MLMR_2.0_exploration__games_1000000_nb_trials_1000_record_length_20_games.gz')

refs['CM_desorder']['TopRank'] = retrieve_data_from_zip(resdir +'purely_simulated__test_CM_order_view_shuffle__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_20_games.gz')


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

plt.figure(figsize=(8, 8))


myplot(ref = refs['CM_desorder']['OSUB_memory_107'], label='Osub, memory =$10^7$', color = 'C1', linestyle = '--')
myplot(ref = refs['CM_desorder']['OSUB_inf_memory'] , label='Osub', color = 'C1', linestyle = '-')
myplot(ref = refs['CM_desorder']['OSUB_PBM'], label='Osub PBM', color = 'C4', linestyle = '-')

myplot(ref = refs['CM_desorder']['TopRank'], label='TopRank', color = 'C2', linestyle = '-')
myplot(ref = refs['CM_order']['TopRank'], label='TopRank', color = 'C2', linestyle = '--')

myplot(ref = refs['CM_desorder']['PB_MHB'], label='PB_MHB, c=100, m=1', color = 'C3', linestyle = '-')

myplot(ref = refs['CM_desorder']['MLMR'], label='MLMR', color = 'C5', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([1, 200000000])
plt.ylim([0, 2000000])
#plt.savefig("./result/graph/Yandex__eGeeedy_20g_10000000t.pdf", bbox_inches = 'tight',
#    pad_inches = 0)


# %% [markdown]
"""
## Timming 
"""

# %%
time_sec_to_HMS((np.array(refs['CM_order']['OSUB'].record_results['time_to_play'])[:,-1].mean())*10)

# %%
