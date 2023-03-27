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


# %%
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
                print(rec['label'], rec['thetas'][0:3], rec['kappas']) 

    else:
        referee_ob = Referee (None, -1, all_time_record=True)
    
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
        #yErrorValues = []
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


# %%
# cd ..

# %% [markdown]
"""
## PBM_decrease
"""

# %%
resdir = "exp_ICML2021/results/simul/"


refs={}
for env_name in ["std", "xxsmall", "big"]: 
    refs[env_name] = {}
    refs[env_name]['UniPBrank'] = retrieve_data_from_zip(resdir +f'purely_simulated__{env_name}__shuffled_kappa__Bandit_UniPBRank_10000000_T__games_10000000_nb_trials_1000_record_length_20_games.gz')


# %% jupyter={"outputs_hidden": true}
resdir = "exp_IDA/results/simul/"

for env_name in ["std", "xxsmall", "big"]: 
    #refs[env_name] = {}
    refs[env_name]['TopRank'] = {}
    refs[env_name]['TopRank']['PBM_decrease'] = retrieve_data_from_zip(resdir +f'purely_simulated__{env_name}__sorted_kappa__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name]['KL-MLMR'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__sorted_kappa__Bandit_KL-MLMR_10000000_horizon__games_10000000_nb_trials_1000_record_length_20_games.gz')


# %%
for env_name in ["std", "xxsmall", "big"]: 
    resdir = "exp_CIKM2020/result/simul/"

    for c in [10000.]:
        refs[env_name][f'eGreedy_c_{c}'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__Bandit_EGreedy_SVD_{c}_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')


    resdir = "exp_AAAI2021/results/simul/"

    cs = [1000.]
    proposal_names = ['TGRW']

    for proposal_name,c in product(proposal_names,cs):
        refs[env_name][f'PB-MHB, proposal={proposal_name}, c={c}, m=1'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__Bandit_PB-MHB_warm-up_start_1_step_{proposal_name}_{c}_c_vari_sigma_proposal__games_10000000_nb_trials_1000_record_length_20_games.gz')
    
    refs[env_name][f'PMED']={}
    
    if env_name =="xxsmall":
        refs[env_name][f'PMED']['PBM_decrease'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_10_games.gz')
    else:
        refs[env_name][f'PMED']['PBM_decrease'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_50000_nb_trials_1000_record_length_5_games.gz')

    for delta in [0.1]:
        refs[env_name][f'TopRank_delta_{delta}'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__extended_kappas__sorted_kappa__Bandit_TopRank_oracle_{delta}_delta____games_100000_nb_trials_1000_record_length_20_games.gz')


# %%
for rec in refs['std']['TopRank'].record_results['env_parameters'][:10]:
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

for env_name in ['std',"xxsmall","big"]:
    plt.figure(figsize=(3, 3))

    myplot(ref = refs[env_name]['eGreedy_c_10000.0'], label='$\epsilon_n$-greedy, c=$10^4$', color = 'C1', linestyle = '-')

    myplot(ref = refs[env_name]['PB-MHB, proposal=TGRW, c=1000.0, m=1'], label='PB_MHB, c=$10^3$, m=1', color = 'C4', linestyle = '-')
    myplot(ref = refs[env_name]['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')

    #myplot(ref = refs[env_name]['KL-MLMR'], label='MLMR', color = 'C4', linestyle = '-')

    myplot(ref = refs[env_name]['TopRank']['PBM_decrease'], label='TopRank', color = 'C2', linestyle = '--')
    #myplot(ref = refs[env_name]['TopRank_delta_0.1'], label='TopRank $delta$=0.1', color = 'C2', linestyle = '--')

    myplot(ref = refs[env_name]['PMED']['PBM_decrease'], label='PMED', color = 'C5', linestyle = '--')



    plt.xlabel('Time-stamp')
    plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    #plt.legend()
    plt.grid(True)
    plt.loglog()
#plt.xscale('log')
    if env_name =="xxsmall":
        plt.xlim([100, 10000000])
        plt.ylim([0.1, 2000])
    else:
        plt.xlim([5, 10000000])
        plt.ylim([1, 200000])
    plt.savefig(f"exp_ICML2021/results/graph/{env_name}_PBM_decrease.pdf", bbox_inches = 'tight',
    pad_inches = 0)

# %%
