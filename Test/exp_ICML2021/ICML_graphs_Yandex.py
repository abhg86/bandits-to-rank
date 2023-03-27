# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
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
## PBM_ decrease
"""

# %%
# cd ..

# %%
resdir = "exp_ICML2021/results/real_Yandex/"

refs={}
refs['PBM_decrease'] = {}

refs['PBM_decrease']['UniPBrank'] = retrieve_data_from_zip(resdir +'Yandex_all__shuffled_kappa__Bandit_UniPBRank_10000000_T__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_decrease']['Top_Rank107'] = retrieve_data_from_zip(resdir +'Yandex_all__sorted_kappa__extended_kappas__Bandit_TopRank_10000000.0_delta_TimeHorizonKnown___games_10000000_nb_trials_1000_record_length_193_games.gz')


# %% jupyter={"outputs_hidden": true}
resdir = "exp_AISTAT2021/results/real_Yandex/"


refs['PBM_decrease']['OSUB_PBM'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_OSUB_PBM_1000000_T__games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_decrease']['Top_Rank106'] = retrieve_data_from_zip(resdir +'Yandex_all__sorted_kappa__extended_kappas__Bandit_TopRank_1000000.0_delta_TimeHorizonKnown___games_1000000_nb_trials_1000_record_length_200_games.gz')

refs['PBM_decrease']['KL-MLMR'] = retrieve_data_from_zip(resdir + 'Yandex_all__sorted_kappa__Bandit_KL-MLMR_1000000_horizon__games_1000000_nb_trials_1000_record_length_200_games.gz')


# %%
resdir = "exp_CIKM2020/result/real_Yandex/"

for c in [10000.]:
    refs['PBM_decrease'][f'eGreedy_c_{c}'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_EGreedy_SVD_{c}_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')


resdir = "exp_AAAI2021/results/real_Yandex/"

cs = [1000.]
proposal_names = ['TGRW']

for proposal_name,c in product(proposal_names,cs):
   # print (proposal_name, c)
    refs['PBM_decrease'][f'PB-MHB, proposal={proposal_name}, c={c}, m=1'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_PB-MHB_warm-up_start_1_step_{proposal_name}_{c}_c_vari_sigma_proposal__games_10000000_nb_trials_1000_record_length_200_games.gz')


refs['PBM_decrease'][f'PMED'] = retrieve_data_from_zip(resdir + f'Yandex_all__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_197_games.gz')


for delta in [0.1]:
    refs['PBM_decrease'][f'TopRank_delta_{delta}'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__Bandit_TopRank_oracle_{delta}_delta____games_10000000_nb_trials_1000_record_length_200_games.gz')
 






# %%
refs['PBM_decrease'].keys()

# %%
for rec in refs['PBM_decrease']['PB-MHB, proposal=TGRW, c=1000.0, m=1'].record_results['env_parameters'][:10]:
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

myplot(ref = refs['PBM_decrease']['eGreedy_c_10000.0'], label='$\epsilon_n$-greedy, c=$10^4$', color = 'C1', linestyle = '-')

myplot(ref = refs['PBM_decrease']['PB-MHB, proposal=TGRW, c=1000.0, m=1'], label='PB_MHB, c=$10^3$, m=1', color = 'C4', linestyle = '-')
myplot(ref = refs['PBM_decrease']['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')
myplot(ref = refs['PBM_decrease']['OSUB_PBM'], label='OSUB_PBM', color = 'C3', linestyle = '--')

#myplot(ref = refs['PBM_decrease']['KL-MLMR'], label='MLMR', color = 'C4', linestyle = '-')

myplot(ref = refs['PBM_decrease']['Top_Rank107'], label='TopRank', color = 'C2', linestyle = '--')
myplot(ref = refs['PBM_decrease']['Top_Rank106'], label='TopRank T=$10^{6}$', color = 'C2', linestyle = '-.')

#myplot(ref = refs['PBM_decrease']['TopRank_delta_0.1'], label='TopRank $delta$=0.1', color = 'C2', linestyle = '--')

myplot(ref = refs['PBM_decrease']['PMED'], label='PMED', color = 'C5', linestyle = '--')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 10000000])
plt.ylim([1, 200000])
plt.savefig("exp_ICML2021/results/graph/Yandex_PBM_decrease.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## PBM_ equi_ 10 
"""

# %%
resdir = "exp_ICML2021/results/real_Yandex/"

refs={}
refs['PBM_equi_10'] = {}

refs['PBM_equi_10']['KL-MLMR'] = retrieve_data_from_zip(resdir +'Yandex_equi_all__shuffled_kappa__Bandit_KL-MLMR_10000000_horizon__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_equi_10']['OSUB_PBM'] = retrieve_data_from_zip(resdir +'Yandex_equi_all__shuffled_kappa__Bandit_OSUB_PBM_10000000_T__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_equi_10']['PB-MHB'] = retrieve_data_from_zip(resdir +'Yandex_equi_all__shuffled_kappa__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_equi_10']['UniPBrank'] = retrieve_data_from_zip(resdir +'Yandex_equi_all__shuffled_kappa__Bandit_UniPBRank_10000000_T__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_equi_10']['TopRank'] = retrieve_data_from_zip(resdir +'Yandex_equi_10_K_all__sorted_kappa__extended_kappas__Bandit_TopRank_10000000.0_delta_TimeHorizonKnown___games_10000000_nb_trials_1000_record_length_200_games.gz')




# %%
refs['PBM_equi_10']['Egreedy']={}
refs['PBM_equi_10']['Egreedy'][0.0] = retrieve_data_from_zip(resdir +f'Yandex_equi_10_K_all__Bandit_EGreedy_SVD_0.0_c_1_update__games_10000000_nb_trials_1000_record_length_150_games.gz')
refs['PBM_equi_10']['Egreedy'][1.0] = retrieve_data_from_zip(resdir +f'Yandex_equi_10_K_all__Bandit_EGreedy_SVD_1.0_c_1_update__games_10000000_nb_trials_1000_record_length_185_games.gz')

for c in [10., 100., 1000., 10000., 100000., 1000000., 10.**20]:
    refs['PBM_equi_10']['Egreedy'][c] = retrieve_data_from_zip(resdir +f'Yandex_equi_10_K_all__Bandit_EGreedy_SVD_{c}_c_1_update__games_10000000_nb_trials_1000_record_length_200_games.gz')


# %%
refs['PBM_equi_10'].keys()

# %%
refs['PBM_equi_10']['Egreedy'].keys()

# %%
for rec in refs['PBM_equi_10']['PB-MHB'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas']), np.array(rec['kappas'])) 


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
i=0
for c in [0., 1., 10., 100., 1000., 10000., 100000., 1000000., 10.**20]:
    i+=1
    myplot(ref = refs['PBM_equi_10']['Egreedy'][c], label=f'Egreedy, c = {c}', color = f'C{i}', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 10000000])
plt.ylim([1, 200000])
plt.savefig("exp_ICML2021/results/graph/Yandex_tun_Egreedy_equi_10.pdf", bbox_inches = 'tight',
    pad_inches = 0)


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

myplot(ref = refs['PBM_equi_10']['KL-MLMR'], label='KL-MLMR', color = 'C1', linestyle = '-')

myplot(ref = refs['PBM_equi_10']['PB-MHB'], label='PB_MHB, c=$10^3$, m=1', color = 'C6', linestyle = '-')
myplot(ref = refs['PBM_equi_10']['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')
myplot(ref = refs['PBM_equi_10']['OSUB_PBM'], label='OSUB_PBM', color = 'C3', linestyle = '--')
myplot(ref = refs['PBM_equi_10']['TopRank'], label='TopRank', color = 'C4', linestyle = '--')
myplot(ref = refs['PBM_equi_10']['Egreedy'][10000.0], label='Egreedy', color = 'C5', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 10000000])
plt.ylim([1, 200000])
plt.savefig("exp_ICML2021/results/graph/Yandex_PBM_equi.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %%
refs['PBM_equi_10']['UniPBrank_tunning'] = {}

for g in [4,15,20,30]:
    refs['PBM_equi_10']['UniPBrank_tunning'][g] = retrieve_data_from_zip(resdir +f'Yandex_equi_10_K_all__shuffled_kappa__Bandit_UniPBRank_100000_T_{g}_gamma__games_100000_nb_trials_1000_record_length_200_games.gz')


# %%
refs['PBM_equi_10']['UniPBrank_tunning'].keys()


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

myplot(ref = refs['PBM_equi_10']['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')
gamma= [4,15,20,30]
color=[1,2,4,5]
for i in range(4):
    myplot(ref = refs['PBM_equi_10']['UniPBrank_tunning'][gamma[i]] , label=f'UniPBrank, $\gamma$ = {gamma[i]}', color = f'C{color[i]}', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 10000000])
plt.ylim([1, 200000])
plt.savefig("exp_ICML2021/results/graph/Yandex_PBM_tunegamma_uniPBRank_equi_10.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## PBM_ equi_ 5
"""

# %%
resdir = "exp_ICML2021/results/real_Yandex/"

refs={}
refs['PBM_equi_5'] = {}

refs['PBM_equi_5']['KL-MLMR'] = retrieve_data_from_zip(resdir +'Yandex_equi_5_K_all__shuffled_kappa__Bandit_KL-MLMR_100000_horizon__games_100000_nb_trials_1000_record_length_200_games.gz')
refs['PBM_equi_5']['OSUB_PBM'] = retrieve_data_from_zip(resdir +'Yandex_equi_5_K_all__shuffled_kappa__Bandit_OSUB_PBM_100000_T__games_100000_nb_trials_1000_record_length_188_games.gz')
refs['PBM_equi_5']['PbubblePBRank'] = retrieve_data_from_zip(resdir +'Yandex_equi_5_K_all__shuffled_kappa__Bandit_PBubblePBRank_100000_T_10_gamma__games_100000_nb_trials_1000_record_length_186_games.gz')
refs['PBM_equi_5']['UniPBrank'] = retrieve_data_from_zip(resdir +'Yandex_equi_5_K_all__shuffled_kappa__Bandit_UniPBRank_100000_T_10_gamma__games_100000_nb_trials_1000_record_length_171_games.gz')
refs['PBM_equi_5']['TopRank'] = retrieve_data_from_zip(resdir +'Yandex_equi_5_K_all__sorted_kappa__extended_kappas__Bandit_TopRank_100000.0_delta_TimeHorizonKnown___games_100000_nb_trials_1000_record_length_200_games.gz')




# %%
refs['PBM_equi_5']['Egreedy']={}
for c in [0., 1., 10., 100.]:
    refs['PBM_equi_5']['Egreedy'][c] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__Bandit_EGreedy_SVD_0.0_c_1_update__games_100000_nb_trials_1000_record_length_130_games.gz')

refs['PBM_equi_5']['Egreedy'][1000] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__Bandit_EGreedy_SVD_1000.0_c_1_update__games_100000_nb_trials_1000_record_length_105_games.gz')
refs['PBM_equi_5']['Egreedy'][10000] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__Bandit_EGreedy_SVD_10000.0_c_1_update__games_100000_nb_trials_1000_record_length_80_games.gz')
refs['PBM_equi_5']['Egreedy'][100000] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__Bandit_EGreedy_SVD_100000.0_c_1_update__games_100000_nb_trials_1000_record_length_80_games.gz')
refs['PBM_equi_5']['Egreedy'][1000000] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__Bandit_EGreedy_SVD_1000000.0_c_1_update__games_100000_nb_trials_1000_record_length_128_games.gz')
refs['PBM_equi_5']['Egreedy'][10.**20] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__Bandit_EGreedy_SVD_1e+20_c_1_update__games_100000_nb_trials_1000_record_length_200_games.gz')


# %%
refs['PBM_equi_5'].keys()

# %%
for rec in refs['PBM_equi']['PB-MHB'].record_results['env_parameters'][:10]:
                print(rec['label'], np.array(rec['thetas']), np.array(rec['kappas'])) 


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
i=0
for c in [0., 1., 10., 100., 1000., 10000., 100000., 1000000., 10.**20]:
    i+=1
    myplot(ref = refs['PBM_equi_5']['Egreedy'][c], label=f'Egreedy, c = {c}', color = f'C{i}', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 100000])
plt.ylim([0.1, 200000])
plt.savefig("exp_ICML2021/results/graph/Yandex_tun_Egreedy_equi_5.pdf", bbox_inches = 'tight',
    pad_inches = 0)


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

myplot(ref = refs['PBM_equi_5']['KL-MLMR'], label='KL-MLMR', color = 'C1', linestyle = '-')

myplot(ref = refs['PBM_equi_5']['PbubblePBRank'], label='PbubblePBRank', color = 'C2', linestyle = '-')
myplot(ref = refs['PBM_equi_5']['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')
myplot(ref = refs['PBM_equi_5']['OSUB_PBM'], label='OSUB_PBM', color = 'C3', linestyle = '--')
myplot(ref = refs['PBM_equi_5']['TopRank'], label='TopRank', color = 'C4', linestyle = '--')
myplot(ref = refs['PBM_equi_5']['Egreedy'][1000.0], label='Egreedy', color = 'C5', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 100000])
plt.ylim([0.01, 2000])
plt.savefig("exp_ICML2021/results/graph/Yandex_PBM_equi_5.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %%
refs['PBM_equi_5']['UniPBrank_tunning'] = {}

for g in [4,15,20,30]:
    refs['PBM_equi_5']['UniPBrank_tunning'][g] = retrieve_data_from_zip(resdir +f'Yandex_equi_5_K_all__shuffled_kappa__Bandit_UniPBRank_100000_T_{g}_gamma__games_100000_nb_trials_1000_record_length_200_games.gz')


# %%
refs['PBM_equi_10']['UniPBrank_tunning'].keys()


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

myplot(ref = refs['PBM_equi_5']['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')
gamma= [4,15,20,30]
color=[1,2,4,5]
for i in range(4):
    myplot(ref = refs['PBM_equi_5']['UniPBrank_tunning'][gamma[i]] , label=f'UniPBrank, $\gamma$ = {gamma[i]}', color = f'C{color[i]}', linestyle = '-')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 100000])
plt.ylim([0.01, 2000])
plt.savefig("exp_ICML2021/results/graph/Yandex_PBM_tunegamma_uniPBRank_equi_5.pdf", bbox_inches = 'tight',
    pad_inches = 0)


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

myplot(ref = refs['PBM_equi']['KL-MLMR'], label='KL-MLMR', color = 'C1', linestyle = '-')

myplot(ref = refs['PBM_equi']['PB-MHB'], label='PB_MHB, c=$10^3$, m=1', color = 'C4', linestyle = '-')
myplot(ref = refs['PBM_equi']['UniPBrank'], label='UniPBrank', color = 'C3', linestyle = '-')
myplot(ref = refs['PBM_equi']['OSUB_PBM'], label='OSUB_PBM', color = 'C3', linestyle = '--')



plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([5, 10000000])
plt.ylim([1, 200000])
plt.savefig("exp_ICML2021/results/graph/Yandex_PBM_equi.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %%

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
time_sec_to_HMS(np.array(refs['CM_order']['MLMR'].record_results['time_to_play'])[:,-1].mean())

# %%
