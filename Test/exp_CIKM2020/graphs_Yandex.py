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
        referee_ob = None
    
    return referee_ob



# %%
def time_sec_to_HMS(sec):
    heure=sec//3600
    rest_h=sec%3600
    minute=rest_h//60
    rest_m=rest_h%60

    return(str(int(heure))+'H '+str(int(minute))+'min '+str(int(rest_m))+'sec')


# %% [markdown]
"""
## eGreedy
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./result/real_Yandex/"

refs = {}
refs["greedy"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs["random"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
for c in [1., 10., 100., 1000., 10000., 100000., 1000000.]:
    refs[c] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_' + str(c) + '_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')



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

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label=label)
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)

plt.figure(figsize=(3, 3))

myplot(ref = refs["random"], label='Random', color = 'black', linestyle = '-')

myplot(ref = refs[1.], label='eGreedy, c=1', color = 'C1', linestyle = '-')
myplot(ref = refs[10.], label='eGreedy, c=10', color = 'C2', linestyle = '-')
myplot(ref = refs[100.], label='eGreedy, c=100', color = 'C3', linestyle = '-')
myplot(ref = refs[1000.], label='eGreedy, c=1,000', color = 'C4', linestyle = '-')
myplot(ref = refs[10000.], label='eGreedy, c=10,000', color = 'C5', linestyle = '-')
myplot(ref = refs[100000.], label='eGreedy, c=100,000', color = 'C6', linestyle = '-')
myplot(ref = refs[1000000.], label='eGreedy, c=1,000,000', color = 'C7', linestyle = '-')

plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([1000, 20000000])
plt.ylim([70, 200000])
plt.savefig("./result/graph/Yandex__eGeeedy_20g_10000000t.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# %% [markdown]
"""
## TopRank / BubleRank
"""

# %%
resdir = "./result/real_Yandex/"

refs = {}
refs["greedy"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs["random"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
for delta in [1e-28]:
    refs[f'BubbleRank, {delta}, semi-oracle'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
    refs[f'BubbleRank, {delta}, sorted'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__sorted_kappa__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs[f'TopRank, semi-oracle'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs[f'TopRank, sorted'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__sorted_kappa__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')


# %%
def myplot(ref, label, color, linestyle, marker=None):
    if ref is not None:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        if marker is None:
            plt.plot(trials, mu, color=color, linestyle=linestyle, label=label)
            #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
        else:
            markersize = 10
            step = len(trials) // 10
            plt.plot(trials, mu, color=color, linestyle=linestyle, label="")
            plt.plot(trials[:len(trials):step], mu[:len(trials):step], color=color, linestyle="", marker=marker, markersize=markersize, label="")
            plt.plot([], [], color=color, linestyle=linestyle, marker=marker, markersize=markersize, label=label)
    else:
        plt.plot([], [], color=color, linestyle=linestyle, label='! no data ! '+label)


i = 1
for label, vals in refs.items():
    if label.find('sorted') == -1:
        myplot(ref = vals, label=label, color = f'C{i%10}', linestyle = '-')
        i += 1
    else:
        i -= 1
        myplot(ref = vals, label=label, color = f'C{i%10}', linestyle = ':')
        i += 1

plt.xlabel('Time')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.loglog()
#plt.xscale('log')
#if env_name == "xxsmall":
#    plt.xlim([1000, 20000000])
#    plt.ylim([1,20000])
#else:
#    plt.xlim([1000, 20000000])
#    plt.ylim([70,200000])
plt.savefig("./result/graph/Yandex__BubbleRank_TopRank_20g_10000000t.pdf", bbox_inches = 'tight',
    pad_inches = 0)





# %%
for rec in refs['TopRank, sorted'].record_results['env_parameters'][:4]:
                print(rec['label'], rec['thetas'][0:3], rec['kappas'])
for rec in refs['TopRank, semi-oracle'].record_results['env_parameters'][:4]:
                print(rec['label'], rec['thetas'][0:3], rec['kappas'])

# %% [markdown]
"""
## PB-MHB parameters
"""

# %%
resdir = "./result/real_Yandex/"

refs = {}
refs["greedy"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs["random"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
nb_steps = [1, 10]
cs = [0.1, 1., 10., 100., 1000.]
random_starts = [True, False]
for nb_step, c, random_start in product(nb_steps, cs, random_starts):
    refs[f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_PB-MHB_{"random_start" if random_start else "warm-up_start"}_{c}_c_{nb_step}_step__games_100000_nb_trials_1000_record_length_200_games.gz')
nb_step = 1
c = 100.
random_start = False
refs[f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_PB-MHB_{"random_start" if random_start else "warm-up_start"}_{c}_c_{nb_step}_step__games_10000000_nb_trials_1000_record_length_200_games.gz')


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

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label=label)
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)
plt.figure(figsize=(3, 3))


i = 0
#for label in ["greedy", "random"]:
#    myplot(ref=refs[label], label=label, color=f'C{i % 10}', linestyle='-')
#    i += 1

nb_steps = [1, 10]
cs = [0.1, 1., 10., 100., 1000.]
random_starts = [ False]
color =['green','orange','purple','red','pink']

for c in cs:
    j = 0
    line_styles = ['-', '--', ':', '-.']
    for nb_step, random_start in product(nb_steps, random_starts):
        label = f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'
        myplot(ref=refs[label], label=label, color=color[i], linestyle=line_styles[j])
        j+=1
    i += 1

plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
# plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
# plt.legend()
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.loglog()
# plt.xscale('log')
plt.xlim([10, 100000])
plt.ylim([3, 5000])
plt.savefig("./result/graph/Yandex_PBMHB_cXpasXstart.pdf", bbox_inches='tight',
            pad_inches=0)

# %%
env_name = 'std'
myplot(ref = refs['PB-MHB, c=0.1, m=1'], label='PB-MHB, c=0.1,     m=1', color = 'blue', linestyle = '-')
myplot(ref = refs['PB-MHB, c=1., m=1'], label='PB-MHB, c=1,        m=1', color = 'green', linestyle = '-')
myplot(ref = refs['PB-MHB, c=10., m=1'], label='PB-MHB, c=10,      m=1', color = 'red', linestyle = '-')
myplot(ref = refs['PB-MHB, c=100., m=1'], label='PB-MHB, c=100,    m=1', color = 'orange', linestyle = '-')
myplot(ref = refs['PB-MHB, c=1000., m=1'], label='PB-MHB, c=1,000, m=1', color = 'purple', linestyle = '-')

plt.xlabel('Time')
plt.ylabel('Cumulative Expected Regret')
plt.legend()
plt.grid(True)
plt.loglog()
plt.xlim([10, 10000000])
plt.ylim([3,50000])
plt.savefig("result/graph/Yandex_PBMHB_cXpas.pdf", bbox_inches = 'tight',
    pad_inches = 0)






# %% [markdown]
"""
## Opponents
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./result/real_Yandex/"

refs = {}
refs["greedy"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs["random"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')

for c in [10000.]:
    refs[f'eGreedy_c_{c}'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_EGreedy_SVD_{c}_c_1_maj__games_10000000_nb_trials_1000_record_length_200_games.gz')

c=100.
refs[f'PB-MHB_c_{c}'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_PB-MHB_warm-up_start_{c}_c_1_step__games_10000000_nb_trials_1000_record_length_200_games.gz')

refs["BC_MPTS_oracle"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_BC-MPTS_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs["BC_MPTS_greedy"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_BC-MPTS_greedy__games_10000000_nb_trials_1000_record_length_200_games.gz')

refs["PBM_TS_oracle"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_PBM-TS_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs["PBM_TS_greedy"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_PBM-TS_greedy__games_10000000_nb_trials_1000_record_length_200_games.gz')

for delta in [1e-28]:
    refs[f'BubbleRank, {delta}, semi-oracle'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
    refs[f'BubbleRank, {delta}, sorted'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__sorted_kappa__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs[f'TopRank, semi-oracle'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')
refs[f'TopRank, sorted'] = retrieve_data_from_zip(resdir + f'Yandex_all__extended_kappas__sorted_kappa__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_200_games.gz')



# %%
plt.figure(figsize=(3, 3))

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

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label=label)
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)
        

myplot(ref = refs["random"], label='Random', color = 'black', linestyle = '-')

myplot(ref = refs["greedy"], label='Greedy', color = 'grey', linestyle = '-')
myplot(ref = refs["eGreedy_c_10000.0"], label='$\epsilon_n$-greedy, c=$10^4$', color = 'orange', linestyle = '-')

myplot(ref = refs["PBM_TS_oracle"], label='PBM-TS, semi-oracle', color = 'blue', linestyle = ':')
myplot(ref = refs["PBM_TS_greedy"], label='PBM-TS, greedy', color = 'blue', linestyle = '-')

myplot(ref = refs["BC_MPTS_oracle"], label='BC-MPTS, semi-oracle', color = 'green', linestyle = ':')
myplot(ref = refs["BC_MPTS_greedy"], label='BC-MPTS, greedy', color = 'green', linestyle = '-')

delta = 1e-28
#myplot(ref = refs[f'BubbleRank, {delta}, sorted'], label=f'BubbleRank, $\\delta={delta}$, sorted', color = 'purple', linestyle = '--')

myplot(ref = refs[f'TopRank, sorted'], label=f'TopRank, sorted', color = 'olive', linestyle = ':')

myplot(ref = refs["PB-MHB_c_100.0"], label='PB-MHB, c=$100$, m=1', color = 'red', linestyle = '-')





plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.legend()
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.loglog()
#plt.xscale('log')
plt.xlim([10, 10000000])
plt.ylim([1,50000])
plt.savefig("./result/graph/Opponents_Yandex.pdf", bbox_inches = 'tight',
    pad_inches = 0)















# %%
refs.keys()


# %%
