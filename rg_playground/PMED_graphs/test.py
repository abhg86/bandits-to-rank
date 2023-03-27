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

# Miscellaneous change

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
        print(file_name)
        referee_ob = Referee (None, -1, all_time_record=True)
    
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
## PMED
"""

# %%
resdir = "../../Test/exp_AAAI2021/results/real_Yandex/"

refs = {}
refs["random"] = retrieve_data_from_zip(resdir + 'Yandex_all__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_100000_nb_trials_1000_record_length_200_games.gz')


refs["PMED old 1.0, 1, 0"] = retrieve_data_from_zip(resdir + f'Yandex_all__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_200_games_old.gz')
refs["PMEDbis 1.0, 1, 0"] = retrieve_data_from_zip(resdir + f'Yandex_all__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100_nb_trials_-1_record_length_80_games.gz')
refs["PMED 1.0, 1, 0"] = retrieve_data_from_zip(resdir + f'Yandex_all__sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q__games_100000_nb_trials_1000_record_length_200_games.gz')



c = 1000.
proposal_name = 'TGRW'
refs[f'PB-MHB, proposal={proposal_name}, c={c}, m=1'] = retrieve_data_from_zip(resdir + f'Yandex_all__Bandit_PB-MHB_warm-up_start_1_step_{proposal_name}_{c}_c_vari_sigma_proposal__games_100000_nb_trials_1000_record_length_200_games.gz')







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

        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label="")
        #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3, linestyle = linestyle, label=label)
    except:
        plt.plot([], [], color = color, linestyle = linestyle, label=label)
 
plt.figure(figsize=(6, 6))

i = 1
for label, vals in refs.items():
    myplot(ref = vals, label=label, color = f'C{i%10}', linestyle = '-')
    i += 1

plt.xlabel('Time-stamp')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.loglog()
#plt.xscale('log')
#plt.xlim([1000, 20000000])
#plt.ylim([1,20000])



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

def myplot_stat(ref, label, color, linestyle,for_theta=True, marker=None,type_errorbar='standart_error'):
    if 'stat_norm' in ref.record_results.keys() :
        trials = ref.get_recorded_trials()
        if for_theta:
            stat_to_draw =np.transpose(ref.record_results['stat_norm'])[0]
            final_label = label
        else :
            stat_to_draw =np.transpose(ref.record_results['stat_norm'])[1]
            final_label = 'kappa'+label
        norm_stat =np.mean(stat_to_draw,axis =1)
        markersize = 10
        step = len(trials) // 10
        
        if type_errorbar is not None:
            X_val,Y_val,Yerror=error_value(ref,stat_to_draw,type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[norm_stat[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[norm_stat[i]+Yerror[i] for i in range(len(Yerror))]
        
        plt.plot(trials, norm_stat, color=color, linestyle=linestyle, label=final_label)
        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
    else:
        plt.plot([], [], color=color, linestyle=linestyle, label='! no data ! '+label)

        
plt.figure(figsize=(6, 6))

i = 1
for label, vals in refs.items():
    myplot_stat(ref = vals, label=label, color = f'C{i%10}', linestyle = '-')
    i += 1

plt.xlabel('Time-stamp')
plt.ylabel('error in theta')
plt.title (env_name)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
#plt.xlim([1, 10000])
plt.ylim([0, 1])


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

def myplot_stat(ref, label, color, linestyle, marker=None,type_errorbar='standart_error'):
    if 'time_to_play' in ref.record_results.keys() :
        trials = ref.get_recorded_trials()
        stat_to_draw =np.transpose(ref.record_results['time_to_play'])
        norm_stat =np.mean(stat_to_draw,axis =1)
        markersize = 10
        step = len(trials) // 10
        
        if type_errorbar is not None:
            X_val,Y_val,Yerror=error_value(ref,stat_to_draw,type_errorbar=type_errorbar)
            nb_trials=len(X_val)
            spars_X_val=[X_val[i] for i in range(0, nb_trials, 200)]
            spars_Y_val=[Y_val[i] for i in range(0, nb_trials, 200)]
            spars_Yerror=[Yerror[i] for i in range(0, nb_trials, 200)]
            #plt.errorbar(spars_X_val, spars_Y_val, yerr = spars_Yerror,
            #fmt = 'none', capsize = 0, ecolor = color)
        neg_Yerror=[norm_stat[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror=[norm_stat[i]+Yerror[i] for i in range(len(Yerror))]
        
        plt.plot(trials, norm_stat, color=color, linestyle=linestyle, label=label)
        plt.fill_between(X_val,neg_Yerror, pos_Yerror , color = color, alpha=0.3, linestyle = linestyle, label='')
    else:
        plt.plot([], [], color=color, linestyle=linestyle, label='! no data ! '+label)

        
plt.figure(figsize=(6, 6))

i = 1
for label, vals in refs.items():
    myplot_stat(ref = vals, label=label, color = f'C{i%10}', linestyle = '-')
    i += 1

plt.xlabel('Time-stamp')
plt.ylabel('computation-time')
#plt.title (env_name)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
#plt.xlim([1, 100000])
#plt.ylim([0, 1])

t = np.array(refs["PMED 1.0, 1, 0"].record_results['time_to_play'])[:,-1].mean()
print(f'real computation time ', time_sec_to_HMS(t),
      'for', refs["PMED 1.0, 1, 0"].get_recorded_trials()[-1], "trials")
print('x 10:', time_sec_to_HMS(10* t), '(assume comp. time is due to optimization of q)')
print('x 100:', time_sec_to_HMS(100* t), '(assume average comp. time is constant)')
deb = np.array(refs["PMED 1.0, 1, 0"].get_recorded_trials()).shape[0]//6
x = np.log(np.array(refs["PMED 1.0, 1, 0"].get_recorded_trials())[deb:])
y = np.log(np.array(refs["PMED 1.0, 1, 0"].record_results['time_to_play']).mean(0)[deb:])
f = np.poly1d(np.polyfit(x, y, 1))
plt.plot(np.exp(x), np.exp(f(x)), color=f'C{i%10}', linestyle='-', label="fit")
xx = np.array([1, 10**5, 3*10**5, 10**6, 3*10**6, 10**7])
#xx = np.array([10**5, 3*10**5])
plt.plot(xx, np.exp(f(np.log(xx))), color=f'C{i%10}', linestyle=':', label="")
print('approximation:', time_sec_to_HMS(np.exp(f(np.log(10**7)))), 'ie.', np.exp(f(np.log(10**7)))/10**7, 'sec/trial')





# %%
last = 0
base = 3
l = [] 
for t in range(10**7):
    #if (np.log(last+1) // np.log(base)) != (np.log(t+1) // np.log(base)):
    if (np.sqrt(last) // base) != (np.sqrt(t) // base):
        last = t
        l.append(t)
        #print(t)

print(len(l))
plt.figure(figsize=(10, 6))
plt.plot(l, 'o')
plt.yscale('log')
plt.grid()


# %%
3.5**(np.log(np.array([1,2,42,34.123]))/np.log(3.5))

# %%
#### Get only legend 
plt.figure(figsize=(8, 8))
ax = plt.axes([0.1, 0.1, 0.67, 0.8])
def myplot(ref, label, color, linestyle):
    trials = ref.get_recorded_trials()
    mu, d_10, d_90 = ref.get_regret_expected()
    plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
    #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)

myplot(ref = refs[env_name]["random"], label='Random', color = 'black', linestyle = '-')
myplot(ref = refs[env_name][1.], label='$\epsilon_n$-greedy, c=1', color = 'C1', linestyle = '-')
myplot(ref = refs[env_name][10.], label='$\epsilon_n$-greedy, c=10', color = 'C2', linestyle = '-')
myplot(ref = refs[env_name][100.], label='$\epsilon_n$-greedy, c=100', color = 'C3', linestyle = '-')
myplot(ref = refs[env_name][1000.], label='$\epsilon_n$-greedy, c=1,000', color = 'C4', linestyle = '-')
myplot(ref = refs[env_name][10000.], label='$\epsilon_n$-greedy, c=10,000', color = 'C5', linestyle = '-')
myplot(ref = refs[env_name][100000.], label='$\epsilon_n$-greedy, c=100,000', color = 'C6', linestyle = '-')
myplot(ref = refs[env_name][1000000.], label='$\epsilon_n$-greedy, c=1,000,000', color = 'C7', linestyle = '-')

figlegend = pylab.figure(figsize=(6.5,1))
ha, lb = ax.get_legend_handles_labels()
print(lb)
figlegend.legend(ha, lb,"center right", fontsize='small', ncol=4)

figlegend.savefig('./result/graph/eGeeedy_20g_10000000t_legend.pdf')
      


# %% [markdown]
"""
## TopRank / BubleRank
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./result/simul/"

refs = {}
for env_name in ["std", "xxsmall", "big"]: 
    refs[env_name] = {}
    refs[env_name]["greedy"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name]["random"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')
    for delta in [1e-28]:
        refs[env_name][f'BubbleRank, {delta}, semi-oracle'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
        refs[env_name][f'BubbleRank, {delta}, sorted'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__sorted_kappa__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name][f'TopRank, semi-oracle'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name][f'TopRank, sorted'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__sorted_kappa__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')

# %%
for env_name in ["std", "xxsmall", "big"]:

    plt.figure(figsize=(8, 8))

    def myplot(ref, label, color, linestyle):
        try:
            trials = ref.get_recorded_trials()
            mu, d_10, d_90 = ref.get_regret_expected()
            plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
            #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
        except:
            plt.plot([], [], color = color, linestyle = linestyle, label=label)

    i = 1
    for label, vals in refs[env_name].items():
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
    plt.savefig("./result/graph/" + env_name + "__BubbleRank_TopRank_20g_10000000t.pdf", bbox_inches = 'tight',
        pad_inches = 0)





# %%
for rec in refs['xxsmall']['TopRank, sorted'].record_results['env_parameters'][:4]:
                print(rec['label'], rec['thetas'][0:3], rec['kappas'])
for rec in refs['xxsmall']['TopRank, semi-oracle'].record_results['env_parameters'][:4]:
                print(rec['label'], rec['thetas'][0:3], rec['kappas'])

# %% [markdown]
"""
## PB-MHB parameters
"""

# %% jupyter={"outputs_hidden": true}
resdir = "./result/simul/"

refs = {}
for env_name in ["std", "xxsmall", "big"]:
    refs[env_name] = {}
    refs[env_name]["greedy"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name]["random"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')
    nb_steps = [1, 10]
    cs = [0.1, 1., 10., 100., 1000.]
    random_starts = [True, False]
    for nb_step, c, random_start in product(nb_steps, cs, random_starts):
        refs[env_name][f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__Bandit_PB-MHB_{"random_start" if random_start else "warm-up_start"}_{c}_c_{nb_step}_step__games_100000_nb_trials_1000_record_length_20_games.gz')
    nb_step = 1
    c = 100.
    random_start = False
    refs[env_name][f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__Bandit_PB-MHB_{"random_start" if random_start else "warm-up_start"}_{c}_c_{nb_step}_step__games_10000000_nb_trials_1000_record_length_20_games.gz')


# %%
def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        # plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)

for env_name in ["std", "xxsmall", "big"]:
    plt.figure(figsize=(3, 3))

    i = 1
    for label in ["greedy", "random"]:
        myplot(ref=refs[env_name][label], label=label, color=f'C{i % 10}', linestyle='-')
        i += 1

    nb_steps = [1, 10]
    cs = [0.1, 1., 10., 100., 1000.]
    random_starts = [True, False]
    for c in cs:
        j = 0
        line_styles = ['-', '--', ':', '-.']
        for nb_step, random_start in product(nb_steps, random_starts):
            label = f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'
            myplot(ref=refs[env_name][label], label=label, color=f'C{i % 10}', linestyle=line_styles[j])
            j+=1
        i += 1

    plt.xlabel('Time-stamp')
    plt.ylabel('Cumulative Expected Regret')
    # plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.loglog()
    # plt.xscale('log')
    if env_name == "xxsmall":
        plt.xlim([10, 10000000])
        plt.ylim([1, 20000])
    else:
        plt.xlim([10, 10000000])
        plt.ylim([3, 50000])
    plt.savefig("./result/graph/" + env_name + "__PBM-BH_20g_10000000t_full.pdf", bbox_inches='tight',
                pad_inches=0)


# %%
def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        # plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)

for env_name in ["std", "xxsmall", "big"]:
    plt.figure(figsize=(3, 3))

    i = 0
    #for label in ["greedy", "random"]:
    #    myplot(ref=refs[env_name][label], label=label, color=f'C{i % 10}', linestyle='-')
    #    i += 1

    nb_steps = [1, 10]
    cs = [0.1, 1., 10., 100., 1000.]
    #random_starts = [True, False]
    random_starts = [False]
    color =['green','orange','purple','red','pink']
    for c in cs:
        j = 0
        line_styles = ['-', '--', ':', '-.']
        for nb_step, random_start in product(nb_steps, random_starts):
            label = f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'
            myplot(ref=refs[env_name][label], label=label, color=color[i], linestyle=line_styles[j])
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
    if env_name == "xxsmall":
        plt.xlim([10, 100000])
        plt.ylim([0.01, 2000])
    else:
        plt.xlim([10, 100000])
        plt.ylim([1, 50000])
    plt.savefig("./result/graph/" + env_name + "__PB-MHB_20g_10000000t_tuneCxPas.pdf", bbox_inches='tight',
                pad_inches=0)

# %%
plt.figure(figsize=(8, 8))
ax = plt.axes()
def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        ax.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        # plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)

for env_name in ["std"]:
    plt.figure(figsize=(8, 8))

    i = 0
    #for label in ["greedy", "random"]:
    #    myplot(ref=refs[env_name][label], label=label, color=f'C{i % 10}', linestyle='-')
    #    i += 1

    nb_steps = [1, 10]
    cs = [0.1, 1., 10., 100., 1000.]
    #random_starts = [True, False]
    random_starts = [False]
    color =['green','orange','purple','red','pink']
    for c in cs:
        j = 0
        line_styles = ['-', '--', ':', '-.']
        for nb_step, random_start in product(nb_steps, random_starts):
            label = f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'
            myplot(ref=refs[env_name][label], label=label, color=color[i], linestyle=line_styles[j])
            if c == 100 :
                if nb_step == 10 :
                    ax.plot([], [], color='red', linestyle='--', label='PB-MHB, c=100, m=10')
            j+=1
        i += 1

    plt.xlabel('Time')
    plt.ylabel('Cumulative Expected Regret')

    plt.grid(True)
figlegend = pylab.figure(figsize=(10,1))
ha, lb = ax.get_legend_handles_labels()
print(lb)
figlegend.legend(ha, lb,"center right", fontsize='small', ncol=5)

figlegend.savefig('./result/graph/PB-MHB_20g_10000000t_legend.pdf')


# %%
def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        # plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)

for env_name in ["std", "xxsmall", "big"]:
    plt.figure(figsize=(3, 2))

    i = 0

    nb_steps = [1, 10]
    cs = [100.]
    random_starts = [False,True]
    color =['red']
    for c in cs:
        j = 0
        line_styles = ['-', '--', ':', '-.']
        for nb_step, random_start in product(nb_steps, random_starts):
            label = f'PB-MHB, c={c}, m={nb_step}{", random start" if random_start else ""}'
            myplot(ref=refs[env_name][label], label=label, color=color[i], linestyle=line_styles[j])
            j+=1
        i += 1

    plt.xlabel('Time-stamp')
    plt.ylabel('Cum. Exp. Reg.')
    # plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
    # plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.6),prop={'size':8})
    plt.grid(True)
    plt.loglog()
    # plt.xscale('log')
    if env_name == "xxsmall":
        plt.xlim([10, 100000])
        plt.ylim([1, 20000])
    else:
        plt.xlim([10, 100000])
        plt.ylim([3, 50000])
    plt.savefig("./result/graph/" + env_name + "__PBM-HB_20g_10000000t_rsdstart.pdf", bbox_inches='tight',
                pad_inches=0)

# %%
env_name = 'std'
myplot(ref = refs[env_name]['PB-MHB, c=0.1, m=1'], label='PB-MHB, c=0.1,     m=1', color = 'blue', linestyle = '-')
myplot(ref = refs[env_name]['PB-MHB, c=1., m=1'], label='PB-MHB, c=1,        m=1', color = 'green', linestyle = '-')
myplot(ref = refs[env_name]['PB-MHB, c=10., m=1'], label='PB-MHB, c=10,      m=1', color = 'red', linestyle = '-')
myplot(ref = refs[env_name]['PB-MHB, c=100., m=1'], label='PB-MHB, c=100,    m=1', color = 'orange', linestyle = '-')
myplot(ref = refs[env_name]['PB-MHB, c=1000., m=1'], label='PB-MHB, c=1,000, m=1', color = 'purple', linestyle = '-')

plt.xlabel('Time')
plt.ylabel('Cumulative Expected Regret')
plt.legend()
plt.grid(True)
plt.loglog()
plt.xlim([10, 10000000])
plt.ylim([3,50000])
plt.savefig("result/graph/TSMH_cXpas_std.pdf", bbox_inches = 'tight',
    pad_inches = 0)






# %% [markdown]
"""
## Opponents
"""

# %%
resdir = "./result/simul/"

refs = {}
for env_name in ["std", "xxsmall", "big"]:
    refs[env_name] = {}
    refs[env_name]["greedy"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_EGreedy_SVD_0.0_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name]["random"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_EGreedy_SVD_1e+20_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')

    for c in [1000., 100000.]:
        refs[env_name][f'eGreedy_c_{c}'] = retrieve_data_from_zip(resdir + f'purely_simulated__{env_name}__Bandit_EGreedy_SVD_{c}_c_1_maj__games_10000000_nb_trials_1000_record_length_20_games.gz')

    c=100.
    refs[env_name][f'PB-MHB_c_{c}'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__Bandit_PB-MHB_warm-up_start_{c}_c_1_step__games_10000000_nb_trials_1000_record_length_20_games.gz')

    refs[env_name]["BC_MPTS_oracle"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_BC-MPTS_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name]["BC_MPTS_greedy"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_BC-MPTS_greedy__games_10000000_nb_trials_1000_record_length_20_games.gz')

    refs[env_name]["PBM_TS_oracle"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_PBM-TS_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name]["PBM_TS_greedy"] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + '__Bandit_PBM-TS_greedy__games_10000000_nb_trials_1000_record_length_20_games.gz')

    for delta in [1e-28]:
        refs[env_name][f'BubbleRank, {delta}, semi-oracle'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
        refs[env_name][f'BubbleRank, {delta}, sorted'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__sorted_kappa__Bandit_BubbleRank_{delta}_delta_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name][f'TopRank, semi-oracle'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')
    refs[env_name][f'TopRank, sorted'] = retrieve_data_from_zip(resdir + 'purely_simulated__' + env_name + f'__extended_kappas__sorted_kappa__Bandit_TopRank_oracle__games_10000000_nb_trials_1000_record_length_20_games.gz')



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
        
for env_name in ["std", "xxsmall", "big"]:


    plt.figure(figsize=(3, 3))


    myplot(ref = refs[env_name]["random"], label='Random', color = 'black', linestyle = '-')

    myplot(ref = refs[env_name]["greedy"], label='Greedy', color = 'grey', linestyle = '-')
    if env_name == 'xxsmall':
        myplot(ref = refs[env_name]["eGreedy_c_100000.0"], label='$\epsilon_n$-greedy, c=$10^5$', color = 'orange', linestyle = '-')
    else:
        myplot(ref = refs[env_name]["eGreedy_c_1000.0"], label='$\epsilon_n$-greedy, c=$10^3$', color = 'orange', linestyle = '-')

    myplot(ref = refs[env_name]["PBM_TS_oracle"], label='PBM-TS, semi-oracle', color = 'blue', linestyle = ':')
    myplot(ref = refs[env_name]["PBM_TS_greedy"], label='PBM-TS, greedy', color = 'blue', linestyle = '-')

    myplot(ref = refs[env_name]["BC_MPTS_oracle"], label='BC-MPTS, semi-oracle', color = 'green', linestyle = ':')
    myplot(ref = refs[env_name]["BC_MPTS_greedy"], label='BC-MPTS, greedy', color = 'green', linestyle = '-')

    delta = 1e-28
    #myplot(ref = refs[env_name][f'BubbleRank, {delta}, sorted'], label=f'BubbleRank, $\\delta={delta}$, sorted', color = 'purple', linestyle = '--')

    myplot(ref = refs[env_name][f'TopRank, sorted'], label=f'TopRank, sorted', color = 'olive', linestyle = ':')

    myplot(ref = refs[env_name]["PB-MHB_c_100.0"], label='PB-MHB, c=$100$, m=1', color = 'red', linestyle = '-')





    plt.xlabel('Time-stamp')
    plt.ylabel('Cumulative Expected Regret')
    #plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.loglog()
    #plt.xscale('log')
    if env_name == "xxsmall":
        plt.xlim([10, 10000000])
        plt.ylim([0.01,20000])
    else:
        plt.xlim([10, 10000000])
        plt.ylim([3,50000])
    plt.savefig("./result/graph/Opponents_simul_" + env_name + ".pdf", bbox_inches = 'tight',
        pad_inches = 0)












# %%
plt.figure(figsize=(8, 8))
ax = plt.axes()
def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        ax.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        # plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)
    

for env_name in ["std"]:


    plt.figure(figsize=(4, 4))


    myplot(ref = refs[env_name]["random"], label='Random', color = 'black', linestyle = '-')

    myplot(ref = refs[env_name]["greedy"], label='Greedy', color = 'grey', linestyle = '-')
    
    myplot(ref = refs[env_name]["eGreedy_c_1000.0"], label='$\epsilon_n$-greedy', color = 'orange', linestyle = '-')
    myplot(ref = refs[env_name][f'TopRank, sorted'], label=f'TopRank, $\kappa$-sorted', color = 'olive', linestyle = ':')

    myplot(ref = refs[env_name]["PBM_TS_oracle"], label='PBM-TS, semi-oracle', color = 'blue', linestyle = ':')
    myplot(ref = refs[env_name]["PBM_TS_greedy"], label='PBM-TS, greedy', color = 'blue', linestyle = '-')

    myplot(ref = refs[env_name]["BC_MPTS_oracle"], label='BC-MPTS, semi-oracle', color = 'green', linestyle = ':')
    myplot(ref = refs[env_name]["BC_MPTS_greedy"], label='BC-MPTS, greedy', color = 'green', linestyle = '-')

    delta = 1e-28
    #myplot(ref = refs[env_name][f'BubbleRank, {delta}, sorted'], label=f'BubbleRank, $\\delta={delta}$, sorted', color = 'purple', linestyle = '--')

   
    myplot(ref = refs[env_name]["PB-MHB_c_100.0"], label='PB-MHB, c=$100$, m=1', color = 'red', linestyle = '-')



figlegend = pylab.figure(figsize=(8,1))
ha, lb = ax.get_legend_handles_labels()
print(lb)
figlegend.legend(ha, lb,"center", fontsize='small', ncol=5)

figlegend.savefig('./result/graph/Opponent_legend.pdf')

# %%
plt.figure(figsize=(8, 8))
ax = plt.axes()
def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        ax.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        # plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)
    

for env_name in ["std"]:


    plt.figure(figsize=(4, 4))


    myplot(ref = refs[env_name]["random"], label='Random', color = 'black', linestyle = '-')

    myplot(ref = refs[env_name]["greedy"], label='Greedy', color = 'grey', linestyle = '-')
    
    myplot(ref = refs[env_name]["eGreedy_c_1000.0"], label='$\epsilon_n$-greedy', color = 'orange', linestyle = '-')
    myplot(ref = refs[env_name][f'TopRank, sorted'], label=f'TopRank, sorted', color = 'olive', linestyle = '--')

    myplot(ref = refs[env_name]["PBM_TS_oracle"], label='PBM-TS, semi-oracle', color = 'blue', linestyle = '--')
    myplot(ref = refs[env_name]["PBM_TS_greedy"], label='PBM-TS, greedy', color = 'blue', linestyle = '-')

    myplot(ref = refs[env_name]["BC_MPTS_oracle"], label='BC-MPTS, semi-oracle', color = 'green', linestyle = '--')
    myplot(ref = refs[env_name]["BC_MPTS_greedy"], label='BC-MPTS, greedy', color = 'green', linestyle = '-')

    delta = 1e-28
    #myplot(ref = refs[env_name][f'BubbleRank, {delta}, sorted'], label=f'BubbleRank, $\\delta={delta}$, sorted', color = 'purple', linestyle = '--')

   
    myplot(ref = refs[env_name]["PB-MHB_c_100.0"], label='PB-MHB, c=$100$, m=1', color = 'red', linestyle = '-')



figlegend = pylab.figure(figsize=(3,2))
ha, lb = ax.get_legend_handles_labels()
print(lb)
figlegend.legend(ha, lb)

figlegend.savefig('./result/graph/Opponent_legend_formatCol.pdf')

# %%
refs['std'].keys()


# %%
