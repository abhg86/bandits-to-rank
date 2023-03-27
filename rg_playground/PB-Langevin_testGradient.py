# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# PB_GB
"""

# %%
# You want bandits_to_rank in your path
import os
os.chdir('..')

# %%
# To ease pairing with .py file
# %autosave 0
# %reload_ext autoreload
# %autoreload 2 

import numpy as np
from itertools import product
from bandits_to_rank.opponents import PB_GB
from bandits_to_rank.tools import tools_Langevin
import matplotlib.pyplot as plt

# %% [markdown]
"""
## Test préliminaires
"""

# %%
theta_star = [1,0.58,0.06,0.04,0.0001]
kappa_star = [0.9,0.5,0.1]
nb_arm = len(theta_star)
nb_pos = len(kappa_star)
theta_kappa= np.kron(kappa_star, np.array([theta_star]).transpose())

print(np.kron(kappa_star, np.array([theta_star]).transpose()))
print(np.asarray(theta_star).reshape((-1,1))@np.asarray(kappa_star).reshape((1,-1)))
print(np.asarray(theta_star).reshape((-1,1))@np.asarray(kappa_star).reshape((1,-1)))

# %%
part0 = [np.random.uniform(0, 1, nb_arm),
                np.random.uniform(0, 1, nb_pos)]
N = 500

# %%
part0

# %%
nb_obs = 100
S = np.floor(theta_kappa *nb_obs)
F = nb_obs -S

# %%
S,F

# %%
np.sum(1/theta_star[0]*S[0]+kappa_star/(1 - theta_kappa[0]))

# %% [markdown]
"""
## Langevin: fixed number of observations
"""

# %%
N=10000
list_param = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
list_obs = [1000, 10000, 100000]
#list_obs = [100000000]
list_min_bound = ['null', 'h', 'sqrt', '1/obs']

# %%
res = {}

for param, min_bound in product(list_param, list_min_bound) :
    part = part0
    for obs in list_obs:
        #print('#########',obs,"###########")
        S = np.floor(theta_kappa *obs)
        F = obs -S
        tag = f'{obs} {param} {min_bound}'
        res[tag] = {'ind' : np.zeros(N//100+1),
                    'dist of part' : np.zeros(N//100+1),
                    'dist of avg' : np.zeros(N//100+1)
                   }
        for n in range(N):
            h = param/obs
            # h = h/np.sqrt(n+1)
            grad = np.asarray(tools_Langevin.compute_gradient(part, S,F))
            part = [part[0] - h * grad[:nb_arm] + np.sqrt(2*h)*np.random.rand(nb_arm)
                    , part[1] - h * grad[nb_arm:] + np.sqrt(2*h)*np.random.rand(nb_pos)]
            if min_bound == 'null':
                thresh = 0
            elif min_bound == 'h':
                thresh = min(h,0.001)
            elif min_bound == 'sqrt':
                thresh = min(np.sqrt(h),0.001)
            elif min_bound == '1/obs':
                thresh = min(1/(obs+2),0.001)
            part = [np.array([1 if v > 1 else thresh if v<0 else v for v in part[0]]),
                np.array([1 if v > 1 else thresh if v<0 else v for v in part[1]])]
            if n == 0:
                mean_part = part
            else:
                mean_part = [ mean_part[0] + (part[0]-mean_part[0])/(n+1),
                             mean_part[1] + (part[1]-mean_part[1])/(n+1)]
            if n % 100 == 0 or n == N-1:
                res[tag]['ind'][(n+1)//100] = n + 1
                res[tag]['dist of part'][(n+1)//100] = np.linalg.norm(theta_kappa - np.kron(part[1], part[0][:,None]))
                res[tag]['dist of avg'][(n+1)//100] = np.linalg.norm(theta_kappa - np.kron(mean_part[1], mean_part[0][:,None]))


# %%
def myplot(list_param, list_min_bound):
    for obs in list_obs:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title (f'part with {obs} obs')
        i=0
        for param, min_bound in product(list_param, list_min_bound) :
            i+=1
            tag = f'{obs} {param} {min_bound}'
            plt.plot(res[tag]['ind'], res[tag]['dist of part'], color=f'C{i}', linestyle='-', label=tag)
        plt.xlabel('Time-stamp')
        plt.ylabel('Eucledian distance to tk_star')
        plt.legend()
        plt.grid(True)
        plt.loglog()
        #plt.yscale('log')

        plt.subplot(122)
        plt.title (f'mean_part with {obs} obs')
        i=0
        for param, min_bound in product(list_param, list_min_bound) :
            i+=1
            tag = f'{obs} {param} {min_bound}'
            plt.plot(res[tag]['ind'], res[tag]['dist of avg'], color=f'C{i}', linestyle='-', label=tag)
        plt.xlabel('Time-stamp')
        plt.ylabel('Eucledian distance to tk_star')
        plt.legend()
        plt.grid(True)
        plt.loglog()
        #plt.yscale('log')
        
myplot(list_param[2:3], list_min_bound)
myplot(list_param[:4], list_min_bound[2:4])

# %% [markdown]
"""
## Langevin: increasing number of observations
"""

# %%
res_inc = {}


# %%
def run_exp(list_N, obs, list_param, list_min_bound, prefix=''):
    for N, param, min_bound in product(list_N, list_param, list_min_bound) :
        part = part0
        n_grad = 0
        tag = f'{prefix}{N} {param} {min_bound}'
        res_inc[tag] = {'nb obs' : np.zeros(obs//100+1),
                    'dist of part' : np.zeros(obs//100+1),
                    'dist of avg' : np.zeros(obs//100+1)
                   }
        for n_obs in range(obs):
            S = np.floor(theta_kappa *n_obs)
            F = n_obs -S
            h = param/(n_obs+1)
            for _ in range(N):
                # h = h/np.sqrt(n+1)
                grad = np.asarray(tools_Langevin.compute_gradient(part, S,F))
                part = [part[0] - h * grad[:nb_arm] + np.sqrt(2*h)*np.random.rand(nb_arm)
                        , part[1] - h * grad[nb_arm:] + np.sqrt(2*h)*np.random.rand(nb_pos)]
                if min_bound == 'null':
                    thresh = 0
                elif min_bound == 'h':
                    thresh = min(h,0.001)
                elif min_bound == 'sqrt':
                    thresh = min(np.sqrt(h),0.001)
                elif min_bound == '1/n_obs':
                    thresh = min(1/(n_obs+2),0.001)
                if prefix == '':
                    part = [np.array([1 if v > 1 else thresh if v<0 else v for v in part[0]]),
                        np.array([1 if v > 1 else thresh if v<0 else v for v in part[1]])]
                elif prefix == 'threshold=min ':
                    part = [np.array([1 if v > 1 else thresh if v<thresh else v for v in part[0]]),
                        np.array([1 if v > 1 else thresh if v<thresh else v for v in part[1]])]
                elif prefix == 'prior ':
                    grad = np.asarray(tools_Langevin.compute_gradient(part, S+1,F+1))
                    part = [np.array([1 if v > 1 else thresh if v<thresh else v for v in part[0]]),
                        np.array([1 if v > 1 else thresh if v<thresh else v for v in part[1]])]
                if n_obs == 0 and n_grad == 0:
                    mean_part = part
                else:
                    mean_part = [ mean_part[0] + (part[0]-mean_part[0])/(n_grad+1),
                                 mean_part[1] + (part[1]-mean_part[1])/(n_grad+1)]
                n_grad += 1
            if n_obs % 100 == 0 or n_obs == obs-1:
                res_inc[tag]['nb obs'][(n_obs+1)//100] = n_obs + 1
                res_inc[tag]['dist of part'][(n_obs+1)//100] = np.linalg.norm(theta_kappa - np.kron(part[1], part[0][:,None]))
                res_inc[tag]['dist of avg'][(n_obs+1)//100] = np.linalg.norm(theta_kappa - np.kron(mean_part[1], mean_part[0][:,None]))



# %%
list_N= [1, 10]
obs = 10000
list_param = [10**-1, 10**-2, 10**-3]
list_min_bound = ['null', 'h', 'sqrt', '1/obs']
run_exp(list_N, obs, list_param, list_min_bound)

# %%
list_N= [1]
obs = 100000
list_param = [10**-1, 10**-2, 10**-3, 10**-4]
list_min_bound = ['sqrt']
run_exp(list_N, obs, list_param, list_min_bound)

# %%
list_N= [1]
obs = 1000000
list_param = [10**-2]
list_min_bound = ['sqrt']
run_exp(list_N, obs, list_param, list_min_bound, prefix='threshold=min ')

# %%
list_N= [1]
obs = 100000
list_param = [10**-2]
list_min_bound = ['sqrt']
run_exp(list_N, obs, list_param, list_min_bound, prefix='prior ')


# %%
def myplot(list_N, list_param, list_min_bound, list_prefix=['']):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title (f'part')
    i=0
    for N, param, min_bound, prefix in product(list_N, list_param, list_min_bound, list_prefix) :
        i+=1
        tag = f'{prefix}{N} {param} {min_bound}'
        try:
            plt.plot(res_inc[tag]['nb obs'], res_inc[tag]['dist of part'], color=f'C{i}', linestyle='-', label=tag)
        except:
            plt.plot([], [], color=f'C{i}', linestyle='-', label='[X]'+tag)
    plt.xlabel('nb obs')
    plt.ylabel('Eucledian distance to tk_star')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.yscale('log')

    plt.subplot(122)
    plt.title (f'mean_part')
    i=0
    for N, param, min_bound, prefix in product(list_N, list_param, list_min_bound, list_prefix) :
        i+=1
        tag = f'{prefix}{N} {param} {min_bound}'
        try:
            plt.plot(res_inc[tag]['nb obs'], res_inc[tag]['dist of avg'], color=f'C{i}', linestyle='-', label=tag)
        except:
            plt.plot([], [], color=f'C{i}', linestyle='-', label='[X]'+tag)
    plt.xlabel('nb obs')
    plt.ylabel('Eucledian distance to tk_star')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.yscale('log')
        
for N in [1, 10, 100]:
    myplot([N], [10**-1, 10**-2, 10**-3], ['null', 'h', 'sqrt', '1/obs'])
myplot([1, 10], [10**-2, 10**-3, 10**-4], ['sqrt'])
myplot([1], [10**-2], ['sqrt', '1/obs'])
myplot([1], [10**-2], ['sqrt'], ['', 'threshold=min ', 'prior '])

# %% [markdown]
"""
## Langevin: test decay
"""

# %%
res_dec = {}


# %%
def run_exp_dec(list_dec, list_N, obs, list_param, list_min_bound):
    for dec, N, param, min_bound in product(list_dec, list_N, list_param, list_min_bound) :
        part = part0
        n_grad = 0
        tag = f'{dec} {N} {param} {min_bound}'
        res_dec[tag] = {'nb obs' : np.zeros(obs//100+1),
                    'dist of part' : np.zeros(obs//100+1),
                    'dist of avg' : np.zeros(obs//100+1)
                   }
        for n_obs in range(obs):
            S = np.floor(theta_kappa *n_obs)
            F = n_obs -S
            if dec == '1/t':
                h = param/(n_obs+1)
            elif dec == '1/sqrt(t)':
                h = param/np.sqrt(n_obs+1)
            elif dec == 'cst':
                h = param
            for _ in range(N):
                # h = h/np.sqrt(n+1)
                grad = np.asarray(tools_Langevin.compute_gradient(part, S,F))
                part = [part[0] - h * grad[:nb_arm] + np.sqrt(2*h)*np.random.rand(nb_arm)
                        , part[1] - h * grad[nb_arm:] + np.sqrt(2*h)*np.random.rand(nb_pos)]
                if min_bound == 'null':
                    thresh = 0
                elif min_bound == 'h':
                    thresh = min(h,0.001)
                elif min_bound == 'sqrt':
                    thresh = min(np.sqrt(h),0.001)
                elif min_bound == '1/obs':
                    thresh = min(1/(n_obs+2),0.001)
                part = [np.array([1 if v > 1 else thresh if v<thresh else v for v in part[0]]),
                    np.array([1 if v > 1 else thresh if v<thresh else v for v in part[1]])]
                if n_obs == 0 and n_grad == 0:
                    mean_part = part
                else:
                    mean_part = [ mean_part[0] + (part[0]-mean_part[0])/(n_grad+1),
                                 mean_part[1] + (part[1]-mean_part[1])/(n_grad+1)]
                n_grad += 1
            if n_obs % 100 == 0 or n_obs == obs-1:
                res_dec[tag]['nb obs'][(n_obs+1)//100] = n_obs + 1
                res_dec[tag]['dist of part'][(n_obs+1)//100] = np.linalg.norm(theta_kappa - np.kron(part[1], part[0][:,None]))
                res_dec[tag]['dist of avg'][(n_obs+1)//100] = np.linalg.norm(theta_kappa - np.kron(mean_part[1], mean_part[0][:,None]))



# %%
list_dec= ['1/t', '1/sqrt(t)', 'cst']
list_N= [1, 10]
obs = 10000
list_param = [10**-1, 10**-2, 10**-3]
list_min_bound = ['null', 'h', 'sqrt', '1/obs']
run_exp_dec(list_dec, list_N, obs, list_param, list_min_bound)


# %%
def myplot(list_dec, list_N, list_param, list_min_bound):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title (f'part')
    i=0
    for dec, N, param, min_bound in product(list_dec, list_N, list_param, list_min_bound) :
        i+=1
        tag = f'{dec} {N} {param} {min_bound}'
        try:
            plt.plot(res_dec[tag]['nb obs'], res_dec[tag]['dist of part'], color=f'C{i}', linestyle='-', label=tag)
        except:
            plt.plot([], [], color=f'C{i}', linestyle='-', label='[X]'+tag)
    plt.xlabel('nb obs')
    plt.ylabel('Eucledian distance to tk_star')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.yscale('log')

    plt.subplot(122)
    plt.title (f'mean_part')
    i=0
    for dec, N, param, min_bound in product(list_dec, list_N, list_param, list_min_bound) :
        i+=1
        tag = f'{dec} {N} {param} {min_bound}'
        try:
            plt.plot(res_dec[tag]['nb obs'], res_dec[tag]['dist of avg'], color=f'C{i}', linestyle='-', label=tag)
        except:
            plt.plot([], [], color=f'C{i}', linestyle='-', label='[X]'+tag)
    plt.xlabel('nb obs')
    plt.ylabel('Eucledian distance to tk_star')
    plt.legend()
    plt.grid(True)
    plt.loglog()
    #plt.yscale('log')
        
for dec in ['1/t', '1/sqrt(t)', 'cst']:
    myplot([dec], [10], [10**-1, 10**-2, 10**-3], ['null', 'h', 'sqrt', '1/obs'])
for dec in ['1/t', '1/sqrt(t)', 'cst']:
    myplot([dec], [1, 10], [10**-2, 10**-3], ['sqrt', '1/obs'])
myplot(['1/t', '1/sqrt(t)', 'cst'], [1], [10**-2], ['sqrt'])

# %% [markdown]
"""
## Test Bandit
"""

# %%
from bandits_to_rank.bandits import *
from bandits_to_rank.environment import *
from bandits_to_rank.referee import *
from bandits_to_rank.opponents import oracle,random_player,greedy,PB_GB
from bandits_to_rank.sampling.pbm_inference import *


import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, log
from random import shuffle
from copy import deepcopy


# %reload_ext autoreload
# %autoreload 2 

# %%
def time_sec_to_HMS(sec):
    heure=sec//3600
    rest_h=sec%3600
    minute=rest_h//60
    rest_m=rest_h%60

    return(str(int(heure))+'H '+str(int(minute))+'min '+str(int(rest_m))+'sec')


# %% [markdown]
"""
### TEST PBM
"""

# %%
kappas = [1,0.1,0.6,0.01,0.3]

#thetas_final = [1,0.6,0.3,0.1,0.05,0.01,0.005,0.005,0.001,0.001]
thetas_short = [1,0.6,0.3,0.1,0.05]

nb_prop = len(thetas_short)  
#nb_prop_10 = len(thetas_final)  

nb_place = len(kappas)


# %%
env_PBM = Environment_PBM(thetas_short,kappas)
#env_us = Environment_PBM(thetas_final,kappas)


nb_trial = 1000


# %% [markdown]
"""
### Game 20 Trial 5000
"""

# %%
nb_game = 5

referee_PBM = Referee (env_PBM,nb_trial,10)



# %%
start = time.time()
  
for i in range(nb_game):
    print ('#### game '+str(i))
    #### Reboot player

    start_game = time.time()
    player_pb_gb = PB_GB.PB_GB(nb_prop,nb_place,h_param=0.0000000001,N = 200)   
    print (player_pb_gb.h)
    print(player_pb_gb.N)
    #### Play game
   
    referee_PBM.play_game(player_pb_gb)

    
    #referee_PBM_TS_glouton_5000trials_20games_thetas_final.play_game(player_PBM_TS_glouton)
    
    end_game = time.time()   
    print ('time_game :',time_sec_to_HMS(end_game-start_game)) 

end = time.time()   
print ('time :',time_sec_to_HMS(end-start)) 


# %%
plt.figure(figsize=(8, 8))
def myplot(ref, label, color, linestyle):
    trials = ref.get_recorded_trials()
    mu, d_10, d_90 = ref.get_regret_expected()
    plt.plot(trials, mu, color = color, linestyle = linestyle, label=label)
    #plt.fill_between(trials, d_10, d_90, color = color, alpha=0.3)

#myplot(ref = referee_Random, label='Random', color = 'black', linestyle = '-')

myplot(ref = referee_PBM, label='PB_GB, PBM K=5 L=5', color = 'C1', linestyle = '-')


plt.title('env_unordered ')
plt.xlabel('Time')
plt.ylabel('Cumulative Expected Regret')
#plt.title ("Test for interaction of step's number and length, on "+ str(nb_game) + ' games  of '+ str(nb_trial)+' trials')
plt.legend()
plt.grid(True)
plt.loglog()
plt.xscale('log')
plt.xlim([0,10000])
plt.ylim([0,15000])
#plt.savefig("./result/graph/std_TSMH_cxpas_20g_100000t.pdf", bbox_inches = 'tight',
#    pad_inches = 0)



# %%
player_pb_gb.choose_next_arm()

# %%
kappas = [1,0.1,0.6,0.01,0.3]

#thetas_final = [1,0.6,0.3,0.1,0.05,0.01,0.005,0.005,0.001,0.001]
thetas_short = [1,0.6,0.3,0.1,0.05]
env_PBM.get_best_index()

# %%
player_pb_gb.get_param_estimation()

# %%
