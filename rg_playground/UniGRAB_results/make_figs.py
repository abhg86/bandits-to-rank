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

from bandits_to_rank.referee import *

import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import glob
import re

np.set_printoptions(precision=3)

# %%
def retrieve_data_from_zip(wilcarded_file_name, my_assert=True):
    files = glob.glob(wilcarded_file_name)

    # no file
    if not files:
        print(f'!!! DOES NOT EXIST: {wilcarded_file_name}')
        referee_ob = Referee(None, -1, all_time_record=True)
        referee_ob.record_results = {'time_recorded': [[]],
                                     'expected_best_reward': np.zeros((0, 0)),
                                     'expected_reward': np.zeros((0, 0)),
                                     'time_to_play': np.zeros((0, 0))
                                     }
        return referee_ob

    # pick best file
    best_file = files[np.argmax([int(re.search('(\d+)_games', file).group(1)) for file in files])]
    print(f'loaded file: {best_file}')

    # load best file
    with gzip.GzipFile(best_file, 'r') as fin:
        json_bytes = fin.read()

    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    data = json.loads(json_str)
    referee_ob = Referee(None, -1, all_time_record=True)
    referee_ob.record_results = data
    if my_assert:
        print(best_file)
        if best_file.find('CM') == -1:
            for rec in referee_ob.record_results['env_parameters'][:4]:
                print('\t', np.array(rec['label']), np.array(rec['thetas'][0:3]), np.array(rec['kappas']))
        else:
            for rec in referee_ob.record_results['env_parameters'][:4]:
                print('\t', np.array(rec['label']), np.array(rec['thetas'][0:3]))

    return referee_ob

def load_data_UniGRAB(refs, resdir, n_trials, env_names, ees, eeus, pes, add=False):
    for env_name in env_names:
        if not add:
            refs[env_name] = {}
        full_env_name = env_name + ('__sorted_kappa' if env_name.find('CM') == -1 else '')
        for ee, eeu, pe in product(ees, eeus, pes):
            refs[env_name][f'UniGRAB {ee} {eeu} {pe}{" bis" if add else ""}'] = retrieve_data_from_zip(f'{resdir}/{full_env_name}__Bandit_UniGRAB_{ee}_{eeu}_{pe}__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'UniRank best'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_UniTopRank_best_explo_oracle_None_T__games_10000000_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'UniRank best loglog'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_UniTopRank_best_explo_loglog_oracle_None_T__games_10000000_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'UniRank IDA'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_UniTopRank_globalTime_loglog_oracle_None_T__games_10000000_nb_trials_1000_record_length_5_rel_pos_*_games.gz')
        refs[env_name][f'TopRank'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_TopRank_10000000.0_delta_TimeHorizonKnown___games_10000000_nb_trials_1000_record_length_None_rel_pos_*_games.gz')

def load_data_paper(refs, resdir, env_names, n_trials, add=False):
    for env_name in env_names:
        if not add:
            refs[env_name] = {}
        full_env_name = env_name + ('__sorted_kappa' if env_name.find('CM') == -1 else '')
        refs[env_name][f'UniRank best'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_UniTopRank_best_explo_oracle_None_T__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'UniRank best loglog'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_UniTopRank_best_explo_loglog_oracle_None_T__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'TopRank'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_TopRank_10000000.0_delta_TimeHorizonKnown___games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'TopRank $10^12$'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_TopRank_1000000000000.0_delta_TimeHorizonKnown___games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'TopRank doubling'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_TopRank_100000.0_delta_TimeHorizonKnown_doubling_trick__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'GRAB'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_UniPBRank_10000000_T_9_gamma__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'Cascade-KL'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_CascadeKL_UCB__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'PB-MHB'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_1000000_nb_trials_1000_record_length_None_rel_pos_*_games.gz')
        refs[env_name][f'PB-MHB b'] = retrieve_data_from_zip(
            f'{resdir}/{full_env_name}__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_{n_trials}_nb_trials_1000_record_length_None_rel_pos_*_games.gz')

# %%
def time_sec_to_HMS(sec):
    heure = sec // 3600
    rest_h = sec % 3600
    minute = rest_h // 60
    rest_m = rest_h % 60

    return (str(int(heure)) + 'H ' + str(int(minute)) + 'min ' + str(int(rest_m)) + 'sec')


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

def myplot(X_val, data, color='red', linestyle='-', label='TopRank', type_errorbar='standart_error'):
    if data.shape[0]:
        average, lower, upper = barerror_value(np.array(data), type_errorbar=type_errorbar)
        plt.plot(X_val, average, color=color, linestyle=linestyle, label=label)
        plt.fill_between(X_val, lower, upper, color=color, alpha=0.3, linestyle=linestyle, label='')
    else:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)

def plot_score(refs, keys, x_lim=None, y_lim=None, selection=None):
    if selection is not None:
        plt.title(selection)
    for i, label in enumerate(keys):
        vals = refs[label]
        data = np.array(vals.record_results['expected_best_reward']) - np.array(vals.record_results['expected_reward'])
        if selection is not None and data.shape[0]:
            data = data[list(selection in params['label'] for params in vals.record_results['env_parameters'])]
        myplot(X_val=vals.get_recorded_trials(),
               data=data,
               color=f'C{i % 10}', linestyle=('-' if i//10==0 else ':'), label=f'({len(vals.record_results["expected_best_reward"])}) {label}')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.grid(True)
    plt.loglog()
    plt.xlim(x_lim)
    plt.ylim(y_lim)

def plot_time(refs, keys, x_lim=None, y_lim=None):
    for i, label in enumerate(keys):
        vals = refs[label]
        myplot(X_val=vals.get_recorded_trials(),
               data=np.array(vals.record_results['time_to_play']),
               color=f'C{i % 10}', linestyle=('-' if i // 10 == 0 else ':'), label=f'({len(vals.record_results["expected_best_reward"])}) {label}')
    plt.xlabel('Iteration')
    plt.ylabel('Computation Time')
    plt.grid(True)
    plt.loglog()
    plt.xlim(x_lim)
    plt.ylim(y_lim)


"""
# %%
# load data 10^7
refs = {}
resdir = "rg_playground/UniGRAB_results/outputs"
env_names = ["Yandex_CM_all_10_items_5_positions", "Yandex_all_10_items_5_positions", "KDD_all"]
ees = ["first", "best", "EO", "T2B"]
eeus = ["best", "all_potentials"]
pes = ["all", "focused"]
load_data_UniGRAB(refs, resdir, 10**7, env_names, ees, eeus, pes)

# %%
# plot data 10^7
for env_name in env_names:

    plt.figure(figsize=(25, 10))

    plt.subplot(2, 5, 1)
    plot_score(refs[env_name], refs[env_name].keys())

    plt.subplot(2, 5, 2)
    plot_time(refs[env_name], refs[env_name].keys())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(2, 5, 4)
    ees = ["first", "best", "EO", "T2B"]
    eeus = ["best", "all_potentials"]
    pes = ["focused"]
    plot_score(refs[env_name], [f'UniGRAB {ee} {eeu} {pe}' for ee, eeu, pe in product(ees, eeus, pes)], x_lim=[10**4,None], y_lim=[8*10**2,None])
    plt.legend()

    plt.subplot(2, 5, 5)
    ees = ["best", "EO"]
    eeus = ["best", "all_potentials"]
    pes = ["focused"]
    plot_score(refs[env_name], [f'UniGRAB {ee} {eeu} {pe}' for ee, eeu, pe in product(ees, eeus, pes)] + ['UniRank best'], x_lim=[10**4,None], y_lim=[8*10**2,None])
    plt.legend()

    ees = ["first", "best", "EO", "T2B"]
    eeus = ["best", "all_potentials"]
    pes = ["all", "focused"]
    for i, ee in enumerate(ees):
        plt.subplot(2, 5, 6 + i)
        plot_score(refs[env_name], [f'UniGRAB {ee} {eeu} {pe}' for eeu, pe in product(eeus, pes)])
        plt.legend()

    plt.subplot(2, 5, 10)
    plot_score(refs[env_name], ['TopRank', 'UniGRAB best best focused', 'UniGRAB EO all_potentials focused', 'UniRank best', 'UniRank best loglog', 'UniRank IDA'], x_lim=[10**5,None], y_lim=[8*10**2,None])
    plt.legend()


    plt.savefig(f'rg_playground/UniGRAB_results/figs/{env_name}__UniGRAB.pdf', bbox_inches='tight', pad_inches=0)

# %%
# load data PBM_10^8
refs = {}
resdir = "rg_playground/UniGRAB_results/outputs"
env_names = ["Yandex_all_10_items_5_positions", "KDD_all"]
ees = ["best", "EO", "T2B"]
eeus = ["best", "all_potentials"]
pes = ["focused"]
load_data_UniGRAB(refs, resdir, 10**8, env_names, ees, eeus, pes)
load_data_UniGRAB(refs, resdir, 10**9, ['KDD_all'], ['EO'], ['all_potentials'], ['focused'], add=True)

# %%
# plot data 10^8
for env_name in env_names:

    plt.figure(figsize=(25, 10))

    plt.subplot(2, 4, 1)
    plot_score(refs[env_name], refs[env_name].keys())

    plt.subplot(2, 4, 2)
    plot_score(refs[env_name], refs[env_name].keys(), x_lim=[10**4,None], y_lim=[8*10**2,None])

    plt.subplot(2, 4, 3)
    plot_time(refs[env_name], refs[env_name].keys())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ees = ["best", "EO", "T2B"]
    eeus = ["best", "all_potentials"]
    pes = ["focused"]
    for i, ee in enumerate(ees):
        plt.subplot(2, 4, 5 + i)
        plot_score(refs[env_name], [f'UniGRAB {ee} {eeu} {pe}' for eeu, pe in product(eeus, pes)])
        if env_name == 'KDD_all':
            plot_score(refs[env_name], ['UniGRAB EO all_potentials focused bis'])
        plt.legend()

    plt.subplot(2, 4, 8)
    plot_score(refs[env_name], ['TopRank', 'UniGRAB best best focused', 'UniGRAB EO all_potentials focused', 'UniRank best', 'UniRank IDA'], x_lim=[10**5,None], y_lim=[8*10**2,None])
    plt.legend()


    plt.savefig(f'rg_playground/UniGRAB_results/figs/{env_name}__UniGRAB_10-8.pdf', bbox_inches='tight', pad_inches=0)
"""

"""
# %%
# artificial data, Yandex_CM_L6, Yandex_L6
refs = {}
resdir = "rg_playground/UniGRAB_results/outputs"
env_names = ["purely_simulated__small_and_close", "purely_simulated__small_and_close_CM", "Yandex_CM_all_6_items_5_positions", "Yandex_all_6_items_5_positions"]
env_names = ["Yandex_9_query_6_items_5_positions"]
load_data_paper(refs, resdir, env_names, n_trials=10**7)

# %%
# plot data 10^7
for env_name in env_names:

    plt.figure(figsize=(25, 10))

    plt.subplot(1, 3, 1)
    plot_score(refs[env_name], refs[env_name].keys())

    plt.subplot(1, 3, 2)
    plot_time(refs[env_name], refs[env_name].keys())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(f'rg_playground/UniGRAB_results/figs/{env_name}__all.pdf', bbox_inches='tight', pad_inches=0)


# %%
# query per query
refs = {}
resdir = "rg_playground/UniGRAB_results/outputs"
env_names = ["Yandex_CM_all_10_items_5_positions", "Yandex_all_10_items_5_positions", "KDD_all"]
env_names = ["Yandex_all_6_items_5_positions"]
load_data_paper(refs, resdir, env_names, n_trials=10**7)

for env_name in env_names:

    plt.figure(figsize=(25, 10))

    for i in range(10):
        if env_name.split('_')[0] == 'KDD':
            if i == 8: break
            selection = ['1 ', '2 ', '4 ', '7 ', '8 ', '9 ', '10 ', '19 '][i]
        elif env_name.startswith('Yandex_CM'):
            if i == 10: break
            selection = f'Yandex {i} '
        else:
            if i == 10: break
            selection = f'({i} for us)'
        plt.subplot(3, 5, i + 1)
        plot_score(refs[env_name], refs[env_name].keys(), selection=selection)
    plt.subplot(3, 5, 11)
    plot_score(refs[env_name], refs[env_name].keys())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(f'rg_playground/UniGRAB_results/figs/{env_name}_per_query.pdf', bbox_inches='tight', pad_inches=0)
"""

"""
# %%
# 6 figs of the paper
refs = {}
resdir = "rg_playground/UniGRAB_results/outputs"
env_names = ["KDD_all", "Yandex_all_10_items_5_positions", "Yandex_CM_all_10_items_5_positions", "Yandex_9_query_6_items_5_positions", "purely_simulated__small_and_close", "purely_simulated__small_and_close_CM"]
load_data_paper(refs, resdir, env_names, n_trials=10**7)

# %%
# plot data 10^7

plt.figure(figsize=(25, 10))
for i, env_name in enumerate(env_names):
    plt.subplot(2, 6, 2*i + 1)
    plot_score(refs[env_name], refs[env_name].keys())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig(f'rg_playground/UniGRAB_results/figs/all.pdf', bbox_inches='tight', pad_inches=0)
"""

# %%
# K=10
refs = {}
resdir = "rg_playground/UniGRAB_results/outputs"
env_names = ["Yandex_all_10_items_10_positions", "Yandex_all_15_items_10_positions", "Yandex_CM_all_15_items_10_positions"]
load_data_paper(refs, resdir, env_names, n_trials=10**6)

plt.figure(figsize=(25, 10))
for i, env_name in enumerate(env_names):
    plt.subplot(1, 6, 2*i + 1)
    plot_score(refs[env_name], refs[env_name].keys())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig(f'rg_playground/UniGRAB_results/figs/all.pdf', bbox_inches='tight', pad_inches=0)
