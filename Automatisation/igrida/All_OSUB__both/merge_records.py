#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from Automatisation.igrida.exp import merge_records, line_args_to_params
from Automatisation.igrida.param import Parameters
import os
from itertools import product

# Path to bandtis-to-rank module
import bandits_to_rank
packagedir = os.path.dirname(bandits_to_rank.__path__[0])


nb_trials = [10000000]

PositionsRankings =['--shuffle_kappa']#, '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
OSUB_types = ['--GRAB']#['--OSUB','--OSUB_PBM','--GRAB','--PBubblePBRank']
finit_memory =[False]
gammas = [0]
memory =[1000]
env_names = ['--std_K100']#['--Yandex_all']+['--Yandex_CM_all'] #["--KDD_all"]#['--test_CM']+["--test"]+ ['--Yandex_CM_all'] +["--std", "--xxsmall", "--big"] + ["--KDD_all"] + ['--Yandex_all']

forced_initialisations = [False]
force = False   # to force run again
input_path = 'Result_Bandit'
record_length = 1000
nb_items = 10
nb_positions = [5]#[5,10]
relevant_position = [5]


if __name__ == "__main__":
    for env_name in env_names:
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_Yandex/'
        elif env_name == '--Yandex_CM_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_Yandex/'
        elif env_name == '--Yandex_equi_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_Yandex/'
        else:
            output_path = f'{packagedir}/Test/exp_ICML2021/results/simul'

        for ranking, OSUB_type,nb_trial, rel_pos,nb_position in product(PositionsRankings, OSUB_types, nb_trials,relevant_position,nb_positions):
            arg_env = f'--merge {nb_trial} -r {record_length} -l {rel_pos} {ranking} {env_name} '# {nb_position} {nb_items}   '
            if OSUB_type == '--OSUB':
                temp_finit_memory = finit_memory
                temp_gamma = ['']
                temp_force_init = ['']
                gamma_tune = False

            elif OSUB_type in ['--GRAB', '--PBubblePBRank', '--OSUB_PBM']:
                temp_finit_memory = [False]
                temp_gamma = gammas
                temp_force_init = forced_initialisations
                gamma_tune = True

            else:
                temp_finit_memory = [False]
                temp_gamma = [None]
                temp_force_init = ['']
                gamma_tune = False
            for is_finit in temp_finit_memory:
                if is_finit:
                    temp_memory = memory
                else:
                    temp_memory = [None]
                for memo, gamma, force_init in product(temp_memory, temp_gamma, temp_force_init):
                    arg_player = f'{OSUB_type} {int(memo) if is_finit else int(nb_trial)} {gamma if  gamma_tune else ""} {"--forced_initiation" if force_init else ""}  {"--finit_memory" if is_finit else ""} {input_path}  {output_path}'
                    args =arg_env +arg_player
                    params = line_args_to_params(args)
                    print(args)
                    merge_records(params)
