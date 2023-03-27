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


nb_trials = [100000000]

PositionsRankings =['--order_kappa']#, '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
env_names =['--Yandex_CM_all']+['--Yandex_all']#["--KDD_all"]#['--test_CM']+["--test"]+ ['--Yandex_CM_all'] +["--std", "--xxsmall", "--big"] + ["--KDD_all"] + ['--Yandex_all']

forced_initialisations = [False]
force = False   # to force run again
input_path = 'Result_Bandit'
record_length = 1000
known_horizon =[False]
versions = [None]
equi_K = 5
nb_items = 10
nb_positions = [5,10]
relevant_positions = [5]


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
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/simul'

        for ranking, nb_trial,relevant_pos,is_known,version,nb_position in product(PositionsRankings, nb_trials, relevant_positions,known_horizon,versions,nb_positions):
            arg_env = f'--merge {nb_trial} -r {record_length} -l {relevant_pos} {ranking} {env_name} {nb_position} {nb_items} '
            arg_player = f'--UniTopRank {f"--known_horizon {nb_trial}" if is_known else "" } {f"--version {version}" if version != None else "" }  {input_path}  {output_path}'
            args =arg_env +arg_player
            params = line_args_to_params(args)
            print(args)
            merge_records(params)
