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

PositionsRankings =['--order_kappa']#, '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
env_names =['--Yandex_CM_all']+['--Yandex_all']#["--KDD_all"]#['--test_CM']+["--test"]+ ['--Yandex_CM_all'] +["--std", "--xxsmall", "--big"] + ["--KDD_all"] + ['--Yandex_all']

forced_initialisations = [False]
force = False   # to force run again
input_path = 'Result_Bandit'
record_length = 1000
equi_K = 5
relevant_positions =[5,10]


if __name__ == "__main__":
    for env_name in env_names:
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/'
        elif env_name == '--Yandex_CM_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_Yandex/'
        elif env_name == '--Yandex_equi_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_Yandex/'
        else:
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/simul'

        for ranking, nb_trial,relevant_pos in product(PositionsRankings, nb_trials, relevant_positions):
            arg_env = f'--merge {nb_trial} -r {record_length} -l {relevant_pos} {ranking} {env_name}  '
            arg_player = f'--UniRank {input_path}  {output_path}'
            args =arg_env +arg_player
            params = line_args_to_params(args)
            print(args)
            merge_records(params)
