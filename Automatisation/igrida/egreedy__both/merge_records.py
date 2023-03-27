#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from Automatisation.igrida.exp import merge_records, line_args_to_params
import os
from itertools import product

# Path to bandtis-to-rank module
import bandits_to_rank
packagedir = os.path.dirname(bandits_to_rank.__path__[0])


nb_trial = 10000000
cs = [0., 1., 10., 100., 1000., 10000., 100000., 1000000., 10.**20]
maj = 1
env_names = ['--Yandex_CM_all']#["--std", "--xxsmall", "--big"] + ["--KDD_all"]+['--Yandex_all']
input_path = 'Result_Bandit'
record_length = 1000
equi_K = 5

if __name__ == "__main__":
    for c, env_name in product(cs, env_names):
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
        args = f'--merge {nb_trial} -r {record_length} {env_name} --eGreedy {c} {maj} {input_path} {output_path}'
        params = line_args_to_params(args)
        merge_records(params)
