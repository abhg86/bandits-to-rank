#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from Automatisation.igrida.exp import merge_records, line_args_to_params
from Automatisation.igrida.param import Parameters
import os
from itertools import product

# Path to bandtis-to-rank module
# todo à mettre dans "tools"
import bandits_to_rank
packagedir = os.path.dirname(bandits_to_rank.__path__[0])

#"""
# --- selected hyper-parameters ---
nb_trial = 10**5
alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]
alphas = [0.01, 1., 100.]
gap_MLEs = [10]
gap_qs = [10000]

env_names = ["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
env_names = ["--std"]
shuffle_kappa = [True, False]

input_path = 'results'
record_length = 1000
#"""


if __name__ == "__main__":
    for env_name, is_shuffle, alpha, gap_MLE, gap_q in product(env_names, shuffle_kappa,alphas, gap_MLEs, gap_qs):
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_AAAI2021/results/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_AAAI2021/results/real_Yandex/'
        else:
            output_path = f'{packagedir}/Test/exp_AAAI2021/results/simul'

        args_player = f'--PMED {alpha} {gap_MLE} {gap_q}'
        args = f'--merge {nb_trial} -r {record_length} {""if is_shuffle else"--order_kappa"} {env_name} {args_player} {input_path} {output_path}'
        params = line_args_to_params(args)
        merge_records(params)
