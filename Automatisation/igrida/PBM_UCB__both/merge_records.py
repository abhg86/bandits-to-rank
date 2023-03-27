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

nb_trial = 100000
oracles = [True, False]
epsilons = [0.001]#[0.001, 0.01, 0.1, 1, 10]
env_names = ["--std", "--xxsmall", "--big"] + ["--KDD_all"] + ['--Yandex_all']
input_path = 'results'
record_length = 1000


if __name__ == "__main__":
    for oracle, env_name, epsilon in product(oracles, env_names,epsilons):
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_AAAI2021/results/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_AAAI2021/results/real_Yandex/'
        else:
            output_path = f'{packagedir}/Test/exp_AAAI2021/results/simul'
        args = f'--merge {nb_trial} -r {record_length} {env_name} --PBM-UCB {epsilon}{" --oracle" if oracle else ""} {input_path} {output_path}'
        params = line_args_to_params(args)
        merge_records(params)
