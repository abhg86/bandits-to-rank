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
deltas = [nb_trial**-4]
sorteds = [True, False]
oracles = [True]
env_names = ["--std", "--xxsmall", "--big"] + ["--KDD_all"]
env_names = ['--Yandex_all']
input_path = 'results'
record_length = 1000


if __name__ == "__main__":
    for delta, oracle, sorted, env_name in product(deltas, oracles, sorteds, env_names):
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_CIKM2020/result/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_CIKM2020/result/real_Yandex/'
        else:
            output_path = f'{packagedir}/Test/exp_CIKM2020/result/simul'
        args = f'--merge {nb_trial} -r {record_length} {env_name} --BubbleRank {delta}{" --oracle" if oracle else ""}{" --sorted" if sorted else ""} {input_path} {output_path}'
        params = line_args_to_params(args)
        merge_records(params)
