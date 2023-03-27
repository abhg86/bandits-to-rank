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



nb_trial = 100000000
oracles = [False]
deltas = [0.00000001, 0.0000001, 0.000001, 0.1]#[0.001,0.01,0.1,1]
Time_horizon_known_possibilities =[True]
doubling_trick_possibilities =[False]
env_names = ["--Yandex_CM_all"] +["--Yandex_all"] #["--test_CM"]+["--test"]+ ['--Yandex_all']+ ['--Yandex_CM_all'] #["--std", "--xxsmall", "--big"] #+ ["--KDD_all"]+ ['--Yandex_all']
PositionsRankings = [ '--order_kappa'] #['--order_kappa', '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
input_path = 'Result_Bandit'#'results'
record_length = 1000
nb_items = 10
nb_positions = [5,10]
relevant_positions = [5]


if __name__ == "__main__":
    for oracle, env_name, ranking,Time_horizon_known in product(oracles, env_names, PositionsRankings,Time_horizon_known_possibilities):
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results'#/real_Yandex/'
        elif env_name == '--Yandex_CM_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_Yandex/'
        elif env_name == '--Yandex_equi_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results'#/real_Yandex/'
        else:
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/simul'
        if Time_horizon_known :
                temps_delta =[float(nb_trial)]
                temps_doubling_trick = doubling_trick_possibilities
        else:
                temps_delta = deltas
                temps_doubling_trick =[False]
        for delta, doubling_trick,rel_pos,nb_position in product(temps_delta,temps_doubling_trick,relevant_positions,nb_positions):
                args = f'--merge {nb_trial} -r {record_length} -l {rel_pos} {ranking} {env_name} {nb_position} {nb_items} --TopRank {nb_trial/10 if doubling_trick else delta}{" --horizon_time_known" if Time_horizon_known else ""}{" --doubling_trick" if doubling_trick else ""}{" --oracle" if oracle else ""} {input_path} {output_path}'
                params = line_args_to_params(args)
                print (args)
                merge_records(params)
