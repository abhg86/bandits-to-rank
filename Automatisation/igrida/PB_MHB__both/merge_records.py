#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""
from typing import List

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
PositionsRankings =['--order_kappa']#['--order_kappa', '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
nb_trial = 1000000
nb_steps = [1]
cs = [1000.]#[0.1, 1., 10., 100., 1000.]
proposal_names = ['--TGRW']#['--TGRW', '--LGRW', "--RR", "--MaxPos","--PseudoView"]
str_prop_possible_RR =["TGRW-LGRW-Max_Position-Pseudo_View"] #["TGRW-LGRW-Max_Position-Pseudo_View","TGRW-LGRW-Max_Position-Pseudo_View"]
vari_sigmas = [True]
random_starts = [False]
env_names = ['--Yandex_all']#["--test_CM"]+["--test"]+ ['--Yandex_all']+ ['--Yandex_CM_all'] #["--std", "--xxsmall", "--big"] + ["--KDD_all"]+ ['--Yandex_all']

input_path = 'Result_Bandit'
record_length = 1000

nb_items = 10
nb_positions = [10]
relevant_positions = 5
#"""



if __name__ == "__main__":
    for env_name, ranking,nb_step, random_start, proposal_name,nb_position in product(env_names, PositionsRankings, nb_steps, random_starts, proposal_names,nb_positions):
        if env_name == '--KDD_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/real_KDD/'
        elif env_name == '--Yandex_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021'#/real_Yandex/'
        elif env_name == '--Yandex_CM_all':
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/'
        else:
            output_path = f'{packagedir}/Test/exp_NeurIPS2021/results/simul'

        if proposal_name in ['--TGRW', '--LGRW']:
            tmp_cs = cs
            tmps_vari_sigma = vari_sigmas
            tmps_str_prop_possible = [None]

        elif proposal_name in ["--RR"]:
            tmp_cs = cs
            tmps_vari_sigma = vari_sigmas
            tmps_str_prop_possible = str_prop_possible_RR

        else:
            tmp_cs = [None]
            tmps_vari_sigma = [False]
            tmps_str_prop_possible = [None]

        for c, vari_sigma, str_prop_possible in product(tmp_cs, tmps_vari_sigma, tmps_str_prop_possible):

            args_proposal = f'{proposal_name} {c if c is not None else ""} {str_prop_possible if str_prop_possible is not None else ""} {"--vari_sigma" if vari_sigma else ""}'
            args_player = f'--PB-MHB {nb_step} {args_proposal} {" --random_start" if random_start else ""} '
            args = f'--merge {nb_trial} -r {record_length} -l {relevant_positions} {ranking} {nb_position} {nb_items}  {env_name} {args_player} {input_path} {output_path}'
            params = line_args_to_params(args)
            merge_records(params)
