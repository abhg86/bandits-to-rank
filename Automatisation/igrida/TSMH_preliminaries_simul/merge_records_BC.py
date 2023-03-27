#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from Automatisation.igrida.exp import merge_records
import os
from itertools import product

# Path to bandtis-to-rank module
import bandits_to_rank
packagedir = os.path.dirname(bandits_to_rank.__path__[0])

nb_trial = 100000
input_path = packagedir + '/Automatisation/igrida/result/igrida'
record_length = 1000




if __name__ == "__main__":
    # KDD
    output_path = packagedir + '/Test/exp_ECAI2020/result/real_KDD/'
    input_env_name = 'KDD_*_query'
    output_env_name = 'KDD'
    player_name = 'Bandit_BC-MPTS_oracle'
    #merge_records(input_env_name, player_name, nb_trial, input_path, output_path=output_path, output_env_name=output_env_name, len_record_short=record_length)
    player_name = 'Bandit_BC-MPTS_greedy'
    #merge_records(input_env_name, player_name, nb_trial, input_path, output_path=output_path, output_env_name=output_env_name, len_record_short=record_length)
    # simul
    env_names = ["purely_simulated__std"]#, "purely_simulated__small", "purely_simulated__big"]
    output_path = packagedir + '/Test/exp_ECAI2020/result/simul'
    for env_name, in product(env_names):
        player_name = 'Bandit_BC-MPTS_oracle'
        merge_records(env_name, player_name, nb_trial, input_path, output_path=output_path, len_record_short=record_length)
        player_name = 'Bandit_BC-MPTS_greedy'
        merge_records(env_name, player_name, nb_trial, input_path, output_path=output_path, len_record_short=record_length)





