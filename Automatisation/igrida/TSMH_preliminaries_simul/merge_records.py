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
nb_steps = [1] #,10 change computation time, so cannot take several values for the same oar_array
cs = [100.]# [10000., 100000.,1000000.,10000000.,100000000]#[0.01, 0.1, 1., 10., 100., 1000., 10000.]
random_starts = [False]
input_path =  packagedir + '/Automatisation/igrida/result/igrida'
record_length = 1000


if __name__ == "__main__":
    for c,nb_step, random_start in product(cs,nb_steps, random_starts):
        random_start_name = "random_start" if random_start else "warm-up_start"
        player_name = 'Bandit_TSMH_'+ random_start_name + '_' + str(c) + '_c_' + str(nb_step) + '_step'
        # KDD
        output_path = packagedir + '/Test/exp_ECAI2020/result/real_KDD/'
        input_env_name = 'KDD_*_query'
        output_env_name = 'KDD'
        #merge_records(input_env_name, player_name, nb_trial, input_path, output_path=output_path, output_env_name=output_env_name, len_record_short=record_length)
        # simul
        env_names = ["purely_simulated__std"]#, "purely_simulated__small", "purely_simulated__big"]
        output_path = packagedir + '/Test/exp_ECAI2020/result/simul'
        for env_name, in product(env_names):
            merge_records(env_name, player_name, nb_trial, input_path, output_path=output_path, len_record_short=record_length)



