#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""
import os
from itertools import product
from bandits_to_rank.tools.tools import get_SCRATCHDIR

total_nb_game = 20
nb_game_per_node = 1
start_games = range(0, total_nb_game, nb_game_per_node)
nb_trial = 100000
nb_steps = [1,10] # change computation time, so cannot take several values for the same oar_array
cs = [100.] #[0.01, 0.1, 1., 10., 100., 1000., 10000.]
random_starts = [True]
env_names =["purely_simulated__std"] # ["purely_simulated__std", "purely_simulated__small", "purely_simulated__big"] + ["KDD_%d_query" % (i) for i in range(8)]
env_keys = ["--std"] #"--std", "--small",  #+ ["--KDD %d" % (i) for i in range(8)]
output_path = os.path.join(get_SCRATCHDIR(), "results/")
record_length = 1000

if __name__ == "__main__":
    with open("python_param.txt", "w") as file:
        for start_game, c,nb_step, random_start, (env_name, env_key) in product(start_games, cs,nb_steps, random_starts, zip(env_names, env_keys)):
            games_name = 'games_' + str(nb_trial) + '_nb_trials_' + str(record_length) + '_record_length_' + str(start_game) + '-' + str(start_game+nb_game_per_node) + '_games'
            random_start_name = "random_start" if random_start else "warm-up_start"
            player_name = 'Bandit_TSMH_' + random_start_name + '_' + str(c) + '_c_' + str(nb_step) + '_step'
            output_file = output_path + "/" + env_name + "__" + player_name + "__" + games_name + '.gz'
            if not os.path.exists(output_file):
                print(output_file)
                random_start_key = "--random_start" if random_start else ""
                file.write("--play %d -s %d %d -r %d %s --TSMH %d %f %s --out %s \n" % (nb_game_per_node, start_game, nb_trial, record_length, env_name, nb_step, c, random_start_key, output_path))
