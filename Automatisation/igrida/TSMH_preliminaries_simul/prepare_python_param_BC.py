#!/usr/bin/python3
# -*- coding: utf-8 -*-
""""""

import os
from itertools import product

from bandits_to_rank.tools.tools import get_SCRATCHDIR

total_nb_game = 1000
nb_game_per_node = 50
start_games = range(0, total_nb_game, nb_game_per_node)
nb_trial = 100000
env_names = ["--std"]# [, "--small", "--big" + ["--KDD %d" % (i) for i in range(8)]
output_path = os.path.join(get_SCRATCHDIR(), "results/")
record_length = 1000


if __name__ == "__main__":
    with open("python_param_BC.txt", "w") as file:
        for start_game, env_name in product(start_games, env_names):
            file.write("--play %d -s %d %d -r %d %s --BC-MPTS --out %s \n" % (nb_game_per_node, start_game, nb_trial, record_length, env_name, output_path))
            file.write("--play %d -s %d %d -r %d %s --BC-MPTS --oracle --out %s \n" % (nb_game_per_node, start_game, nb_trial, record_length, env_name, output_path))
