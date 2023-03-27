#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play


#"""
# --- selected hyper-parameters ---
# 1 game = 33h   =>   1 game per node
total_nb_game = 20
nb_game_per_node = 1
start_games = range(0, total_nb_game, nb_game_per_node)
nb_trial = 100000
nb_check_points = 100   # run of more than 10 hours => more checkpoints
         # WARNING: change computation time
alphas = [0.001, 0.01, 0.1, 1., 10., 100, 1000.]
gap_MLEs = [10,1]
gap_qs = [1000,100,10]

env_names = ["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
force = False   # to force run again
output_path = 'results'
record_length = 1000
#"""


if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param.txt", "w") as file:
        for start_game,env_name, alpha, gap_MLE, gap_q in product(start_games, env_names, alphas, gap_MLEs, gap_qs):
            args_env = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} {env_name} '
            args_player = f'--PMED {alpha} {gap_MLE} {gap_q} {output_path}{" --force" if force else ""} --nb_checkpoints {nb_check_points}'
            args = args_env + args_player
            if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                file.write('{args}\n'.format(args=args))
                nb_to_run += 1
            else:
                #print('Skipped as already done: {args}'.format(args=args))
                nb_skipped += 1
    print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
