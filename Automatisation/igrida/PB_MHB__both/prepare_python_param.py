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
nb_trial = 10000000
nb_check_points = 100   # run of more than 10 hours => more checkpoints
nb_steps = [1]          # WARNING: change computation time
cs = [0.1, 1., 10., 1000.]
proposal_names = [ "--RR"] #['--TGRW', '--LGRW', "--RR", "--MaxPos","--PseudoView"]
random_starts = [False]
vari_sigmas = [True,False]

env_names = ["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] #+ ["--Yandex %d" % (i) for i in range(10)]
force = False   # to force run again
output_path = 'results'
record_length = 1000
#"""

if __name__ == "__main__":
    for proposal_name in proposal_names:
        if proposal_name in ['--TGRW', '--LGRW', "--RR"]:
            tmp_cs = cs
            tmps_vari_sigma = vari_sigmas
        else:
            tmp_cs = [None]
            tmps_vari_sigma = [False]
        for c, vari_sigma in product(tmp_cs, tmps_vari_sigma):
            nb_to_run = 0
            nb_skipped = 0
            args_proposal = f'{proposal_name} {c if c is not None else ""} {"--vari_sigma" if vari_sigma else ""}'
            with open(f'python_param_{proposal_name}_{vari_sigma}_{c}_.txt', "w") as file:
                for start_game, nb_step, random_start, env_name in product(start_games, nb_steps, random_starts, env_names): 
                    args_player = f'--PB-MHB {nb_step} {args_proposal} {" --random_start" if random_start else ""} {output_path}{" --force" if force else ""} --nb_checkpoints {nb_check_points}'
                    args_env = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} {env_name} '
                    args = args_env + args_player
                    if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                        file.write('{args}\n'.format(args=args))
                        nb_to_run += 1
                    else:
                          #print('Skipped as already done: {args}'.format(args=args))
                        nb_skipped += 1
            print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')