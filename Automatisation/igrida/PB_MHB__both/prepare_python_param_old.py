#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play


#"""
# --- selected hyper-parameters ---
# 1 game = 33h   =>   1 game per node
total_nb_game = 20
nb_game_per_node = 5
start_games = range(0, total_nb_game, nb_game_per_node)
PositionsRankings = ['--order_kappa', '--shuffle_kappa', '--increasing_kappa'] #['--order_kappa', '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
nb_trial = 10000000
nb_check_points = 100   # run of more than 10 hours => more checkpoints
nb_steps = [1]          # WARNING: change computation time
cs = [0.1, 1., 10.,100., 1000.] #[0.1, 1., 10.,100., 1000.]
proposal_names = ["--TGRW"]#['--TGRW', '--LGRW', "--RR", "--MaxPos","--PseudoView"]
str_prop_possible_RR =["TGRW-LGRW-Max_Position-Pseudo_View"] #["TGRW-LGRW-Max_Position-Pseudo_View","TGRW-LGRW-Max_Position-Pseudo_View"]

random_starts = [False]
vari_sigmas = [True] #[True, False]

env_names = ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)] #["--Yandex_equi %d" % (i) for i in range(10)] +["--test_CM"]+ ["--Yandex_CM %d" % (i) for i in range(10)] #["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
force = False   # to force run again
output_path = 'results'
record_length = 1000
#"""

if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param_tunning_TGRW_real.txt", "w") as file:
        for start_game, nb_step, random_start, env_name, ranking, proposal_name in product(start_games, nb_steps, random_starts, env_names, PositionsRankings ,proposal_names):

            if proposal_name in ['--TGRW', '--LGRW']:
                tmp_cs = cs
                tmps_vari_sigma = vari_sigmas
                tmps_str_prop_possible =[None]

            elif proposal_name in ["--RR"]:
                tmp_cs = cs
                tmps_vari_sigma = vari_sigmas
                tmps_str_prop_possible = str_prop_possible_RR

            else:
                tmp_cs = [None]
                tmps_vari_sigma = [False]
                tmps_str_prop_possible = [None]

            for c, vari_sigma,str_prop_possible in product(tmp_cs, tmps_vari_sigma,tmps_str_prop_possible):

                args_proposal = f'{proposal_name} {c if c is not None else ""} {str_prop_possible if str_prop_possible is not None else ""}  {"--vari_sigma" if vari_sigma else ""}'
                args_env = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} {ranking}  {env_name} '
                args_player = f'--PB-MHB {nb_step} {args_proposal} {" --random_start" if random_start else ""} {output_path}{" --force" if force else ""} --nb_checkpoints {nb_check_points}'
                args = args_env + args_player

                if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                    file.write('{args}\n'.format(args=args))
                    nb_to_run += 1
                else:
                    #print('Skipped as already done: {args}'.format(args=args))
                    nb_skipped += 1
    print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
