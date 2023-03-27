#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play

# 1 game = 2h30   =>   5 games per node (to target 12 hours)
total_nb_game = 20
nb_game_per_node = 5
start_games = range(0, total_nb_game, nb_game_per_node)
PositionsRankings =['--order_kappa', '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
nb_trial = 100000
nb_check_points = 100
env_names = ["--std"]#["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
force = False   # to force run again
output_path = 'results'
record_length = 1000
is_KL = True

if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param.txt", "w") as file:
        for start_game, env_name, ranking in product(start_games, env_names, PositionsRankings):
            args_env = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} {ranking} {env_name} '
            if is_KL:
                args_player = f'--KL-MLMR {nb_trial} {output_path}{" --force" if force else ""} --nb_checkpoints {nb_check_points}'
            else:
                args_player = f'--MLMR {output_path}{" --force" if force else ""} --nb_checkpoints {nb_check_points}'
            args = args_env + args_player
            if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                file.write('{args}\n'.format(args=args))
                nb_to_run += 1
            else:
                #print('Skipped as already done: {args}'.format(args=args))
                nb_skipped += 1
    print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
