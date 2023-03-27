#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play

# 1 game = 2h30   =>   5 games per node (to target 12 hours)
total_nb_game = 20
nb_game_per_node = 10
start_games = range(0, total_nb_game, nb_game_per_node)
PositionsRankings = ['--shuffle_kappa']  # ['--order_kappa', '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
nb_trial = 10000000
nb_check_points = 100
finit_memory = [False]
forced_initialisations = [True]
memory = [1000]
gammas = [0]
OSUB_types = ["--PBubblePBRank",'--OSUB_PBM']  # ["--PBubblePBRank",'--OSUB','--OSUB_PBM']
env_names = ["--Yandex %d" % (i) for i in range(10)]+["--KDD %d" % (i) for i in range(8)] # ["--test_CM"]+["--test"]+["--Yandex_CM %d" % (i) for i in range(10)] +["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
force = False # to force run again
output_path = 'results'
record_length = 1000


if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param_Yandex_KDD.txt", "w") as file:
        for start_game, env_name, ranking, OSUB_type in product(start_games, env_names, PositionsRankings, OSUB_types):
            args_env = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} {ranking} {env_name} '
            if OSUB_type == '--OSUB':
                temp_finit_memory = finit_memory
                temp_gamma = ['']
                temp_force_init = ['']
                gamma_tune = False

            elif OSUB_type in ['--GRAB','--PBubblePBRank','--OSUB_PBM']:
                temp_finit_memory = [False]
                temp_gamma = gammas
                temp_force_init = forced_initialisations
                gamma_tune = True

            else:
                temp_finit_memory = [False]
                temp_gamma = [None]
                temp_force_init = ['']
                gamma_tune = False

            for is_finit in temp_finit_memory:
                if is_finit:
                    temp_memory = memory
                else:
                    temp_memory = [None]
            for memo, gamma, force_init in product(temp_memory, temp_gamma,temp_force_init):
                args_player = f'{OSUB_type} {int(memo) if is_finit else int(nb_trial)} {gamma if  gamma_tune else ""} {"--forced_initiation" if force_init else ""} {"--finit_memory" if is_finit else ""} {output_path}{" --force" if force else ""} --nb_checkpoints {nb_check_points}'
                args = args_env + args_player
                if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                    file.write('{args}\n'.format(args=args))
                    nb_to_run += 1
                else:
                # print('Skipped as already done: {args}'.format(args=args))
                    nb_skipped += 1
print(f'Among {nb_to_run + nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
