#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play

# todo: ÃƒÂ  dÃƒÂ©terminer
# 1 game = 2h30   =>   5 games per node (to target 12 hours)
total_nb_game = 20
nb_game_per_node = 1
start_games = range(0, total_nb_game, nb_game_per_node)
PositionsRankings =[ '--order_kappa'] #['--order_kappa', '--shuffle_kappa', '--shuffle_kappa_except_first', '--increasing_kappa','--increasing_kappa_except_first']
nb_trial = 100000000
oracles = [False]
deltas =  [0.00000001, 0.0000001, 0.000001, 0.1]
Time_horizon_known_possibilities =[True]
doubling_trick_possibilities =[False]
env_names = ["--Yandex %d" % (i) for i in range(10)]  +["--Yandex_CM %d" % (i) for i in range(10)] #["--Yandex_equi %d" % (i) for i in range(10)] ["--Yandex_CM %d" % (i) for i in range(10)] #["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
force = False   # to force run again
output_path = 'results'
record_length = 1000
equi_K = 5
nb_item = 10
nb_positions = [10,5]
relevant_positions = [5]


if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param_Yandex_K10.txt", "w") as file:
        for start_game, oracle, env_name,nb_position, ranking, Time_horizon_known,relevant_pos in product(start_games, oracles, env_names,nb_positions, PositionsRankings, Time_horizon_known_possibilities,relevant_positions):
            args_env = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} -l {relevant_pos} {ranking} {env_name} {nb_position} {nb_item}'
            if Time_horizon_known:
                temps_delta =[float(nb_trial)]
                temps_doubling_trick = doubling_trick_possibilities
            else :
                temps_delta = deltas
                temps_doubling_trick =[False]
            for delta, doubling_trick in product(temps_delta,temps_doubling_trick):
                args_player = f'--TopRank {int(delta/10) if doubling_trick else delta}{" --horizon_time_known" if Time_horizon_known else ""}{" --doubling_trick" if doubling_trick else ""}{" --oracle" if oracle else ""} {output_path}{" --force" if force else ""}'
                args = args_env + args_player
                if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                    file.write('{args}\n'.format(args=args))
                    nb_to_run += 1
                else:
                    #print('Skipped as already done: {args}'.format(args=args))
                    nb_skipped += 1
    print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
