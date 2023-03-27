#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play

# 1 game = 1h30   =>   8 games per node (to target 12 hours)
total_nb_game = 20
nb_game_per_node = 1
start_games = range(0, total_nb_game, nb_game_per_node)
nb_trial = 10000000
cs = [0., 1., 10., 100., 1000., 10000., 100000., 1000000., 10.**20]
maj = 1
env_names = ["--std", "--xxsmall", "--big"] + ["--KDD %d" % (i) for i in range(8)] + ["--Yandex %d" % (i) for i in range(10)]
force = False   # to force run again
output_path = 'results'
record_length = 1000


if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param.txt", "w") as file:
        for start_game, c, env_name in product(start_games, cs, env_names):
            args = f'--play {min(total_nb_game - start_game, nb_game_per_node)} -s {start_game} {nb_trial} -r {record_length} {env_name} --eGreedy {c} {maj} {output_path}{" --force" if force else ""}'
            if force or play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                file.write('{args}\n'.format(args=args))
                nb_to_run += 1
            else:
                #print('Skipped as already done: {args}'.format(args=args))
                nb_skipped += 1
    print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
