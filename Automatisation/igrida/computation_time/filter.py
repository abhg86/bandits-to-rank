#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

from itertools import product
from Automatisation.igrida.exp import line_args_to_params, play

if __name__ == "__main__":
    nb_to_run = 0
    nb_skipped = 0
    with open("python_param_all.txt", "r") as in_file:
        with open("python_param.txt", "w") as out_file:
            for args in in_file:
                args = args.rstrip()
                if play(line_args_to_params(args), dry_run=True, verbose=False) != 0:
                    out_file.write('{args}\n'.format(args=args))
                    nb_to_run += 1
                else:
                    #print('Skipped as already done: {args}'.format(args=args))
                    nb_skipped += 1
    print(f'Among {nb_to_run+nb_skipped}, {nb_to_run} will be ran, and {nb_skipped} are skipped.')
