#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""

Manage Epsilon-greedy experiments.

Usage:
  exp.py --play <nb_game> [-s <start_game>] <nb_trials>  [-r <record_len>] [-l <nb_relevant_pos>]
            ( | [--order_kappa] | [--shuffle_kappa] | [--shuffle_kappa_except_first] | [--increasing_kappa] | [--increasing_kappa_except_first])
            (--KDD <query>
            |--Yandex <query> <nb_position> <nb_item>|--Yandex_equi <query> <K>
            | --std | --small | --big | --xsmall | --xxsmall | --test | --std_K100 | --small_and_close | --delta_variation_01
            |--Yandex_CM <query> <nb_position> <nb_item>
            |--std_CM | --small_CM | --big_CM | --xsmall_CM | --xxsmall_CM | --test_CM | --small_and_close_CM)
            (--eGreedy <c> <maj> [--noSVD]
                | --PBM-TS [--oracle|--greedyMLE|--greedySVD]
                | --PBM-PIE <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --PBM-UCB <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --BC-MPTS [--oracle|--greedyMLE|--greedySVD]
                | --PB-MHB <nb_steps> (--TGRW <c> [--vari_sigma]|--LGRW <c> [--vari_sigma]|--RR <c> <str_proposal_possible> [--vari_sigma]|--MaxPos|--PseudoView ) [--random_start]
                | --PB-GB <N> <h_param> <gamma> <L_smooth_param>
                | --PMED <alpha> <gap_MLE> <gap_q>
                | --PMED_test <alpha> <gap_MLE> <gap_q>
                | --TopRank <T> [--horizon_time_known] [--doubling_trick] [--oracle]
                | --OSUB <memory> [--finit_memory]
                | --OSUB_PBM <T>  <gamma> [--forced_initiation]
                | --GRAB [--known_horizon <T>] [--gamma <gamma>] [--forced_initiation] [--gap_type <gap_type>] [--optimism <optimism>]
                | --UniRank [--known_horizon <T>] [--gamma <gamma>]
                | --UniTopRank [--known_horizon <T>] [--version <vUTR>]
                | --UniGRAB [--ee <ee>] [--eeu <eeu>] [--pe <pe>]
                | --DCGUniRank [--known_horizon <T>] [--gamma <gamma>]
                | --PBubblePBRank <T> <gamma>
                | --BubbleOSUB <T>
                | --BubbleRank <delta> [--oracle]
                | --BubbleRank_OSUB2 [--R_init <R_init>] [--nb_shuffles <nb_shuffles>]
                | --MLMR
                | --KL-MLMR <T>
                | --CascadeKL-UCB
            )
            (<output_path> [--force] [--nb_checkpoints <nb_checkpoints>])
  exp.py --merge <nb_trials>  [-r <record_len>] [-l <nb_relevant_pos>]
            ( | [--order_kappa] | [--shuffle_kappa] | [--shuffle_kappa_except_first] | [--increasing_kappa] | [--increasing_kappa_except_first])
            (--KDD_all | --KDD <query>
            | --Yandex_all <nb_position> <nb_item>| --Yandex <query> <nb_position> <nb_item>
            | --Yandex_equi_all <K> | --Yandex_equi <query> <K>
            | --std | --small | --big | --xsmall | --xxsmall | --test| --std_K100 | --small_and_close | --delta_variation_01
            | --Yandex_CM_all <nb_position> <nb_item> | --Yandex_CM <query> <nb_position> <nb_item>
            |--std_CM | --small_CM | --big_CM | --xsmall_CM | --xxsmall_CM | --test_CM | --small_and_close_CM )
            (--eGreedy <c> <maj> [--noSVD]
                | --PBM-TS [--oracle|--greedyMLE|--greedySVD]
                | --PBM-PIE <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --PBM-UCB <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --BC-MPTS [--oracle|--greedyMLE|--greedySVD]
                | --PB-MHB <nb_steps> (--TGRW <c> [--vari_sigma]|--LGRW <c> [--vari_sigma]|--RR <c>  <str_proposal_possible> [--vari_sigma]|--MaxPos|--PseudoView ) [--random_start]
                | --PB-GB <N> <h_param> <gamma> <L_smooth_param>
                | --PMED <alpha> <gap_MLE> <gap_q>
                | --TopRank <T> [--horizon_time_known] [--doubling_trick] [--oracle]
                | --OSUB <memory> [--finit_memory]
                | --OSUB_PBM <T>  <gamma> [--forced_initiation]
                | --GRAB [--known_horizon <T>] [--gamma <gamma>] [--forced_initiation] [--gap_type <gap_type>] [--optimism <optimism>]
                | --UniRank [--known_horizon <T>] [--gamma <gamma>]
                | --UniTopRank [--known_horizon <T>] [--version <vUTR>]
                | --UniGRAB [--ee <ee>] [--eeu <eeu>] [--pe <pe>]
                | --DCGUniRank [--known_horizon <T>] [--gamma <gamma>]
                | --PBubblePBRank <T> <gamma>
                | --BubbleOSUB <T>
                | --BubbleRank <delta> [--oracle]
                | --BubbleRank_OSUB2 [--R_init <R_init>] [--nb_shuffles <nb_shuffles>]
                | --MLMR
                | --KL-MLMR <T>
                | --CascadeKL-UCB
            )
            (<input_path> [<output_path>])
  exp.py (-h | --help)

Options:
  -h --help         Show this screen
  -r <record_len>   Number of recorded trials [default: 1000]
  -l <nb_relevant_pos>  The reward is computed with respect to positions 0:l. l=0 means "all possible positions" [default: 0]
  -s <start_game>   Run games from <start_game> to <start_game> + <nb_games> - 1 [default: 0]
  <output_path>     Where to put the merged file [default: <input_path>]
                    ! WARNING ! has to be relative wrt. $SCRATCHDIR or absolute
  -KDD_all          Round-robin on KDD queries
  --nb_checkpoints <nb_checkpoints>     [default: 10]
  --known_horizon <T>   [default: -1]
  --gamma <gamma>       [default: -1]
  --version <vUTR>  Can take values 1, 2 or 3 [default: 1]
  -- ee <ee>        couples to be look at for exploration-exploitation [default: best]
  -- eeu <eeu>      undisplayed items to be look at for exploration-exploitation [default: best]
  -- pe <pe>        pure exploration strategy [default: focused]
  --gap_type <gap_type>     which statistic use to compare two arms [default: reward]
  --optimism <optimism>     which approach to use to induce optimism [default: KL]
  --R_init <R_init>     [default: None]
  --nb_shuffles <nb_shuffles>     [default: 0]
"""


# Todo: job with automatic resubmission
"""
How can a checkpointable job be resubmitted automatically?
You have to specify that your job is idempotent and exit from your script with the exit code 99. So, after a successful checkpoint, if the job is resubmitted then all will go right and there will have no problem (like file creation, deletion, ...).

Example:

oarsub --checkpoint 600 --signal 2 -t idempotent /path/to/prog
So this job will send a signal SIGINT (see man kill to know signal numbers) 10 minutes before the walltime ends. Then if everything goes well and the exit code is 99 it will be resubmitted.
"""

from Automatisation.igrida.param import Parameters, record_zip
from bandits_to_rank.environment import PositionsRanking

import os
from glob import glob
import json
import gzip
from docopt import docopt
import pickle
import time
from shutil import move

# --- Useful ---
def retrieve_data_from_zip(file_name):
    print(file_name)
    with gzip.GzipFile(file_name, 'r') as fin:
        json_bytes = fin.read()

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    return json.loads(json_str)


# --- Functions ---
def args_to_params(args):
    #### Init parameters
    params = Parameters()

    #### Init environment
    if args['--KDD']:
        params.set_env_KDD(int(args['<query>']))
    elif args['--KDD_all']:
        params.set_env_KDD_all()
    elif args['--Yandex']:
        params.set_env_Yandex(int(args['<query>']),nb_position=int(args['<nb_position>']), nb_item=int(args['<nb_item>']))
    elif args['--Yandex_all']:
        params.set_env_Yandex_all(nb_position=int(args['<nb_position>']), nb_item=int(args['<nb_item>']))
    elif args['--Yandex_equi']:
        params.set_env_Yandex_equi(query=int(args['<query>']), K=int(args['<K>']))
    elif args['--Yandex_equi_all']:
        params.set_env_Yandex_equi_all(K=int(args['<K>']))
    elif args['--test']:
        params.set_env_test()
    elif args['--std']:
        params.set_env_std()
    elif args['--std_K100']:
        params.set_env_std_K_100()
    elif args['--small']:
        params.set_env_small()
    elif args['--xsmall']:
        params.set_env_extra_small()
    elif args['--xxsmall']:
        params.set_env_xx_small()
    elif args['--big']:
        params.set_env_big()
    elif args['--small_and_close']:
        params.set_env_small_and_close()
    elif args['--delta_variation_01']:
        params.set_env_delta_variation_01()
    elif args['--Yandex_CM']:
        params.set_env_Yandex_CM(query=int(args['<query>']), nb_position=int(args['<nb_position>']), nb_item=int(args['<nb_item>']))
    elif args['--Yandex_CM_all']:
        params.set_env_Yandex_CM_all(nb_position=int(args['<nb_position>']), nb_item=int(args['<nb_item>']))
    elif args['--test_CM']:
        params.set_env_test_CM()
    elif args['--std_CM']:
        params.set_env_std_CM()
    elif args['--small_CM']:
        params.set_env_small_CM()
    elif args['--xsmall_CM']:
        params.set_env_extra_small_CM()
    elif args['--xxsmall_CM']:
        params.set_env_xx_small_CM()
    elif args['--big_CM']:
        params.set_env_big_CM()
    elif args['--small_and_close_CM']:
        params.set_env_small_and_close_CM()
    else:
        raise ValueError("unknown environment")

    #### Init environment shuffling
    if args['--order_kappa']:
        params.set_positions_ranking(PositionsRanking.DECREASING)
    elif args['--shuffle_kappa']:
        params.set_positions_ranking(PositionsRanking.SHUFFLE)
    elif args['--shuffle_kappa_except_first']:
        params.set_positions_ranking(PositionsRanking.SHUFFLE_EXCEPT_FIRST)
    elif args['--increasing_kappa']:
        params.set_positions_ranking(PositionsRanking.INCREASING)
    elif args['--increasing_kappa_except_first']:
        params.set_positions_ranking(PositionsRanking.INCREASING_EXCEPT_FIRST)
    else:  # default
        # TODO: simplify default behavior
        if args['--Yandex_CM_all'] or args['--Yandex_CM'] or args['--test_CM']:
            params.set_positions_ranking(PositionsRanking.SHUFFLE)
        else:
            params.set_positions_ranking(PositionsRanking.SHUFFLE_EXCEPT_FIRST)

    #### Init player
    if args['--eGreedy']:
        params.set_player_eGreedy(float(args['<c>']), int(args['<maj>']), args['--noSVD'])
    elif args['--PBM-TS']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_PBM_TS(type=type)
    elif args['--PBM-PIE']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_PBM_PIE(float(epsilon=args['<epsilon>']), T=int(args['<nb_trials>']),type = type)
    elif args['--PBM-UCB']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_PBM_UCB(epsilon=float(args['<epsilon>']), type=type)
    elif args['--BC-MPTS']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_BC_MPTS(type)
    elif args['--PB-MHB']:  # --PB-MHB <nb_steps> <c> [--random_start]
        if args['--TGRW']:
            params.set_proposal_TGRW(float(args['<c>']), args['--vari_sigma'])
        elif args['--LGRW']:
            params.set_proposal_LGRW(float(args['<c>']), args['--vari_sigma'])
        elif args['--RR']:
            params.set_proposal_RR(float(args['<c>']),args['<str_proposal_possible>'], args['--vari_sigma'])
        elif args['--MaxPos']:
            params.set_proposal_MaxPos()
        elif args['--PseudoView']:
            params.set_proposal_PseudoView()
        params.set_player_PB_MHB(int(args['<nb_steps>']),  args['--random_start'])
    elif args['--PB-GB']:
        params.set_player_PB_GB(N=int(args['<N>']), h_param=float(args['<h_param>']), gamma=float(args['<gamma>']), L_smooth_param=float(args['<L_smooth_param>']))
    elif args['--PMED']:  # --PMED <alpha> <gap_MLE> <gap_q>
        params.set_player_PMED(float(args['<alpha>']), int(args['<gap_MLE>']), int(args['<gap_q>']), run=args['--play'])
    elif args['--PMED_test']:
        params.set_player_PMED_test(float(args['<alpha>']), int(args['<gap_MLE>']), int(args['<gap_q>']), run=args['--play'])
    elif args['--TopRank']:  # --TopRank [--sorted] [--oracle]
        params.set_player_TopRank(float(args['<T>']), horizon_time_known=args['--horizon_time_known'], doubling_trick=args['--doubling_trick'], oracle=args['--oracle'])
    elif args['--OSUB']:
        if args['--finit_memory']:
            params.set_player_OSUB(int(args['<memory>']))
        else :
            params.set_player_OSUB()
    elif args['--OSUB_PBM']:
        params.set_player_OSUB_PBM(int(args['<T>']), gamma=int(args['<gamma>']), forced_initiation=args['--forced_initiation'])
    elif args['--GRAB']:
        params.set_player_GRAB(T=int(args['--known_horizon']), gamma=int(args['--gamma']), gap_type=args['--gap_type'], forced_initiation=args['--forced_initiation'], optimism=args['--optimism'])
    elif args['--UniRank']:
        params.set_player_UniRank(T=int(args['--known_horizon']), gamma=int(args['--gamma']))
    elif args['--UniTopRank']:
        params.set_player_UniTopRank(T=int(args['--known_horizon']), version=int(args['--version']))
    elif args['--UniGRAB']:
        params.set_player_UniGRAB(potential_explore_exploit=args['<ee>'], undisplayed_explore_exploit=args['<eeu>'], pure_explore=args['<pe>'])
    elif args['--DCGUniRank']:
        params.set_player_DCGUniRank(T=int(args['--known_horizon']), gamma=int(args['--gamma']))
    elif args['--PBubblePBRank']:
        params.set_player_PBubblePBRank(T=int(args['<T>']), gamma=int(args['<gamma>']))
    elif args['--BubbleOSUB']:
        params.set_player_bubbleOSUB(int(args['<T>']))
    elif args['--BubbleRank']:  # --BubbleRank <delta> [--sorted] [--oracle]
        params.set_player_BubbleRank(float(args['<delta>']), oracle=args['--oracle'])
    elif args['--BubbleRank_OSUB2']:
        params.set_player_bubblerank_OSUB2(R_init=list(args['--R_init']), nb_shuffles=int(args['--nb_shuffles']))
    elif args['--MLMR']:
        params.set_player_MLMR()
    elif args['--KL-MLMR']:
        params.set_player_KL_MLMR(horizon=int(args['<T>']))
    elif args['--CascadeKL-UCB']:
        params.set_player_CascadeKL_UCB()
    else:
        raise ValueError("unknown player")

    #### Set rules
    params.set_rules(int(args['<nb_trials>']), nb_records=int(args['-r']), nb_relevant_positions=int(args['-l']))

    #### Set experiment
    if args['<nb_game>'] is None:
        args['<nb_game>'] = -1
    params.set_exp(first_game=int(args['-s']), nb_games=int(args['<nb_game>']), nb_checkpoints=int(args['--nb_checkpoints']),
                   input_path=args['<input_path>'], output_path=args['<output_path>'],
                   force=args['--force']
                   )

    return params


def line_args_to_params(line_args):
    try:
        args = docopt(__doc__, argv=line_args.split())
        return args_to_params(args)
    except:
        print(f'Error bad arguments: {line_args}\n')
        print(f'Split as: {line_args.split()}\nWhile expecting:')
        raise


def play(params, dry_run=False, verbose=True):
    """
    # Parameters
        params
        dry_run : bool
            if True, do not play games, only count the number of games to be played

    # Returns
        nb_played_games : int
    """
    nb_trials_between_check_points = params.referee.nb_trials // params.nb_checkpoints
    nb_played_games = 0
    for id_g in range(params.first_game, params.end_game):
        if verbose:
            print('#### GAME '+str(id_g))
        base_output_file_name = f'{params.output_path}/{params.env_name}__{params.player_name}__{params.rules_name}_{id_g}_game_id'
        output_file_name = f'{base_output_file_name}.gz'
        if os.path.exists(output_file_name) and not params.force:
            if verbose:
                print('File', output_file_name, 'already exists. Keep it.')
        else:
            nb_played_games += 1
            if verbose and os.path.exists(output_file_name):
                print('File', output_file_name, 'already exists. Replace with a new one.')
            if not dry_run:
                # --- play one game, in multiple chunks ---

                loaded = False
                if not params.force and (os.path.isfile(f'{base_output_file_name}.ckpt.pickle.gz') or os.path.isfile(f'{base_output_file_name}.ckpt.pickle.old')):
                    # load save game state ...
                    if verbose:
                        start_time = time.time()
                        print('--- loading game state ---')
                    if os.path.isfile(f'{base_output_file_name}.ckpt.pickle.gz'):
                        try:
                            saved_params = pickle.load(gzip.GzipFile(f'{base_output_file_name}.ckpt.pickle.gz', 'rb'))
                            loaded = True
                        except EOFError as e:
                            print(f'!!! {type(e)}:{e}')
                            print('    => load skipped')
                    if not loaded:
                        try:
                            saved_params = pickle.load(gzip.GzipFile(f'{base_output_file_name}.ckpt.pickle.old', 'rb'))
                            loaded = True
                        except EOFError as e:
                            print(f'!!! {type(e)}:{e}')
                            print('    => load skipped')
                    if loaded:
                        if verbose:
                            print(f'--- done in {time.time()-start_time} sec ---')
                        params.env = saved_params.env
                        params.player = saved_params.player
                        params.referee = saved_params.referee
                if not loaded:
                    # ... or init and play first trial
                    params.env.shuffle(positions_ranking=params.positions_ranking)
                    params.player.clean()
                    params.referee.clean_recorded_results()
                    params.referee.prepare_new_game()

                # play the remaining game
                while params.referee.running_t < params.referee.nb_trials-1:
                    params.referee.play_game(params.player, new_game=False, nb_trials_before_break=nb_trials_between_check_points, nb_relevant_positions=params.nb_relevant_positions)

                    # save game state
                    if params.referee.running_t < params.referee.nb_trials-1:
                        if verbose:
                            start_time = time.time()
                            print('--- saving game state ---')
                        pickle.dump(params, gzip.GzipFile(f'{base_output_file_name}.ckpt.pickle.tmp', 'wb'))
                        if verbose:
                            print(f'--- done in {time.time() - start_time} sec ---')
                        if os.path.exists(f'{base_output_file_name}.ckpt.pickle.gz'):
                            #os.remove(f'{base_output_file_name}.ckpt.pickle.old')
                            move(f'{base_output_file_name}.ckpt.pickle.gz', f'{base_output_file_name}.ckpt.pickle.old')
                            #os.rename(f'{base_output_file_name}.ckpt.pickle.gz', f'{base_output_file_name}.ckpt.pickle.old')
                        #os.remove(f'{base_output_file_name}.ckpt.pickle.gz')
                        move(f'{base_output_file_name}.ckpt.pickle.tmp', f'{base_output_file_name}.ckpt.pickle.gz')
                        #os.rename(f'{base_output_file_name}.ckpt.pickle.tmp', f'{base_output_file_name}.ckpt.pickle.gz')

                # save the results
                record_zip(output_file_name, params.referee.record_results)
                for ext in ['tmp', 'old', 'gz']:
                    file = f'{base_output_file_name}.ckpt.pickle.{ext}'
                    if os.path.isfile(file):
                        os.remove(file)

    return nb_played_games


def merge_records(params, verbose=False):
    # Merge
    params.referee.clean_recorded_results()
    nb_games = 0
    logs_env_name = params.logs_env_name
    print(f'{params.input_path}/{logs_env_name}__{params.player_name}__{params.rules_name}_*_game_id')
    for file_name in glob(f'{params.input_path}/{logs_env_name}__{params.player_name}__{params.rules_name}_*_game_id.gz'):

        if verbose:
            print('#### Read ' + os.path.basename(file_name))
        try:
            record = retrieve_data_from_zip(file_name)
            params.referee.add_record(record)
            nb_games += 1
            #print (file_name)
            print(nb_games)
        except (EOFError, json.decoder.JSONDecodeError) as e:
            print(f'!!! {type(e)}:{e}')
            print('    => file renamed "bad_file__XXX"')
            move(file_name, f'{params.input_path}/bad_file__{os.path.basename(file_name)}')

    # Save results
    if nb_games != 0:
        record_zip(f'{params.output_path}/{params.env_name}__{params.player_name}__{params.rules_name}_{nb_games}_games.gz', params.referee.record_results)


if __name__ == "__main__":
    import sys
    print(sys.argv)
    arguments = docopt(__doc__)
    print(arguments)
    params = args_to_params(arguments)

    if arguments['--play']:
        play(params)
    elif arguments['--merge']:
        merge_records(params)
