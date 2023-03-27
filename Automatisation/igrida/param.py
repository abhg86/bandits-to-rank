## Requirements

### Packages
import os
import json
import gzip
import numpy as np

from bandits_to_rank.environment import Environment_PBM, Environment_Cascade, PositionsRanking
from bandits_to_rank.opponents import greedy
from bandits_to_rank.opponents.pbm_pie import PBM_PIE_Greedy_SVD, PBM_PIE_semi_oracle, PBM_PIE_Greedy_MLE
from bandits_to_rank.opponents.pbm_ucb import PBM_UCB_Greedy_SVD, PBM_UCB_semi_oracle, PBM_UCB_Greedy_MLE
from bandits_to_rank.opponents.pbm_ts import PBM_TS_Greedy_SVD, PBM_TS_semi_oracle, PBM_TS_Greedy_MLE
from bandits_to_rank.opponents.bc_mpts import BC_MPTS_Greedy_SVD, BC_MPTS_semi_oracle,BC_MPTS_Greedy_MLE
#from bandits_to_rank.opponents.pmed import PMED   # loaded only before usage to load tensorflow library only when required
from bandits_to_rank.opponents.top_rank import TOP_RANK
from bandits_to_rank.opponents.bubblerank import BUBBLERANK
from bandits_to_rank.opponents.bubblerank_OSUB2 import BUBBLERANK_OSUB2
from bandits_to_rank.opponents.OSRUB_bis import OSUB
from bandits_to_rank.opponents.OSRUB_PBM_bis import OSUB_PBM
from bandits_to_rank.opponents.grab import GRAB
from bandits_to_rank.opponents.PseudoBubblePBRank import PBB_PBRank
from bandits_to_rank.opponents.uni_rank import DCGUniRank, UniRankMaxGap, UniRankWithMemory, OSUB_TOP_RANK, UniGRAB
from bandits_to_rank.opponents.BubbleOSRUB_BAL import BUBBLEOSUB
from bandits_to_rank.opponents.cascadekl_ucb import CASCADEKL_UCB
from bandits_to_rank.opponents.mlmr import MLMR, KL_MLMR
from bandits_to_rank.opponents.PB_GB import PB_GB
from bandits_to_rank.bandits import *
from bandits_to_rank.referee import Referee
from bandits_to_rank.tools.tools import get_SCRATCHDIR


# set.seed(123)


# Path to bandits-to-rank module
import bandits_to_rank

packagedir = os.path.dirname(bandits_to_rank.__path__[0])


class NdArrayEncoder(json.JSONEncoder):
     def default(self, obj):
        if isinstance(obj, np.ndarray):
             return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
             return int(obj)
         # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def record_zip(filename, dico):
    print(type(dico))
    print('file', filename)
    json_str = json.dumps(dico, cls=NdArrayEncoder)
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile(filename, 'w') as fout:
        fout.write(json_bytes)
    return 'done'

class Parameters():
    """ Parameters used for the experiment

    # Environement
        env
        env_name        str used for name of files
        logs_env_name   (only for merge)

    # Player
        player
        player_name

    # Rules
        rules_name
        referee

    # Sub-experiment
        first_game      (only for play)
        end_game        (only for play)
        input_path      (only for merge)
        output_path     ! WARNING ! has to be relative wrt. $SCRATCHDIR or absolute
        force           (only for play)
    """

    def __init__(self):
        self.env = Environment_PBM([1], [1], label="fake")
        self.positions_ranking = PositionsRanking.SHUFFLE_EXCEPT_FIRST  # default: shuffle kappas before each game
        self.nb_relevant_positions = None   # default: compute reward at each position
        self.rng = np.random.default_rng()

    #########" PBM_Setting
    def set_positions_ranking(self, positions_ranking):
        self.positions_ranking = positions_ranking

        # tag for file names and logs
        if positions_ranking == PositionsRanking.FIXED:
            raise ValueError('fixed ranking of positions should be set by the player')
        elif positions_ranking == PositionsRanking.DECREASING:
            # TODO: better naming for PBM '__decreasing_kappa'
            # TODO: better naming for CM '__std_order_on_views'
            tag = '__sorted_kappa' if type(self.env) == Environment_PBM else ''
        elif positions_ranking == PositionsRanking.SHUFFLE:
            # TODO: better naming for CM '__random_order_on_views'
            tag = '__shuffled_kappa' if type(self.env) == Environment_PBM else '_order_view_shuffle'
        elif positions_ranking == PositionsRanking.SHUFFLE_EXCEPT_FIRST:
            # TODO: better naming for PBM __shuffled_kappa_except_first
            tag = '' if type(self.env) == Environment_PBM else '__random_order_on_views_except_first'
        elif positions_ranking == PositionsRanking.INCREASING:
            tag = ( '__increasing_kappa' if type(self.env) == Environment_PBM else '__reverse_order_on_views')
        elif positions_ranking == PositionsRanking.INCREASING_EXCEPT_FIRST:
            tag = ( '__increasing_kappa_except_first' if type(self.env) == Environment_PBM
                    else '__reverse_order_on_views_except_first')
        else:
            raise ValueError(f'unhandled ranking on positions: {positions_ranking}')
        self.env_name += tag
        self.logs_env_name += tag
        self.env.label += tag

    def set_env_KDD_all(self):
        """!!! only to merge logs from several queries of KDD data !!!"""
        self.env_name = f'KDD_all'
        self.logs_env_name = f'KDD_[0-9]*_query'

    def set_env_KDD(self, query):
        # load KDD data
        # todo: to put in bandits_to_rank.data
        with open(packagedir + '/Test/KDD/param_KDD.txt', 'r') as file:
            dict_theta_query = json.load(file)
        query_name, query_params = list(dict_theta_query.items())[query]

        # set environement
        self.env_name = f'KDD_{query}_query'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(query_params['thetas'], query_params['kappas']
                                   , label='%s (%d for us)' % (query_name, query))

    def set_env_Yandex_all(self,nb_position, nb_item):
        """!!! only to merge logs from several queries of Yandex data !!!"""
        self.env_name = f'Yandex_all_{nb_item}_items_{nb_position}_positions'
        self.logs_env_name = f'Yandex_[0-9]*_query_{nb_item}_items_{nb_position}_positions'

    def set_env_Yandex(self, query, nb_position, nb_item):
        # load Yandex data
        # todo: to put in bandits_to_rank.data
        with open(packagedir + '/Test/KDD/param_Yandex.txt', 'r') as file:
            dict_theta_query = json.load(file)
        query_name, query_params = list(dict_theta_query.items())[query]

        # variable number of products and positions
        thetas = np.sort(query_params['thetas'])[:-(nb_item+1):-1]
        kappas = np.sort(query_params['kappas'])[:-(nb_position+1):-1]

        # set environement
        self.env_name = f'Yandex_{query}_query_{nb_item}_items_{nb_position}_positions'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas
                                , label='%s (%d for us)' % (query_name, query))

    def set_env_Yandex_equi_all(self, K):
        """!!! only to merge logs from several queries of Yandex data !!!"""
        self.env_name = f'Yandex_equi_{K}_K_all'
        self.logs_env_name = f'Yandex_equi_{K}_K__[0-9]*_query'

    def set_env_Yandex_equi(self, query, K):
        # load Yandex data
        with open(packagedir + '/Test/KDD/param_Yandex.txt', 'r') as file:
            dict_theta_query = json.load(file)
        query_name, query_params = list(dict_theta_query.items())[query]

        # reduce to 10 products, 10 positions
        index_max = 1+K
        thetas = np.sort(query_params['thetas'])[:-index_max:-1]
        kappas = np.sort(query_params['kappas'])[:-index_max:-1]

        # set environement
        self.env_name = f'Yandex_equi_{K}_K__{query}_query'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label=f'Yandex equi K={K} {query} ({query_name} for Yandex)')

    def set_env_test(self):
        """Purely simulated environment with standard click's probabilities"""
        kappas = [1, 0.6, 0.3]
        thetas = [0.1, 0.5, 0.1, 0.6, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1]
        self.env_name = f'purely_simulated__test'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, test")

    def set_env_std(self):
        """Purely simulated environment with standard click's probabilities"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.3, 0.2, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.01, 0.01]
        self.env_name = f'purely_simulated__std'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, std")

    def set_env_std_K_100(self):
        """Purely simulated environment with standard click's probabilities"""
        kappas = [1, 0.9, 0.8, 0.7, 0.6]
        thetas = [0.3, 0.2, 0.15, 0.15, 0.15]+[0.10]*30+[0.05]*30+[0.01]*35
        self.env_name = f'purely_simulated__std_K_100'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, std with K=100")

    def set_env_small(self):
        """Purely simulated environment with click's probabilities close to 0"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.15, 0.1, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.env_name = f'purely_simulated__small'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, small")

    def set_env_big(self):
        """Purely simulated environment with click's probabilities close to 1"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75]
        self.env_name = f'purely_simulated__big'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, big")

    def set_env_extra_small(self):
        """Purely simulated environment with click's probabilities close to 0"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
        self.env_name = f'purely_simulated__xsmall'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, extra small")

    def set_env_xx_small(self):
        """Purely simulated environment with click's probabilities close to 0"""
        kappas = [1, 0.75, 0.6, 0.3, 0.1]
        thetas = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
        self.env_name = f'purely_simulated__xxsmall'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, xx small")

    def set_env_small_and_close(self):
        """Purely simulated environment with click's probabilities close to 0.1"""
        kappas = [1, 0.8, 0.6, 0.4, 0.2]
        thetas = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05]    # V1
        thetas = [0.1, 0.095, 0.09, 0.085, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]  # V2
        thetas = [0.1, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]    # V3
        thetas = [0.1, 0.098, 0.096, 0.094, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092]  # V4
        thetas = [0.1, 0.098, 0.096, 0.094, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]     # V5
        kappas = [1, 0.9, 0.83, 0.78, 0.75]                                         # V6
        thetas = [0.1, 0.08, 0.06, 0.04, 0.02, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]   # V6
        self.env_name = f'purely_simulated__small_and_close'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, small and close")

    def set_env_delta_variation_01(self):
        """Purely simulated environment to test the difference for a variation of delta = 0.1"""
        kappas = [1, 0.9, 0.8, 0.7, 0.6]
        thetas = [0.5, 0.5-0.1, 0.5-0.2, 0.5-0.3, 0.5-0.4]
        self.env_name = f'purely_simulated__delta_variation_01'
        self.logs_env_name = self.env_name
        self.env = Environment_PBM(thetas, kappas, label="purely simulated, delta variation delta = 0.1")

    #########CM Setting
    def set_env_Yandex_CM_all(self,nb_position, nb_item):
        """!!! only to merge logs from several queries of Yandex data !!!"""
        self.env_name = f'Yandex_CM_all_{nb_item}_items_{nb_position}_positions'
        self.logs_env_name = f'Yandex_CM_[0-9]*_query_{nb_item}_items_{nb_position}_positions'
        self.env = Environment_Cascade(thetas=np.arange(nb_item), order_view=np.arange(nb_position), label='fake')

    def set_env_Yandex_CM(self, query, nb_position, nb_item):
        # load Yandex data
        # todo: to put in bandits_to_rank.data
        with open(packagedir + '/Test/KDD/param_Yandex_CM.txt', 'r') as file:
            dict_theta_query = json.load(file)

        query_name, query_params = list(dict_theta_query.items())[query]

        # TODO: back to 10 items
        # reduce to 10 items and number of positions as argument
        thetas = np.sort(query_params)[:-(nb_item+1):-1]
        order_view = np.arange(nb_position)

        # set environement
        self.env_name = f'Yandex_CM_{query}_query_{nb_item}_items_{nb_position}_positions'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'Yandex {query} ({query_name} for Yandex)')


    def set_env_test_CM(self):
        """Purely simulated environment with standard click's probabilities"""
        thetas = [0.1, 0.5, 0.1, 0.6, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1]
        nb_position = 3
        order_view = np.arange(nb_position)

        self.env_name = f'purely_simulated__test_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, test Cascading')


    def set_env_std_CM(self):
        """Purely simulated environment with standard click's probabilities"""
        nb_position = 5
        thetas = [0.3, 0.2, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.01, 0.01]
        order_view = np.arange(nb_position)

        self.env_name = f'purely_simulated__std_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, std Cascading')



    def set_env_small_CM(self):
        """Purely simulated environment with click's probabilities close to 0"""
        nb_position = 5
        thetas = [0.15, 0.1, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
        order_view = np.arange(nb_position)
        
        self.env_name = f'purely_simulated__small_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, small Cascading')


    def set_env_big_CM(self):
        """Purely simulated environment with click's probabilities close to 1"""
        nb_position = 5
        thetas = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75]
        order_view = np.arange(nb_position)

        self.env_name = f'purely_simulated__big_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, big Cascading')


    def set_env_extra_small_CM(self):
        """Purely simulated environment with click's probabilities close to 0"""
        nb_position = 5
        thetas = [0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
        order_view = np.arange(nb_position)

        self.env_name = f'purely_simulated__xsmall_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, extra small Cascading')


    def set_env_xx_small_CM(self):
        """Purely simulated environment with click's probabilities close to 0"""
        nb_position = 5
        thetas = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
        order_view = np.arange(nb_position)

        self.env_name = f'purely_simulated__xxsmall_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, xx small Cascading')

    def set_env_small_and_close_CM(self):
        """Purely simulated environment with click's probabilities close to 0.1"""
        nb_position = 5
        thetas = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05]    # V1
        thetas = [0.1, 0.095, 0.09, 0.085, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]  # V2
        thetas = [0.1, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]    # V3
        thetas = [0.1, 0.098, 0.096, 0.094, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092]  # V4
        thetas = [0.1, 0.098, 0.096, 0.094, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]     # V5
        thetas = [0.1, 0.08, 0.06, 0.04, 0.02, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]   # V6
        order_view = np.arange(nb_position)

        self.env_name = f'purely_simulated__small_and_close_CM'
        self.logs_env_name = self.env_name
        self.env = Environment_Cascade(thetas=thetas, order_view=order_view, label=f'purely simulated, small and close Cascading')

    def set_env_simul(self, label):
        if label == "std":
            self.set_env_std()
        elif label == "big":
            self.set_env_big()
        elif label == "small":
            self.set_env_small()
        elif label == "xsmall":
            self.set_env_extra_small()
        elif label == "xxsmall":
            self.set_env_xx_small()
        elif label == "delta_variation_01":
            self.set_env_delta_variation_01()
        else:
            raise ValueError("unknown label of environment")

    def set_rules(self, nb_trials, nb_records=1000, nb_relevant_positions=0):
        # Check inputs
        if nb_records > nb_trials:
            nb_records = -1
        if nb_relevant_positions == 0:
            self.nb_relevant_positions = None
        else:
            self.nb_relevant_positions = nb_relevant_positions

        self.rules_name = f'games_{nb_trials}_nb_trials_{nb_records}_record_length_{self.nb_relevant_positions}_rel_pos'
        #self.rules_name = f'games_{nb_trials}_nb_trials_{nb_records}_record_length'
        self.referee = Referee(self.env, nb_trials, all_time_record=False, len_record_short=nb_records)

    def set_player_eGreedy(self, c, update=100, noSVD=False):
        nb_prop, nb_place = self.env.get_setting()
        if noSVD:
            self.player_name = f'Bandit_EGreedy_EM_{c}_c_{update}_update'
            self.player = greedy.greedy_EGreedy_EM(c, nb_prop, nb_place, update)
        else:
            self.player_name = f'Bandit_EGreedy_SVD_{c}_c_{update}_update'
            self.player = greedy.greedy_EGreedy(c, nb_prop, nb_place, update)

    def set_player_PBM_TS(self, type="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = 'Bandit_PBM-TS_oracle'
            self.player = PBM_TS_semi_oracle(nb_prop, nb_place, discount_factor=self.env.kappas, count_update=1)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_PBM_TS_greedy_MLE'
            self.player = PBM_TS_Greedy_MLE(nb_prop, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_PBM_TS_greedy_SVD'
            self.player = PBM_TS_Greedy_SVD(nb_prop, nb_place, count_update=1)
        else:
            self.player_name = 'Bandit_PBM-TS_greedy_SVD'
            self.player = PBM_TS_Greedy_SVD(nb_prop, nb_place, count_update=1)

    def set_player_PBM_PIE(self, epsilon, T, type ="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = f'Bandit_PBM-PIE_oracle_{epsilon}_epsilon'
            self.player = PBM_PIE_semi_oracle(nb_prop, epsilon, T, nb_place, discount_factor=self.env.kappas, count_update=1)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_PBM_PIE_greedy_MLE'
            self.player = PBM_PIE_Greedy_MLE(nb_prop, epsilon, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_PBM_PIE_greedy_SVD'
            self.player = PBM_PIE_Greedy_SVD(nb_prop, epsilon, nb_place, count_update=1)
        else:
            self.player_name = f'Bandit_PBM-PIE_greedy_SVD_{epsilon}_epsilon'
            self.player = PBM_PIE_Greedy_SVD(nb_prop, epsilon, T, nb_place, count_update=1)

    def set_player_PBM_UCB(self, epsilon, type ="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = f'Bandit_PBM_UCB_oracle_{epsilon}_epsilon'
            self.player = PBM_UCB_semi_oracle(nb_prop, epsilon, nb_place, discount_factor=self.env.kappas, count_update=1)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_PBM_UCB_greedy_MLE'
            self.player = PBM_UCB_Greedy_MLE(nb_prop, epsilon, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_PBM_UCB_greedy_SVD'
            self.player = PBM_UCB_Greedy_SVD(nb_prop, epsilon, nb_place, count_update=1)
        else:
            self.player_name = f'Bandit_PBM_UCB_greedy_SVD_{epsilon}_epsilon'
            self.player = PBM_UCB_Greedy_SVD(nb_prop, epsilon, nb_place, count_update=1)


    def set_player_BC_MPTS(self, type ="oracle"):
        nb_prop, nb_place = self.env.get_setting()
        if type =="oracle":
            self.player_name = 'Bandit_BC-MPTS_oracle'
            self.player = BC_MPTS_semi_oracle(nb_prop, nb_place, self.env.kappas)
            self.positions_ranking = PositionsRanking.FIXED
        elif type =="greedyMLE":
            self.player_name = 'Bandit_BC-MPTS_greedy_MLE'
            self.player = BC_MPTS_Greedy_MLE(nb_prop, nb_place, count_update=1)
        elif type =="greedySVD":
            self.player_name = 'Bandit_BC-MPTS_greedy_SVD'
            self.player = BC_MPTS_Greedy_SVD(nb_prop, nb_place, count_update=1)
        else:
            self.player_name = 'Bandit_BC-MPTS_greedy_SVD'
            self.player = BC_MPTS_Greedy_SVD(nb_prop, nb_place, count_update=1)

    def set_player_PMED(self, alpha, gap_MLE, gap_q, run=True):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_PMED_{alpha}_alpha_{gap_MLE}_gap_MLE_{gap_q}_gap_q'

        if run:
            from bandits_to_rank.opponents.pmed import PMED
            self.player = PMED(nb_prop, nb_place, alpha, gap_MLE, gap_q)

    def set_player_PMED_test(self, alpha, gap_MLE, gap_q, run=True):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_PMED_less_tracing_{alpha}_alpha_{gap_MLE}_gap_MLE_{gap_q}_gap_q'

        if run:
            from bandits_to_rank.opponents.pmed_test import PMED
            self.player = PMED(nb_prop, nb_place, alpha, gap_MLE, gap_q)

    def set_player_MLMR(self, exploration_factor=2.):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_MLMR_{exploration_factor}_exploration'
        self.player = MLMR(nb_arms=nb_prop, nb_positions=nb_place, exploration_factor=exploration_factor)

    def set_player_KL_MLMR(self, horizon):
        nb_prop, nb_place = self.env.get_setting()

        self.player_name = f'Bandit_KL-MLMR_{horizon}_horizon'
        self.player = KL_MLMR(nb_arms=nb_prop, nb_positions=nb_place, horizon=horizon)

    def set_player_PB_MHB(self, nb_steps, random_start=False):
        nb_prop, nb_place = self.env.get_setting()
        if random_start:
            self.player_name = f'Bandit_PB-MHB_random_start_{nb_steps}_step_{self.proposal_name}_proposal'
            self.player = TS_MH_kappa_desordonne(nb_prop, nb_place, proposal_method=self.proposal, step=nb_steps, part_followed=False)
        else:
            self.player_name = f'Bandit_PB-MHB_warm-up_start_{nb_steps}_step_{self.proposal_name}_proposal'
            self.player = TS_MH_kappa_desordonne(nb_prop, nb_place, proposal_method=self.proposal,  step=nb_steps, part_followed=True)

    def set_player_TopRank(self, T,horizon_time_known=True,doubling_trick=False, oracle=False):
        nb_prop, nb_place = self.env.get_setting()
        if oracle:
            self.player_name = f'Bandit_TopRank_oracle_{T}_delta_{"TimeHorizonKnown" if horizon_time_known else ""}_{"doubling_trick" if doubling_trick else ""}'
            self.player = TOP_RANK(nb_arms=nb_prop,
                                   T=T, horizon_time_known=horizon_time_known,doubling_trick_active=doubling_trick,
                                   discount_factor=self.env.kappas)
            self.positions_ranking = PositionsRanking.FIXED
        else:
            self.player_name = f'Bandit_TopRank_{T}_delta_{"TimeHorizonKnown" if horizon_time_known else ""}_{"doubling_trick" if doubling_trick else ""}'
            self.player = TOP_RANK(nb_arms=nb_prop,
                                   T=T, horizon_time_known=horizon_time_known, doubling_trick_active=doubling_trick,
                                   discount_factor=np.arange(nb_place - 1, -1, -1))
        """
            self.player_name = f'Bandit_TopRank_greedy_{T}_delta_{"TimeHorizonKnown" if horizon_time_known else ""}_{"doubling_trick" if doubling_trick else ""}'
            self.player = TOP_RANK(nb_arms=nb_prop,
                                   T=T, horizon_time_known=horizon_time_known,doubling_trick_active=doubling_trick,
                                   nb_positions=nb_place, lag=1)
        """

    def set_player_CascadeKL_UCB(self):
        nb_prop, nb_place = self.env.get_setting()
        self.player_name = f'Bandit_CascadeKL_UCB'
        self.player = CASCADEKL_UCB(nb_arms=nb_prop, nb_position=nb_place)

    def set_player_OSUB(self, memory=np.inf):
        nb_prop, nb_place = self.env.get_setting()
        self.player_name = f'Bandit_OSUB_{memory}_memory'
        self.player = OSUB(nb_arms=nb_prop, nb_positions=nb_place, memory_size=memory)


    def set_player_OSUB_PBM(self, T, gamma, forced_initiation):
        nb_prop, nb_place = self.env.get_setting()
        if gamma == 0:
            gamma_use = nb_prop * nb_place
        else :
            gamma_use = gamma
        self.player_name = f'Bandit_OSUB_PBM_{T}_T_{gamma_use}_gamma{"_forced" if forced_initiation else ""}'
        self.player = OSUB_PBM(nb_arms=nb_prop, nb_positions=nb_place, T=T)

    def set_player_GRAB(self, T, gamma, forced_initiation, gap_type='reward', optimism='KL'):
        if gamma == -1:
            gamma = None
        nb_prop, nb_place = self.env.get_setting()
        self.player = GRAB(nb_arms=nb_prop, nb_positions=nb_place, T=T, gamma=gamma, forced_initiation=forced_initiation, gap_type=gap_type, optimism=optimism)
        self.player_name = f'Bandit_GRAB_{T}_T_{gamma}_gamma{"_forced" if forced_initiation else ""}_{gap_type}_gap_{optimism}_optimism'

    def set_player_UniRank(self, T, gamma, bound_l='o', bound_n='o', lead_l='o', lead_n='a', oracle=True):
        if gamma == -1:
            gamma = None
        if T == -1:
            T = None
        nb_prop, nb_place = self.env.get_setting()
        if oracle:
            try:
                sigma = np.argsort(-self.env.kappas)
            except:
                print(f'Env with unknown kappa. We assume the positions are oredered from the first to the last')
                sigma = np.arange(nb_place - 1, -1, -1)
        else:
            sigma = None
        self.player_name = f'Bandit_UniRankWithSharedMemory{"_oracle" if oracle else ""}_bl{bound_l}_bn{bound_n}_ll{lead_l}_ln{lead_n}_{T}_T_{gamma}_gamma'
        self.player = UniRankWithMemory(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma, gamma=gamma, bound_l=bound_l, bound_n=bound_n, lead_l=lead_l, lead_n=lead_n)

    def set_player_UniTopRank(self, T, oracle=True, version=0):
        if T == -1:
            T = None
        nb_prop, nb_place = self.env.get_setting()
        if oracle:
            try:
                sigma = np.argsort(-self.env.kappas)
            except:
                print(f'Env with unknown kappa. We assume the positions are oredered from the first to the last')
                #sigma = np.arange(nb_place - 1, -1, -1)
                sigma = np.arange(nb_place)
            print(sigma)
            if (sigma != np.arange(nb_place)).any():
                raise ValueError('Current implementation of OSUB_TOP_RANK requires positions to be ordered from the first to the last.')
        else:
            sigma = None
        if version == 0: # used after NeurIPS'21 (fix: correct time used by choose_next_arm)
            self.player_name = f'Bandit_UniTopRank_loglog{"_oracle" if oracle else ""}_{T}_T'
            self.player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma, global_time_for_threshold=False)
        elif version == 1: # used for NeurIPS'21 submission
            self.player_name = f'Bandit_UniTopRank_globalTime_loglog{"_oracle" if oracle else ""}_{T}_T'
            self.player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma, global_time_for_threshold=True)
        elif version == 2:
            self.player_name = f'Bandit_UniTopRank_sqrtlog{"_oracle" if oracle else ""}_{T}_T'
            self.player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma, slight_optimism='sqrt log')
        elif version == 3:
            self.player_name = f'Bandit_UniTopRank_loglog_finePartition{"_oracle" if oracle else ""}_{T}_T'
            self.player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma, fine_grained_partition=True)
        elif version == 4:
            self.player_name = f'Bandit_UniTopRank_best_explo{"_oracle" if oracle else ""}_{T}_T'
            self.player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma,
                                        recommended_partition_choice='best merge or best remaining item',
                                        slight_optimism='tau hat',
                                        global_time_for_threshold=False)
        elif version == 5:
            self.player_name = f'Bandit_UniTopRank_best_explo_loglog{"_oracle" if oracle else ""}_{T}_T'
            self.player = OSUB_TOP_RANK(nb_arms=nb_prop, nb_positions=nb_place, T=T, sigma=sigma,
                                        recommended_partition_choice='best merge or best remaining item',
                                        slight_optimism='log log',
                                        global_time_for_threshold=False)
        else:
            raise ValueError("unknown version of UniTopRank")

    def set_player_UniGRAB(self, potential_explore_exploit, undisplayed_explore_exploit, pure_explore):
        nb_prop, nb_place = self.env.get_setting()
        print(potential_explore_exploit)
        PEE = {'best': 'best',
               'first': 'first',
               'even-odd': 'EO',
               'as_much_as_possible_from_top_to_bottom': 'T2B'
              }[potential_explore_exploit]

        self.player_name = f'Bandit_UniGRAB_{PEE}_{undisplayed_explore_exploit}_{pure_explore}'
        self.player = UniGRAB(nb_arms=nb_prop, nb_positions=nb_place,
                              potential_explore_exploit=potential_explore_exploit,
                              undisplayed_explore_exploit=undisplayed_explore_exploit,
                              pure_explore=pure_explore
                             )

    def set_player_DCGUniRank(self, T, gamma):
        if gamma == -1:
            gamma = None
        if T == -1:
            T = None
        nb_prop, nb_place = self.env.get_setting()
        self.player_name = f'Bandit_DCGUniRank_{T}_T_{gamma}_gamma'
        self.player = DCGUniRank(nb_arms=nb_prop, nb_positions=nb_place, T=T, gamma=gamma)

    def set_player_PBubblePBRank(self, T, gamma):
        nb_prop, nb_place = self.env.get_setting()
        if gamma == 0:
            gamma_use = nb_prop + nb_place
        else:
            gamma_use = gamma
        self.player_name = f'Bandit_PBubblePBRank_{T}_T_{gamma_use}_gamma'
        self.player = PBB_PBRank(nb_arms=nb_prop, nb_positions=nb_place, T=T, gamma=gamma_use)

    def set_player_bubbleOSUB(self, T):
        nb_prop, nb_place = self.env.get_setting()
        self.player_name = f'Bandit_BubbleOSUB_{T}_T'
        self.player = BUBBLEOSUB(nb_prop, T, nb_place)
    
    def set_player_bubblerank_OSUB2(self, R_init=None, nb_shuffles=0):
        nb_arms, _ = self.env.get_setting()
        self.player_name = f'Bandit_BubbleRank_OSUB2'
        if R_init == ['N', 'o', 'n', 'e']:
            R_init = None
        self.player = BUBBLERANK_OSUB2(nb_arms, discount_factor=np.linspace(1,0,nb_arms), nb_positions=nb_arms, R_init=R_init, nb_shuffles=nb_shuffles)


    def set_player_BubbleRank(self, delta, oracle=True):
        nb_prop, nb_place = self.env.get_setting()
        # requires environment with nb_prop == nb_place
        self.env_name += '__extended_kappas'
        self.logs_env_name += '__extended_kappas'
        if nb_prop > nb_place:
            # TODO: to debug, self.nb_relevant_positions should be set after the algorithm
            if self.nb_relevant_positions is None:
                self.nb_relevant_positions = nb_place
            kappas = np.ones(nb_prop) * min(self.env.kappas)/ 10
            kappas[:nb_place] = self.env.kappas
            self.env = Environment_PBM(self.env.thetas, kappas, label=self.env.label+'__extended_kappas')
            nb_prop, nb_place = self.env.get_setting()

        if oracle:
            self.player_name = f'Bandit_BubbleRank_{delta}_delta_oracle'
            self.player = BUBBLERANK(nb_prop, delta=delta, discount_factor=self.env.kappas, lag=1)
            self.positions_ranking = PositionsRanking.FIXED
        else:
            self.player_name = f'Bandit_BubbleRank_{delta}_delta_greedy'
            self.player = BUBBLERANK(nb_prop, delta=delta, nb_positions=nb_place, lag=1)

    def set_player_PB_GB(self, N, h_param, gamma, L_smooth_param):
        nb_prop, nb_place = self.env.get_setting()
        self.player_name = f'Bandit_PB_GB_{N}_N_{h_param}_h_param_{gamma}_gamma_{L_smooth_param}_L_smooth_param'
        self.player = PB_GB(nb_arms=nb_prop, nb_position=nb_place, N=N, h_param=h_param, gamma=gamma, L_smooth_param=L_smooth_param)


    def set_proposal_TGRW (self, c, vari_sigma=True):
        self.proposal_name = f'TGRW_{c}_c{"_vari_sigma" if vari_sigma else ""}'
        self.proposal = propos_trunk_GRW(c,vari_sigma)

    def set_proposal_LGRW (self, c, vari_sigma=True):
        self.proposal_name = f'LGRW_{c}_c{"_vari_sigma" if vari_sigma else ""}'
        self.proposal = propos_logit_RW(c,vari_sigma)

    def set_proposal_RR (self, c, str_proposal_possible,vari_sigma=True):
        list_proposal_possible = list(str_proposal_possible.split("-"))
        self.proposal_name = f'RR_{c}_c_{len(list_proposal_possible)}_proposals'
        self.proposal = propos_Round_Robin(c,vari_sigma,list_proposal_possible)

    def set_proposal_MaxPos (self):
        self.proposal_name = f'MaxPos'
        self.proposal = propos_max_position()

    def set_proposal_PseudoView (self):
        self.proposal_name = f'PseudoView'
        self.proposal = propos_pseudo_view()




    def set_exp(self, first_game=-1, nb_games=-1, nb_checkpoints=10, input_path=None, output_path=None, force=True):
        self.first_game = first_game
        self.end_game = first_game + nb_games
        self.nb_checkpoints=nb_checkpoints
        self.input_path = input_path if input_path is not None else output_path
        self.output_path = output_path if output_path is not None else input_path
        if not os.path.isabs(self.input_path):
            self.input_path = os.path.join(get_SCRATCHDIR(), self.input_path)
        if not os.path.isabs(self.output_path):
            self.output_path = os.path.join(get_SCRATCHDIR(), self.output_path)
        self.force = force


