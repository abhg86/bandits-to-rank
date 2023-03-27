# -*- coding: utf-8 -*-
# #### Referee

# ## Package

from bandits_to_rank.environment import *
from bandits_to_rank.tools.tools import build_scale, time_sec_to_DHMS

import time
import numpy as np
from math import log, exp, sqrt

from copy import deepcopy



class Referee_multiRequest:
    """
    Monitors the interaction between a bandit and the environement
    """

    def __init__(self, environment, nb_trials, all_time_record=True, len_record_short=1000, print_trial=100):
        """

        Parameters
        ----------
        environment
        nb_trials
        all_time_record
        len_record_short
        print_trial
        nb_relevant_positions : int (default: None)
            Cumulative reward and regret are computed only at positions [0, nb_relevant_positions-1].
            If set to `None`, each position is relevant.
        """
        self.env = environment
        self.nb_trials = nb_trials
        if all_time_record or len_record_short == -1:
            self.time_record_expected = [t for t in range(self.nb_trials)]
        else:
            self.time_record_expected = build_scale(nb_trials, len_record_short)

        self.print_trial=print_trial
        self.record_results = self._empty_record_results()
        
        # Reflexion a avoir sur la gestion des matrices de clic dans le cas de plusieur requete.. tensor... mais comment on le regarde ?....

    @staticmethod
    def _empty_record_results():
        return {'query_asked': [],
                'env_parameters': [],
                'expected_best_reward': [],
                'reward': [],
                'expected_reward': [],
                'reject_time': [],
                'time_recorded': [],
                'time_to_play': [],
                'dico_placed':[],
                'dico_clicked':[]
               }

    def clean_recorded_results(self):
        self.record_results = self._empty_record_results()

    def prepare_new_game(self, query=None):
        """Init game state
        """
        # --- logs ---
        self.running_cumulative_reward = 0
        self.running_cumulative_expected_reward = 0
        self.running_cumulative_expected_best_reward = 0
        self.running_cumulative_reject_time = 0

        self.running_time_recorded = []
        self.running_queries= []
        self.running_rewards_recorded = []
        self.running_expected_reward_recorded = []
        self.running_expected_best_reward = []
        self.running_expected_best_reward_recorded = []
        self.running_reject_time_recorded = []
        self.running_time_to_play_recorded = []
        
        self.running_dico_placed, self.running_dico_clicked = self.dico_interactions()

        # --- iteration / time counter ---
        self.running_t = -1
        self.total_time = 0     # total time spent for that game since beginning

    def dico_interactions(self):
        dico_placed = {}
        dico_clicked = {}
        if isinstance(self.env, Environment_multirequest_PBM):
            queries = self.env._query_list()    
            self.nb_position = self.env._kappas()
            self.nb_items = {}
            for query in queries:
                self.nb_item[query] = self.env._thetas_query(query)
                self.dico_placed[query] = np.zeros([self.nb_item[query], self.nb_position])
                self.dico_clicked[query] = np.zeros([self.nb_item[query], self.nb_position])
    
        return dico_placed, dico_clicked 
                
    def play_game(self, player, new_game=True, nb_trials_before_break=None, nb_relevant_positions=None):
        """Play one game vs the environment corresponding to the chosen query
        The query index is given to be stored in the record_result dedicated field

        Parameters
        ----------
        player
        query
        new_game : bool
            if `False`, continue previous game
        nb_trials_before_break
        start_time

        Returns
        -------

        """
        # --- Variables initialization ---
        if new_game:
            self.prepare_new_game()
        

        # --- trials to be played ---
        start = self.running_t+1
        if nb_trials_before_break is None:
            end = self.nb_trials
        else:
            end = min(start + nb_trials_before_break, self.nb_trials)

        # --- play trials ---
        start_time = time.time()
        for self.running_t in range(start, end):
            query = self.env.get_next_query()
            self.running_queries.append(query)
            propositions, reject = player.choose_next_arm(query)
            expected_best_reward = self.env.get_expected_reward(self.env.get_best_index(query), query)
            expected_reward = self.env.get_expected_reward(propositions, query)
            rewards = self.env.get_reward(propositions, query)
            
            
            self.running_cumulative_expected_best_reward += sum(expected_best_reward[:nb_relevant_positions])
            self.running_cumulative_expected_reward += sum(expected_reward[:nb_relevant_positions])
            self.running_cumulative_reward += sum(rewards[:nb_relevant_positions])
            self.running_cumulative_reject_time += reject

            player.update(propositions, rewards, query)
            if isinstance(self.env, Environment_multirequest_PBM):
                self.update_matrix(propositions[:nb_relevant_positions], rewards[:nb_relevant_positions])

            if self.running_t in self.time_record_expected:

                self.running_time_recorded.append(self.running_t + 1)
                self.running_expected_reward_recorded.append(self.running_cumulative_expected_reward)
                self.running_rewards_recorded.append(self.running_cumulative_reward)
                self.running_expected_best_reward_recorded.append(self.running_cumulative_expected_best_reward)

                self.running_reject_time_recorded.append(self.running_cumulative_reject_time)
                self.running_time_to_play_recorded.append(self.total_time + time.time() - start_time)

            # TODO: remove that temporary catch
            try:
                if self.running_t % (self.nb_trials / self.print_trial) == 0:
                    print("Trial %d/%d\t(%s)" % (self.running_t, self.nb_trials, time_sec_to_DHMS(self.total_time + time.time() - start_time)), flush=True)
            except:
                if self.running_t % (self.nb_trials / 100) == 0:
                    print("Trial %d/%d\t(%s)" % (
                    self.running_t, self.nb_trials, time_sec_to_DHMS(self.total_time + time.time() - start_time)),
                          flush=True)

            if self.running_t == self.nb_trials - 1:
                
                self.record_results['expected_best_reward'].append(self.running_expected_best_reward_recorded)
                self.record_results['expected_reward'].append(self.running_expected_reward_recorded)
                self.record_results['reward'].append(self.running_rewards_recorded)
                self.record_results['reject_time'].append(self.running_reject_time_recorded)
                self.record_results['time_recorded'].append(self.running_time_recorded)
                self.record_results['time_to_play'].append(self.running_time_to_play_recorded)
                self.record_results['query_asked'].append(self.running_queries)
                self.record_results['dico_placed'].append(self.running_dico_placed)
                self.record_results['dico_clicked'].append(self.running_dico_clicked)
                if query is None:
                    self.record_results['env_parameters'].append(deepcopy(self.env.get_params()))

        self.total_time += time.time() - start_time

        return self.running_expected_best_reward - sum(expected_reward[:nb_relevant_positions])     # instantaneous regret

    def update_matrix(self, query, propositions, rewards):
        for i in range(len(propositions)):
            place = i
            item = propositions[i]
            rew = rewards[i]
            self.running_dico_placed[query][item][place] += 1
            self.running_dico_clicked[query][item][place] += rew

    def add_record(self, record):
        ### First record
        if len(self.record_results['time_recorded']) == 0:
            self.record_results = record
            return

        ### Checks
        # same values recorded
        if self.record_results.keys() != record.keys():
            raise ValueError(
                "Both records should contain the same information. Currently referee record contains " + str(
                    self.record_results.keys()) + " and added record contains " + str(record.keys()))

        ### Merge
        for key, val in self.record_results.items():
            val.extend(record[key])

    def get_ext_decile(self, var):
        decile = np.percentile(var, np.arange(0, 100, 10))  # deciles
        return decile[0], decile[8]

    def barerror_value(self,type_errorbar='confidence'):
        regret_pergame = np.array(self.record_results['expected_best_reward']) - np.array(self.record_results['expected_reward'])
        nb_game = len(regret_pergame)
        xValues = self.record_results['time_recorded'][0]
        regret_pertrial =np.mean(regret_pergame,axis=0)
        yValues = regret_pertrial
        yErrorValues = []
        yErrorValues=np.std(regret_pergame,axis=0)
        if type_errorbar=='std':
            return xValues,yValues,yErrorValues
        elif type_errorbar=='standart_error':
            yErrorValues/=sqrt(nb_game)
            return xValues,yValues,yErrorValues
        elif type_errorbar=='confidence':
            yErrorValues/=sqrt(nb_game)
            yErrorValues*=4
            return xValues,yValues,yErrorValues

    def sparser_(self,liste,step):
        sparse_indice=range(0, self.nb_trials, step)
        return [liste[i] for i in sparse_indice]

    def get_regret_expected_withoutdec(self):
        expec_best = np.mean(self.record_results['expected_best_reward'], axis=0)
        expec_prop = np.mean(self.record_results['expected_reward'], axis=0)
        return expec_best - expec_prop

    def get_regret_expected(self):
        sous = np.array(self.record_results['expected_best_reward']) - np.array(self.record_results['expected_reward'])
        print('test',sous)
        d_10_list = []
        d_90_list = []
        for i in sous.transpose():
            d_10, d_90 = self.get_ext_decile(i)
            d_10_list.append(d_10)
            d_90_list.append(d_90)
        reg = np.mean(sous, axis=0)
        return reg, d_10_list, d_90_list

    def get_regret_withoutdec(self):
        expec_best = np.mean(self.record_results['expected_best_reward'], axis=0)
        rew_prop = np.mean(self.record_results['reward'], axis=0)
        print(rew_prop)
        return expec_best - rew_prop

    def get_regret(self):
        sous = np.array(self.record_results['expected_best_reward']) - np.array(self.record_results['reward'])
        d_10_list = []
        d_90_list = []
        for i in sous.transpose():
            d_10, d_90 = self.get_ext_decile(i)
            d_10_list.append(d_10)
            d_90_list.append(d_90)
        reg = np.mean(sous, axis=0)
        return reg, d_10_list, d_90_list

    def get_time_reject(self):
        mean_reject_time = np.mean(self.record_results['reject_time'], axis=0)
        return mean_reject_time

    def get_query_asked(self):
        return self.record_results['query_asked']

    def get_recorded_trials(self):
        return self.record_results['time_recorded'][0]

    def show(self):
        trials = [i for i in range(self.nb_trials)]
        plt.plot(trials, np.cumsum(self.get_regret_expected(), axis=0)[trials], label='player')

        plt.xlabel('Time')
        plt.ylabel('Cumulative Expected')
        plt.legend()
        plt.grid(True)
        plt.loglog()
        plt.show()

        plt.plot(trials, np.cumsum(self.get_regret(), axis=0)[trials], label='player')

        plt.xlabel('Time')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.grid(True)
        plt.loglog()
        plt.show()

    def hist_finalperf(referee):
        sous = np.array(referee.record_results['expected_best_reward']) - np.array(
            referee.record_results['expected_reward'])
        sous_class_perf_fin = [round(i[-1], 0) for i in sous]
        plt.hist(sous_class_perf_fin)

    def function_DCG(mat,proposition):
        nb_trials, nb_choix = mat.shape
        nb_position=len(self.env.kappas)
        indice_kappa_ordonne =  np.array(self.env.kappas).argsort()[::-1][:nb_position]
        log_DCG = []
        for trial in range(nb_trials):
            DCG = 0
            for pos in indice_kappa_ordonne:
                relevance = pow(2, mat[trial][pos]) - 1
                rang = log(pos + 2,
                           2)  ## +2 car DCG c'est rang plus 1 et qu'on a comme position ici les index python qui commencent à 0
                DCG += relevance / rang
            log_DCG.append(DCG)

        return log_DCG
