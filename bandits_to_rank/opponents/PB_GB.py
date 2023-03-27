
#### Bandits

## Packages


import numpy as np
from math import floor


from bandits_to_rank.opponents.greedy import GetSVD
from bandits_to_rank.tools.tools import order_theta_according_to_kappa_index
from bandits_to_rank.tools.tools_Langevin import Langevin
from copy import deepcopy


#from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import partial

## PB_GB

    
class PB_GB:
    """
    Adaption of PB-MHB with a Langevin algorithm as MCMC method to approximate the draw of our target distribution
    """
    def __init__(self, nb_arms, nb_position, initial_particule=None, h_param=0.01, N=1, gamma=1, L_smooth_param=1,
                 m_strongconcav_param=1, prior_s=0.5, prior_f=0.5, part_followed=True,  store_eff=False):
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.initial_particule = initial_particule
        self.part_followed = part_followed
        self.positions = np.arange(nb_position)
        self.store_eff = store_eff
        if h_param is not None:
            self.h_param = h_param
            self.N = N
        self.L_smooth_param = L_smooth_param
        self.m_strongconcav_param = m_strongconcav_param
        self.gamma = gamma

        self.get_model = GetSVD(self.nb_arms, self.nb_position)
        self.clean()

    def _random_particule(self):
        return [np.random.uniform(0, 1, self.nb_arms),
                np.random.uniform(0, 1, self.nb_position)]

    def clean(self):
        """ Clean log data.
        To be ran before playing a new game.
        """
        self.success = np.ones([self.nb_arms, self.nb_position], dtype=np.uint)*self.prior_s
        self.fail = np.ones([self.nb_arms, self.nb_position], dtype=np.uint)*self.prior_f
        if self.initial_particule is not None:
            self.particule = deepcopy(self.initial_particule)
        else:
            self.particule = self._random_particule()

        if self.store_eff:
            self.eff = []
        self.turn = 1

        self.reject_time = 0
        self.pbm_model = self.get_model()

    def choose_next_arm(self):
        if self.h_param is not None:
            h = self.h_param/self.turn
        else:
            h = self.m_strongconcav_param/(32*self.turn*(self.L_smooth_param+self.L_smooth_param/self.turn)**2)
            self.N = floor(640*(self.L_smooth_param + self.L_smooth_param / self.turn) ** 2/ (self.m_strongconcav_param**2))

        param_final_cov = self.turn * self.L_smooth_param * self.gamma

        self.particule, samples = Langevin(self.particule, self.success, self.fail,
                                           self.N, h, param_final_cov, self.nb_position, self.nb_arms)
        reject_time = 0

        thetas = samples[0]
        kappas = samples[1]
        return order_theta_according_to_kappa_index(thetas, kappas), reject_time
    
    def update(self, propositions, rewards):
        self.turn += 1
        self.fail[propositions, self.positions] += 1 - rewards
        self.success[propositions, self.positions] += rewards

        self.pbm_model.add_session(propositions, rewards)

    def get_param_estimation(self):
        self.pbm_model.learn()
        self.thetas_estim, self.kappas_estim = self.pbm_model.get_params()
        return self.thetas_estim, self.kappas_estim


    def type(self):
        return 'PB-MHB'
    

  