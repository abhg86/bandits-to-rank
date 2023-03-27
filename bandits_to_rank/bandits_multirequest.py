
#### Bandits

## Packages


import numpy as np
import random as rd

from bandits_to_rank.bandits import *
from bandits_to_rank.sampling.metropolis_hasting import *
from bandits_to_rank.sampling.proposal import *
from bandits_to_rank.sampling.target import *


from numpy.random import beta 
from random import uniform
from copy import deepcopy


import scipy.stats as stats
from functools import partial

## Fonction Auxiliaires


## TS_MH

    
class TS_MH_kappa_desordonne_multirequest:
    """
    Our Algo (aka.PB-MHB)
    """
    def __init__(self, nb_queries,nb_arms, nb_position, proposal_method=propos_trunk_GRW(vari_sigma=True, c=3), initial_particule=None, pas=10, prior_s=0.5, prior_f=0.5, part_suivie=True,  store_eff=False):
        self.nb_queries = nb_queries
        self.nb_arms = nb_arms
        self.nb_position = nb_position
        self.pas = pas
        
        self.players = [TS_MH_kappa_desordonne(nb_arms[q], nb_position, proposal_method, initial_particule, pas, prior_s, prior_f, part_suivie,  store_eff) for q in range(nb_queries) ]
        
        
        self.positions = np.arange(nb_position)
        self.store_eff = store_eff
        
        self.clean()

    def _random_particule(self,query):
        return [np.random.uniform(0, 1, self.nb_arms[query]),
                np.array([1] + list(np.random.uniform(0, 1, self.nb_position - 1)))]

    def clean(self):
        """ Clean log data.
        To be ran before playing a new game.
        """
        self.dico_success ={}
        self.dico_fail ={}
        for q in range(self.nb_queries) :
            self.success = np.ones([self.nb_arms[q], self.nb_position], dtype=np.uint)*self.players[q].prior_s
            self.fail = np.ones([self.nb_arms[q], self.nb_position], dtype=np.uint)*self.players[q].prior_f
            if self.players[q].initial_particule is not None:
                self.players[q].particule = deepcopy(self.players[q].initial_particule)
            else:
                self.players[q].particule = self._random_particule(q)
        if self.store_eff:
            self.eff = []
        self.tour = 0
        self.reject_time = 0
        


    def choose_next_arm(self,query):
        return self.players[query].choose_next_arm()
    
    def update(self, propositions, rewards,query):
        self.players[query].update(propositions, rewards)
        
    def type(self):
        return 'TS_MH'
    

  