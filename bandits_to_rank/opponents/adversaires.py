
#### Bandits

## Packages


import numpy as np
import random as rd


from bandits_to_rank.sampling.target import *
from bandits_to_rank.sampling.pbm_inference import EM


from pyclick.click_models.Evaluation import LogLikelihood, Perplexity
from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.DCM import DCM
from pyclick.click_models.CCM import CCM
from pyclick.click_models.CTR import DCTR, RCTR, GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult




from numpy.random import beta 
from random import uniform
from copy import deepcopy


from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import partial

## Fonction Auxiliaires

def maximum_K_indice(liste,K=3):
    new=np.argsort(liste)
    return new[-1:-(int(K)+1):-1]

def maximum_K(liste,K=3):
    new=np.sort(liste)
    return new[-1:-(int(K)+1):-1]


def ordonne_indice_theta_function_kappa(thetas,kappas):
    """

    :param thetas:
    :param nb_position:
    :param kappas:
    :return:

    >>> import numpy as np
    >>> thetas = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.4])
    >>> kappas = np.array([1, 0.9, 0.8])
    >>> propositions = ordonne_theta_function_kappa(thetas, 2, kappas)
    >>> assert(np.all(propositions == np.array([0, 1])), str(propositions))
    >>> propositions = ordonne_theta_function_kappa(thetas, 3, kappas)
    >>> assert(np.all(propositions == np.array([0, 1, 2])), str(propositions))

    >>> thetas = np.array([0.6, 0.6, 0.8, 0.9, 0.7, 0.4])
    >>> kappas = np.array([1, 0.8, 0.9])
    >>> propositions = ordonne_theta_function_kappa(thetas, 2, kappas)
    >>> assert(np.all(propositions == np.array([3, 4])), str(propositions))
    >>> propositions = ordonne_theta_function_kappa(thetas, 3, kappas)
    >>> assert(np.all(propositions == np.array([3, 4, 2])), str(propositions))
    """
    nb_position = len(kappas)
    indice_theta_ordonne = np.array(thetas).argsort()[::-1][:nb_position]
    indice_kappa_ordonne =  np.array(kappas).argsort()[::-1][:nb_position]
    res = np.ones(nb_position, dtype=np.int)
    nb_put_in_res = 0
    for i in indice_kappa_ordonne:
        res[i]=indice_theta_ordonne[nb_put_in_res]
        nb_put_in_res+=1
    return res


def simule_log_Pyclick(nb_reco,theta,kappa):
    search_sessions=[]
    nb_position = len(kappa)
    nb_item = len(theta)
    indice_item = [x for x in range(nb_item)]
    for reco in range(nb_reco):
        #print('recommandation numero = ',reco)
        web_results =[]
        ### tire aléatoirement la présentation des produits :
        proposition=sample(indice_item,nb_position)
        #print ('produits proposes',proposition)
        for pos in range(len(proposition)):
            #print ('a la position', pos )
            index_produit = proposition[pos]
            #print('je propose',index_produit)
            is_view = int(random() < kappa[pos])
            is_click= int(random() < theta[index_produit])
            web_results.append(SearchResult(index_produit,is_view*is_click))
        ###simulation comportement
        search_sessions.append(TaskCentricSearchSession(reco,'Reco'))
        search_sessions[-1].web_results = web_results
    return(search_sessions)

def extract_kappa(clickmodel,nb_position):
    param =[]
    for i,j in enumerate(clickmodel.params) :
        param.append(clickmodel.params[j])
    kappa=[]
    for i in range(nb_position):
        kappa.append(param[1]._container[i].value())
    return kappa

def give_kappa_Pyclick(sessions,nb_position):
    click_model = PBM()
    click_model.train(sessions)
    return extract_kappa (click_model,nb_position)


## Bench

class Oracle:
    """
    Player which plays the best arm
    """
    def __init__(self, best_arm):
        self.best_arm = best_arm

    def clean(self):
        pass

    def choose_next_arm(self):
        return self.best_arm
    
    def update(self, arm, reward):
        pass
    
    def type(self):
        return 'Oracle'


class Random:
    
    """
    Player  which plays random arms
    """
    def __init__(self,nb_arm,nb_choix):
        self.nb_arm=nb_arm
        self.nb_choix=nb_choix

    def clean(self):
        pass

    def choose_next_arm(self):
        return rd.sample(range(self.nb_arm),self.nb_choix)

    def update(self, arm, reward):
        pass
    
    def type(self):
        return 'Random'
    
    
class greedy_Cascade:
    """Construction d'un Bandit 
    Source: 
      Parameter
    nb_arms : nombre de bras a etudier
    kappas : cllick rate of prositions
      Attributs:  
    performance = cumule des fois ou le bras a ete clique,
    nb_trials : array(n,2) nombre de fois ou les categories sont vues,
    nb_position
    """
    
    def __init__(self, nb_arms, kappas):
        self.nb_arms = nb_arms
        self.nb_position = len(kappas)
        self.kappas = kappas
        self.clean()

    def clean(self):
        self.performance = np.zeros(self.nb_arms)
        self.nb_trials = np.zeros(self.nb_arms, dtype=np.uint)

    def choose_next_arm(self, temps_init=10**(-5)):
        # exploit
        taux_perf=self.performance/(self.nb_trials+temps_init)
        return maximum_K_indice(taux_perf,self.nb_position)
        
    def update(self, propositions, rewards):
        for pos in range(len(propositions)):
            if random() <self.kappas[pos]: ### Alors le produit a probablement été vu, on peut mettre à jour 
                item = propositions[pos]
                rew = rewards[pos]
                self.performance[item] += rew
                self.nb_trials[item] += 1
        
    def type(self):
        return 'greedy_Cascade'
    
    

    
### E_greedy


class greedy_EGreedy_ (object):
    """Construction d'un Bandit 
    Source: 
      Parameter
    c = coefficient de cadrage de l'epsilon,
    nb_arms = nombre de bras a etudier
      Attributs:  
    Mise_a_jour_liste : Bool permettant de savoir si une mise a jour a ete faite et si l'epsilon doit etre modifie dans la Strategie
    performance = cumule des fois ou le bras a ete clique,
    nb_trials : array(n,2) nombre de fois ou les categories sont vues,
    learning_rate : array(n,2)taux d'apprentissage de chaque categories, soit la probabilite actuelle
                    de chaque categorie d'etre la meilleure categorie\,
    """
    
    def __init__(self, c, nb_arms, kappas):
        self.nb_arms = nb_arms
        self.c = c
        self.nb_position = len(kappas)
        self.Mise_a_jour = False
        self.kappas = kappas
    
    def clean(self):
        self.performance = np.zeros(self.nb_arms)
        self.nb_trials = np.zeros(self.nb_arms, dtype=np.uint)

    def choose_next_arm(self, temps_init=10**(-5)):
        t = sum(self.nb_trials) + temps_init
        if random() < self.c/t:
            # explore
            return rd.sample(range(len(self.performance)-1), self.nb_position)
        else:
            # exploit
            taux_perf=self.performance/(self.nb_trials+temps_init)
            return maximum_K_indice(taux_perf,self.nb_position)
        
    def update(self, propositions, rewards):
        for pos in range(len(propositions)):
            if random() <self.kappas[pos]: ### Alors le produit a probablement été vu, on peut mettre à jour 
                item = propositions[pos]
                rew = rewards[pos]
                self.performance[item] += rew
                self.nb_trials[item] += 1
        
    def type(self):
        return 'E_greedy'
    


class Bandit_EGreedy_X_rep (object):
    """Construction d'un Bandit 
    Source: 
      Parameter
    c = coefficient de cadrage de l'epsilon,
    nb_arms = nombre de bras a etudier
      Attributs:  
    Mise_a_jour_liste : Bool permettant de savoir si une mise a jour a ete faite et si l'epsilon doit etre modifie dans la Strategie
    performance = cumule des fois ou le bras a ete clique,
    nb_trials : array(n,2) nombre de fois ou les categories sont vues,
    learning_rate : array(n,2)taux d'apprentissage de chaque categories, soit la probabilite actuelle
                    de chaque categorie d'etre la meilleure categorie\,
    """
    
    def __init__(self,c,nb_arms,nb_position=3):
        self.performance= np.zeros(nb_arms)
        self.nb_trials = np.zeros(nb_arms, dtype=np.uint)
        self.c = c
        self.nb_position=nb_position
        self.Mise_a_jour=False
    
    def choose_next_arm(self, temps_init=10**(-5)):
        t = sum(self.nb_trials) + temps_init
        if random() < self.c/t:
            # explore
            return rd.sample(range(len(self.performance)-1), self.nb_position)
        else:
            # exploit
            taux_perf=self.performance/(self.nb_trials+temps_init)
            return maximum_K_indice(taux_perf,self.nb_position)
        
    def update(self, propositions, rewards):
        for pos in range(len(propositions)):
            item = propositions[pos]
            rew = rewards[pos]
            self.performance[item] += rew
            self.nb_trials[item] += 1
        
    def type(self):
        return 'E_greedy'
    

    
    
    
## TS_V0

class ThompsonSamplingBernoulli_V0:
    """
    Source : "Optimal Regret Analysis of TS in Stochastic MAB Problem with multiple Play"_Komiyama,Honda,Nakagawa
    Approximate random sampling given posterior probability to be optimal,
    """
    def __init__(self, nb_arms,nb_position, prior_s=0.5, prior_f=0.5):
        self.success = np.ones([nb_arms, nb_position], dtype=np.uint)*prior_s
        self.fail = np.ones([nb_arms, nb_position], dtype=np.uint)*prior_f
        self.nb_position = nb_position
        
    def choose_next_arm(self):
        thetas = beta(np.sum(self.success,axis=1), np.sum(self.fail,axis=1))
        return maximum_K_indice(thetas,self.nb_position)
    
    def update(self, propositions, rewards):
        for pos in range(len(propositions)):
            item = propositions[pos]
            rew = rewards[pos]
            self.success[item][pos] += rew
            self.fail[item][pos] += 1 - rew
        
    def type(self):
        return 'TS_V0'


class BC_MPTS:
    """
    Source : "Optimal Regret Analysis of TS in Stochastic MAB Problem with multiple Play"_Komiyama,Honda,Nakagawa
    Approximate random sampling given posterior probability to be optimal,
    """
    def __init__(self, nb_arms,nb_position,discount_factor, prior_s=0.5, prior_f=0.5):
        self.success = np.ones([nb_arms, nb_position], dtype=np.uint)*prior_s
        self.vu_place = np.array([np.zeros(nb_position)]*nb_arms) + prior_s*2
        self.nb_position = nb_position
        self.discount_factor = discount_factor
        
    def choose_next_arm(self):
        fail= self.vu_place - self.success
        good_vue =[]
        for i in np.sum(fail,axis=1):
            if i>0 :
                good_vue.append(i+1)
            else:
                good_vue.append(1)
        thetas = beta(np.sum(self.success,axis=1)+1, good_vue)
        return maximum_K_indice(thetas,self.nb_position)

    def update(self, propositions, rewards):
        for pos in range(len(propositions)):
            item = propositions[pos]
            rew = rewards[pos]
            self.success[item][pos] += rew
            self.vu_place[item][pos] += self.discount_factor[pos]
        
    def type(self):
        return 'BC_MPTS'
    

        