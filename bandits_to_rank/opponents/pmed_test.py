#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

import tensorflow as tf
import numpy as np
import scipy.optimize
import scipy.special
import scipy.sparse
import time
import warnings

from numpy import sqrt, log
from numpy.random import beta
from scipy.optimize import linear_sum_assignment
from tensorflow.python.ops.gen_array_ops import matrix_diag_v2  # waiting for a debug of tf.linalg.diag

import bandits_to_rank.tools.tfp_math_minimize as tfprg


## usefull functions

def dKL_Bernoulli(P, Q, N=None):
    if N is None:
        N = np.ones(P.shape)
    res = N*scipy.special.rel_entr(P, Q) + N*scipy.special.rel_entr(1.-P, 1.-Q)
    return res

# TODOs
# * optim_hat change uniquement la bonne variable
# * optim_prime dépend de l => change uniquement la bonne variable
# * optim_prime projection limitée à bonne variable
# * condition de fin d'optim



## PMED
class PMED:
    """
    algorithm PMED.
    Junpei Komiyama, Junya Honda, Akiko Takeda. Position-based Multiple-play Bandit Problem with Unknown Position Bias. NIPS'17

    Assume kappas ordered in decreasing order
    """

    def __init__(self, nb_arms, nb_positions, alpha, gap_MLE=20, gap_q=1000):
        """

        Parameters
        ----------
        nb_arms
        nb_positions
        alpha
            PMED explores each couple (arm, position) at least `alpha * log(t)` times.
        gap_MLE
            minimum number of trials between two updates of inferred values `(thetas_hat, kappas_hat)`
        gap_q
            minimum number of trials between two updates of inferred value `q`


        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(precision=3)
        >>> player = PMED(5, 3, 1.)
        >>> player.l_current = {(0,1,2)}
        >>> player.nb_prints = 1000 * np.array([[30, 3, 3],
        ...        [5, 30, 1],
        ...        [1, 3, 30],
        ...        [1, 1, 1],
        ...        [1, 1, 3]])
        >>> player.nb_trials = player.nb_prints[:,0].sum()
        >>> player.nb_trials
        38000
        >>> thetas = np.array([0.9, 0.7, 0.5, 0.45, 0.1])
        >>> kappas = np.array([1, 0.9, 0.8])
        >>> player.nb_clics = np.array((thetas.reshape((-1,1)) @ kappas.reshape((1,-1))) * player.nb_prints, dtype=np.int)
        >>> player.nb_clics
        array([[27000,  2430,  2160],
               [ 3500, 18900,   559],
               [  500,  1350, 12000],
               [  450,   405,   360],
               [  100,    90,   240]])
        >>> player.mus_hat = player.nb_clics / player.nb_prints
        >>> prop, _ = player.choose_next_arm()
        >>> prop
        array([0, 1, 2])
        >>> for _ in range(100):
        ...     player.update_thetas_hat_kappas_hat()
        >>> player.kappas_hat
        array([1. , 0.9, 0.8])
        >>> player.thetas_hat
        array([0.9 , 0.7 , 0.5 , 0.45, 0.1 ])
        >>> player.update(prop, np.array([0,1,0]))

        >>> player.kappas_hat
        array([1. , 0.9, 0.8])
        >>> player.thetas_hat
        array([0.9 , 0.7 , 0.5 , 0.45, 0.1 ])
        >>> np.array(player.q * np.log(player.nb_trials), dtype=np.int)
        array([[3501,   43,    0,    0,    0],
               [  43, 3467,   34,    0,    0],
               [   0,   32,  343, 1584, 1584],
               [   0,    0, 3129,  208,  208],
               [   0,    1,   38, 1752, 1752]])
        >>> print([(np.round(c,3), perm) for c, perm in player.decompose_nb_prints_tilde()])
        [(1584.419, array([0, 1, 3, 2, 4])), (1539.614, array([0, 1, 3, 4, 2])), (175.106, array([0, 1, 2, 4, 3])), (168.432, array([0, 1, 2, 3, 4])), (38.133, array([1, 0, 4, 3, 2])), (32.946, array([0, 2, 1, 4, 3])), (5.185, array([1, 0, 3, 4, 2])), (1.488, array([0, 4, 1, 3, 2])), (0.0, array([4, 0, 1, 3, 2])), (0.0, array([0, 3, 1, 2, 4]))]
        >>> player.l_current
        {(0, 1, 2), (1, 0, 3), (0, 1, 3)}
        """
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.alpha = alpha
        self.gap_MLE = gap_MLE
        self.gap_q = gap_q
        try:
            self.rng = np.random.default_rng()
        except:
            self.rng = np.random
        # --- variables ---
        self.clean()

    def clean(self):
        self.positions = np.arange(self.nb_positions)
        self.pseudo_positions = np.arange(self.nb_arms)

        # statistics
        self.nb_trials = 0
        self.last_MLE = -np.inf
        self.last_optimize_q = -np.inf
        self.nb_empty_l_current = 0
        self.nb_error_with_optimize_q = 0
        self.nb_clics = np.ones((self.nb_arms, self.nb_positions), dtype=np.float64)
        self.nb_prints = np.ones((self.nb_arms, self.nb_positions), dtype=np.uint)
        self.mus_hat = np.zeros((self.nb_arms, self.nb_positions), dtype=np.float64)
        self.thetas_hat = self.rng.random(self.nb_arms)
        #  kappas is expected to be sorted, with kappas[0] = 1
        self.kappas_hat = np.sort(self.rng.random(self.nb_positions))[::-1]
        self.kappas_hat[0] = 1.
        self.q = np.zeros((self.nb_arms, self.nb_arms), dtype=np.float64)

        # constant and variables used for optimisation of theta_hat and kappas_hat
        self.thetas_hat_tf = tf.Variable(self.thetas_hat.reshape((-1, 1)))
        self.kappas_hat_1end_tf = tf.Variable(self.kappas_hat[1:])
        self.p_hat = tf.Variable(tf.zeros((self.nb_arms * self.nb_positions, 2), dtype='float64'))    # flatten mu_hat and (1-mu_hat)
        self.min_p = tf.Variable(0., dtype='float64')
        self.max_p = tf.Variable(1., dtype='float64')
        self.nb_prints_update_thetas_kappas = tf.Variable(self.nb_prints.reshape(-1), dtype='float64')

        # constant and variables used for optimisation of theta_prime
        self.qo_tf = tf.Variable(self.q[:, :self.nb_positions])
        self.keep_value = tf.constant(matrix_diag_v2(np.zeros(self.nb_positions, dtype='float64'), k=0,
                                                     num_rows=self.nb_arms, num_cols=-1, padding_value=1))
        self.thetas_hato = tf.Variable(self.thetas_hat)
        self.thetas_primeo_minus0andl = tf.Variable(self.thetas_hat[2:])
        tf.print('initial thetas prime', self.thetas_primeo_minus0andl)
        self.thetas_hato_kappas_hat = tf.Variable(tf.zeros(self.nb_positions, dtype='float64'))
        self.mus_hato = tf.Variable(self.mus_hat)
        self.minus_mus_hato = tf.Variable(self.mus_hat)

        # list of permutations to play
        self.basic_permutations = [tuple(np.arange(i, i + self.nb_positions) % self.nb_arms) for i in range(self.nb_arms)]
        self.l_current = set(self.basic_permutations)

    def choose_next_arm(self):
        try:
            propositions = list(self.l_current)[self.rng.integers(len(self.l_current))]
        except:
            propositions = list(self.l_current)[self.rng.randint(len(self.l_current))]
        self.l_current.remove(propositions)
        return np.array(propositions), 0

    def update(self, propositions, rewards, verbose=True):
        # update statistics
        self.nb_trials += 1
        self.nb_prints[propositions, self.positions] += 1
        self.nb_clics[propositions, self.positions] += rewards
        self.mus_hat[propositions, self.positions] = (self.nb_clics[propositions, self.positions]
                                                      / self.nb_prints[propositions, self.positions] )

        # find next arms to play
        # done only when self.l_current is empty to reduce computation time
        if not self.l_current:
            self.nb_empty_l_current += 1

            # heavy updates
            if self.nb_trials >= self.last_MLE + self.gap_MLE:
                if verbose:
                    print(f'infer thetas_hat and kappas_hat at iteration {self.nb_trials}')
                self.last_MLE = self.nb_trials
                t0 = time.time()
                self.update_thetas_hat_kappas_hat()
                if verbose:
                    print('computing time for inference of thetas_hat and kappas_hat:', time.time()-t0)

            if (self.last_optimize_q < 0 or
                (self.gap_q > 0 and self.nb_trials >= self.last_optimize_q + self.gap_q) or
                (self.gap_q == 0 and np.sqrt(self.last_optimize_q) // 3 != np.sqrt(self.nb_trials) // 3)
                ):
                if verbose:
                    print(f'optimize q at iteration {self.nb_trials}')
                self.last_optimize_q = self.nb_trials
                t0 = time.time()
                self.optimize_q()
                if verbose:
                    print('computing time for optimization of q:', time.time()-t0)

            # exploitation
            self.l_current.add(tuple(np.array(self.thetas_hat).argsort()[::-1][:self.nb_positions]))

            # exploration
            self.add_flat_exploration()
            self.add_optimal_exploration()

        if verbose and self.nb_trials % 1000 == 0:
            print(f'L_C has been renewed {self.nb_empty_l_current} times during the first {self.nb_trials} iterations')
            print(f'optimize_q() has failed {self.nb_error_with_optimize_q} times during the first {self.nb_trials} iterations')

    def get_param_estimation(self):
        return self.thetas_hat, self.kappas_hat

    @tf.function
    def kl_loss_fn(self):
        """ KL-like loss function used to infer thetas and kappas
        from empirical click-probability per couple (item,  position)
        """
        print('Tracing kl_loss_fn()')
        #tf.print('be', self.kappas_hat_1end_tf)
        #tf.print('be', tf.reshape(self.thetas_hat_tf, (-1,)))
        one = tf.constant(1., shape=(1,), dtype='float64')
        kappas_hat = tf.reshape(tf.concat((one, self.kappas_hat_1end_tf), 0), (1, -1))
        mus_hathat = tf.clip_by_value(tf.matmul(self.thetas_hat_tf, kappas_hat),
                                      clip_value_min=self.min_p, clip_value_max=self.max_p)
        p_hathat = tf.concat((tf.reshape(mus_hathat, [-1, 1]), tf.reshape(1. - mus_hathat, [-1, 1])), 1)
        return tf.reduce_sum(tf.reduce_sum(self.p_hat * tf.math.log(self.p_hat / p_hathat), axis=1)
                             * self.nb_prints_update_thetas_kappas)

    @tf.function
    def projection_theta_hat_kappas_hat(self):
        """
        By clipping, enforces constraints on thetas_hat and kappas_hat
        * kappas_hat is non-increasing
        * thetas_hat and kappas_hat[:L-1] are in [epsilon, 1.-epsilon]
        Some constraints are not enforced has the optimization procedure do not change corresponding values
        * kappas_hat[0]=1


        We enforce the "non-increasing" constraints for i from 1 to L-1

        Remark: these is a clipping, not a projection
        """
        print('Tracing projection_theta_hat_kappas_hat()')
        # kappas_hat is non-increasing and in [epsilon, 1.-epsilon]
        self.kappas_hat_1end_tf[0].assign(tf.clip_by_value(self.kappas_hat_1end_tf[0],
                                                           clip_value_min=self.min_p,
                                                           clip_value_max=self.max_p))
        for i in tf.range(self.nb_positions-1):
            self.kappas_hat_1end_tf[i].assign(tf.clip_by_value(self.kappas_hat_1end_tf[i],
                                                           clip_value_min=self.min_p,
                                                           clip_value_max=self.kappas_hat_1end_tf[i - 1]))
        # thetas_hat is in [epsilon, 1.-epsilon]
        self.thetas_hat_tf.assign(tf.clip_by_value(self.thetas_hat_tf,
                                                   clip_value_min=self.min_p, clip_value_max=self.max_p))

    def update_thetas_hat_kappas_hat(self, num_steps=100, verbose=False, epsilon=10**-14):
        if verbose:
            print('clics:', self.nb_clics)
            print('prints:', self.nb_prints)
            print('mus_hat:', self.mus_hat)

        # --- useful constants ---
        # TODO: adapt epsilon to dataset size
        # epsilon = tf.constant(min(0.001/np.sqrt(self.nb_trials), 0.001))

        # --- update variables/parameters of loss function ---
        self.thetas_hat_tf.assign(self.thetas_hat.reshape((-1, 1)))
        self.kappas_hat_1end_tf.assign(self.kappas_hat[1:])
        mus_hat = np.clip(self.mus_hat, a_min=epsilon, a_max=1. - epsilon)
        self.p_hat.assign(tf.concat((tf.reshape(mus_hat, [-1, 1]), tf.reshape(1.-mus_hat, [-1, 1])), 1))
        self.nb_prints_update_thetas_kappas.assign(self.nb_prints.reshape(-1))
        self.min_p.assign(epsilon)
        self.max_p.assign(1. - epsilon)

        if verbose:
            kappas_hat = np.concatenate((np.ones(1), self.kappas_hat_1end_tf.numpy()))
            print(f'begin: loss {self.kl_loss_fn().numpy()}, for thetas = {self.thetas_hat_tf.numpy()}, and kappas[1:] = {kappas_hat}')
            mus_hathat_np = self.thetas_hat_tf.numpy() @ kappas_hat
            kl_np = float(np.sum(self.nb_prints * (mus_hat * np.log(mus_hat / mus_hathat_np)
                                 + (1 - mus_hat) * np.log((1 - mus_hat) / (1 - mus_hathat_np)))))
            mus_hat_cliped = np.clip(mus_hat, epsilon, 1.-epsilon)
            mus_hathat_cliped = np.clip(mus_hathat_np, epsilon, 1.-epsilon)
            kl_np_clipped = float(np.sum(self.nb_prints * (mus_hat_cliped * np.log(mus_hat_cliped / mus_hathat_cliped)
                                           + (1 - mus_hat_cliped) * np.log((1 - mus_hat_cliped) / (1 - mus_hathat_cliped)))))
            print(f'given np, KL = {kl_np}     and clipped KL = {kl_np_clipped}')
            trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': self.thetas_hat_tf, 'kappa': self.kappas_hat_1end_tf}
        else:
            trace_fn=lambda loss, grads, variables: {}
        trace = tfprg.minimize(self.kl_loss_fn, num_steps=num_steps,
                               trainable_variables=(self.thetas_hat_tf, self.kappas_hat_1end_tf),
                               optimizer=tf.optimizers.Adam(0.001), projection=self.projection_theta_hat_kappas_hat,
                               trace_fn=trace_fn)
        if verbose:
            kappas_hat = np.concatenate((np.ones(1), self.kappas_hat_1end_tf.numpy()))
            print(f'begin: loss {self.kl_loss_fn().numpy()}, for thetas = {self.thetas_hat_tf.numpy()}, and kappas[1:] = {kappas_hat}')
            mus_hathat_np = self.thetas_hat_tf.numpy() @ kappas_hat
            kl_np = float(np.sum(self.nb_prints * (mus_hat * np.log(mus_hat / mus_hathat_np)
                                 + (1 - mus_hat) * np.log((1 - mus_hat) / (1 - mus_hathat_np)))))
            mus_hat_cliped = np.clip(mus_hat, epsilon, 1.-epsilon)
            mus_hathat_cliped = np.clip(mus_hathat_np, epsilon, 1.-epsilon)
            kl_np_clipped = float(np.sum(self.nb_prints * (mus_hat_cliped * np.log(mus_hat_cliped / mus_hathat_cliped)
                                           + (1 - mus_hat_cliped) * np.log((1 - mus_hat_cliped) / (1 - mus_hathat_cliped)))))
            print(f'given np, KL = {kl_np}     and clipped KL = {kl_np_clipped}')

        self.thetas_hat = self.thetas_hat_tf.numpy().reshape(-1)
        self.kappas_hat = np.concatenate((np.ones(1), self.kappas_hat_1end_tf.numpy()))
        self.assert_thetas_and_kappas(self.thetas_hat, self.kappas_hat)

    def assert_thetas_and_kappas(self, thetas, kappas):
        """ some assertions on thetas and kappas
        """
        if np.any(thetas < 0):
            warnings.warn(f'thetas not positive: {thetas} (kappas: {kappas})', stacklevel=2)
        if np.any(thetas > 1):
            warnings.warn(f'thetas not smaller than 1: {thetas} (kappas: {kappas})', stacklevel=2)
        # kappas[0] is 1
        if np.abs(kappas[0] - 1.) > 10 ** -5:
            warnings.warn(f'kappas[0] not equal to 1: {kappas} (thetas: {thetas})', stacklevel=2)
        # kappas is in [0, 1]
        if np.any(kappas < 0):
            warnings.warn(f'kappas not positive: {kappas} (thetas: {thetas})', stacklevel=2)
        if np.any(kappas > 1):
            warnings.warn(f'kappas not smaller than 1: {kappas} (thetas: {thetas})', stacklevel=2)
        # kappas is decreasing
        if np.any(np.diff(kappas) > 10 ** -5):
            warnings.warn(f'kappas not decreasing: {kappas} (thetas: {thetas})', stacklevel=2)

    def optimize_q(self, num_constraints=100, epsilon=10**-5, epsilon_clip=10**-14, plot=False, verbose=True):
        """ Assumes kappas_hat is non-increasing """
        # TODO: q eloigné de epsilon du bord
        if verbose:
            print('clics:', self.nb_clics)
            print('prints:', self.nb_prints)
            print('mus_hat:', self.mus_hat)
            print('theta_hat:', self.thetas_hat)
            print('kappas_hat:', self.kappas_hat)
        if plot:
            sum_delta_qs = []
            max_constraints = []
            new_constraints = [-np.inf]

        # WARNING: thetas_xxx, q, and mus_xxx are manipulated given the order indicated by thetas_hat
        order_thetas_hat = np.argsort(self.thetas_hat)[::-1]
        thetas_hato = self.thetas_hat[order_thetas_hat]
        mus_hato = np.clip(self.mus_hat[order_thetas_hat, :], epsilon_clip, 1.-epsilon_clip)

        # - init optimization problem -
        # min sum_{i<K, l<L} delta[i,l]q[i,l]
        # With delta[i,l] = thetas_hat[l]*kappas_hat[l] - thetas_hat[i]*kappas_hat[l]
        delta = np.concatenate((np.ones((self.nb_arms, 1))
                                @ (thetas_hato[:self.nb_positions]*self.kappas_hat).reshape([1, -1])
                                - thetas_hato.reshape([-1, 1]) @ self.kappas_hat.reshape([1, -1]),
                                np.zeros((self.nb_arms, self.nb_arms-self.nb_positions))), axis=1)
        c = delta.flatten()
        # q[i,l] >= 0  => default in scipy.optimize.linprog
        A_eq = np.zeros((self.nb_arms + self.nb_arms - 2, self.nb_arms * self.nb_arms))
        b_eq = np.zeros(self.nb_arms + self.nb_arms - 2)
        # \sum_l q[i,l] = \sum_l q[i+1,l] = \sum_l q[0,l]
        A_eq[:(self.nb_arms-1), :self.nb_arms] = 1
        for i in range(self.nb_arms-1):
            A_eq[i, ((i+1) * self.nb_arms):((i + 2) * self.nb_arms)] = -1
        # \sum_i q[i,l] = \sum_i q[i,l+1] = \sum_i q[i,0]
        A_eq[(self.nb_arms-1):, ::self.nb_arms] = 1
        for i in range(self.nb_arms-1):
            A_eq[(self.nb_arms-1+i), (i+1)::self.nb_arms] = -1
        # for all s, \sum_{i<K, l<L : i!=l} q[i,l] dKL(mus_hato, mus_primeo) >= 1
        A_ub = np.zeros((num_constraints, self.nb_arms * self.nb_arms))
        b_ub = np.zeros(num_constraints)

        previous_fun_val = -np.inf

        for s in range(num_constraints):
            # --- add a constraint --- if constraint not strong enough, stop optimization of q
            if s == 0:
                thetas_primeo = self.get_initial_thetas_primeo()
            else:
                t0 = time.time()
                tmp = self.optimize_constraint_with_increasing_num_steps(res.x.reshape((self.nb_arms, self.nb_arms)), epsilon=epsilon)
                print('computing time for optimization of theta prime:', time.time()-t0)
                if tmp['val'] > 1. - epsilon:
                    # constraint not strong enough
                    if verbose:
                        print(f'early stopping of optimize_q() as constraints {s} is not strong enough')
                    break
                thetas_primeo = tmp['x']
            if verbose:
                print("thetas_primeo^(s):", thetas_primeo)

            # for all s, \sum_{i<K, l<L : i!=l} q[i,l] dKL(mus_hato[i,l], mus_primeo[i,l]) >= 1
            kappas_primeo = thetas_hato[:self.nb_positions] * self.kappas_hat / thetas_primeo[:self.nb_positions]
            mus_primeo = np.clip(thetas_primeo.reshape((-1, 1)) @ kappas_primeo.reshape((1, -1))
                                 , epsilon_clip, 1.-epsilon_clip)
            constraints = np.concatenate((-dKL_Bernoulli(mus_hato, mus_primeo),
                                          np.zeros((self.nb_arms, self.nb_arms-self.nb_positions))), axis=1)
            A_ub[s, :] = constraints.flatten()
            A_ub[s, ::(self.nb_arms + 1)] = 0       # remove i == l
            b_ub[s] = -1
            if plot and s > 0:
                new_constraints.append(-A_ub[s, :] @ res.x)
            # WARNING: matrices are flatten in row-major (C-style) order

            try:
                res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=None,
                                             method='interior-point', callback=None, options=None, x0=None)
            except TypeError as e:
                res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=None,
                                             method='interior-point', callback=None, options=None)

            if verbose:
                print("q (ordered):", res.x.reshape((self.nb_arms, self.nb_arms)))
                print("sum delta.q:", res.fun)
                print(f'sum q.d_KL:', A_ub @ res.x)
                print(f' vs up. b.:', b_ub)
                print(f'sum q.d_KL[s]:', A_ub[s, :] @ res.x)
                print("error:", not res.success)
                print("status:", res.status, res.message)
                print("nb it:", res.nit)

            # --- error during optimization ---
            if not res.success:
                warnings.warn(f'Warning: optimization of q failed with {s} constraints. We early stop optimize_q() and keep previous value of q.\n' +
                              f'Failure: {res.status} {res.message}\n' +
                              f'Nb it for scipy.optimize.linprog: {res.nit}')
                print(f'Warning: optimization of q failed with {s} constraints. We early stop optimize_q() and keep previous value of q.\n' +
                              f'Failure: {res.status} {res.message}\n' +
                              f'Nb it for scipy.optimize.linprog: {res.nit}')
                print('context')
                print('clics:', self.nb_clics)
                print('prints:', self.nb_prints)
                print('mus_hat:', self.mus_hat)
                print('theta_hat:', self.thetas_hat)
                print('kappas_hat:', self.kappas_hat)
                self.nb_error_with_optimize_q += 1
                break

            self.q = res.x.reshape((self.nb_arms, self.nb_arms))[np.argsort(order_thetas_hat), :]

            if plot:
                sum_delta_qs.append(res.fun)
                max_constraints.append(-(A_ub[:(s+1), :] @ res.x).max())


            # --- early stopping --- when the new constraint does not enough increase the score to minimize
            if res.fun < previous_fun_val * (1. + epsilon):
                if verbose:
                    print(f'early stopping of optimize_q() as constraints {s} does not enough increase the score to minimize')
                break
            previous_fun_val = res.fun

        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(2, 2, 1)
            plt.ylabel("to be minimized")
            plt.plot(sum_delta_qs, label='sum delta.q')
            plt.legend()
            plt.grid()

            plt.subplot(2, 2, 2)
            plt.ylabel("to be minimized")
            plt.plot(sum_delta_qs, label='sum delta.q')
            plt.legend()
            plt.grid()
            plt.yscale('log')

            plt.subplot(2, 2, 3)
            plt.ylabel("constraints (>1)")
            plt.plot(max_constraints, label='worst constraint')
            plt.plot(new_constraints, label='added constraint')
            plt.legend()
            plt.grid()

            plt.subplot(2, 2, 4)
            plt.ylabel("constraints (>1)")
            plt.plot(max_constraints, label='worst constraint')
            plt.plot(new_constraints, label='added constraint')
            plt.legend()
            plt.grid()
            plt.yscale('log')

            plt.show()

    def assert_thetas_prime(self, thetas_primeo, thetas_hat_max, kappas_prime, l, epsilon=10**-5):
        """ some assertions on thetas_primeo
        """
        self_nb_positions = kappas_prime.shape[0]

        # thetas_primeo[0] = thetas_hat[0]
        if np.abs(thetas_primeo[0] - thetas_hat_max) > epsilon:
            warnings.warn(f'thetas_primeo[0] not equal to thetas_hato[0]: {thetas_primeo}', stacklevel=2)
        # thetas_primeo[min(l-1, L-1)] = thetas_primeo[l]
        if np.abs(thetas_primeo[min(l - 1, self_nb_positions - 1)] - thetas_primeo[l]) > 10 ** -5:
            warnings.warn(
                f'thetas_primeo[l] not equal to thetas_primeo[min(l-1, L-1)]: {thetas_primeo} (with l = {l} and min(l-1, L-1) = {min(l - 1, self_nb_positions - 1)})', stacklevel=2)
        # thetas_primeo is in [0, 1]
        if np.any(thetas_primeo < 0):
            warnings.warn(f'thetas_primeo not positive: {thetas_primeo}', stacklevel=2)
        if np.any(thetas_primeo > 1):
            warnings.warn(f'thetas_primeo not smaller than 1: {thetas_primeo}', stacklevel=2)
        # thetas_primeo[:L] is decreasing
        if np.any(np.diff(thetas_primeo[:self_nb_positions]) > epsilon):
            warnings.warn(f'thetas_primeo not decreasing: {thetas_primeo}', stacklevel=2)
        # thetas_primeo[L-1] >= thetas_primeo[L:]
        if np.any(thetas_primeo[self_nb_positions - 1] + epsilon < thetas_primeo[self_nb_positions:]):
            warnings.warn(f'thetas_primeo not decreasing: {thetas_primeo}', stacklevel=2)
        # some assertions regarding corresponding kappas_prime
        # kappas_prime[0] is 1
        if np.abs(kappas_prime[0] - 1.) > epsilon:
            warnings.warn(f'kappas_prime[0] not equal to 1: {kappas_prime} (thetas_primeo: {thetas_primeo})', stacklevel=2)
        # kappas_prime is in [0, 1]
        if np.any(kappas_prime < 0):
            warnings.warn(f'kappas_prime not positive: {kappas_prime} (thetas_primeo: {thetas_primeo})', stacklevel=2)
        if np.any(kappas_prime > 1):
            warnings.warn(f'kappas_prime not smaller than 1: {kappas_prime} (thetas_primeo: {thetas_primeo})', stacklevel=2)
        # kappas_prime is decreasing
        if np.any(np.diff(kappas_prime) > 10 ** -5):
            warnings.warn(f'kappas_prime not decreasing: {kappas_prime} (thetas_primeo: {thetas_primeo})', stacklevel=2)

    def get_initial_thetas_primeo(self, l=1):
        """
        return a vector thetas_primeo
        * thetas_primeo[:L] is non-increasing
        * thetas_primeo[L:] is smaller than thetas_primeo[L-1]
        * thetas_primeo[l] = thetas_primeo[min(l-1, L-1)]
        * corresponding kappas_prime is non-increasing
        * thetas_prime[i] * kappas_prime[i] = thetas_hat[i] * kappas_hat[i]
        * kappas_prime[0] = 1

        Start from thetas_primeo = thetas_hato and clip it baised on these constraints which sum_up to
        * thetas_primeo[0] =  thetas_hat[0]
        * thetas_primeo[l] = thetas_primeo[min(l-1, L-1)]
        * for i in 1:L, i \neq l
            thetas_primeo[i] is in [ thetas_primeo[i-1] * thetas_hato[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1]),
                                     thetas_primeo[i-1] ]
        * thetas_primeo[L:] is smaller than thetas_primeo[L-1]
        We enforce these constraints for i from 1 to K-1 (except i = l)

        Remark: these is a clipping, not a projection
        """
        # consider thetas in the right order
        order_thetas_hat = np.argsort(self.thetas_hat)[::-1]
        thetas_hato = self.thetas_hat[order_thetas_hat]

        # proposed vector
        thetas_primeo = thetas_hato.copy()
        # thetas_prime[:L] is non-increasing & thetas_prime[l] = thetas_prime[l - 1]
        for i in range(1, self.nb_positions):
            if i == l:
                thetas_primeo[i] = thetas_primeo[i - 1]
            else:
                a = thetas_primeo[i - 1] * thetas_hato[i] * self.kappas_hat[i] / (
                            thetas_hato[i - 1] * self.kappas_hat[i - 1])
                b = thetas_primeo[i - 1]
                thetas_primeo[i] = np.clip(thetas_primeo[i], a, b)
        # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
        for i in range(self.nb_positions, self.nb_arms):
            if i == l:
                thetas_primeo[i] = thetas_primeo[self.nb_positions - 1]
            else:
                a = 0
                b = thetas_primeo[self.nb_positions - 1]
                thetas_primeo[i] = np.clip(thetas_primeo[i], a, b)

        return thetas_primeo

    def optimize_constraint_with_increasing_num_steps(self, qo, log10_num_steps_max=3, epsilon=10**-7, verbose=True):
        for num_steps in 10**np.arange(2, log10_num_steps_max+1):
            res = self.optimize_constraint(qo, num_steps=num_steps)
            if res['val'] < 1.-epsilon:
                break
        if verbose:
            print(f'constraint found after {num_steps} iterations')
            exit(0)
        return res

    @tf.function
    def projection_theta_primeo(self, l):
        """
        By clipping, enforces constraints on thetas_prime and kappas_prime
        * thetas_prime[:L] is non-increasing
        * thetas_prime[L:] is smaller than thetas_prime[L-1]
        * thetas_prime[l] = thetas_prime[min(l-1, L-1)]
        * corresponding kappas_prime is non-increasing
        * thetas_prime[i] * kappas_prime[i] = thetas_hat[i] * kappas_hat[i]
        * kappas_prime[0] = 1

        which sum_up to
        * (not changed by the optimization) thetas_prime[0] = thetas_hat[0]
        * (not changed by the optimization) thetas_prime[l] = thetas_prime[min(l-1, L-1)]
        * for i in 1:L, i \neq l
            thetas_prime[i] is in [ thetas_prime[i-1] * thetas_hat[i] * kappas_hat[i] / (thetas_hat[i-1] * kappas_hat[i-1]),
                                     thetas_prime[i-1] ]
        * thetas_prime[L:] is smaller than thetas_prime[L-1]

        We enforce these constraints for i from 1 to K-1 (except i = l)

        Remark: these is a clipping, not a projection
        """
        print(f'Tracing projection_theta_primeo({l})')
        if l < self.nb_positions:
            b = self.thetas_hato[0]
            for i_pos in range(1, l):
                i_pos_minus0andl = i_pos - 1
                a = b * self.thetas_hato_kappas_hat[i_pos] / self.thetas_hato_kappas_hat[i_pos-1]
                b = tf.clip_by_value(self.thetas_primeo_minus0andl[i_pos_minus0andl], clip_value_min=a, clip_value_max=b)
                self.thetas_primeo_minus0andl[i_pos_minus0andl].assign(b)
            for i_pos in range(l+1, self.nb_positions):
                i_pos_minus0andl = i_pos - 2
                a = b * self.thetas_hato_kappas_hat[i_pos] / self.thetas_hato_kappas_hat[i_pos-1]
                b = tf.clip_by_value(self.thetas_primeo_minus0andl[i_pos_minus0andl], clip_value_min=a, clip_value_max=b)
                self.thetas_primeo_minus0andl[i_pos_minus0andl].assign(b)
            # thetas_prime[L:] is smaller than thetas_prime[L-1]
            self.thetas_primeo_minus0andl[(self.nb_positions - 2):].assign(tf.clip_by_value(self.thetas_primeo_minus0andl[(self.nb_positions - 2):], clip_value_min=0., clip_value_max=b))
        else:
            # thetas_prime[:L] is non-increasing
            b = self.thetas_hato[0]
            for i_pos in range(1, self.nb_positions):
                i_pos_minus0andl = i_pos - 1
                a = b * self.thetas_hato_kappas_hat[i_pos] / self.thetas_hato_kappas_hat[i_pos-1]
                b = tf.clip_by_value(self.thetas_primeo_minus0andl[i_pos_minus0andl], clip_value_min=a, clip_value_max=b)
                self.thetas_primeo_minus0andl[i_pos_minus0andl].assign(b)
            # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
            self.thetas_primeo_minus0andl[(self.nb_positions-1):].assign(tf.clip_by_value(self.thetas_primeo_minus0andl[(self.nb_positions-1):], clip_value_min=0., clip_value_max=b))

    @tf.function
    def get_thetas_prime(self, l):
        print(f'Tracing get_thetas_prime({l})')
        if l == 1:
            return tf.concat((self.thetas_hato[0:1],
                              self.thetas_hato[0:1],
                              self.thetas_primeo_minus0andl),
                             0)
        else:
            i_previous = min(self.nb_positions - 1, l - 1)
            return tf.concat((self.thetas_hato[0:1],
                              self.thetas_primeo_minus0andl[:(l-1)],
                              self.thetas_primeo_minus0andl[(i_previous - 1):i_previous],
                              self.thetas_primeo_minus0andl[(l-1):]),
                             0)

    @tf.function
    def loss_fn_theta_prime(self, l):
        """ loss function used to get worst constraint when optimizing q.

        rem: diagonal terms leads to a constant cost as kappas_prime = thetas_hat * kappas_hat / thetas_prime; we remove them
        """
        # TODO: Vérifier formule et notemment ordonner quoi parmis thetas_prime, kappas.thetas, mus_prime...
        print(f'Tracing loss_fn_theta_prime({l})')

        thetas_primeo = self.get_thetas_prime(l)
        kappas_primeo = self.thetas_hato_kappas_hat / thetas_primeo[:self.nb_positions]
        mus_primeo = tf.clip_by_value(tf.matmul(tf.reshape(thetas_primeo, (-1, 1)),
                                                tf.reshape(kappas_primeo, (1, -1))),
                                      clip_value_min=self.min_p, clip_value_max=self.max_p)
        return tf.reduce_sum(self.keep_value * self.qo_tf *
                             (self.mus_hato * tf.math.log(self.mus_hato / mus_primeo)
                              + self.minus_mus_hato * tf.math.log(self.minus_mus_hato / (1. - mus_primeo))))

    def optimize_constraint(self, qo, num_steps=10000, verbose=False, plot=True, epsilon=10**-14):
        if verbose:
            print('theta_hat:', self.thetas_hat)
            print('kappas_hat:', self.kappas_hat)

        # WARNING: thetas_xxx, etas_xxx, and q are manipulated given the order indicated by thetas_hat
        order_thetas_hat = np.argsort(self.thetas_hat)[::-1]
        thetas_hato = self.thetas_hat[order_thetas_hat]

        # --- constant parameters of loss function ---
        self.thetas_hato.assign(thetas_hato)
        self.thetas_hato_kappas_hat.assign(thetas_hato[:self.nb_positions] * self.kappas_hat)
        self.qo_tf.assign(qo[:, :self.nb_positions])
        self.min_p.assign(epsilon)
        self.max_p.assign(1. - epsilon)
        self.mus_hato.assign(np.clip(self.mus_hat[order_thetas_hat, :], a_min=epsilon, a_max=1. - epsilon))
        self.minus_mus_hato.assign(1. - self.mus_hato)

        best_constraint_value = np.inf
        for l in range(1, self.nb_arms):
            # --- variables ---
            # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
            # so there is only `self.nb_arms-2` variables
            if verbose:
                print(f'l = {l}')
            # --- init variables of the loss function ---
            self.thetas_primeo_minus0andl.assign(np.concatenate((thetas_hato[1:l], thetas_hato[(l + 1):]), 0))
            self.projection_theta_primeo(l)

            if verbose:
                thetas_primeo = self.get_thetas_prime(l).numpy()
                kappas_prime = thetas_hato[:self.nb_positions] * self.kappas_hat / thetas_primeo[:self.nb_positions]
                print('before optimization')
                print('theta_primeo:', thetas_primeo)
                print('kappas_prime:', kappas_prime)
                print(f'loss {self.loss_fn_theta_prime(l).numpy()}')
            if np.isnan(self.loss_fn_theta_prime(l).numpy()):
                thetas_prime = self.get_thetas_prime(l).numpy()
                kappas_prime = self.thetas_hato_kappas_hat / thetas_prime[:self.nb_positions]
                mus_prime = tf.clip_by_value(tf.matmul(tf.reshape(thetas_prime, [-1, 1]),
                                                       tf.reshape(kappas_prime, [1, -1])),
                                             clip_value_min=epsilon, clip_value_max=1. - epsilon)
                print('theta_prime', thetas_prime.numpy())
                print('kappas_prime', kappas_prime.numpy())
                print('mus_prime', mus_prime.numpy())
                print('mus_hat', self.mus_hato.numpy())
                print('log(mus_hat / mus_prime)', tf.math.log(self.mus_hato / mus_prime).numpy())
                print('1. - mus_prime', (1.-mus_prime).numpy())
                print('1. - mus_hat', self.minus_mus_hato.numpy())
                print('log(minus_mus_hat / (1 - mus_prime))', tf.math.log(self.minus_mus_hato / (1 - mus_prime)).numpy())
                raise ValueError('loss function = NaN (before optimization)')
            # TODO: switch to stopping criterion
            if plot:
                trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': self.get_thetas_prime(l)}
            else:
                trace_fn = lambda loss, grads, variables: {}
            trace = tfprg.minimize(lambda: self.loss_fn_theta_prime(l), num_steps=num_steps, trainable_variables=(self.thetas_primeo_minus0andl,),
                                   optimizer=tf.optimizers.Adam(0.001), projection=lambda: self.projection_theta_primeo(l),
                                   trace_fn=trace_fn)
            thetas_primeo = self.get_thetas_prime(l).numpy()
            kappas_prime = thetas_hato[:self.nb_positions] * self.kappas_hat / thetas_primeo[:self.nb_positions]

            if verbose:
                print('after optimization')
                print('theta_primeo:', thetas_primeo)
                print('kappas_prime:', kappas_prime)
                print(f'loss {self.loss_fn_theta_prime(l).numpy()}')
            if np.isnan(self.loss_fn_theta_prime(l).numpy()):
                thetas_prime = self.get_thetas_prime(l).numpy()
                kappas_prime = self.thetas_hato_kappas_hat / thetas_prime[:self.nb_positions]
                mus_prime = tf.clip_by_value(tf.matmul(tf.reshape(thetas_prime, [-1, 1]),
                                                       tf.reshape(kappas_prime, [1, -1])),
                                             clip_value_min=epsilon, clip_value_max=1. - epsilon)
                print('theta_prime', thetas_prime.numpy())
                print('kappas_prime', kappas_prime.numpy())
                print('mus_prime', mus_prime.numpy())
                print('mus_hat', self.mus_hato.numpy())
                print('log(mus_hat / mus_prime)', tf.math.log(self.mus_hato / mus_prime).numpy())
                print('1. - mus_prime', (1.-mus_prime).numpy())
                print('1. - mus_hat', self.minus_mus_hato.numpy())
                print('log(minus_mus_hat / (1 - mus_prime))', tf.math.log(self.minus_mus_hato / (1 - mus_prime)).numpy())
                raise ValueError('loss function = NaN (after optimization)')

            if plot:
                import matplotlib.pyplot as plt
                plt.subplot(2, 2, 1)
                plt.ylabel("f(x)")
                plt.plot(trace['loss'], label='loss')
                plt.legend()
                plt.grid()

                plt.subplot(2, 2, 2)
                plt.ylabel("f(x)")
                plt.plot(trace['loss'], label='loss')
                plt.legend()
                plt.grid()
                plt.yscale('log')

                plt.subplot(2, 2, 3)
                plt.ylabel("thetas")
                plt.plot(trace['theta'][:, 0], ':', color=f'C{1}')
                plt.axhline(y=thetas_hato[0], color=f'C{1}')
                plt.plot(trace['theta'][:, l - 1], ':', color=f'C{2}')
                plt.axhline(y=thetas_hato[l - 1], color=f'C{2}')
                plt.axhline(y=thetas_hato[l], color=f'C{2}')
                plt.grid()
                plt.ylim([-0.05, 1.05])

                plt.show()

            # some assertions on thetas_primeo
#            for node in self.projection_theta_primeo.graph.as_graph_def().node:
#                print(f'{node.input} -> {node.name}')
            self.assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)
            """
            print(thetas_hato[0], l)
            print(thetas_primeo)
            print(thetas_hato)
            tf.print(self.thetas_hato)
            self.projection_theta_primeo()
            print(thetas_primeo)
            print(thetas_hato)
            thetas_primeo = self.get_thetas_prime(l).numpy().reshape(-1)
            kappas_prime = thetas_hato[:self.nb_positions] * self.kappas_hat / thetas_primeo[:self.nb_positions]
            self.assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)
            exit(0)
            """


            current_constraint_value = self.loss_fn_theta_prime(l).numpy()
            if current_constraint_value < best_constraint_value:
                best_constraint_value = current_constraint_value
                best_thetas_primeo = self.get_thetas_prime(l).numpy()

        print({'x': best_thetas_primeo, 'val': best_constraint_value})
        return {'x': best_thetas_primeo, 'val': best_constraint_value}

    def add_flat_exploration(self):
        for perm in self.basic_permutations:
            for l, i in enumerate(perm):
                if self.nb_prints[i, l] < self.alpha * sqrt(log(self.nb_trials)):
                    self.l_current.add(perm)
                    break

    def add_optimal_exploration(self, verbose=False):
        """
        ...

        Examples
        --------
        >>> import numpy as np
        >>> player = PMED(5, 3, 1.)
        >>> player.nb_trials = np.exp(1.)
        >>> player.q = np.array([[3., 0., 1., 2., 0.],
        ...        [0., 3., 0., 0., 3.],
        ...        [0., 2., 0., 4., 0.],
        ...        [2., 1., 3., 0., 0.],
        ...        [1., 0., 2., 0., 3.]])
        >>> player.decompose_nb_prints_tilde()
        [(3.0, (0, 1, 3)), (2.0, (3, 2, 4)), (1.0, (4, 3, 0))]
        >>> player.l_current = {(0, 1, 2)}
        >>> player.nb_prints = np.array([[2., 0., 1.],
        ...        [0., 3., 0.],
        ...        [0., 2., 0.],
        ...        [1., 1., 3.],
        ...        [1., 0., 2.]])
        >>> player.add_optimal_exploration(verbose=False)
        >>> player.l_current
        {(0, 1, 2), (3, 2, 4), (0, 1, 3)}
        """
        decomposition = self.decompose_nb_prints_tilde()
        r = np.array(self.nb_prints.copy(), dtype=np.float)
        if verbose:
            print('r initial:', r)
            print('required exploration:', np.array(self.q * np.log(self.nb_trials), dtype=np.uint)[:,:self.nb_positions])
        for c_req, perm in decomposition:
            l_min = np.argmin(r[perm, self.positions])
            i_min = perm[l_min]
            c_aff = r[i_min, l_min]     # max_c {c>0: ....}
            if verbose:
                print(perm, c_req, c_aff)
            if c_aff < c_req:
                for l, vl in enumerate(perm):
                    found = False
                    for p in self.l_current:
                        if p[l] == vl:
                            found = True
                            break
                    if not found:
                        if verbose:
                            print('add')
                        self.l_current.add(perm)
                        break
            r[perm, self.positions] -= min(c_aff, c_req)
            if verbose:
                print('r:', r)

    def decompose_nb_prints_tilde(self, epsilon=0.5, verbose=False):
        """
        Compute decomposition of the K x K matrix, and return the corresponding (weighted) L-permutations

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(precision=3)

        >>> player = PMED(5, 3, 1.)
        >>> player.nb_trials = np.exp(1.)
        >>> player.q = np.array([[3., 0., 1., 2., 0.],
        ...        [0., 3., 0., 0., 3.],
        ...        [0., 2., 0., 4., 0.],
        ...        [2., 1., 3., 0., 0.],
        ...        [1., 0., 2., 0., 3.]])
        >>> player.decompose_nb_prints_tilde()
        [(3.0, array([0, 1, 3])), (2.0, array([3, 2, 4])), (1.0, array([4, 3, 0]))]

        >>> target = np.array([[5.04, 9.04, 1.04, 6.04, 3.84],
        ...        [3.24, 8.24, 2.24, 5.24, 6.04],
        ...        [3.84, 2.84, 9.84, 4.84, 3.64],
        ...        [6.04, 4.04, 5.04, 4.04, 5.84],
        ...        [6.84, 0.84, 6.84, 4.84, 5.64]])
        >>> player.q = target
        >>> decomposition = player.decompose_nb_prints_tilde()
        >>> np.any([c>0 for c, _ in decomposition])
        True
        >>> recomposition = np.zeros((5,3))
        >>> for c, perm in decomposition:
        ...     recomposition[perm, np.arange(3)] += c
        >>> np.any((recomposition - target[:,:3]) <= 0.5)
        True
        >>> recomposition
        array([[4.84, 8.88, 1.04],
               [3.24, 8.24, 1.96],
               [3.84, 2.84, 9.76],
               [6.04, 3.84, 5.04],
               [6.68, 0.84, 6.84]])
        """
        n_tilde = self.q * np.log(self.nb_trials)
        from collections import defaultdict
        res = defaultdict(lambda: 0)
        for _ in range(self.nb_arms**2):
            if verbose:
                print('N_tilde', n_tilde)
                #print('int(N_tilde)', np.array(n_tilde, dtype=np.int))
                print('bipartite-graph', np.array(n_tilde > epsilon, dtype=np.int))
                #print('max(N_tilde)', np.max(n_tilde))
                #print('min(N_tilde != 0)', np.min(n_tilde[n_tilde > epsilon]))
                print('nb(N_tilde > epsilon)', np.sum(n_tilde > epsilon))
            try:
                _, perm = linear_sum_assignment(np.array(n_tilde > epsilon, dtype=np.int).T, maximize=True)
            except TypeError as e:
                _, perm = linear_sum_assignment(-np.array(n_tilde > epsilon, dtype=np.int).T)
            c = np.min(n_tilde[perm, self.pseudo_positions])
            n_tilde[perm, self.pseudo_positions] -= c
            res[tuple(perm[:self.nb_positions])] += c
            if verbose:
                print(f'n_tilde[perm, [0,..,L-1]]: {n_tilde[perm, self.pseudo_positions]}')
                print(f'remove {c} * {perm}')
                print(f'max(n_tilde): {np.max(n_tilde)}')
            if np.max(n_tilde) < epsilon:
                # --- precision reached => terminate ---
                break

        if np.max(n_tilde) >= epsilon:
            # --- decomposition with too much terms ---
            warnings.warn(f'decompose_nb_prints_tilde() terminated due to nb_iter_max = {self.nb_arms**2}' +
                          f' with max(N_tilde) = {np.max(n_tilde)} and N_tilde =\n{n_tilde}')
            print(f'warning: decompose_nb_prints_tilde() terminated due to nb_iter_max = {self.nb_arms**2}' +
                  f' with max(N_tilde) = {np.max(n_tilde)} and N_tilde =\n{n_tilde}')
        return sorted([(weight, perm) for perm, weight in res.items()], key=lambda x: x[0], reverse=True)

    def type(self):
        return 'PMED'


if __name__ == "__main__":
    import doctest

    doctest.testmod()
