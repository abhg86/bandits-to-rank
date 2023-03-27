#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

import tensorflow as tf
import numpy as np
import scipy.optimize
import scipy.special
import scipy.sparse
import time
from tensorflow.python.ops.gen_array_ops import matrix_diag_v2  # waiting for a debug of tf.linalg.diag


import bandits_to_rank.tools.tfp_math_minimize as tfprg

def dKL_Bernoulli(P, Q, N=None):
    if N is None:
        N = np.ones(P.shape)
    res = N*scipy.special.rel_entr(P, Q) + N*scipy.special.rel_entr(1.-P, 1.-Q)
    print('dKL:', res)
    print("p:", P)
    print('q:', Q)
    return res


def assert_thetas_and_kappas(thetas, kappas):
    """ some assertions on thetas and kappas
    """
    if np.any(thetas < 0):
        raise Warning(f'thetas not positive: {thetas} (kappas: {kappas})')
    if np.any(thetas > 1):
        raise Warning(f'thetas not smaller than 1: {thetas} (kappas: {kappas})')
    # kappas[0] is 1
    if np.abs(kappas[0] - 1.) > 10 ** -5:
        raise Warning(f'kappas[0] not equal to 1: {kappas} (thetas: {thetas})')
    # kappas is in [0, 1]
    if np.any(kappas < 0):
        raise Warning(f'kappas not positive: {kappas} (thetas: {thetas})')
    if np.any(kappas > 1):
        raise Warning(f'kappas not smaller than 1: {kappas} (thetas: {thetas})')
    # kappas is decreasing
    if np.any(np.diff(kappas) > 10 ** -5):
        raise Warning(f'kappas not decreasing: {kappas} (thetas: {thetas})')


def assert_thetas_prime(thetas_primeo, thetas_hat_max, kappas_prime, l):
        """ some assertions on thetas_primeo
        """
        nb_positions = kappas_prime.shape[0]

        # thetas_primeo[0] = thetas_hat[0]
        if np.abs(thetas_primeo[0] - thetas_hat_max) > 10 ** -5:
            raise Warning(f'thetas_primeo[0] not equal to thetas_hato[0]: {thetas_primeo}')
        # thetas_primeo[min(l-1, L-1)] = thetas_primeo[l]
        if np.abs(thetas_primeo[min(l - 1, nb_positions - 1)] - thetas_primeo[l]) > 10 ** -5:
            raise Warning(
                f'thetas_primeo[l] not equal to thetas_primeo[min(l-1, L-1)]: {thetas_primeo} (with l = {l} and min(l-1, L-1) = {min(l - 1, nb_positions - 1)})')
        # thetas_primeo is in [0, 1]
        if np.any(thetas_primeo < 0):
            raise Warning(f'thetas_primeo not positive: {thetas_primeo}')
        if np.any(thetas_primeo > 1):
            raise Warning(f'thetas_primeo not smaller than 1: {thetas_primeo}')
        # thetas_primeo[:L] is decreasing
        if np.any(np.diff(thetas_primeo[:nb_positions]) > 0):
            raise Warning(f'thetas_primeo not decreasing: {thetas_primeo}')
        # thetas_primeo[L-1] >= thetas_primeo[L:]
        if np.any(thetas_primeo[nb_positions - 1] < thetas_primeo[nb_positions:]):
            raise Warning(f'thetas_primeo not decreasing: {thetas_primeo}')
        # some assertions regarding corresponding kappas_prime
        # kappas_prime[0] is 1
        if np.abs(kappas_prime[0] - 1.) > 10 ** -5:
            raise Warning(f'kappas_prime[0] not equal to 1: {kappas_prime} (thetas_primeo: {thetas_primeo})')
        # kappas_prime is in [0, 1]
        if np.any(kappas_prime < 0):
            raise Warning(f'kappas_prime not positive: {kappas_prime} (thetas_primeo: {thetas_primeo})')
        if np.any(kappas_prime > 1):
            raise Warning(f'kappas_prime not smaller than 1: {kappas_prime} (thetas_primeo: {thetas_primeo})')
        # kappas_prime is decreasing
        if np.any(np.diff(kappas_prime) > 10 ** -5):
            raise Warning(f'kappas_prime not decreasing: {kappas_prime} (thetas_primeo: {thetas_primeo})')


class Rank1_factorization():
    def __init__(self, nb_arms=10, nb_positions=5):
        self.rng = np.random.default_rng()
        self.nb_arms = nb_arms
        self.nb_positions = nb_positions
        self.init_data()

    def _small_noise(self, a, noise=0.1, epsilon=10**-14):
        res = a * (1. + (self.rng.random(a.shape) - 0.5) * noise)
        res[res > 1] = 1. - epsilon
        res[res < 0] = epsilon
        return res

    def init_data(self, verbose=True, noise=0.1):
        self.thetas_star = self.rng.random(self.nb_arms)
        self.thetas_star[0] = 0.01 * self.rng.random()  # to be resilient when theta close to zero
        self.thetas_star[1] = 1. - 0.01 * self.rng.random()  # to be resilient when theta close to one
        self.kappas_star = np.sort(self.rng.random(self.nb_positions))[::-1]
        self.kappas_star[0] = 1.
        self.mus_star = self.thetas_star.reshape((-1, 1)) @ self.kappas_star.reshape((1, -1))
        self.mus_hat = self._small_noise(self.mus_star, noise=noise)
        self.N = self.rng.integers(1, 10, (self.nb_arms, self.nb_positions))
        if verbose:
            self.print_data()
        assert_thetas_and_kappas(self.thetas_star, self.kappas_star)

    def print_data(self):
        print('theta_star:', self.thetas_star)
        print('kappas_star:', self.kappas_star)
        print('mus_star:', self.mus_star)
        print('mus_hat:', self.mus_hat)
        print('N:', self.N)

    def optimize_PBM(self, metric="MSE", num_steps=10000, plot=True, verbose=True, start_close=False, noise=0.1, epsilon=10**-10):
        mus_hat = tf.constant(self.mus_hat)

        if start_close:
            thetas_hat = tf.Variable(self._small_noise(self.thetas_star, noise=noise))
            kappas_hat_var = tf.Variable(self._small_noise(self.kappas_star[1:], noise=noise))
        else:
            thetas_hat = tf.Variable(self.rng.random(self.nb_arms)*(1-2*epsilon) + epsilon)
            kappas_hat_var = tf.Variable(np.sort(self.rng.random(self.nb_positions-1)*(1-2*epsilon) + epsilon)[::-1]) # kappas is expected to be sorted, with kappas[0] = 1

        #tf.funtcion
        def get_kappas_hat():
            return tf.concat([tf.constant([1.], dtype='float64'), kappas_hat_var], 0)

        if metric == 'MSE':
            def loss_fn():
                mus_hathat = tf.matmul(tf.reshape(thetas_hat, [-1, 1]), tf.reshape(get_kappas_hat(), [1, -1]))
                return tf.reduce_sum(tf.math.squared_difference(mus_hat, mus_hathat)*self.N)
        elif metric == 'KL':
            kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
            def loss_fn():
                mus_hathat = tf.matmul(tf.reshape(thetas_hat, [-1, 1]), tf.reshape(get_kappas_hat(), [1, -1]))
                p_hat = tf.concat((tf.reshape(mus_hat, [-1, 1]), tf.reshape(1.-mus_hat, [-1, 1])), 1)
                p_hathat = tf.concat((tf.reshape(mus_hathat, [-1, 1]), tf.reshape(1.-mus_hathat, [-1, 1])), 1)
                return tf.reduce_sum(tf.reduce_sum(p_hat * tf.math.log(p_hat / p_hathat), axis=1) * self.N.reshape([-1]))
                #return kld(p_hat, p_hathat, sample_weight=self.N.reshape([-1]))    # kl clipped by tf
        else:
            raise ValueError(f'unknown metric "{metric}"')

        @tf.function
        def projection():
            """
            By clipping, enforces constraints on thetas_hat and kappas_hat
            * kappas_hat is non-increasing
            * thetas_hat and kappas_hat[:L-1] are in [epsilon, 1.-epsilon]

            We enforce the "non-increasing" constraints for i from 1 to L-1

            Remark: these is a clipping, not a projection
            """
            # kappas_hat is non-increasing and smaller than 1.-epsilon
            new_kappas_hat1 = 1.-epsilon
            for i in range(0, self.nb_positions-1):
                new_kappas_hat1 = tf.clip_by_value(kappas_hat_var[i], clip_value_min=0., clip_value_max=new_kappas_hat1)
                kappas_hat_var[i].assign(new_kappas_hat1)
            # kappas_hat is in [epsilon, 1.-epsilon]
            kappas_hat_var.assign(tf.clip_by_value(kappas_hat_var, clip_value_min=epsilon, clip_value_max=1. -epsilon))
            # thetas_hat is in [epsilon, 1.-epsilon]
            thetas_hat.assign(tf.clip_by_value(thetas_hat, clip_value_min=epsilon, clip_value_max=1. -epsilon))

        if verbose:
            print(f'begin: loss {loss_fn().numpy()}, for thetas = {thetas_hat.numpy()}, and kappas = {get_kappas_hat().numpy()}')
            mus_hathat_np = thetas_hat.numpy().reshape((-1, 1)) @ get_kappas_hat().numpy().reshape((1, -1))
            mse_np = float(np.sum(self.N * (self.mus_hat - mus_hathat_np) ** 2))
            kl_np = float(np.sum(self.N * (self.mus_hat * np.log(self.mus_hat / mus_hathat_np)
                                           + (1 - self.mus_hat) * np.log((1 - self.mus_hat) / (1 - mus_hathat_np)))))
            mus_hat_cliped = np.clip(self.mus_hat, 10**-7, 1.)
            mus_hathat_cliped = np.clip(mus_hathat_np, 10**-7, 1.)
            kl_np_clipped = float(np.sum(self.N * (mus_hat_cliped * np.log(mus_hat_cliped / mus_hathat_cliped)
                                           + (1 - mus_hat_cliped) * np.log((1 - mus_hat_cliped) / (1 - mus_hathat_cliped)))))
            print(f'given np, MSE = {mse_np}     KL = {kl_np}     clipped KL = {kl_np_clipped}')
        trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': thetas_hat, 'kappa': get_kappas_hat()}
        trace = tfprg.minimize(loss_fn, num_steps=num_steps, trainable_variables=None,#[thetas_hat],
                                optimizer=tf.optimizers.Adam(0.001), projection=projection,
                                trace_fn=trace_fn)
        if verbose:
            print(f'end:   loss {loss_fn().numpy()}, for thetas = {thetas_hat.numpy()}, and kappas = {get_kappas_hat().numpy()}')
            mus_hathat_np = thetas_hat.numpy().reshape((-1, 1)) @ get_kappas_hat().numpy().reshape((1, -1))
            mse_np = float(np.sum(self.N * (self.mus_hat - mus_hathat_np) ** 2))
            kl_np = float(np.sum(self.N * (self.mus_hat * np.log(self.mus_hat / mus_hathat_np)
                                           + (1 - self.mus_hat) * np.log((1 - self.mus_hat) / (1 - mus_hathat_np)))))
            mus_hat_cliped = np.clip(self.mus_hat, 10**-7, 1.)
            mus_hathat_cliped = np.clip(mus_hathat_np, 10**-7, 1.)
            kl_np_clipped = float(np.sum(self.N * (mus_hat_cliped * np.log(mus_hat_cliped / mus_hathat_cliped)
                                           + (1 - mus_hat_cliped) * np.log((1 - mus_hat_cliped) / (1 - mus_hathat_cliped)))))
            print(f'given np, MSE = {mse_np}     KL = {kl_np}     clipped KL = {kl_np_clipped}')

        self.thetas_hat = thetas_hat.numpy()
        self.kappas_hat = get_kappas_hat().numpy()

        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(2, 2, 1)
            plt.ylabel("f(x)")
            plt.plot(trace['loss'])
            plt.grid()

            plt.subplot(2, 2, 2)
            plt.ylabel("f(x)")
            plt.plot(trace['loss'])
            plt.grid()
            plt.yscale('log')

            plt.subplot(2, 2, 3)
            plt.ylabel("thetas[0]")
            for i in range(4):
                plt.plot(trace['theta'][:, i], ':', color=f'C{i+1}', label='')
                plt.axhline(y=self.thetas_star[i], color=f'C{i+1}', label=f'$\\theta_{i}$')
            plt.grid()
            plt.ylim([-0.05, 1.05])
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.ylabel("kappas[0]")
            for i in range(4):
                plt.plot(trace['kappa'][:, i], ':', color=f'C{i+1}', label='')
                plt.axhline(y=self.kappas_star[i], color=f'C{i+1}', label=f'$\\theta_{i}$')
            plt.grid()
            plt.ylim([-0.05, 1.05])
            plt.legend()

            plt.show()

        assert_thetas_and_kappas(self.thetas_hat, self.kappas_hat)

    def assert_thetas_prime(thetas_primeo, thetas_hat_max, kappas_prime, l):
        """ some assertions on thetas_primeo
        """
        self_nb_positions = kappas_prime.shape[0]

        # thetas_primeo[0] = thetas_hat[0]
        if np.abs(thetas_primeo[0] - thetas_hat_max) > 10 ** -5:
            raise Warning(f'thetas_primeo[0] not equal to thetas_hato[0]: {thetas_primeo}')
        # thetas_primeo[min(l-1, L-1)] = thetas_primeo[l]
        if np.abs(thetas_primeo[min(l - 1, self_nb_positions - 1)] - thetas_primeo[l]) > 10 ** -5:
            raise Warning(
                f'thetas_primeo[l] not equal to thetas_primeo[min(l-1, L-1)]: {thetas_primeo} (with l = {l} and min(l-1, L-1) = {min(l - 1, self_nb_positions - 1)})')
        # thetas_primeo is in [0, 1]
        if np.any(thetas_primeo < 0):
            raise Warning(f'thetas_primeo not positive: {thetas_primeo}')
        if np.any(thetas_primeo > 1):
            raise Warning(f'thetas_primeo not smaller than 1: {thetas_primeo}')
        # thetas_primeo[:L] is decreasing
        if np.any(np.diff(thetas_primeo[:self_nb_positions]) > 0):
            raise Warning(f'thetas_primeo not decreasing: {thetas_primeo}')
        # thetas_primeo[L-1] >= thetas_primeo[L:]
        if np.any(thetas_primeo[self_nb_positions - 1] < thetas_primeo[self_nb_positions:]):
            raise Warning(f'thetas_primeo not decreasing: {thetas_primeo}')
        # some assertions regarding corresponding kappas_prime
        # kappas_prime[0] is 1
        if np.abs(kappas_prime[0] - 1.) > 10 ** -5:
            raise Warning(f'kappas_prime[0] not equal to 1: {kappas_prime} (thetas_primeo: {thetas_primeo})')
        # kappas_prime is in [0, 1]
        if np.any(kappas_prime < 0):
            raise Warning(f'kappas_prime not positive: {kappas_prime} (thetas_primeo: {thetas_primeo})')
        if np.any(kappas_prime > 1):
            raise Warning(f'kappas_prime not smaller than 1: {kappas_prime} (thetas_primeo: {thetas_primeo})')
        # kappas_prime is decreasing
        if np.any(np.diff(kappas_prime) > 10 ** -5):
            raise Warning(f'kappas_prime not decreasing: {kappas_prime} (thetas_primeo: {thetas_primeo})')

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

    def optimize_q(self, num_constraints=100, epsilon=10**-5, plot=False, inner_plot=None, verbose=False, inner_verbose=None):
        if np.any(np.diff(self.kappas_hat) > 0):
            raise Warning(f'kappas_hat not decreasing: {self.kappas_hat}')
        if inner_verbose is None:
            inner_verbose = verbose
        if inner_plot is None:
            inner_plot = plot
        if verbose:
            print('theta_hat:', self.thetas_hat)
            print('kappas_hat:', self.kappas_hat)
            sum_delta_qs = []
            max_constraints = []
            new_constraints = [-np.inf]

        # TODO: q eloigné de epsilon du bord

        # WARNING: thetas_xxx, q, and mus_xxx are manipulated given the order indicated by thetas_hat
        order_thetas_hat = np.argsort(self.thetas_hat)[::-1]
        mus_hato = self.mus_hat[order_thetas_hat, :]
        thetas_hato = self.thetas_hat[order_thetas_hat]

        # - init optimization problem -
        # min sum_{i<K, l<L} delta[i,l]q[i,l]
        """ With delta[i,l] = mus_hat[l,l] - mus_hat[i,l]
        # leads to problems when mus_hat is not of rank 1
        delta = np.concatenate((np.ones((self.nb_arms,1)) @ np.diag(mus_hato).reshape((1, -1)) - mus_hato,
                                np.zeros((self.nb_arms, self.nb_arms-self.nb_positions))), axis=1)
        """
        #""" With delta[i,l] = thetas_hat[l]*kappas_hat[l] - thetas_hat[i]*kappas_hat[l]
        delta = np.concatenate((np.ones((self.nb_arms, 1))
                                @ (thetas_hato[:self.nb_positions]*self.kappas_hat).reshape([1, -1])
                                - thetas_hato.reshape([-1, 1]) @ self.kappas_hat.reshape([1, -1]),
                                np.zeros((self.nb_arms, self.nb_arms-self.nb_positions))), axis=1)
        #"""
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
                tmp = self.optimize_constraint_with_increasing_num_steps(res.x.reshape((self.nb_arms, self.nb_arms)), epsilon=epsilon, plot=inner_plot, verbose=inner_verbose)
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
            mus_primeo = thetas_primeo.reshape((-1, 1)) @ kappas_primeo.reshape((1, -1))
            constraints = np.concatenate((-dKL_Bernoulli(mus_hato, mus_primeo),
                                          np.zeros((self.nb_arms, self.nb_arms-self.nb_positions))), axis=1)
            A_ub[s, :] = constraints.flatten()
            A_ub[s, ::(self.nb_arms + 1)] = 0       # remove i == l
            b_ub[s] = -1
            if verbose and s > 0:
                new_constraints.append(-A_ub[s, :] @ res.x)
            # WARNING: matrices are flatten in row-major (C-style) order

            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=None, method='interior-point',
                               callback=None, options=None, x0=None)

            if verbose:
                print("q (ordered):", res.x.reshape((self.nb_arms, self.nb_arms)))
                print("sum delta.q:", res.fun)
                print(f'sum q.d_KL:', A_ub @ res.x)
                print(f' vs up. b.:', b_ub)
                print(f'sum q.d_KL[s=0]:', A_ub[0, :] @ res.x)
                sum_delta_qs.append(res.fun)
                max_constraints.append(-(A_ub[:(s+1), :] @ res.x).max())

            # --- early stopping when the new constraint does not enough increase the score to minimize
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

        return res.x.reshape((self.nb_arms, self.nb_arms))[np.argsort(order_thetas_hat), :]

    def optimize_constraint_with_increasing_num_steps(self, qo, log10_num_steps_max=3, epsilon=10**-7, plot=True, verbose=True):
        for num_steps in 10**np.arange(2, log10_num_steps_max+1):
            res = self.optimize_constraint(qo, num_steps=num_steps, plot=plot, verbose=verbose)
            if res['val'] < 1.-epsilon:
                break
        if verbose:
            print(f'constraint found after {num_steps} iterations')
        return res

    def optimize_constraint(self, qo, num_steps=10000, plot=True, verbose=True):
        if verbose:
            print('theta_hat:', self.thetas_hat)
            print('kappas_hat:', self.kappas_hat)

        # WARNING: thetas_xxx, etas_xxx, and q are manipulated given the order indicated by thetas_hat
        order_thetas_hat = np.argsort(self.thetas_hat)[::-1]
        thetas_hato = self.thetas_hat[order_thetas_hat]

        mus_hat = tf.constant(self.mus_hat[order_thetas_hat, :], dtype='float64')
        thetas_hat = tf.constant(thetas_hato, dtype='float64')
        kappas_hat = tf.constant(self.kappas_hat, dtype='float64')
        q = tf.constant(qo[:, :self.nb_positions], dtype='float64')

        best_constraint_value = np.inf
        for l in range(1, self.nb_arms):
            if verbose:
                print(f'l = {l}')
            # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
            # so there is only `self.nb_arms-2` variables
            initial_thetas_prime = self.get_initial_thetas_primeo(l=l)
            thetas_prime_var = tf.Variable(np.concatenate((initial_thetas_prime[1:l], initial_thetas_prime[(l + 1):]), axis=0), dtype='float64')

            @tf.function
            def get_thetas_prime():
                thetas_prime0 = tf.constant(thetas_hato[0], shape=(1,))
                if l == 1:
                    return tf.concat([thetas_prime0, thetas_prime0, thetas_prime_var], 0)
                else:
                    ll = min(l - 1, self.nb_positions - 1) - 1      # the -1 is due to $\theta_{(0)}'$ being a constant
                    thetas_primel = tf.reshape(thetas_prime_var[ll], [1])
                    return tf.concat([thetas_prime0, thetas_prime_var[:l-1], thetas_primel, thetas_prime_var[l-1:]], 0)

            @tf.function
            def projection():
                """
                By clipping, enforces constraints on thetas_prime and kappas_prime
                * thetas_prime[:L] is non-increasing
                * thetas_prime[L:] is smaller than thetas_prime[L-1]
                * thetas_prime[l] = thetas_prime[min(l-1, L-1)]
                * corresponding kappas_prime is non-increasing
                * thetas_prime[i] * kappas_prime[i] = thetas_hat[i] * kappas_hat[i]
                * kappas_prime[0] = 1

                which sum_up to
                * (already enforced, it is a constant for tf) thetas_prime[0] = thetas_hat[0]
                * (already enforced, it is a copy for tf) thetas_prime[l] = thetas_prime[min(l-1, L-1)]
                * for i in 1:L, i \neq l
                    thetas_prime[i] is in [ thetas_prime[i-1] * thetas_hat[i] * kappas_hat[i] / (thetas_hat[i-1] * kappas_hat[i-1]),
                                             thetas_prime[i-1] ]
                * thetas_prime[L:] is smaller than thetas_prime[L-1]

                We enforce these constraints for i from 1 to K-1 (except i = l)

                Remark: these is a clipping, not a projection
                """
                thetas_prime = get_thetas_prime()
                new_thetas_primei1 = thetas_prime[0]
                # thetas_prime[:L] is non-increasing
                for i in range(1, min(self.nb_positions, l)):
                    a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
                    b = new_thetas_primei1
                    # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
                    new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b)
                    thetas_prime_var[i-1].assign(new_thetas_primei1)
                for i in range(l+1, self.nb_positions):
                    a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
                    b = new_thetas_primei1
                    # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
                    new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b)
                    thetas_prime_var[i-2].assign(new_thetas_primei1)
                # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
                a = tf.constant(0., dtype='float64')
                b = new_thetas_primei1
                for i in range(self.nb_positions, l):
                    # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
                    thetas_prime_var[i-1].assign(tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b))
                for i in range(max(self.nb_positions, l+1), self.nb_arms):
                    # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
                    #   and thetas_prime[l] being a copy of thetas_prime[l-1]
                    thetas_prime_var[i-2].assign(tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b))

            @tf.function
            def loss_fn():
                thetas_prime = get_thetas_prime()
                kappas_prime = thetas_hat[:self.nb_positions] * kappas_hat / thetas_prime[:self.nb_positions]
                # loss
                loss = tf.constant(0, dtype='float64')
                mus_prime = tf.matmul(tf.reshape(thetas_prime, [-1, 1]), tf.reshape(kappas_prime, [1, -1]))
                minus_mus_hat = 1. - mus_hat
                loss += tf.reduce_sum(q * (mus_hat * tf.math.log(mus_hat / mus_prime)
                                          + minus_mus_hat * tf.math.log(minus_mus_hat / (1 - mus_prime))))
                # rem: diagonal terms are constant as kappas_prime = thetas_hat * kappas_hat / thetas_prime; no need to remove them
                return loss

            @tf.function
            def constraint_for_learning_q():
                """similar loss_fn, but removing the diagonal terms of q"""
                thetas_prime = get_thetas_prime()
                kappas_prime = thetas_hat[:self.nb_positions] * kappas_hat / thetas_prime[:self.nb_positions]
                # loss
                loss = tf.constant(0, dtype='float64')
                mus_prime = tf.matmul(tf.reshape(thetas_prime, [-1, 1]), tf.reshape(kappas_prime, [1, -1]))
                minus_mus_hat = 1. - mus_hat
                keep_value = matrix_diag_v2(np.zeros(self.nb_positions), k=0, num_rows=self.nb_arms, num_cols=-1, padding_value=1)
                loss += tf.reduce_sum(keep_value * q * (mus_hat * tf.math.log(mus_hat / mus_prime)
                                          + minus_mus_hat * tf.math.log(minus_mus_hat / (1 - mus_prime))))
                return loss

            if verbose:
                thetas_primeo = get_thetas_prime().numpy()
                kappas_prime = thetas_hato[:self.nb_positions] * self.kappas_hat / thetas_primeo[:self.nb_positions]
                print('before optimization')
                print('theta_primeo:', thetas_primeo)
                print('kappas_prime:', kappas_prime)
                print(f'loss {loss_fn().numpy()}')
                print(f'constraint for learning q {constraint_for_learning_q().numpy()}')

            t0 = time.time()
            trace_fn = lambda loss, grads, variables: {'loss': loss_fn(), 'constraint_for_learning_q':constraint_for_learning_q(), 'theta': get_thetas_prime()}
            trace = tfprg.minimize(loss_fn, num_steps=num_steps, trainable_variables=None,
                                          optimizer=tf.optimizers.Adam(0.001), projection=projection,
                                          trace_fn=trace_fn)
            thetas_primeo = get_thetas_prime().numpy()
            kappas_prime = thetas_hato[:self.nb_positions] * self.kappas_hat / thetas_primeo[:self.nb_positions]

            if verbose:
                print('computing time:', time.time()-t0)
                print('after optimization')
                print('theta_primeo:', thetas_primeo)
                print('kappas_prime:', kappas_prime)
                print(f'loss {loss_fn().numpy()}')
                print(f'constraint for learning q {constraint_for_learning_q().numpy()}')

            if plot:
                import matplotlib.pyplot as plt
                plt.subplot(2, 2, 1)
                plt.ylabel("f(x)")
                plt.plot(trace['loss'], label='loss')
                plt.plot(trace['constraint_for_learning_q'], label='constraint for learning q')
                plt.legend()
                plt.grid()

                plt.subplot(2, 2, 2)
                plt.ylabel("f(x)")
                plt.plot(trace['loss'], label='loss')
                plt.plot(trace['constraint_for_learning_q'], label='constraint for learning q')
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
            #self.assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

            current_constraint_value = constraint_for_learning_q().numpy()
            if current_constraint_value < best_constraint_value:
                best_constraint_value = current_constraint_value
                best_thetas_primeo = get_thetas_prime().numpy()

        return {'x': best_thetas_primeo, 'val': best_constraint_value}


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    exp = Rank1_factorization(nb_arms=10, nb_positions=5)

    """ # need for multiple starting points ? (it seems not)
    for _ in range(3):
        exp.optimize_PBM(metric="MSE", num_steps=5000, plot=False, verbose=True, start_close=False)
    for _ in range(3):
        exp.optimize_PBM(metric="KL", num_steps=5000, plot=False, verbose=True, start_close=False)
    """

    """ # recover thetas_star and kappas_star if noise small enough
    exp.init_data(verbose=False, noise=0.00001)
    exp.optimize_PBM(metric="MSE", num_steps=5000, plot=True, verbose=True, start_close=False)
    exp.optimize_PBM(metric="KL", num_steps=5000, plot=True, verbose=True, start_close=False)
    """

    """ # keep kappas_hat non-increasing
    exp.init_data(verbose=False, noise=0.001)
    exp.kappas_star[2:0:-1] = exp.kappas_star[1:3]
    exp.mus_star[:, 2:0:-1] = exp.mus_star[:, 1:3]
    exp.mus_hat[:, 2:0:-1] = exp.mus_hat[:, 1:3]
    exp.print_data()
    exp.optimize_PBM(metric="MSE", num_steps=5000, plot=True, verbose=True, start_close=False)
    exp.optimize_PBM(metric="KL", num_steps=5000, plot=True, verbose=True, start_close=False)
    """

    #""" # able to learn theta_prime
    exp.init_data(verbose=False, noise=0.1)     # to be tested with small and "big" noise
    exp.optimize_PBM(metric="KL", num_steps=5000, plot=False, verbose=True, start_close=False)
    exp.optimize_q(num_constraints=100, verbose=True, plot=True, inner_verbose=None, inner_plot=False)
    #"""
