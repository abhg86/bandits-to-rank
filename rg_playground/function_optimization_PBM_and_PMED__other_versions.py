#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

import bandits_to_rank.tools.tfp_math_minimize as tfprg

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.python.ops.gen_array_ops import matrix_diag_v2  # waiting for a debug of tf.linalg.diag
from scipy.optimize import linear_sum_assignment

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


def get_initial_thetas_primeo_v1(l=1, verbose=False):
    """
    return a vector thetas_primeo
    * thetas_primeo[:L] is non-increasing
    * thetas_primeo[L:] is smaller than thetas_primeo[L-1]
    * thetas_primeo[l] = thetas_primeo[min(l-1, L-1)]
    * corresponding kappas_prime is non-increasing
    * thetas_prime[i] * kappas_prime[i] = thetas_hat[i] * kappas_hat[i]
    * kappas_prime[0] = 1

    obtained by taking
    * kappas_prime[:l] = 1,
    * and kappas_prime[l:] = 1/c
    * with c = thetas_hato[l-1] * self_kappas_hat[l-1] / (thetas_hato[l] * self_kappas_hat[l])
    equivalent to
    * thetas_primeo[:l] = thetas_hato[:l] * kappas_hat[l:]
    * thetas_primeo[l:L] = thetas_hato[l:L] * kappas_hat[l:] * c

    Pb: most values for thetas_prime[L:] are taken equal to zero, which disable the existence of matrix Q proportional
    to the number of required exploration.
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_nb_positions = self_kappas_hat.shape[0]

    # début
    # consider thetas in the right order
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # proposed vector
    if l <= self_nb_positions-1:
        thetas_primeo = np.zeros(thetas_hato.shape)
        thetas_primeo[:l] = thetas_hato[:l] * self_kappas_hat[:l]
        c = thetas_hato[l-1] * self_kappas_hat[l-1] / (thetas_hato[l] * self_kappas_hat[l])
        thetas_primeo[l:self_nb_positions] = thetas_hato[l:self_nb_positions] * self_kappas_hat[l:] * c
    else:
        thetas_primeo = np.zeros(thetas_hato.shape)
        thetas_primeo[:self_nb_positions] = thetas_hato[:self_nb_positions] * self_kappas_hat
        thetas_primeo[l] = thetas_primeo[self_nb_positions-1]

    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def get_initial_thetas_primeo(l=1, verbose=False):
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

    remark: these is a clipping, not a projection
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_thetas_hat = np.array([0.43362092, 0.16051062, 0.64824003, 0.31249062, 0.24481692])
    self_kappas_hat = np.array([1.,         0.36517494, 0.29870629, 0.28619078, 0.17552095])
    self_nb_positions = self_kappas_hat.shape[0]
    self_nb_arms = self_thetas_hat.shape[0]

    # début
    # consider thetas in the right order
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # proposed vector
    thetas_primeo = thetas_hato.copy()
    # thetas_prime[:L] is non-increasing & thetas_prime[l] = thetas_prime[l - 1]
    for i in range(1, self_nb_positions):
        if i == l:
            thetas_primeo[i] = thetas_primeo[i - 1]
        else:
            a = thetas_primeo[i-1] * thetas_hato[i] * self_kappas_hat[i] / (thetas_hato[i - 1] * self_kappas_hat[i - 1])
            b = thetas_primeo[i-1]
            if verbose:
                print('i:', i)
                print('lim:', a, b)
                print(thetas_primeo[i], '->', np.clip(thetas_primeo[i], a, b))
            thetas_primeo[i] = np.clip(thetas_primeo[i], a, b)
    # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
    for i in range(self_nb_positions, self_nb_arms):
        if i == l:
            thetas_primeo[i] = thetas_primeo[self_nb_positions - 1]
        else:
            a = 0
            b = thetas_primeo[self_nb_positions - 1]
            thetas_primeo[i] = np.clip(thetas_primeo[i], a, b)

    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def get_random_thetas_primeoV1(l=1, num_steps=1000, verbose=False, plot=False):
    """
    !!! do not find a correct vector in some configurations !!!

    return a random vector thetas_prime
    * items are ordered given theta_hat
    * thetas_primeo[:L] is decreasing
    * corresponding kappas_prime is in [0,1]
    * thetas_prime[l] is equal to thetas_prime[min(l-1, L-1)]

    Minize a loss function enforcing these constraints
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_nb_positions = self_kappas_hat.shape[0]

    # début
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
    # so there is only `self_nb_arms-2` variables
    thetas_prime_var = tf.Variable(np.concatenate((thetas_hato[1:l], thetas_hato[(l + 1):]), axis=0), dtype='float64')
    thetas_hat = tf.constant(thetas_hato, dtype='float64')
    kappas_hat = tf.constant(self_kappas_hat, dtype='float64')

    @tf.function
    def get_thetas_prime():
        thetas_prime0 = tf.constant(thetas_hato[0], shape=(1,))
        if l == 1:
            return tf.concat([thetas_prime0, thetas_prime0, thetas_prime_var], 0)
        else:
            ll = min(l - 1, self_nb_positions - 1) - 1      # the -1 is due to $\theta_{(0)}'$ being a constant
            thetas_primel = tf.reshape(thetas_prime_var[ll], [1])
            return tf.concat([thetas_prime0, thetas_prime_var[:l-1], thetas_primel, thetas_prime_var[l-1:]], 0)

    @tf.function
    def loss_fn():
        # return self_regularization_thetas_prime_kappas_prime(thetas_prime, thetas_hat, kappas_hat, l)
        thetas_prime = get_thetas_prime()
        kappas_prime = thetas_hat[:self_nb_positions] * kappas_hat / thetas_prime[:self_nb_positions]
        # loss
        loss = tf.constant(0, dtype='float64')
        # thetas_prime[:L] is decreasing
        alpha2 = tf.constant(1, dtype='float64')
        phi2 = tf.constant(10, dtype='float64')
        loss += alpha2 * tf.reduce_sum(tf.math.sigmoid(
            - phi2 * (thetas_prime[:(self_nb_positions - 1)] - thetas_prime[1:self_nb_positions])))
        # thetas_prime[L-1] > thetas_prime[L:]
        alpha2b = tf.constant(1, dtype='float64')
        phi2b = tf.constant(10, dtype='float64')
        loss += alpha2b * tf.reduce_sum(tf.math.sigmoid(
            - phi2b * (thetas_prime[self_nb_positions - 1] - thetas_prime[self_nb_positions:])))
        # thetas_prime is in [0, 1]
        alpha3 = tf.constant(1, dtype='float64')
        phi3 = tf.constant(10, dtype='float64')
        loss += alpha3 * tf.reduce_sum(tf.math.sigmoid(- phi3 * thetas_prime))
        loss += alpha3 * tf.reduce_sum(tf.math.sigmoid(- phi3 * (1 - thetas_prime)))
        # kappas_prime is decreasing
        alpha4 = tf.constant(1, dtype='float64')
        phi4 = tf.constant(10, dtype='float64')
        loss += alpha4 * tf.reduce_sum(tf.math.sigmoid(- phi4 * (kappas_prime[:-1] - kappas_prime[1:])))
        # kappas_prime[1:] is in [0, 1]
        alpha5 = tf.constant(1, dtype='float64')
        phi5 = tf.constant(10, dtype='float64')
        loss += alpha5 * tf.reduce_sum(tf.math.sigmoid(- phi5 * kappas_prime[1:]))
        loss += alpha5 * tf.reduce_sum(tf.math.sigmoid(- phi5 * (1- kappas_prime[1:])))
        # kappas_prime[0] is 1
        alpha6 = tf.constant(1, dtype='float64')
        loss += alpha6 * (kappas_prime[0]-1.)**2
        return loss

    trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': get_thetas_prime(),
                                               'blop': grads}
    trace = tfp.math.minimize(loss_fn, num_steps=num_steps, trainable_variables=None,
                              optimizer=tf.optimizers.Adam(0.001),
                              trace_fn=trace_fn)
    #print(trace['blop'])
    #exit(0)
    # """

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
        plt.ylabel("thetas")
        plt.plot(trace['theta'][:, 0], ':', color=f'C{1}')
        plt.axhline(y=thetas_hato[0], color=f'C{1}')
        plt.plot(trace['theta'][:, l - 1], ':', color=f'C{2}')
        plt.axhline(y=thetas_hato[l - 1], color=f'C{2}')
        plt.axhline(y=thetas_hato[l], color=f'C{2}')
        plt.grid()
        plt.ylim([-0.05, 1.05])

        plt.show()

    thetas_primeo = get_thetas_prime().numpy()
    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def get_random_thetas_primeoV2(l=1, num_steps=1000, verbose=False, plot=False):
    """
    !!! do not find a correct vector in some configurations !!!

    return a random vector thetas_prime
    * items are ordered given theta_hat
    * thetas_primeo[:L] is decreasing
    * corresponding kappas_prime is in [0,1]
    * thetas_prime[l] is equal to thetas_prime[min(l-1, L-1)]

    Minize a loss function enforcing these constraints
    V2 : based on adding etas_prime parameters
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_nb_positions = self_kappas_hat.shape[0]

    # début
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
    # so there is only `self_nb_arms-2` variables
    #thetas_prime_var = tf.Variable(np.concatenate((thetas_hato[1:l], thetas_hato[(l + 1):]), axis=0), dtype='float64')
    initial_thetas_prime = get_initial_thetas_primeo(l=l, verbose=False)
    thetas_prime_var = tf.Variable(np.concatenate((initial_thetas_prime[1:l], initial_thetas_prime[(l + 1):]), axis=0), dtype='float64')

    @tf.function
    def get_thetas_prime():
        thetas_prime0 = tf.constant(thetas_hato[0], shape=(1,))
        if l == 1:
            return tf.concat([thetas_prime0, thetas_prime0, thetas_prime_var], 0)
        else:
            ll = min(l - 1, self_nb_positions - 1) - 1      # the -1 is due to $\theta_{(0)}'$ being a constant
            thetas_primel = tf.reshape(thetas_prime_var[ll], [1])
            return tf.concat([thetas_prime0, thetas_prime_var[:l-1], thetas_primel, thetas_prime_var[l-1:]], 0)

    etas_prime = tf.Variable(1/(get_thetas_prime().numpy()[:self_nb_positions]), dtype='float64')
    thetas_hat = tf.constant(thetas_hato, dtype='float64')
    kappas_hat = tf.constant(self_kappas_hat, dtype='float64')


    @tf.function
    def loss_fn():
        # return self_regularization_thetas_prime_kappas_prime(thetas_prime, thetas_hat, kappas_hat, l)
        thetas_prime = get_thetas_prime()
        kappas_prime = thetas_hat[:self_nb_positions] * kappas_hat * etas_prime
        # loss
        loss = tf.constant(0, dtype='float64')
        # etas_prime = 1/thetas_pre[:L]
        alpha = tf.constant(1, dtype='float64')
        loss += alpha * tf.reduce_sum((thetas_prime[:self_nb_positions] * etas_prime - 1) ** 2)
        # thetas_prime[:L] is decreasing
        alpha2 = tf.constant(1, dtype='float64')
        phi2 = tf.constant(10, dtype='float64')
        loss += alpha2 * tf.reduce_sum(tf.math.sigmoid(
            - phi2 * (thetas_prime[:(self_nb_positions - 1)] - thetas_prime[1:self_nb_positions])))
        # thetas_prime[L-1] > thetas_prime[L:]
        alpha2b = tf.constant(1, dtype='float64')
        phi2b = tf.constant(10, dtype='float64')
        loss += alpha2b * tf.reduce_sum(tf.math.sigmoid(
            - phi2b * (thetas_prime[self_nb_positions - 1] - thetas_prime[self_nb_positions:])))
        # thetas_prime is in [0, 1]
        alpha3 = tf.constant(1, dtype='float64')
        phi3 = tf.constant(10, dtype='float64')
        loss += alpha3 * tf.reduce_sum(tf.math.sigmoid(- phi3 * thetas_prime))
        loss += alpha3 * tf.reduce_sum(tf.math.sigmoid(- phi3 * (1 - thetas_prime)))
        # kappas_prime is decreasing
        alpha4 = tf.constant(1, dtype='float64')
        phi4 = tf.constant(10, dtype='float64')
        loss += alpha4 * tf.reduce_sum(tf.math.sigmoid(- phi4 * (kappas_prime[:-1] - kappas_prime[1:])))
        # kappas_prime[1:] is in [0, 1]
        alpha5 = tf.constant(1, dtype='float64')
        phi5 = tf.constant(10, dtype='float64')
        loss += alpha5 * tf.reduce_sum(tf.math.sigmoid(- phi5 * kappas_prime[1:]))
        loss += alpha5 * tf.reduce_sum(tf.math.sigmoid(- phi5 * (1- kappas_prime[1:])))
        # kappas_prime[0] is 1
        alpha6 = tf.constant(1, dtype='float64')
        loss += alpha6 * (kappas_prime[0]-1.)**2
        return loss

    trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': get_thetas_prime(),
                                               'blop': grads}
    trace = tfp.math.minimize(loss_fn, num_steps=num_steps, trainable_variables=None,
                              optimizer=tf.optimizers.Adam(0.001),
                              trace_fn=trace_fn)
    #print(trace['blop'])
    #exit(0)
    # """

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
        plt.ylabel("thetas")
        plt.plot(trace['theta'][:, 0], ':', color=f'C{1}')
        plt.axhline(y=thetas_hato[0], color=f'C{1}')
        plt.plot(trace['theta'][:, l - 1], ':', color=f'C{2}')
        plt.axhline(y=thetas_hato[l - 1], color=f'C{2}')
        plt.axhline(y=thetas_hato[l], color=f'C{2}')
        plt.grid()
        plt.ylim([-0.05, 1.05])

        plt.show()

    thetas_primeo = get_thetas_prime().numpy()
    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def test_tf_projection(l=1, verbose=False, plot=False):
    """
    test random projection by applying one step of of optimisation of a constant function (+ projection).
    Returns a vector thetas_prime s.t.
    * items are ordered given theta_hat
    * thetas_primeo[:L] is decreasing
    * corresponding kappas_prime is in [0,1]
    * thetas_prime[l] is equal to thetas_prime[min(l-1, L-1)]

    rem: start with thetas_prime[anything but l] = thetas_hat[anything but l]
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_nb_positions = self_kappas_hat.shape[0]
    self_nb_arms = self_thetas_hat.shape[0]

    # début
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
    # so there is only `self_nb_arms-2` variables
    thetas_prime_var = tf.Variable(np.concatenate((thetas_hato[1:l], thetas_hato[(l + 1):]), axis=0), dtype='float64')
    thetas_hat = tf.constant(thetas_hato, dtype='float64')
    kappas_hat = tf.constant(self_kappas_hat, dtype='float64')

    @tf.function
    def get_thetas_prime():
        thetas_prime0 = tf.constant(thetas_hato[0], shape=(1,))
        if l == 1:
            return tf.concat([thetas_prime0, thetas_prime0, thetas_prime_var], 0)
        else:
            ll = min(l - 1, self_nb_positions - 1) - 1      # the -1 is due to $\theta_{(0)}'$ being a constant
            thetas_primel = tf.reshape(thetas_prime_var[ll], [1])
            return tf.concat([thetas_prime0, thetas_prime_var[:l-1], thetas_primel, thetas_prime_var[l-1:]], 0)

    @tf.function
    def projection():
        """
        * thetas_prime[:L] is non-increasing
        * kappas_prime is non-increasing
        * thetas_prime[i] * kappas_prime[i] = thetas_hat[i] * kappas_hat[i]
        * kappas_prime[0] = 1
        => sum_up to
        thetas_prime[i] is in [ thetas_prime[i-1] * thetas_hat[i] * kappas_hat[i] / (thetas_hat[i-1] * kappas_hat[i-1]),
                                  thetas_prime[i-1] ]
        which we apply for i from 1 to L-1 (except i = l)
        """
        thetas_prime = get_thetas_prime()
        new_thetas_primei1 = thetas_prime[0]
        # thetas_prime[:L] is non-increasing
        for i in range(1, min(self_nb_positions, l)):
            a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
            b = new_thetas_primei1
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b)
            thetas_prime_var[i-1].assign(new_thetas_primei1)
        for i in range(l+1, self_nb_positions):
            a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
            b = new_thetas_primei1
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b)
            thetas_prime_var[i-2].assign(new_thetas_primei1)
        # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
        a = tf.constant(0., dtype='float64')
        b = new_thetas_primei1
        for i in range(self_nb_positions, l):
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            thetas_prime_var[i-1].assign(tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b))
        for i in range(max(self_nb_positions, l+1), self_nb_arms):
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            #   and thetas_prime[l] being a copy of thetas_prime[l-1]
            thetas_prime_var[i-2].assign(tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b))


    @tf.function
    def loss_fn():
        # return self_regularization_thetas_prime_kappas_prime(thetas_prime, thetas_hat, kappas_hat, l)
        thetas_prime = get_thetas_prime()
        kappas_prime = thetas_hat[:self_nb_positions] * kappas_hat / thetas_prime[:self_nb_positions]
        # loss
        loss = tf.constant(0, dtype='float64')
        # thetas_prime[:L] is decreasing
        zero = tf.constant(np.zeros(thetas_prime.shape), dtype='float64')
        loss += tf.reduce_sum(thetas_prime * zero)
        return loss

    trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': get_thetas_prime(),
                                               'blop': grads}
    trace = tfprg.minimize(loss_fn, num_steps=1, trainable_variables=None,
                           optimizer=tf.optimizers.Adam(0.001), projection=projection,
                           trace_fn=trace_fn)
    #print(trace['blop'])
    #exit(0)
    # """

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
        plt.ylabel("thetas")
        plt.plot(trace['theta'][:, 0], ':', color=f'C{1}')
        plt.axhline(y=thetas_hato[0], color=f'C{1}')
        plt.plot(trace['theta'][:, l - 1], ':', color=f'C{2}')
        plt.axhline(y=thetas_hato[l - 1], color=f'C{2}')
        plt.axhline(y=thetas_hato[l], color=f'C{2}')
        plt.grid()
        plt.ylim([-0.05, 1.05])

        plt.show()

    thetas_primeo = get_thetas_prime().numpy()
    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def get_higher_thetas_primeo(l=1, num_steps=1000, verbose=False, plot=False):
    """
    return the vector thetas_prime with higher values s.t.
    * items are ordered given theta_hat
    * thetas_primeo[:L] is decreasing
    * corresponding kappas_prime is in [0,1]
    * thetas_prime[l] is equal to thetas_prime[min(l-1, L-1)]

    constraints are enforced by clipping values after each gradient-step
    initial value is fulfill the constraint (comes from get_initial_thetas_primeo()) but may be upgraded
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_mus_hat = np.array([[0.625, 0.479, 0.238, 0.15,  0.039],
                             [0.352, 0.279, 0.139, 0.09,  0.022],
                             [0.585, 0.434, 0.216, 0.146, 0.037],
                             [0.868, 0.655, 0.335, 0.222, 0.055],
                             [0.292, 0.235, 0.108, 0.072, 0.017],
                             [0.739, 0.582, 0.263, 0.176, 0.046],
                             [0.664, 0.488, 0.257, 0.164, 0.04 ],
                             [0.547, 0.42,  0.196, 0.139, 0.035],
                             [0.608, 0.489, 0.243, 0.159, 0.04 ],
                             [0.447, 0.358, 0.179, 0.119, 0.029]])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_nb_positions = self_kappas_hat.shape[0]
    self_nb_arms = self_thetas_hat.shape[0]
    qo = np.ones((self_nb_arms, self_nb_arms))

    # début
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
    # so there is only `self_nb_arms-2` variables
    initial_thetas_prime = get_initial_thetas_primeo(l=l, verbose=False)
    thetas_prime_var = tf.Variable(np.concatenate((initial_thetas_prime[1:l], initial_thetas_prime[(l + 1):]), axis=0), dtype='float64')
    thetas_hat = tf.constant(thetas_hato, dtype='float64')
    kappas_hat = tf.constant(self_kappas_hat, dtype='float64')
    mus_hat = tf.constant(self_mus_hat[order_thetas_hat, :], dtype='float64')
    q = tf.constant(qo[:, :self_nb_positions], dtype='float64')

    @tf.function
    def get_thetas_prime():
        thetas_prime0 = tf.constant(thetas_hato[0], shape=(1,))
        if l == 1:
            return tf.concat([thetas_prime0, thetas_prime0, thetas_prime_var], 0)
        else:
            ll = min(l - 1, self_nb_positions - 1) - 1      # the -1 is due to $\theta_{(0)}'$ being a constant
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
        for i in range(1, min(self_nb_positions, l)):
            a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
            b = new_thetas_primei1
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b)
            thetas_prime_var[i-1].assign(new_thetas_primei1)
        for i in range(l+1, self_nb_positions):
            a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
            b = new_thetas_primei1
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b)
            thetas_prime_var[i-2].assign(new_thetas_primei1)
        # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
        a = tf.constant(0., dtype='float64')
        b = new_thetas_primei1
        for i in range(self_nb_positions, l):
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            thetas_prime_var[i-1].assign(tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b))
        for i in range(max(self_nb_positions, l+1), self_nb_arms):
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            #   and thetas_prime[l] being a copy of thetas_prime[l-1]
            thetas_prime_var[i-2].assign(tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b))


    @tf.function
    def loss_fn():
        # return self_regularization_thetas_prime_kappas_prime(thetas_prime, thetas_hat, kappas_hat, l)
        thetas_prime = get_thetas_prime()
        kappas_prime = thetas_hat[:self_nb_positions] * kappas_hat / thetas_prime[:self_nb_positions]
        # loss
        loss = tf.constant(0, dtype='float64')
        loss += -tf.reduce_sum(thetas_prime)
        return loss

    if verbose:
        thetas_primeo = get_thetas_prime().numpy()
        kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]
        print('before optimization')
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

        print(loss_fn().numpy())

    trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': get_thetas_prime(),
                                               'blop': grads}
    trace = tfprg.minimize(loss_fn, num_steps=num_steps, trainable_variables=None,
                           optimizer=tf.optimizers.Adam(0.001), projection=projection,
                           trace_fn=trace_fn)
    #print(trace['blop'])
    #exit(0)
    # """

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
        plt.ylabel("thetas")
        plt.plot(trace['theta'][:, 0], ':', color=f'C{1}')
        plt.axhline(y=thetas_hato[0], color=f'C{1}')
        plt.plot(trace['theta'][:, l - 1], ':', color=f'C{2}')
        plt.axhline(y=thetas_hato[l - 1], color=f'C{2}')
        plt.axhline(y=thetas_hato[l], color=f'C{2}')
        plt.grid()
        plt.ylim([-0.05, 1.05])

        plt.show()

    thetas_primeo = get_thetas_prime().numpy()
    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('after optimization')
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)

        print(loss_fn().numpy())

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def get_best_thetas_primeo_wrt_q(l=1, num_steps=1000, verbose=False, plot=False):
    """
    return the vector thetas_prime with best values given q s.t.
    * items are ordered given theta_hat
    * thetas_primeo[:L] is decreasing
    * corresponding kappas_prime is in [0,1]
    * thetas_prime[l] is equal to thetas_prime[min(l-1, L-1)]

    constraints are enforced by clipping values after each gradient-step
    initial value is fulfill the constraint (comes from get_initial_thetas_primeo()) but may be upgraded
    """
    # params
    self_thetas_star = np.array([0.61,  0.349, 0.567, 0.854, 0.279, 0.706, 0.638, 0.523, 0.639, 0.463])
    self_mus_hat = np.array([[0.625, 0.479, 0.238, 0.15,  0.039],
                             [0.352, 0.279, 0.139, 0.09,  0.022],
                             [0.585, 0.434, 0.216, 0.146, 0.037],
                             [0.868, 0.655, 0.335, 0.222, 0.055],
                             [0.292, 0.235, 0.108, 0.072, 0.017],
                             [0.739, 0.582, 0.263, 0.176, 0.046],
                             [0.664, 0.488, 0.257, 0.164, 0.04 ],
                             [0.547, 0.42,  0.196, 0.139, 0.035],
                             [0.608, 0.489, 0.243, 0.159, 0.04 ],
                             [0.447, 0.358, 0.179, 0.119, 0.029]])
    self_thetas_hat = np.array([0.051, 0.576, 0.063, 0.085, 0.326, 0.039, 0.947, 0.072, 0.718, 0.21])
    self_kappas_hat = np.array([1., 0.526, 0.254, 0.112, 0.11])
    self_nb_positions = self_kappas_hat.shape[0]
    self_nb_arms = self_thetas_hat.shape[0]
    qo = np.ones((self_nb_arms, self_nb_arms))

    # début
    order_thetas_hat = np.argsort(self_thetas_hat)[::-1]
    thetas_hato = self_thetas_hat[order_thetas_hat]

    # $\theta_{(0)}' = \hat\theta_{(0)}$ as $\kappa_0' = \hat\kappa_0 = 1$ and $\theta_{(0)}'\kappa_0' = \hat\theta_{(0)}\hat\kappa_0$
    # so there is only `self_nb_arms-2` variables
    initial_thetas_prime = get_initial_thetas_primeo(l=l, verbose=False)
    thetas_prime_var = tf.Variable(np.concatenate((initial_thetas_prime[1:l], initial_thetas_prime[(l + 1):]), axis=0), dtype='float64')
    thetas_hat = tf.constant(thetas_hato, dtype='float64')
    kappas_hat = tf.constant(self_kappas_hat, dtype='float64')
    mus_hat = tf.constant(self_mus_hat[order_thetas_hat, :], dtype='float64')
    q = tf.constant(qo[:, :self_nb_positions], dtype='float64')

    @tf.function
    def get_thetas_prime():
        thetas_prime0 = tf.constant(thetas_hato[0], shape=(1,))
        if l == 1:
            return tf.concat([thetas_prime0, thetas_prime0, thetas_prime_var], 0)
        else:
            ll = min(l - 1, self_nb_positions - 1) - 1      # the -1 is due to $\theta_{(0)}'$ being a constant
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
        for i in range(1, min(self_nb_positions, l)):
            a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
            b = new_thetas_primei1
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b)
            thetas_prime_var[i-1].assign(new_thetas_primei1)
        for i in range(l+1, self_nb_positions):
            a = new_thetas_primei1 * thetas_hat[i] * kappas_hat[i] / (thetas_hato[i-1] * kappas_hat[i-1])
            b = new_thetas_primei1
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            new_thetas_primei1 = tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b)
            thetas_prime_var[i-2].assign(new_thetas_primei1)
        # thetas_prime[L:] is smaller than thetas_prime[L-1] (except for thetas_prime[l])
        a = tf.constant(0., dtype='float64')
        b = new_thetas_primei1
        for i in range(self_nb_positions, l):
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            thetas_prime_var[i-1].assign(tf.clip_by_value(thetas_prime_var[i-1], clip_value_min=a, clip_value_max=b))
        for i in range(max(self_nb_positions, l+1), self_nb_arms):
            # thetas_prime_var[i-1] corresponds to thetas_prime[i] due to thetas_prime[0] being a constant
            #   and thetas_prime[l] being a copy of thetas_prime[l-1]
            thetas_prime_var[i-2].assign(tf.clip_by_value(thetas_prime_var[i-2], clip_value_min=a, clip_value_max=b))


    @tf.function
    def loss_fn():
        # return self_regularization_thetas_prime_kappas_prime(thetas_prime, thetas_hat, kappas_hat, l)
        thetas_prime = get_thetas_prime()
        kappas_prime = thetas_hat[:self_nb_positions] * kappas_hat / thetas_prime[:self_nb_positions]
        # loss
        loss = tf.constant(0, dtype='float64')
        #loss += -tf.reduce_sum(thetas_prime)
        mus_prime = tf.matmul(tf.reshape(thetas_prime, [-1, 1]), tf.reshape(kappas_prime, [1, -1]))
        minus_mus_hat = 1. - mus_hat
        loss += tf.reduce_sum(q * (mus_hat * tf.math.log(mus_hat / mus_prime)
                                  + minus_mus_hat * tf.math.log(minus_mus_hat / (1 - mus_prime))))
        # TODOS : minus the diagonal ( i=l )
        return loss

    @tf.function
    def constraint_for_learning_q():
        """similar loss_fn, but removing the diagonal terms of q"""
        thetas_prime = get_thetas_prime()
        kappas_prime = thetas_hat[:self_nb_positions] * kappas_hat / thetas_prime[:self_nb_positions]
        # loss
        loss = tf.constant(0, dtype='float64')
        mus_prime = tf.matmul(tf.reshape(thetas_prime, [-1, 1]), tf.reshape(kappas_prime, [1, -1]))
        minus_mus_hat = 1. - mus_hat
        keep_value = matrix_diag_v2(np.zeros(self_nb_positions), k=0, num_rows=self_nb_arms, num_cols=-1,
                                    padding_value=1)
        loss += tf.reduce_sum(keep_value * q * (mus_hat * tf.math.log(mus_hat / mus_prime)
                                                + minus_mus_hat * tf.math.log(minus_mus_hat / (1 - mus_prime))))
        return loss

    if verbose:
        thetas_primeo = get_thetas_prime().numpy()
        kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]
        print('before optimization')
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)
        print(f'loss {loss_fn().numpy()}')
        print(f'constraint for learning q {constraint_for_learning_q().numpy()}')

    trace_fn = lambda loss, grads, variables: {'loss': loss, 'theta': get_thetas_prime(),
                                               'blop': grads}
    trace = tfprg.minimize(loss_fn, num_steps=num_steps, trainable_variables=None,
                           optimizer=tf.optimizers.Adam(0.001), projection=projection,
                           trace_fn=trace_fn)
    #print(trace['blop'])
    #exit(0)
    # """

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

    thetas_primeo = get_thetas_prime().numpy()
    kappas_prime = thetas_hato[:self_nb_positions] * self_kappas_hat / thetas_primeo[:self_nb_positions]

    if verbose:
        print('after optimization')
        print('theta_primeo:', thetas_primeo)
        print('kappas_prime:', kappas_prime)
        print(f'loss {loss_fn().numpy()}')
        print(f'constraint for learning q {constraint_for_learning_q().numpy()}')

    # some assertions on thetas_primeo
    assert_thetas_prime(thetas_primeo, thetas_hato[0], kappas_prime, l)

    return thetas_primeo


def decompose_nb_prints_tilde(n_bar, epsilon=1e-10):
    """
    Compute decomposition of the K x K matrix, but return only the corresponding (weighted) L-permutations

    Examples
    --------
    >>> import numpy as np
    >>> nbar = np.array([[3., 0., 1., 2., 0.],
    ...        [0., 3., 0., 0., 3.],
    ...        [0., 2., 0., 4., 0.],
    ...        [2., 1., 3., 0., 0.],
    ...        [1., 0., 2., 0., 3.]])
    >>> decompose_nb_prints_tilde(nbar)
    [(3.0, array([0, 1, 3, 2, 4])), (2.0, array([3, 2, 4, 0, 1])), (1.0, array([4, 3, 0, 2, 1]))]

    >>> nbar = np.array([[5.04, 9.04, 1.04, 6.04, 3.84],
    ...        [3.24, 8.24, 2.24, 5.24, 6.04],
    ...        [3.84, 2.84, 9.84, 4.84, 3.64],
    ...        [6.04, 4.04, 5.04, 4.04, 5.84],
    ...        [6.84, 0.84, 6.84, 4.84, 5.64]])
    >>> decomposition = decompose_nb_prints_tilde(nbar)
    >>> [c>0 for c, _ in decomposition]
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    >>> [perm for _, perm in decomposition]
    [array([4, 1, 2, 0, 3]), array([3, 0, 4, 2, 1]), array([0, 3, 2, 1, 4]), array([1, 0, 3, 4, 2]), array([2, 1, 4, 3, 0]), array([4, 2, 1, 3, 0]), array([0, 2, 3, 1, 4]), array([2, 0, 3, 4, 1]), array([2, 4, 1, 3, 0]), array([3, 2, 0, 4, 1]), array([3, 1, 0, 4, 2]), array([3, 2, 1, 0, 4]), array([2, 0, 1, 3, 4]), array([3, 2, 0, 1, 4]), array([2, 3, 0, 1, 4])]
    >>> recomposition = np.zeros((5,5))
    >>> for c, perm in decomposition:
    ...     recomposition[perm, np.arange(5)] += c
    >>> recomposition
    array([[5.04, 9.04, 1.04, 6.04, 3.84],
           [3.24, 8.24, 2.24, 5.24, 6.04],
           [3.84, 2.84, 9.84, 4.84, 3.64],
           [6.04, 4.04, 5.04, 4.04, 5.84],
           [6.84, 0.84, 6.84, 4.84, 5.64]])

    >>> nbar = np.array([[2.62558568e+02, 3.83313916e-08, 7.91387933e-09, 7.25792378e-09, 4.13308337e-09],
    ...     [3.46307143e-08, 2.62558568e+02, 4.41553030e-08, 4.31593788e-08, 1.84943466e-08],
    ...     [9.81964535e-09, 5.06297774e-08, 1.87115315e+01, 2.43847035e+02, 1.43126922e-06],
    ...     [7.29621214e-09, 3.71827847e-08, 2.43847035e+02, 1.87115325e+01, 1.50750773e-08],
    ...     [4.95086473e-09, 1.39764962e-08, 9.44586036e-07, 5.04736426e-07, 2.62558567e+02]])
    >>> decomposition = decompose_nb_prints_tilde(nbar)
    >>> [c>0 for c, _ in decomposition]
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    >>> [perm for _, perm in decomposition]
    [array([4, 1, 2, 0, 3]), array([3, 0, 4, 2, 1]), array([0, 3, 2, 1, 4]), array([1, 0, 3, 4, 2]), array([2, 1, 4, 3, 0]), array([4, 2, 1, 3, 0]), array([0, 2, 3, 1, 4]), array([2, 0, 3, 4, 1]), array([2, 4, 1, 3, 0]), array([3, 2, 0, 4, 1]), array([3, 1, 0, 4, 2]), array([3, 2, 1, 0, 4]), array([2, 0, 1, 3, 4]), array([3, 2, 0, 1, 4]), array([2, 3, 0, 1, 4])]
    >>> recomposition = np.zeros((5,5))
    >>> for c, perm in decomposition:
    ...     recomposition[perm, np.arange(5)] += c
    array([[262   0   0   0   0]
           [  0 262   0   0   0]
           [  0   0  18 243   0]
           [  0   0 243  18   0]
           [  0   0   0   0 262]])
    """
    pseudo_positions = np.arange(n_bar.shape[0])

    n_bar = np.array(n_bar, dtype=np.int)
    res = []
    nb_turn = 10
    while np.max(n_bar) > epsilon and nb_turn > 0:
        #print('n_bar\n', n_bar)
        #print('n_bar\n', np.array(n_bar, dtype=np.int))
        #print('max', np.max(n_bar))
        _, perm = linear_sum_assignment(-n_bar.T)
        c = np.min(n_bar[perm, pseudo_positions])
        #print('perm', perm)
        #print('n_perm', n_bar[perm, pseudo_positions])
        #print('c', c)
        n_bar[perm, pseudo_positions] -= c
        res.append((c, perm))
        nb_turn -= 1
    return res


if __name__ == "__main__":
    n_bar = np.array([[2.62558568e+02, 3.83313916e-08, 7.91387933e-09, 7.25792378e-09, 4.13308337e-09],
                     [3.46307143e-08, 2.62558568e+02, 4.41553030e-08, 4.31593788e-08, 1.84943466e-08],
                     [9.81964535e-09, 5.06297774e-08, 1.87115315e+01, 2.43847035e+02, 1.43126922e-06],
                     [7.29621214e-09, 3.71827847e-08, 2.43847035e+02, 1.87115325e+01, 1.50750773e-08],
                     [4.95086473e-09, 1.39764962e-08, 9.44586036e-07, 5.04736426e-07, 2.62558567e+02]])
    decompose_nb_prints_tilde(n_bar, epsilon=np.min(n_bar))

    for l in range(1, 10):
        print(f'l: {l}')
        #get_initial_thetas_primeo_v1(l=l, verbose=True)
        get_initial_thetas_primeo(l=l, verbose=True)
        #test_tf_projection(l=l, verbose=True, plot=False)
        #get_higher_thetas_primeo(l=l, num_steps=1, verbose=True, plot=False)
        #get_higher_thetas_primeo(l=l, num_steps=1000, verbose=True, plot=False)
        #get_best_thetas_primeo_wrt_q(l=l, num_steps=1, verbose=True, plot=False)
        get_best_thetas_primeo_wrt_q(l=l, num_steps=1000, verbose=True, plot=False)
        #get_random_thetas_primeoV2(l=l, num_steps=1000, verbose=True, plot=False)
        #get_random_thetas_primeoV1(l=l, num_steps=1000, verbose=True, plot=False)

