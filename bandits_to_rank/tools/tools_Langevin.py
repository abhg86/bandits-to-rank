#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

import os
import numpy as np
from math import log, exp
from scipy.optimize import fminbound, root_scalar


def compute_gradient(part,success,fail):
    """ Compute the gradient of the log of our product of beta
    """
    theta_kappa = np.kron(part[1], np.array([part[0]]).transpose())
    #print(part)
    grad_U_theta = -np.sum(
        success * (np.array([1 / part[0]])).transpose() - fail * (1 / (1 - theta_kappa)) * np.array(
            [part[1]]), axis=1)
    grad_U_kappa = -np.sum(success.transpose() * (np.array([1 / part[1]])).transpose() - fail.transpose() *
                          (1 / (1 - theta_kappa.transpose())) * np.array([part[0]]), axis=1)
    return grad_U_theta, grad_U_kappa


def Langevin(part0, success,fail, N, h, param_final_cov, nb_position, nb_arms):
    """ Apply Langevin gradient descent on our posterior law cf "On Thompson Sampling with "
    Langevin algorithm
    """
    cap_min_val = min(np.sqrt(h), 0.001)
    cap_max_val = 1. - cap_min_val
    sqrt_2h = np.sqrt(2 * h)
    part = [part0[0].copy(), part0[1].copy()]
    for _ in range(N):
        grad_theta, grad_kappa = compute_gradient(part, success, fail)
        part[0] -= h * grad_theta + sqrt_2h * np.random.randn(nb_arms)
        part[1] -= h * grad_kappa + sqrt_2h * np.random.randn(nb_position)
        part = [np.array([cap_max_val if v > cap_max_val else cap_min_val if v < cap_min_val else v for v in part[0]]),
                np.array([cap_max_val if v > cap_max_val else cap_min_val if v < cap_min_val else v for v in part[1]])]

    estim_param = [part[0] + np.sqrt(1 / param_final_cov) * np.random.randn(nb_arms),
                   part[1] + np.sqrt(1 / param_final_cov) * np.random.randn(nb_position)]
    estim_param = [np.array([cap_max_val if v > cap_max_val else cap_min_val if v < cap_min_val else v for v in estim_param[0]]),
                   np.array([cap_max_val if v > cap_max_val else cap_min_val if v < cap_min_val else v for v in estim_param[1]])]

    return part, estim_param


if __name__ == "__main__":
    import doctest

    doctest.testmod()
