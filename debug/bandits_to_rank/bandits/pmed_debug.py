#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

if __name__ == "__main__":
    import numpy as np
    from bandits_to_rank.opponents.pmed import PMED

    #""" 13/08/2020: decompose_nb_prints_tilde() does not terminate
    # => maximum weight matching has to be done on the bi-patite graph of weight matrix N_tilde != 0 (while previous implementation was using N_tilde directly)
    player = PMED(9, 5, alpha=1)

    player.nb_trials = 8015
    player.nb_clics = np.array([[4.479e+03, 4.420e+02, 3.400e+02, 1.000e+00, 1.000e+00],
                                 [4.380e+02, 3.189e+03, 3.270e+02, 2.890e+02, 1.000e+00],
                                 [1.000e+00, 3.240e+02, 2.271e+03, 5.010e+02, 1.920e+02],
                                 [2.790e+02, 1.000e+00, 5.020e+02, 1.710e+03, 1.000e+00],
                                 [1.000e+00, 2.390e+02, 1.000e+00, 6.000e+00, 1.354e+03],
                                 [1.000e+00, 1.000e+00, 1.000e+00, 1.360e+02, 3.000e+00],
                                 [1.000e+00, 1.000e+00, 1.000e+00, 1.000e+00, 9.000e+00],
                                 [2.000e+00, 1.000e+00, 4.000e+00, 1.000e+00, 1.000e+00],
                                 [1.000e+00, 2.000e+00, 1.000e+00, 1.000e+00, 2.000e+00]])
    player.nb_prints = np.array([[4982,  522,  502,    2,    2],
                                 [ 522, 4482,  502,  502,    2],
                                 [   2,  502, 3992, 1012,  502],
                                 [ 502,    2, 1002, 3962,    2],
                                 [   2,  502,    2,   22, 5412],
                                 [   2,    2,    2,  502,   12],
                                 [   2,    2,    2,    2,   52],
                                 [   2,    2,   12,   12,   12],
                                 [   2,    2,    2,    2,   22]])
    player.mus_hat = player.nb_clics / player.nb_prints
    player.thetas_hat = np.array([0.8998144,  0.80119604, 0.71593631, 0.60487086, 0.5074083,  0.38710492, 0.40554945, 0.3363748,  0.42316151])
    player.kappas_hat = np.array([1.,         0.8982153,  0.79424251, 0.70823014, 0.49743397])

    player.update_thetas_hat_kappas_hat(verbose=False, plot=False)
    player.optimize_q(verbose=False, plot=False)
    print(player.q)
    player.decompose_nb_prints_tilde(verbose=True)

    #"""

    """ 12/08/2020: scipy.optimize.linprog in optimize_q manipulats an ill-conditionned matrix  
    # => catch errors from scipy.optimize.linprog and return previous matrix q
    player = PMED(9, 5, alpha=1)

    player.nb_trials = 8015
    player.nb_clics = np.array([[5.691e+03, 5.580e+02, 3.560e+02, 3.080e+02, 1.000e+00],
                                 [3.950e+02, 4.581e+03, 2.250e+02, 6.000e+00, 1.000e+00],
                                 [4.850e+02, 3.210e+02, 3.113e+03, 3.430e+02, 1.730e+02],
                                 [2.970e+02, 1.400e+01, 1.630e+02, 2.141e+03, 3.550e+02],
                                 [1.000e+00, 1.540e+02, 1.970e+02, 1.400e+01, 1.235e+03],
                                 [1.000e+00, 1.000e+00, 2.020e+02, 1.500e+02, 3.000e+00],
                                 [1.000e+00, 1.000e+00, 1.000e+00, 2.000e+00, 5.600e+01],
                                 [1.000e+00, 1.000e+00, 1.000e+00, 8.300e+01, 4.500e+01],
                                 [1.000e+00, 1.000e+00, 1.000e+00, 2.800e+01, 2.200e+01]])
    player.nb_prints = np.array([[6324,  684,  502,  502,    2],
                                 [ 502, 6448,  350,    9,    2],
                                 [ 684,  523, 5628,  677,  502],
                                 [ 502,   23,  336, 5273, 1184],
                                 [   2,  336,  509,   40, 5106],
                                 [   2,    2,  691,  502,    9],
                                 [   2,    2,    2,    9,  373],
                                 [   2,    2,    2,  674,  501],
                                 [   2,    2,    2,  336,  343]])
    player.mus_hat = player.nb_clics / player.nb_prints
    player.thetas_hat = np.array([0.90004585, 0.79046743, 0.70284136, 0.59103486, 0.49170117, 0.39552317,
 0.31209396, 0.18207646, 0.13015086])
    player.kappas_hat = np.array([1.,         0.89947016, 0.78835097, 0.6924595,  0.49486388])

    player.update_thetas_hat_kappas_hat(verbose=True, plot=False)
    player.optimize_q(verbose=True, plot=True)
    print(player.q)
    """
