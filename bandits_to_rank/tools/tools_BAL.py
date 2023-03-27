from math import log

def newton(mu, certitude, n, x0, tol=1.48e-8, maxiter=50):
    #C'est une légère modification de la méthode de newton codée dans scipy.optimize
    #mu, certitude et n vont définir f et fprime:
    #certitude = -log(p) où p est la confiance de l'intervalle de confiance que nous voulons définir
    # f = lambda x: mu * log(mu / x) + (1 - mu) * log((1 - mu) / (1 - x)) - certitude / n
    #fprime = lambda x: (x - mu) / (x * (1 - x))
    p0 = 1.0 * x0
    if mu == 0:
        for itr in range(maxiter):
        # first evaluate fval
            fval = - log(1 - p0) - certitude / n
        # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return p0
            newton_step = fval *(1 - p0)
            p = p0 - newton_step
            if abs(newton_step) <= tol:
                return p
            p0 = p
    elif mu == 1:
        return 1
    else:
        for itr in range(maxiter):
        # first evaluate fval
            fval = mu * log(mu / p0) + (1 - mu) * log((1 - mu) / (1 - p0)) - certitude / n
        # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return p0
            newton_step = (p0 * (1 - p0)) * fval / (p0 - mu)
            p = p0 - newton_step
            if abs(newton_step) <= tol:
                return p
            p0 = p

def start_up(mu, certitude, n):
    #mu, certitude et n servent à définir f
    # f = lambda x: mu * log(mu / x) + (1 - mu) * log((1 - mu) / (1 - x)) - certitude / n

    # On cherche k tel que f(r_k) < 0 et f(r_k+1) > 0 et renvoie r_k où r_0 = (1 + mu)/ 2 et 1 - r_k+1 = (1 - r_k)/10
    # Voir la courbe de f (i.e de KL)
    # mu est le point où f atteint son minimum
    res = (1 + mu) / 2
    if mu == 0:
        while (- log(1 - res) - certitude / n < 0):
            next_res = 1 - (1 - res) / 10
            if next_res == 1: #a cause des erreurs d'approximation des réels
                return res
            res = next_res
        return res
    elif mu == 1:
        return 1
    else:
        while (mu * log(mu / res) + (1 - mu) * log((1 - mu) / (1 - res)) - certitude / n < 0):
            next_res = 1 - (1 - res) / 10
            if next_res == 1: #a cause des erreurs d'approximation des réels
                return res
            res = next_res
        return res

