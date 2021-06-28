
import numpy as np


def kl_bernoulli(p, q, eps=1e-14):
    """
    Compute kl_distance-divergence for Bernoulli distributions
    """
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    result = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return result


def klucb(x, d, kl_distance, upperbound, lowerbound=float('-inf'), precision=1e-6, max_iterations=50, ):
    value = max(x, lowerbound)
    u = upperbound
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) * 0.5
        if kl_distance(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) * 0.5


def klucb_bern(x, d, precision=1e-6):
    def klucbGauss(x, d, sig2x=0.25):
        return x + np.sqrt(abs(2 * sig2x * d))
    upperbound = min(1., klucbGauss(x, d, sig2x=0.25))
    klucb(x, d, kl_distance=kl_bernoulli, upperbound=upperbound, precision=precision)


def osub_upper_bisection(kl_distance, num, s, value, precision=1e-6, max_iterations=50):
    """
    Compute the upper confidence bound for each arm with bisection method
    """
    upperbound = np.log(value) + np.log(np.log(value + 1e-11))
    reward = s

    u = upperbound
    value = reward

    _count_iteration = 0

    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        q = (value + u) / 2
        if num * kl_distance(reward, q) > upperbound:
            u = q
        else:
            value = q

    return (value + u) / 2