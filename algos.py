import numpy as np

from scipy.stats import rv_discrete


# Algorithm from "Online Learning with Feedback Graphs: Beyond Bandits".
def Exp3G(eta, gamma, U, graph, T, n_runs):
    print('Exp3G')
    n = len(graph.arms)
    u = np.zeros(n)
    u[U] = 1.0 / len(U)
    rewards = np.zeros(T)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        q = np.ones(n) / n
        for t in range(T):
            p = (1 - gamma) * q + gamma * u
            p = p / np.sum(p)
            rv = rv_discrete(values=(range(n), p))
            I = rv.rvs()
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward) / n_runs
            all_rewards[t][run] += float(reward)
            r = np.zeros(n, dtype=float)
            for i in range(n):
                if feedback[i] is None:
                    r[i] = 0.0
                else:
                    r[i] = feedback[i] / np.sum(p[graph.in_neighbors(i)])
            q = q * np.exp(eta * r)
            q /= np.sum(q)
    return rewards, all_rewards


def TS(graph, T, n_runs):
    """
    Thompson sampling
    """
    print('TS')
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(T):
            theta = np.random.beta(S + 1, F + 1)
            J = np.argmax(theta)
            reward, feedback = graph.draw(J)

            rewards[t] += float(reward) / n_runs
            all_rewards[t][run] += float(reward)
            if feedback[J] is not None:
                S[J] += feedback[J]
                F[J] += 1 - feedback[J]
    return rewards, all_rewards


def UTS(graph, T, n_runs):
    print('UTS')
    n = len(graph.arms)
    rewards = np.zeros(T + 1)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(1, T + 1):
            theta = np.random.beta(S + 1, F + 1)
            I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(((S + 1) / (S + F + 2))[graph.in_neighbors(I)])]

            reward, feedback = graph.draw(J)

            rewards[t] += float(reward) / n_runs
            all_rewards[t-1][run] += float(reward)
            if not feedback[J] is None:
                S[J] += feedback[J]
                F[J] += 1 - feedback[J]
    return rewards[1:], all_rewards

################################################################################


def kl_bernoulli(p, q, eps=1e-14):
    """
    Compute kl_distance-divergence for Bernoulli distributions
    """
    p = np.minimum(np.maximum(p, eps), 1 - eps)
    q = np.minimum(np.maximum(q, eps), 1 - eps)
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
    return klucb(x, d, kl_distance=kl_bernoulli, upperbound=upperbound, precision=precision)


def OSUB(graph, T, n_runs):
    print('OSUB')
    n = len(graph.arms)
    rewards = np.zeros(T + 1)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        X = np.zeros(n)
        O = np.ones(n)
        L = np.ones(n)  # number of being leader up to time t
        for t in range(1, T + 1):
            I = np.random.choice([k for k in range(n) if O[k] == np.max(O)])

            l = L[I]
            d_ = (np.log(l) + np.log(np.log(l + 1e-11)))

            neighbors = np.nonzero(graph.in_neighbors(I))[0]
            B = np.zeros(len(neighbors))
            for idx, j in enumerate(neighbors):
                B[idx] = klucb_bern(X[j], d_/O[j])
            J = neighbors[np.argmax(B)]
            # J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(B[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            all_rewards[t-1][run] += float(reward)
            for i in range(n):
                if feedback[i] is not None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards[1:], all_rewards


def IMED_UB(graph, T, n_runs):
    print('IMED-UB')

    n = len(graph.arms)
    rewards = np.zeros(T + 1)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        X = np.zeros(n)
        O = np.ones(n)
        for t in range(1, T + 1):
            u_star = np.max(X)
            indices = O * kl_bernoulli(X, np.array([u_star] * n)) + np.log(O)
            I = np.random.choice([k for k in range(n) if O[k] == np.max(O)])
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmin(indices[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            all_rewards[t-1][run] += float(reward)
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards[1:], all_rewards


def KLUCB_UB(graph, T, n_runs):
    print('KLUCB-UB')
    K = len(graph.arms)
    rewards = np.zeros(T+1, dtype=float)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        X = np.zeros(K)
        O = np.ones(K)
        for t in range(1, T + 1):
            I = np.random.choice([k for k in range(K) if O[k] == np.max(O)])
            neighbors = np.nonzero(graph.in_neighbors(I))[0]
            B = np.zeros(len(neighbors))
            for idx, k in enumerate(neighbors):
                d = np.log(O[I]) - np.log(O[k])
                B[idx] = klucb_bern(X[k], d / O[k])
            J = neighbors[np.argmax(B)]
            # J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(indices[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            all_rewards[t-1][run] += float(reward)
            for i in range(K):
                if feedback[i] is not None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards[1:], all_rewards


def FEF_UTS(graph, T, n_runs):
    print('FEF-UTS')
    K = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    all_rewards = np.zeros((T, n_runs))
    for run in range(n_runs):
        S = np.zeros(K)
        F = np.zeros(K)
        X = np.zeros(K)
        O = np.zeros(K)
        for t in range(T):
            I = np.random.choice([k for k in range(K) if X[k] == np.max(X)])
            # I = np.argmax(X)
            # Argmax of I's in-neighbours.
            # J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            theta = np.random.beta(S + 1, F + 1)
            # I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(theta[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            all_rewards[t][run] += float(reward)
            for i in range(K):
                if feedback[i] is not None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards, all_rewards
