

import numpy as np

from scipy.stats import rv_discrete


# Algorithm from "Online Learning with Feedback Graphs: Beyond Bandits".
def Exp3G(eta, gamma, U, graph, T, n_runs):
    print('Exp3G')
    n = len(graph.arms)
    u = np.zeros(n)
    u[U] = 1.0 / len(U)
    rewards = np.zeros(T)
    for run in range(n_runs):
        q = np.ones(n) / n
        for t in range(T):
            p = (1 - gamma) * q + gamma * u
            p = p / sum(p)
            rv = rv_discrete(values=(range(n), p))
            I = rv.rvs()
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward) / n_runs
            r = np.zeros(n, dtype=float)
            for i in range(n):
                if feedback[i] is None:
                    r[i] = 0.0
                else:
                    r[i] = feedback[i] / np.sum(p[graph.in_neighbors(i)])
            q = q * np.exp(eta * r)
            q /= np.sum(q)
    print("Exp3G done")
    return rewards


# Algorithm from "Leveraging Side Observations in Stochastic Bandits".
def UCB(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)

        for t in range(n):
            reward, feedback = graph.draw(t)
            rewards[t] += reward / n_runs
            O[t] += 1
            if feedback[t]:
                X[t] = feedback[t] / O[t] + (1. - 1./O[t]) * X[t]

        for t in range(n, T):
            I = np.argmax(X + np.sqrt(2 * np.log(t) / O))
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward) / n_runs
            O[I] += 1
            if not feedback[I] is None:
                X[I] = float(feedback[I]) / O[I] + (1.0 - 1.0 / O[I]) * X[I]
    # =============================================================================
    #             for i in range(n):
    #                 if not feedback[i] is None:
    #                     O[i] += 1
    #                     X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    # =============================================================================
    return rewards


def UCB_N(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            I = np.argmax(X + np.sqrt(2*np.log(t)/O))
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    return rewards


# Algorithm from "Leveraging Side Observations in Stochastic Bandits".
def UCB_MaxN(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T+1, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(1, T+1):
            I = np.argmax(X + np.sqrt(2 * np.log(t) / O))
            # Argmax of I's in-neighbours.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards[1:]


# Algorithm attempt for the stochastic general feedback graph case.
# This not a UCB algorithm but it was inspired from it: exploration and 
# exploitation moves are decoupled.
def Generalized_UCB(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            I = np.argmax(np.hstack((X, np.sqrt(2 * np.log(t) / O))))
            if I < n:  # Exploitation move.
                J = I
            else:  # Exploration move.
                # Argmax of I's in-neighbours.
                J = np.nonzero(graph.in_neighbors(I - n))[0][np.argmax(X[graph.in_neighbors(I - n)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards


def Generalized_UCB_inv(graph, alpha, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            if np.random.rand() > 1 / (1 + alpha * t):  # Exploitation move.
                I = np.argmax(np.hstack(X))
            else:  # Exploration move.
                I = np.argmax(np.hstack(X + np.sqrt(2 * np.log(t) / O)))
                # Argmax of I's in-neighbours.
                J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards


def Generalized_UCB_exp(graph, alpha, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            if np.random.rand() > np.exp(-alpha * t):  # Exploitation move.
                J = np.argmax(np.hstack(X))
            else:  # Exploration move.
                I = np.argmax(np.hstack(X + np.sqrt(2 * np.log(t) / O)))
                # Argmax of I's in-neighbours.
                J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards


# Generalization of Thompson sampling to side-observation graphs.
def Thompson_N(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.ones(n)
        F = np.ones(n)

        for t in range(T):
            theta = np.random.beta(S, F)
            I = np.argmax(theta)
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]
    return rewards


# Generalization of Thompson sampling to side-observation graphs with the idea
# of using the in-neighbor of max reward taken from UCB-MaxN.
def UTS_MaxN(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(T):
            theta = np.random.beta(S + 1, F + 1)
            I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax((S / (S + F))[graph.in_neighbors(I)])]

            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs

            for i in range(n):
                if not feedback[i] is None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]

    return rewards


def OSUB(graph, T, n_runs):
    print('OSUB')
    n = len(graph.arms)
    rewards = np.zeros(T+1)
    for run in range(n_runs):
        X = np.zeros(n)
        O = np.ones(n)
        for t in range(1, T+1):
            ucbs = X + np.sqrt(2 * np.log(t) / O)
            I = np.argmax(ucbs)
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(ucbs[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards[1:]


def IMED_UB(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T+1)
    for run in range(n_runs):
        X = np.zeros(n)
        O = np.ones(n)
        for t in range(1, T + 1):
            O *  + np.log(O)
            ucbs = X + np.sqrt(2 * np.log(t) / O)
            I = np.argmax(ucbs)
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(ucbs[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards[1:]


def TS_UB(graph, T, n_runs):
    K = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.zeros(K)
        F = np.zeros(K)
        X = np.zeros(K)
        O = np.zeros(K)
        for t in range(T):
            I = np.argmax(X)
            # Argmax of I's in-neighbours.
            # J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            theta = np.random.beta(S + 1, F + 1)
            # I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(theta[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs

            for i in range(K):
                if feedback[i] is not None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards


def UTS(graph, T, n_runs):
    print('UTS')
    n = len(graph.arms)
    rewards = np.zeros(T+1)
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(1, T+1):
            theta = np.random.beta(S + 1, F + 1)
            I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(((S+1) / (S + F + 2))[graph.in_neighbors(I)])]

            reward, feedback = graph.draw(J)

            rewards[t] += float(reward) / n_runs
            if not feedback[J] is None:
                S[J] += feedback[J]
                F[J] += 1 - feedback[J]
    return rewards[1:]


def TS(graph, T, n_runs):
    """
    Thompson sampling
    """
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(T):
            theta = np.random.beta(S + 1, F + 1)
            J = np.argmax(theta)
            reward, feedback = graph.draw(J)

            rewards[t] += float(reward) / n_runs
            if feedback[J] is not None:
                S[J] += feedback[J]
                F[J] += 1 - feedback[J]
    return rewards


# Generalization of Thompson sampling to side-observation graphs with the idea
# of using the in-neighbor of max reward taken from UCB-MaxN.
def Thompson_MaxN(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(T):
            theta = np.random.beta(S+1, F+1)
            I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax((S/(S+F))[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]
    return rewards