import numpy as np
from scipy.stats import rv_discrete


# Algorithm from "Online Learning with Feedback Graphs: Beyond Bandits".
class Exp3G:

    def __init__(self, eta, gamma, U):
        self.eta = eta
        self.gamma = gamma
        self.U = U

    def __call__(self, graph, T, n_runs):
        U, gamma, eta = self.U, self.gamma, self.eta
        n = len(graph.arms)
        u = np.zeros(n, dtype=float)
        u[U] = 1.0 / len(U)
        rewards = np.zeros(T, dtype=float)
        for run in range(n_runs):
            q = np.ones(n, dtype=float) / n
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
        return rewards
