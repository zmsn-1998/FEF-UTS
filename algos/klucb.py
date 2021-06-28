import numpy as np

import os.path
import pickle


def write_data(sub_arm_list, total_rewards_list, name):
    """
    Write data in file
    """
    filename = 'data/scenario_1_' + name + '.pkl'

    if os.path.exists(filename):
        update_data_file(sub_arm_list, total_rewards_list, filename)
    else:
        create_data_file(sub_arm_list, total_rewards_list, filename)


def create_data_file(sub_arm_list, total_rewards_list, filename):
    """
    Create file and add data
    """
    parameters = {
        'sub_arm_list': sub_arm_list,
        'total_rewards_list': total_rewards_list
    }
    output = open(filename, 'wb')
    pickle.dump(parameters, output)
    output.close()


def update_data_file(sub_arm_list, total_rewards_list, filename):
    """
    Update data in file
    """
    parameters = read_file(filename)
    # update
    parameters['sub_arm_list'] += sub_arm_list
    parameters['total_rewards_list'] = np.concatenate((parameters['total_rewards_list'], total_rewards_list), axis=0)
    # write
    write = open(filename, 'wb')
    pickle.dump(parameters, write)
    write.close()


def get_data(filename):
    """
    Get data from file
    """
    parameters = read_file(filename)
    return parameters['sub_arm_list'], parameters['total_rewards_list']


def read_file(filename):
    """
    Read file
    """
    read = open(filename, 'rb')
    parameters = pickle.load(read)
    read.close()
    return parameters


def get_results(total_rewards_list, sub_arm_list):
    """
    return results data for scenario 1 and 2 graphs
    """
    runs = len(sub_arm_list)
    mean_total_rewards_list = np.mean(total_rewards_list, axis=0)
    mean_sub_arm = np.mean(sub_arm_list, axis=0)  # Mean number of the suboptimal arm as a function of time
    sub_arm_draws_T = np.zeros(runs)  # number of draws of the suboptimal arm at tim n=5000
    for i in range(runs):
        sub_arm_draws_T[i] = np.sum(sub_arm_list[i][:5000])

    return mean_total_rewards_list, mean_sub_arm, sub_arm_draws_T

#########################


def kl_bernoulli(p, q):
    """
    Compute kl_distance-divergence for Bernoulli distributions
    """
    result = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return result


def dkl_bernoulli(p, q):
    result = (q - p) / (q * (1.0 - q))
    return result


def kl_exponential(p, q):
    """
    Compute kl_distance-divergence for Exponential distributions
    """
    result = (p / q) - 1 - np.log(p / q)
    return result


def dkl_exponential(p, q):
    result = (q - p) / (q ** 2)
    return result


def klucb_upper_newton(kl_distance, N, S, k, t, precision=1e-6, max_iterations=50, dkl=dkl_bernoulli):
    """
    Compute the upper confidence bound for each arm using Newton's iterations method
    """
    delta = 0.1
    logtdt = np.log(t) / N[k]
    p = max(S[k] / N[k], delta)
    if p >= 1:
        return 1

    converged = False
    q = p + delta

    f, df = 0., 0.
    for n in range(max_iterations):
        f = logtdt - kl_distance(p, q)
        df = - dkl(p, q)

        if f * f < precision:
            converged = True
            break

    q = min(1 - delta, max(q - f / df, p + delta))

    if not converged:
        print("KL-UCB algorithm: Newton iteration did not converge!", "p=", p, "logtdt=", logtdt)

    return q


def klucb_upper_bisection(kl_distance, N, S, k, t, precision=1e-6, max_iterations=50):
    """
    Compute the upper confidence bound for each arm with bisection method
    """
    upperbound = np.log(t) / N[k]
    reward = S[k] / N[k]

    u = upperbound
    l = reward
    n = 0

    while n < max_iterations and u - l > precision:
        q = (l + u) / 2
        if kl_distance(reward, q) > upperbound:
            u = q
        else:
            l = q
        n += 1

    return (l + u) / 2


class KLUCBPolicy:
    """
    KL-UCB algorithm
    """

    def __init__(self, K, klucb_upper=klucb_upper_bisection, kl_distance=kl_bernoulli, precision=1e-6,
                 max_iterations=50):
        self.K = K
        self.kl_distance = kl_distance
        self.klucb_upper = klucb_upper
        self.precision = precision
        self.max_iterations = max_iterations
        self.N = None
        self.S = None
        self.reset()

    def reset(self):
        self.N = np.zeros(self.K)
        self.S = np.zeros(self.K)

    def select_next_arm(self):
        t = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K):
            if self.N[k] == 0:
                return k

            # KL-UCB index
            indices[k] = self.klucb_upper(self.kl_distance, self.N, self.S, k, t, self.precision, self.max_iterations)

        selected_arm = np.argmax(indices)
        return selected_arm

    def update_state(self, k, r):
        self.N[k] += 1
        self.S[k] += r


def KLUCB(graph, T, n_runs):
    K = len(graph.arms)
    klucb = KLUCBPolicy(K)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        klucb.reset()
        for t in range(T):
            J = klucb.select_next_arm()
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs

            for k in range(K):
                if feedback[k] is not None:
                    klucb.update_state(k, feedback[k])
    return rewards


def KLUCB(graph, T, n_runs):

    def klucb_upper_bisection(kl_distance, N, S, k, t, precision=1e-6, max_iterations=50):
        """
        Compute the upper confidence bound for each arm with bisection method
        """
        upperbound = np.log(t) / N[k]
        reward = S[k]

        u = upperbound
        l = reward
        n = 0

        while n < max_iterations and u - l > precision:
            q = (l + u) / 2
            if kl_distance(reward, q) > upperbound:
                u = q
            else:
                l = q
            n += 1

        return (l + u) / 2

    def kl_bernoulli(p, q):
        """
        Compute kl_distance-divergence for Bernoulli distributions
        """
        result = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        return result

    K = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(K)
        O = np.zeros(K)
        for t in range(T):
            indices = np.zeros(K)
            J = K
            for k in range(K): # only neighbor
                if O[k] == 0:
                    J = k
                    break
                indices[k] = klucb_upper_bisection(kl_bernoulli, N=O, S=X, k=k, t=t)
            if not J:
                J = np.argmax(indices)

            reward, feedback = graph.draw(J)
            rewards[t] += float(reward) / n_runs

            for i in range(K):
                if feedback[i] is not None:
                    O[i] += 1
                    X[i] = float(feedback[i]) / O[i] + (1.0 - 1.0 / O[i]) * X[i]
    return rewards