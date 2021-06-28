
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from algos import *
from bandit import BernoulliArm
from feedback_graph import FeedbackGraph

# np.random.seed(seed=0)
np_rng = np.random.RandomState(10)

#
# pre_arm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
# pre_arm = [0.1,0.3,0.5,0.7,0.9,0.7,0.5,0.3,0.1]

# 45-fine-arms [0.46 ~ 0.9 ~ 0.46]
a = list(np.arange(0.46, 0.9, 0.02))
b = list(np.arange(0.9, 0.44, -0.02))
pre_arm = a + b
pre_arm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# pre_arm = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
#                     0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.9, 0.85, 0.8, 0.75,
#                     0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
#                     0.15, 0.1])

# 40- fine-arms [0.1,0.9,0.01]
# a = list(np.arange(0.6,0.9,0.01))
# b = list(np.arange(0.9,0.6,-0.01))
# pre_arm = a + b

# 129 - fine-arms [0.1,0.9,0.01]
a = list(np.arange(0.24, 0.85, 0.01))
b = list(np.arange(0.85, 0.24, -0.01))
# pre_arm = a + b

# pre_arm = list(np.random.random(127))
# pre_arm = sorted(pre_arm)
# a = pre_arm[::2]
# b = pre_arm[1::2]
# b = b[::-1]
# pre_arm = a + b
assert sum(x == max(pre_arm) for x in pre_arm) == 1, 'not unimodal'
###########################
print(pre_arm)
n = len(pre_arm)
print('num of arm', n)
print(pre_arm[0], pre_arm[-1])

#T = 30000 if n > 40 else 5000
#n_runs = 50
T = 100
n_runs=50

A_line = np.asarray(np.tri(n, k=1) - np.tri(n, k=-2), dtype=bool)
A_line_2 = np.asarray(np.tri(n, k=2) - np.tri(n, k=-3), dtype=bool)
arms = [BernoulliArm(mean=pre_arm[i], np_rng=np_rng) for i in range(n)]

line = FeedbackGraph(A_line, 'line', arms=arms, np_rng=np_rng)
line2 = FeedbackGraph(A_line_2, 'line_2', arms=arms, np_rng=np_rng)

# Learning rate.
eta = 0.1
# Regularisation parameter.
gamma = 0.01
# List of nodes.
U = list(range(n))


def exp(graphs):
    # p = Pool(mp.cpu_count())
    for graph in graphs:
        print(graph.name)
        exp3g_rewards,_ = Exp3G(eta, gamma, U, graph, T, n_runs)
        ts_rewards, _ = TS(graph, T, n_runs)
        osub_rewards, _ = OSUB(graph, T, n_runs)
        uts_rewards,_ = UTS(graph, T, n_runs)
        imed_ub_rewards,_ = IMED_UB(graph, T, n_runs)
        klucb_ub_rewards,_ = KLUCB_UB(graph, T, n_runs)
        fef_uts_rewards,_ = FEF_UTS(graph, T, n_runs)

        with open(f'{graph.name}_{graph.n_arms}.pkl', 'wb') as f:
            pickle.dump(([exp3g_rewards, ts_rewards, osub_rewards, uts_rewards,
                          imed_ub_rewards, klucb_ub_rewards, fef_uts_rewards], graph), file=f)

        plt.figure()
        x = np.arange(T)
        best_cum_reward = graph.best_mean() * x
        plt.plot(x, best_cum_reward - np.cumsum(exp3g_rewards), label='Exp3.G')
        plt.plot(x, best_cum_reward - np.cumsum(ts_rewards), label='TS')
        plt.plot(x, best_cum_reward - np.cumsum(osub_rewards), label='OSUB')
        plt.plot(x, best_cum_reward - np.cumsum(uts_rewards), label='UTS')
        plt.plot(x, best_cum_reward - np.cumsum(imed_ub_rewards), label='IMED-UB')
        plt.plot(x, best_cum_reward - np.cumsum(klucb_ub_rewards), label='KLUCB-UB')
        plt.plot(x, best_cum_reward - np.cumsum(fef_uts_rewards), label='FEF-UTS')

        plt.legend(loc="upper left")
        # plt.suptitle(graph.name, fontsize=20)
        plt.xlabel("Round")
        plt.ylabel("Regret")
        plt.xlim(0, T)
        # plt.ylim(0, 1500)
        plt.grid(ls='--')
        plt.show()
        #plt.savefig("figure/" + graph.name + '.pdf', bbox_inches='tight')


def plot(filename, xlim, ylim, ts_vs_fef=False):
    with open(filename, 'rb') as f:
        rewards, graph = pickle.load(f)
        if filename == 'line_2_45.pkl':
            exp3g_rewards, ts_rewards, osub_rewards, uts_rewards, fef_uts_rewards, imed_ub_rewards, klucb_ub_rewards = rewards
            rewards = [exp3g_rewards, ts_rewards, osub_rewards, uts_rewards, imed_ub_rewards, klucb_ub_rewards, fef_uts_rewards]

        exp3g_rewards, ts_rewards, osub_rewards, uts_rewards, imed_ub_rewards, klucb_ub_rewards, fef_uts_rewards = rewards
        labels = ['Exp3.G', 'TS', 'OSUB', 'UTS', 'IMED-UB', 'KLUCB-UB', 'FEF-UTS']
        colors = ['b', 'indigo', 'c', 'y', 'g', 'm', 'r']
        T = len(ts_rewards)
        x = np.arange(T)
        best_cum_reward = graph.best_mean() * x
        plt.figure()
        if ts_vs_fef:
            sns.lineplot(x=x, y=best_cum_reward - np.cumsum(ts_rewards), label='TS', color='steelblue')
            sns.lineplot(x=x, y=best_cum_reward - np.cumsum(fef_uts_rewards), label='FEF-UTS', color='r')
        else:
            for reward, label, c in zip(rewards, labels, colors):
                if label == 'TS':
                    continue
                sns.lineplot(x=x, y=best_cum_reward - np.cumsum(reward), label=label, color=c)

        plt.xlabel("Round", fontdict={'fontsize':20})
        plt.ylabel("Regret", fontdict={'fontsize':20})
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.grid(ls='--')
        if ts_vs_fef:
            plt.savefig(f"figure/paper/{graph.name}_arm_{graph.n_arms}_ts_vs_fef.pdf", bbox_inches='tight')
        else:
            plt.savefig(f"figure/paper/{graph.name}_arm_{graph.n_arms}.pdf", bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    graphs = [line]

    exp(graphs)

    # ts_vs_fef
    # xlims = [5000, 5000, 5000, 5000]
    # ylims = [100, 120, 250, 200]
    # #
    # xlims = [5000, 5000, 5000, 5000]
    # ylims = [2000, 2000, 1000, 1200]
    #
    # for i, file in enumerate(['line_17.pkl', 'rew_line_35.pkl', 'line_2_45.pkl', 'rew_star_101.pkl']):
    #     if i != 3: continue
    #     print(file)
    #     xlim = xlims[i]
    #     ylim = ylims[i]
    #     plot(file, xlim, ylim)
    #     # plot(file, xlim, ylim, ts_vs_fef=True)
