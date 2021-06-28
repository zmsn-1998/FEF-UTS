

import matplotlib.pyplot as plt

from algorithms import *
from bandit import BernoulliArm
from feedback_graph import FeedbackGraph

# np.random.seed(seed=0)
np_rng = np.random.RandomState(0)

#
# pre_arm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# pre_arm = [0.1,0.3,0.5,0.7,0.9,0.7,0.5,0.3,0.1]

# 16-fine-arms
a = list(np.arange(0.46, 0.9, 0.02))
b = list(np.arange(0.9, 0.44, -0.02))
# pre_arm is a list with prob
pre_arm = a + b


# 40- fine-arms [0.1,0.9,0.01]
# a = list(np.arange(0.6,0.9,0.01))
# b = list(np.arange(0.9,0.6,-0.01))
# pre_arm = a + b

###########################


# 60- fine-arms [0.1,0.9,0.01]
# a = list(np.arange(0.6,0.8,0.01))
# b = list(np.arange(0.8,0.6,-0.01))
# pre_arm = a + b


n = len(pre_arm)

T = 3000
n_runs = 20

A_full = np.ones((n, n), dtype=bool)
A_bandit = np.eye(n, dtype=bool)
A_line = np.asarray(np.tri(n, k=1) - np.tri(n, k=-2), dtype=bool)
A_line_2 = np.asarray(np.tri(n, k=2)-np.tri(n, k=-3), dtype=bool)

print("A_full", A_full)
print("--------")
print("A_bandit", A_bandit)


A_loopless = A_full ^ A_bandit

print("A_loopless", A_loopless)
A_revealing = np.vstack((np.ones((1, n), dtype=bool), np.zeros((n - 1, n), dtype=bool)))
A_unobservable = np.ones((n, n), dtype=bool)

# arms = [BernoulliArm(mean=(i+1)/(n+1), np_rng=np_rng) for i in range(n)]
arms = [BernoulliArm(mean=pre_arm[i], np_rng=np_rng) for i in range(n)]


# arms = [0.1,0.2,0.6,0.3,0.1]


full = FeedbackGraph(A_full,"full", arms=arms, np_rng=np_rng)
bandit = FeedbackGraph(A_line_2, "bandit", arms=arms, np_rng=np_rng)
loopless = FeedbackGraph(A_loopless, "loopless", arms=arms, np_rng=np_rng)
revealing = FeedbackGraph(A_revealing, "revealing", arms=arms, np_rng=np_rng)
# weak = FeedbackGraph(A_weak, arms=arms, np_rng=np_rng)


# Learning rate.
eta = 0.1
# Regularisation parameter.
gamma = 0.01
# List of nodes.
U = list(range(n))

graphs = [bandit]
# graphs = [full, bandit, loopless, revealing]
# names = ["Full", "Bandit", "Loopless", "Revealing"]
names = ["45-second-bandit-Arms-2"]
full_flag = True

# TS_rewards-UB
# OSUB / KL-UB
# UTS_rewards
# TS_rewards
# IMDE-UB /
# Exp3G


if full_flag:
    for graph, name in zip(graphs, names):
        print("Working with", name)

        Exp3G_rewards = Exp3G(eta, gamma, U, graph, T, n_runs)

        UCB_rewards = UCB(graph, T, n_runs)

        UCB_MaxN_rewards = UCB_MaxN(graph, T, n_runs)

        Generalized_UCB_rewards = Generalized_UCB(graph, T, n_runs)

        # Thompson_N_rew = Thompson_N(graph, T, n_runs)

        TS_N_rewards = Thompson_N(graph, T, n_runs)

        TS_UB_rewards = TS_UB(graph, T, n_runs)

        UTS_rewards = UTS(graph, T, n_runs)

        TS_rewards = TS(graph, T, n_runs)

        # plot
        plt.figure()
        x = np.arange(T)
        best_cum_reward = graph.best_mean() * x
        plt.plot(x, best_cum_reward - np.cumsum(UCB_rewards))
        UCB_cum = best_cum_reward - np.cumsum(UCB_rewards)
        #np.save("./data2/" + name + "UCB_N.npy", UCB_cum)

        plt.plot(x, best_cum_reward - np.cumsum(UCB_MaxN_rewards))
        UCB_MaxN_cum = best_cum_reward - np.cumsum(UCB_MaxN_rewards)
        #np.save("./data2/" + name + "UCB_MaxN.npy", UCB_MaxN_cum)

        plt.plot(x, best_cum_reward - np.cumsum(Generalized_UCB_rewards))
        Generalized_UCB_cum = best_cum_reward - np.cumsum(Generalized_UCB_rewards)
        #np.save("./date2/" + name + "OSUB.npy", Generalized_UCB_cum)

        plt.plot(x, best_cum_reward - np.cumsum(Exp3G_rewards))
        Exp3G_cum = best_cum_reward - np.cumsum(Exp3G_rewards)
        #np.save("./date2/" + name + "Exp3G.npy", Exp3G_cum)

        plt.plot(x, best_cum_reward - np.cumsum(UTS_rewards))
        UTS_cum = best_cum_reward - np.cumsum(UTS_rewards)
        #np.save("./date2/" + name + "UTS_rewards.npy", UTS_cum)

        plt.plot(x, best_cum_reward - np.cumsum(TS_N_rewards))
        TS_N_cum = best_cum_reward - np.cumsum(TS_N_rewards)
        #np.save("./date2/" + name + "TS_N_rewards.npy", TS_N_cum)
        # plt.legend(["Exp3.G","UCB-N","UCB-MaxN","UCB","Thompson-N","UTS_rewards","UTS_rewards-MaxN"], loc="upper left")
        # plt.plot(x, graph.best_mean()*np.array(range(T)) - np.cumsum(Generalized_UCB_inv_rew))
        # plt.plot(x, graph.best_mean()*np.array(range(T)) - np.cumsum(Generalized_UCB_exp_rew))
        plt.plot(x, best_cum_reward - np.cumsum(TS_rewards))
        TS_cum = best_cum_reward - np.cumsum(TS_rewards)
        #np.save("./date2/" + name + "TS_rewards.npy", TS_cum)

        plt.plot(x, best_cum_reward - np.cumsum(TS_UB_rewards))
        TS_UB_cum = best_cum_reward - np.cumsum(TS_UB_rewards)
        #np.save("./date2/" + name + "TS_UB_rewards.npy", TS_UB_cum)

        plt.legend(["UCB", "UCB-MaxN", "OSUB", "EXP3", "UTS_rewards", "TS_rewards-N", "TS_rewards", "TS_UB_rewards"], loc="upper left")
        plt.suptitle(name, fontsize=20)
        plt.xlabel("Round")
        plt.ylabel("Regret")
        plt.show()
        #plt.savefig("./date2/" + name + '.pdf', bbox_inches='tight')

else:
    # Comparison of generalized UCBs.
    for graph, name in zip(graphs, names):
        print("Working with", name)
        Exp3G_rewards = Exp3G(eta, gamma, U, graph, T, n_runs)
        Generalized_UCB_rewards = Generalized_UCB(graph, T, n_runs)
        Generalized_UCB_inv_rew = Generalized_UCB_inv(graph, 0.1, T, n_runs)
        Generalized_UCB_exp_rew = Generalized_UCB_exp(graph, 0.1, T, n_runs)
        f = plt.figure()
        plt.plot(range(T), graph.best_mean() * np.array(range(T)) - np.cumsum(Exp3G_rewards))
        plt.plot(range(T), graph.best_mean() * np.array(range(T)) - np.cumsum(Generalized_UCB_rewards))
        plt.plot(range(T), graph.best_mean() * np.array(range(T)) - np.cumsum(Generalized_UCB_inv_rew))
        plt.plot(range(T), graph.best_mean() * np.array(range(T)) - np.cumsum(Generalized_UCB_exp_rew))
        plt.legend(["Exp3G", "Generalized-UCB", "Generalized-UCB-inv", "Generalized-UCB-exp"], loc="upper left")
        plt.suptitle(name, fontsize=20)
        plt.xlabel("Time")
        plt.ylabel("Regret")
        f.savefig("./date2/" + name + '.pdf', bbox_inches='tight')
