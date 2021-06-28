
import matplotlib.pyplot as plt
import numpy as np
from algorithms import *
from bandit import BernoulliArm
from feedback_graph import FeedbackGraph

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


def data_catch():
    f = plt.figure()
    names = ["17-fine-arms", "35-fine-arms","45-fine-arms","101-fine-arms"]
    pre_arms = list()
    # pre_arm is a list with prob
    #17-fine-arms
    pre_arm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pre_arms.append(pre_arm)
    # 35-fine-arms
    pre_arm = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                        0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.9, 0.85, 0.8, 0.75,
                        0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
                        0.15, 0.1])
    pre_arms.append(pre_arm)

    # 45-fine-arms

    a = list(np.arange(0.46, 0.9, 0.02))
    b = list(np.arange(0.9, 0.44, -0.02))
    pre_arm = a + b
    pre_arms.append(pre_arm)

    # 101-fine-arms
    a = list(np.arange(0.4, 0.9, 0.01))#61
    b = list(np.arange(0.9, 0.39, -0.01))#
    pre_arm = a + b
    pre_arms.append(pre_arm)



    for pre_arm,name in zip(pre_arms,names):
        yy_ts = []
        yy_fef = []
        data = dict()
        for T in range(5):
            np_rng = np.random.RandomState(np.random.randint(0,2000))
            print(len(pre_arm))
            n = len(pre_arm)
            # the number of round
            T = 5000
            # the number of runs in each round
            n_runs = 20
            arms = [BernoulliArm(mean=pre_arm[i], np_rng=np_rng) for i in range(n)]
            A_full = np.ones((n, n), dtype=bool)
            A_bandit = np.eye(n, dtype=bool)
            A_line = np.asarray(np.tri(n, k=1) - np.tri(n, k=-2), dtype=bool)
            A_line_2 = np.asarray(np.tri(n, k=2)-np.tri(n, k=-3), dtype=bool)
            A_loopless = A_full ^ A_bandit

            A_revealing = np.vstack((np.ones((1, n), dtype=bool), np.zeros((n - 1, n), dtype=bool)))
            A_unobservable = np.ones((n, n), dtype=bool)
            bandit = FeedbackGraph(A_line_2, "bandit", arms=arms, np_rng=np_rng)

            graph = bandit

            
            x = np.arange(T)

            # load 
            data["x"] = x 

            best_cum_reward = graph.best_mean() * x
            TS_rewards = TS(bandit, T, n_runs)
            fef_uts_rewards,_ = FEF_UTS(bandit, T, n_runs)
            plt.plot(x, best_cum_reward - np.cumsum(TS_rewards), label="TS")
            yy_ts.append(best_cum_reward - np.cumsum(TS_rewards))
            plt.plot(x, best_cum_reward - np.cumsum(fef_uts_rewards), label='FEF-UTS')
            yy_fef.append(best_cum_reward - np.cumsum(fef_uts_rewards))
            

        data["yy_ts"] = yy_ts
        data["yy_fef"] = yy_fef
        #plt.show()
        np.save("./data_ts/"+ name+".npy",data)

def plot_pict():
    names = ["17-fine-arms", "35-fine-arms","45-fine-arms","101-fine-arms"]
    
    for name in names:


        data = np.load("./data_ts/" + name + ".npy",allow_pickle=True).item()


        x = data['x']
        yy_ts = data['yy_ts']
        yy_fef = data["yy_fef"]

        f = plt.figure()
        yy_ts = np.array(yy_ts)
        yy_ts_mean = yy_ts.mean(axis=0)
        yy_ts_max = yy_ts.max(axis=0)
        yy_ts_min = yy_ts.min(axis=0)
        yy_ts_std = yy_ts.std(axis=0)

        plt.errorbar(x[::120], yy_ts_mean[::120], yerr=yy_ts_std[::120], fmt='-s',ms=4,label = "TS",color="c")
        
        yy_fef = np.array(yy_fef)
        yy_fef_mean = yy_fef.mean(axis=0)
        yy_fef_max = yy_fef.max(axis=0)
        yy_fef_min = yy_fef.min(axis=0)
        yy_fef_std = yy_fef.std(axis=0)

        print(x.shape)
        plt.errorbar(x[::125], yy_fef_mean[::125], yerr=yy_fef_std[::125], fmt='-s',ms=4,label = "FEF-UTS",color="r")
        # plt.plot(x, yy_fef_mean, label = "FEF-UTS")
        # plt.fill_between(x,
        #                 yy_fef_mean-0.5*yy_fef_std,
        #                 yy_fef_mean+0.5*yy_fef_std,
        #                 color='b',
        #                 alpha=0.2)
        plt.xlabel("Round")
        plt.ylabel("Regret")
        plt.legend(loc="upper left")
        plt.xlim(0, 5000)
        plt.ylim(0)
        #plt.title(name) 
        f.savefig("./picts_ts/" + name + '.pdf', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    data_catch()
    #plot_pict()