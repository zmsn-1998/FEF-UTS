

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from algos import *
from bandit import BernoulliArm
from feedback_graph import FeedbackGraph

# np.random.seed(seed=0)




def exp():
    names = ["17-fine-arms", "35-fine-arms","45-fine-arms","101-fine-arms"]
    pre_arms = list()
    # # pre_arm is a list with prob
    # 17-fine-arms
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

    for name,pre_arm in zip(names, pre_arms):

        data = dict()
        data.clear()
        data["exp3g"] = []
        data['ts'] = []
        data["osub"] = []
        data["uts"] = []
        data["imed_ub"] = []
        data["klucb_ub"] = []
        data["fef_uts"] = []
        for rd in range(5):
            np_rng = np.random.RandomState(np.random.randint(0,2000))
            assert sum(x == max(pre_arm) for x in pre_arm) == 1, 'not unimodal'
            ###########################
            print(pre_arm)
            n = len(pre_arm)
            print('num of arm', n)
            print(pre_arm[0], pre_arm[-1])
            T=2000
            #T = 30000 if n > 40 else 5000
            n_runs = 20

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
            graphs = [line]
            
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
                data["x"] = x
                best_cum_reward = graph.best_mean() * x

                plt.plot(x, best_cum_reward - np.cumsum(exp3g_rewards), label='Exp3.G')
                #plt.plot(x, best_cum_reward - np.cumsum(ts_rewards), label='TS')
                plt.plot(x, best_cum_reward - np.cumsum(osub_rewards), label='OSUB')
                plt.plot(x, best_cum_reward - np.cumsum(uts_rewards), label='UTS')
                plt.plot(x, best_cum_reward - np.cumsum(imed_ub_rewards), label='IMED-UB')
                plt.plot(x, best_cum_reward - np.cumsum(klucb_ub_rewards), label='KLUCB-UB')
                plt.plot(x, best_cum_reward - np.cumsum(fef_uts_rewards), label='FEF-UTS')
                data["exp3g"].append(best_cum_reward - np.cumsum(exp3g_rewards))
                data['ts'].append(best_cum_reward - np.cumsum(ts_rewards))
                data["osub"].append(best_cum_reward - np.cumsum(osub_rewards))
                data["uts"].append(best_cum_reward - np.cumsum(uts_rewards))
                data["imed_ub"].append(best_cum_reward - np.cumsum(imed_ub_rewards))
                data["klucb_ub"].append(best_cum_reward - np.cumsum(klucb_ub_rewards))
                data["fef_uts"].append(best_cum_reward - np.cumsum(fef_uts_rewards))
                np.save("./data_fef/"+ name +".npy",data)
                print("save the result of " + name)
                plt.legend(loc="upper left")
                # plt.suptitle(graph.name, fontsize=20)
                plt.xlabel("Round")
                plt.ylabel("Regret")
                plt.xlim(0, T)
                # plt.ylim(0, 1500)
                plt.grid(ls='--')
                plt.show()


def plotx():
    names = ["17-fine-arms", "35-fine-arms","45-fine-arms","101-fine-arms"]
    
    for name in names:


        data = np.load("./data_fef/" + name + ".npy",allow_pickle=True).item()
        print(data.keys())


        x = data['x']
        for key in data.keys():
            if str(key) == 'x':
                break
            if str(key) == 'ts':
                continue
            y = np.array(data[key])
            y_mean = y.mean(axis=0)
            y_max = y.max(axis=0)
            y_min = y.min(axis=0)
            y_std = y.std(axis=0)
            print(name, ":", str(key), ",")
            print(x.shape)
            print(y_std.shape)
            print(y_mean.shape)
            print("--------")
            # exit(0)
            labels = ''
            lens=0
            colors = ""
            if key == "exp3g":
                labels = "Exp3.G"
                colors = "dodgerblue"
            elif key == "ts":
                labels = "TS"
                colors = "b"
            elif key == "osub":
                labels = "OSUB"
                colors = "c"
            elif key == "uts":
                labels = "UTS"
                colors = "gray"
            elif key == "imed_ub":
                colors = "fuchsia"
                labels = "IMED-UB"
            elif key == "klucb_ub":
                labels = "KLUCB-UB"
                colors = "saddlebrown"
            elif key == "fef_uts":
                labels = "FEF-UTS"
                colors = "r"
            #if labels != "FEF-UTS":
            #    plt.errorbar(x[::50], y_mean[::50], yerr=y_std[::50], fmt='-s',ms=4,label = labels)
            #else:
            plt.errorbar(x[::50], y_mean[::50], yerr=y_std[::50], fmt='-s',ms=4,label = labels,color=colors)
            #plt.plot(x, y_mean, label = str(key))
            # plt.fill_between(x,
            #                 y_mean-y_std,
            #                 y_mean+y_std,
            #                 alpha=0.2)

        plt.xlabel("Round")
        plt.ylabel("Regret")
        plt.legend()
        plt.xlim(0, 2000)
        plt.ylim(0)
        #plt.savefig("./picts_fef/" + name + '.pdf', bbox_inches='tight')
        plt.show()





if __name__ == '__main__':

    #exp()
    plotx()
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
