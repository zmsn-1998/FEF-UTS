
import numpy as np

import bandit


class FeedbackGraph(object):
    """ General feedback graph.
    The underlying directed graph is specified by the adjacency matrix.
    """

    # TODO: Extend to graphs with observation probabilities
    def __init__(self, adjacency, name, arms=None, np_rng=None):
        self.adj = adjacency
        self.n_arms = len(adjacency)
        self.name = name
        self.np_rng = np_rng or np.random.RandomState(0)
        self.arms = arms or [bandit.BernoulliArm(np_rng=self.np_rng) for _ in range(self.n_arms)]

    def draw(self, index):
        """ Draw from an arm.
        Returns the reward for regret computations and the feedback according to
        the given graph.
        The feedback is a list of size n_arms with None when there is no feedback.
        """
        if index >= self.n_arms:
            raise IndexError
        reward = self.arms[index].draw()
        feedback = [self.arms[j].draw() if self.adj[index][j] else None for j in range(self.n_arms)]
        if self.adj[index][index]:
            feedback[index] = reward
        # TODO: changed
        # feedback = [None] * self.n_arms
        # feedback[index] = reward
        return reward, feedback

    def best_mean(self):
        result = -np.inf
        for arm in self.arms:
            if arm.get_mean() > result:
                result = arm.get_mean()
        return result

    def in_neighbors(self, index):
        """incoming neighbors"""
        return self.adj[:, index]

    def out_neighbors(self, index):
        """outgoing neighbors"""
        return self.adj[index, :]
