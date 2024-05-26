import numpy as np
import mab
import matplotlib.pyplot as plt
import tqdm

class exp3:
    def __init__(self, n_arms, eta=0.1, gamma=0):
        self.n_arms = n_arms
        self.L = np.zeros((n_arms,))
        self.eta = eta
        self.gamma = gamma
        self.p = np.ones((n_arms,)) / n_arms
        self.i_prior = np.copy(self.p)

    def update_p(self):
        tmp = np.exp(-self.eta * self.L)
        self.p = tmp / np.sum(tmp)

    def reset(self):
        self.L = np.zeros((self.n_arms,))
        self.p = np.ones((self.n_arms,)) / self.n_arms

    def pull_arm(self):
        a = np.random.choice(self.n_arms, p=self.p)
        return a

    def update(self, action, reward):
        prev_dist = np.copy(self.p)
        self.L[action] = self.L[action] + (1 - reward) / (self.gamma + self.p[action])
        self.update_p()
        if np.any(np.isnan(self.p)):
            self.p = prev_dist
        if np.all(self.p == 0):
            kl = 0
        else:
            kl = np.sum(self.p * np.log2(np.divide(self.p, self.i_prior)))
        return kl

    def run_trial(self, bandit, n_steps):
        regret = np.zeros((n_steps,))
        mi = np.zeros((n_steps,))
        for i in range(n_steps):
            action = self.pull_arm()
            reward = bandit.pull(action)
            kl = self.update(action, reward)
            if i == 0:
                regret[i] = bandit.get_optimal() - bandit.get_means()[action]
                mi[i] = kl
            else:
                regret[i] = bandit.get_optimal() - bandit.get_means()[action] + regret[i-1]
                mi[i] = kl
        return regret, mi

    def run_experiment(self, bandit, n_steps, n_trials):
        regret = np.zeros((n_trials, n_steps))
        mi = np.zeros((n_trials, n_steps))
        for i in tqdm.tqdm(range(n_trials)):
            regret[i], mi[i] = self.run_trial(bandit, n_steps)
            self.reset()
        return regret, mi

    def run_experiment_bayesian(self, bandit, n_steps, n_trials):
        regret = np.zeros((n_trials, n_steps))
        mi = np.zeros((n_trials, n_steps))
        for i in tqdm.tqdm(range(n_trials)):
            new_bandit = bandit.new_problem()
            regret[i], mi[i] = self.run_trial(new_bandit, n_steps)
            self.reset()
        return regret, mi
