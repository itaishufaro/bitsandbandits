import numpy as np
import mab
import matplotlib.pyplot as plt
import tqdm
import thompson_sampling

class aps:
    def __init__(self, n_arms, eta):
        self.n_arms = n_arms
        self.p = np.ones((n_arms,)) / n_arms
        self.eta = eta
        self.i_prior = np.copy(self.p)

    def pull_arm(self):
        a = np.random.choice(self.n_arms, p=self.p)
        return a

    def reset(self):
        self.p = np.ones((self.n_arms,)) / self.n_arms

    def update(self, action, reward):
        prev_dist = np.copy(self.p)
        if reward == 1:
            x = ((1 - np.exp(-self.eta, dtype=np.longdouble)) /
                 (1 - np.exp(-self.eta / self.p[action], dtype=np.longdouble)))
        else:
            x = (1 - np.exp(self.eta)) / (1 - np.exp(self.eta / self.p[action]))
        for a in range(self.n_arms):
            if a != action:
                self.p[a] *= (1 - x) / (1 - self.p[action])
            else:
                self.p[a] = x
        self.p /= np.sum(self.p)
        if np.any(np.isnan(self.p)):
            self.p = prev_dist
            kl = 0
        elif np.any(self.p == 0):
            kl = 0
        else:
            kl = np.sum(self.p * np.log2(np.divide(self.p, self.i_prior, where=(self.p != 0), dtype=np.longdouble)))
        return kl

    def run_trial(self, bandit, n_steps):
        regret = np.zeros((n_steps,))
        mi = np.zeros((n_steps,))
        N = self.n_arms
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
