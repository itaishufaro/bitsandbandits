import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import tqdm

import mab

def entropy(p):
    return -np.sum(p * np.log2(p))


class k_models_ts:
    def __init__(self, n_arms, epsilon=0.1, prior=None):
        self.n_arms = n_arms
        if prior is None:
            self.prior = np.ones((n_arms,)) / n_arms
        else:
            self.prior = prior
        self.posterior = np.copy(self.prior)
        self.N_successes = np.zeros((n_arms,))
        self.N_failures = np.zeros((n_arms,))
        self.epsilon = epsilon

    def reset(self):
        self.posterior = np.copy(self.prior)
        self.N_successes = np.zeros((self.n_arms,))
        self.N_failures = np.zeros((self.n_arms,))

    def choose(self):
        ind = np.random.choice(self.n_arms, 1, p=self.posterior)
        return ind[0]

    def update_posterior(self):
        prev_posterior = np.copy(self.posterior)
        rem_failure = np.sum(self.N_failures) - self.N_failures
        rem_success = np.sum(self.N_successes) - self.N_successes
        mean_pos = np.log(0.5 + 0.5 * self.epsilon)
        mean_neg = np.log(0.5 - 0.5 * self.epsilon)
        tmp = (self.N_successes + rem_failure) * mean_pos + (self.N_failures + rem_success) * mean_neg
        x = np.exp(tmp, dtype=np.longdouble) * self.prior
        x = np.divide(x, np.sum(x), dtype=np.longdouble)
        if np.any(np.isnan(x)):
            self.posterior = prev_posterior
        else:
            self.posterior = x
        kl = np.sum(self.posterior * np.log2(np.divide(self.posterior, self.prior), where=(self.posterior != 0),
                                             dtype=np.longdouble))
        return kl

    def update(self, action, reward):
        if reward == 1:
            self.N_successes[action] += 1
        else:
            self.N_failures[action] += 1
        kl = self.update_posterior()
        return kl

    def run_trial(self, bandit, episodes):
        regret = np.zeros((episodes,))
        mi = np.zeros((episodes,))
        for i in range(episodes):
            action = self.choose()
            reward = bandit.pull(int(action))
            kl = self.update(int(action), reward)
            if i == 0:
                regret[i] = bandit.get_optimal() - bandit.get_means()[action]
                mi[i] = kl
            else:
                regret[i] = bandit.get_optimal() - bandit.get_means()[action] + regret[i-1]
                mi[i] = kl
        return regret, mi

    def run_experiment(self, bandit, episodes, trials):
        regret = np.zeros((trials, episodes))
        mi = np.zeros((trials, episodes))
        for i in range(trials):
            regret[i] = self.run_trial(bandit, episodes)
            mi[i] = self.run_trial(bandit, episodes)
            self.reset()
        return regret, mi

    def run_experiment_bayesian(self, bandit, episodes, trials):
        regret = np.zeros((trials, episodes))
        mi = np.zeros((trials, episodes))
        for i in tqdm.tqdm(range(trials)):
            new_bandit = bandit.new_problem()
            regret[i], mi[i] = self.run_trial(new_bandit, episodes)
            self.reset()
        return regret, mi


class beta_bernoulli_ts:
    def __init__(self, n_arms, alphas=None, betas=None):
        self.n_arms = n_arms
        if alphas is None:
            self.alpha = np.ones((n_arms,))
        else:
            self.alpha = alphas
        if betas is None:
            self.beta = np.ones((n_arms,))
        else:
            self.beta = betas

        self.alpha0 = np.copy(self.alpha)
        self.beta0 = np.copy(self.beta)

    def reset(self):
        self.alpha = np.copy(self.alpha0)
        self.beta = np.copy(self.beta0)

    def choose(self):
        samples = np.random.beta(self.alpha, self.beta)
        ind = np.argmax(samples)
        return ind

    def update(self, action, reward):
        if reward == 1:
            self.alpha[int(action)] += 1
        else:
            self.beta[int(action)] += 1

    def run_trial(self, bandit, episodes):
        regret = np.zeros((episodes,))
        for i in range(episodes):
            action = self.choose()
            reward = bandit.pull(action)
            if i == 0:
                regret[i] = bandit.get_optimal() - bandit.get_means()[action]
            else:
                regret[i] = bandit.get_optimal() - bandit.get_means()[action] + regret[i-1]
            self.update(action, reward)
        return regret

    def run_experiment(self, bandit, episodes, trials):
        regret = np.zeros((trials, episodes))
        for i in range(trials):
            regret[i] = self.run_trial(bandit, episodes)
            self.reset()
        return regret

    def run_experiment_bayesian(self, bandit, episodes, trials):
        regret = np.zeros((trials, episodes))
        for i in range(trials):
            new_bandit = bandit.new_problem()
            regret[i] = self.run_trial(new_bandit, episodes)
            self.reset()
        return regret

