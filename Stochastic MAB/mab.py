import numpy as np


class bernoulli_bandit:
    def __init__(self, n_arms, means):
        self.n_arms = n_arms
        self.means = means
        self.optimal = np.argmax(self.means)

    def pull(self, action):
        return np.random.binomial(1, self.means[action])

    def reset(self):
        self.means = np.random.rand(self.n_arms)
        self.optimal = np.argmax(self.means)

    def step(self):
        return self.pull(np.random.choice(self.n_arms))

    def get_optimal(self):
        return self.means[self.optimal]

    def get_means(self):
        return self.means


class k_models_bandit:
    def __init__(self, n_arms, epsilon=0.1, prior=None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        if prior is None:
            self.prior = np.ones((n_arms,)) / n_arms
        else:
            self.prior = prior

    def new_problem(self):
        opt_arm = np.random.choice(self.n_arms,1, p=self.prior)
        new_means = np.ones((self.n_arms,)) * 0.5 - 0.5 * self.epsilon
        new_means[opt_arm] = 0.5 + 0.5 * self.epsilon
        return bernoulli_bandit(self.n_arms, new_means)


class beta_bernoulli_bandit:
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

    def new_problem(self):
        means = np.random.beta(self.alpha, self.beta)
        return bernoulli_bandit(self.n_arms, means)