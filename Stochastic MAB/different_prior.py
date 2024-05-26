import numpy as np
import mab
import thompson_sampling
import matplotlib.pyplot as plt
import matplotlib
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Comparing Thompson sampling with three different priors for '
                                                 'Bayesian bandits')
    parser.add_argument('--n_arms', type=int, default=8, help='Number of arms')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Gap between rewards')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials')
    parser.add_argument('--p2', type=float, default=0.0205, help='Probability of the best arm')
    parser.add_argument('--p1', type=float, default=0.0532, help='Probability of the second best arm')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    n_arms = args.n_arms
    epsilon = args.epsilon
    episodes = args.episodes
    trials = args.trials
    p2 = args.p2
    p1 = args.p1
    matplotlib.rcParams.update({'font.size': 14})
    prior_list = [np.ones((n_arms,)) / n_arms,
                  np.ones((n_arms,)) * p1,
                  np.ones((n_arms,)) * p2]
    prior_list[1][0] = 1 - n_arms * p1
    prior_list[1] /= prior_list[1].sum()
    prior_list[2][0] = 1 - n_arms * p2
    prior_list[2] /= prior_list[2].sum()
    regret_array = np.zeros((len(prior_list), episodes))
    mi_array = np.zeros((len(prior_list), episodes))
    regret_std = np.zeros((len(prior_list), episodes))
    mi_std = np.zeros((len(prior_list), episodes))
    entropy = np.zeros((len(prior_list),))
    for i in range(len(prior_list)):
        prior = prior_list[i]
        bandit = mab.k_models_bandit(n_arms, epsilon, prior)
        ts_alg = thompson_sampling.k_models_ts(n_arms, epsilon, prior)
        regret_ts, mi_ts = ts_alg.run_experiment_bayesian(bandit, episodes, trials)
        entropy[i] = thompson_sampling.entropy(prior)
        regret_array[i] = np.mean(regret_ts, axis=0)
        mi_array[i] = np.mean(mi_ts, axis=0)
        regret_std[i] = np.std(regret_ts, axis=0) / np.sqrt(trials)
        mi_std[i] = np.std(mi_ts, axis=0) / np.sqrt(trials)
    plt.figure()
    colors = ['b', 'g', 'r']
    for i in range(len(prior_list)):
        plt.plot(regret_array[i], label="{:.2f}".format(entropy[i]), color=colors[i])
        plt.fill_between(range(episodes), regret_array[i] - regret_std[i], regret_array[i] + regret_std[i], alpha=0.5,
                         color=colors[i])
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.grid()
    plt.show()
    plt.figure()
    for i in range(len(prior_list)):
        plt.plot(mi_array[i], label="{:.2f}".format(entropy[i]), color=colors[i])
        plt.fill_between(range(episodes), mi_array[i] - mi_std[i], mi_array[i] + mi_std[i], alpha=0.5, color=colors[i])
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Mutual information')
    plt.grid()
    plt.show()