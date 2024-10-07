"""
    Main file with experiment settings
"""
import numpy as np
import mab
import thompson_sampling, aps, exp3
import matplotlib.pyplot as plt
import matplotlib
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Comparing three different algorithms for Bayesian bandits')
    parser.add_argument('--n_arms', type=int, default=8, help='Number of arms')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Gap between rewards')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    n_arms = args.n_arms
    epsilon = args.epsilon
    episodes = args.episodes
    real_episodes = episodes
    trials = args.trials
    matplotlib.rcParams.update({'font.size': 14})
    prior = np.ones((n_arms,)) / n_arms
    print(thompson_sampling.entropy(prior))
    bandit = mab.k_models_bandit(n_arms, epsilon, prior)
    ts_alg = thompson_sampling.k_models_ts(n_arms, epsilon, prior)
    aps_alg = aps.aps(n_arms, np.sqrt(2 * np.log(n_arms) / real_episodes))
    exp3_alg = exp3.exp3(n_arms, np.sqrt(2 * np.log(n_arms) / real_episodes), gamma=0)
    regret_ts, mi_ts = ts_alg.run_experiment_bayesian(bandit, episodes, trials)
    regret_aps, mi_aps = aps_alg.run_experiment_bayesian(bandit, episodes, trials)
    regret_exp3, mi_exp3 = exp3_alg.run_experiment_bayesian(bandit, episodes, trials)
    regret_ts, mi_ts = regret_ts[:, :real_episodes], mi_ts[:, :real_episodes]
    regret_aps, mi_aps = regret_aps[:, :real_episodes], mi_aps[:, :real_episodes]
    regret_exp3, mi_exp3 = regret_exp3[:, :real_episodes], mi_exp3[:, :real_episodes]
    plt.plot(np.mean(regret_ts, axis=0), label='TS', color='r')
    plt.fill_between(range(real_episodes), np.mean(regret_ts, axis=0) - np.std(regret_ts, axis=0) / np.sqrt(trials),
                     np.mean(regret_ts, axis=0) + np.std(regret_ts, axis=0) / np.sqrt(trials), alpha=0.5, color='r')
    plt.plot(np.mean(regret_aps, axis=0), label='APS', color='b')
    plt.fill_between(range(real_episodes), np.mean(regret_aps, axis=0) - np.std(regret_aps, axis=0) / np.sqrt(trials),
                     np.mean(regret_aps, axis=0) + np.std(regret_aps, axis=0) / np.sqrt(trials), alpha=0.5, color='b')
    plt.plot(np.mean(regret_exp3, axis=0), label='EXP3', color='g')
    plt.fill_between(range(real_episodes), np.mean(regret_exp3, axis=0) - np.std(regret_exp3, axis=0) / np.sqrt(trials),
                     np.mean(regret_exp3, axis=0) + np.std(regret_exp3, axis=0) / np.sqrt(trials), alpha=0.5, color='g')
    plt.legend()
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.show()
    time = np.arange(real_episodes)
    ts_mi_bound = np.mean(mi_ts, axis=0)
    aps_mi_bound = np.mean(mi_aps, axis=0)
    exp3_mi_bound = np.mean(mi_exp3, axis=0)
    std_ts = np.std(mi_ts, axis=0) / np.sqrt(trials)
    std_aps = np.std(mi_aps, axis=0) / np.sqrt(trials)
    std_exp3 = np.std(mi_exp3, axis=0) / np.sqrt(trials)
    plt.plot(ts_mi_bound, label='TS', color='r')
    plt.fill_between(time, ts_mi_bound - std_ts, ts_mi_bound + std_ts, alpha=0.5, color='r')
    plt.plot(aps_mi_bound, label='APS', color='b')
    plt.fill_between(time, aps_mi_bound - std_aps, aps_mi_bound + std_aps, alpha=0.5, color='b')
    plt.plot(exp3_mi_bound, label='EXP3', color='g')
    plt.fill_between(time, exp3_mi_bound - std_exp3, exp3_mi_bound + std_exp3, alpha=0.5, color='g')
    plt.grid()
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Mutual information')
    plt.show()
    ratio1 = np.mean(regret_ts * mi_ts, axis=0)
    ratio2 = np.mean(regret_exp3 * mi_exp3, axis=0)
    ratio3 = np.mean(regret_aps * mi_aps, axis=0)
    std1 = np.std(regret_ts * mi_ts, axis=0) / np.sqrt(trials)
    std2 = np.std(regret_exp3 * mi_exp3, axis=0) / np.sqrt(trials)
    std3 = np.std(regret_aps * mi_aps, axis=0) / np.sqrt(trials)
    plt.plot(ratio1, label='TS', color='r')
    plt.fill_between(time, ratio1 - std1, ratio1 + std1, alpha=0.5, color='r')
    plt.plot(ratio3, label='APS', color='b')
    plt.fill_between(time, ratio3 - std3, ratio3 + std3, alpha=0.5, color='b')
    plt.plot(ratio2, label='EXP3', color='g')
    plt.fill_between(time, ratio2 - std2, ratio2 + std2, alpha=0.5, color='g')
    plt.grid()
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Ratio')
    plt.show()