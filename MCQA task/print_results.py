import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import argparse


def plot_results(seed, n_episodes, n_trails, percentile):
    large_filename = 'results/llm_mcqa_{ms}_{s}_{h}.json'.format(ms='large', s=seed, h=n_episodes)
    small_filename = 'results/llm_mcqa_{ms}_{s}_{h}.json'.format(ms='small', s=seed, h=n_episodes)
    with open(large_filename) as f:
        large_results = json.load(f)
    with open(small_filename) as g:
        small_results = json.load(g)
    l = 0
    h = percentile * 100
    large_rewards = np.zeros((n_trails, n_episodes))
    small_rewards = np.zeros((n_trails, n_episodes))
    small_mi = np.zeros((n_trails, n_episodes))
    for i in range(n_trails):
        large_rewards[i] = large_results[i]['rewards'][0:n_episodes]
        small_rewards[i] = small_results[i]['rewards'][0:n_episodes]
        small_mi[i] = small_results[i]['mutual_information'][0:n_episodes]
    large_rewards -= 0.1
    mi_lower = np.percentile(small_mi, l)
    mi_upper = np.percentile(small_mi, h)
    selected_indices = np.where(np.logical_and(small_mi <= mi_upper, small_mi >= mi_lower), False, True)
    selected_rewards = np.copy(small_rewards)
    selected_rewards[selected_indices] = large_rewards[selected_indices]
    random_rewards = (percentile) * small_rewards + large_rewards * (1-percentile)
    best_reward = np.where(small_rewards >= large_rewards, small_rewards, large_rewards)
    selected_regret = np.cumsum(best_reward - selected_rewards, axis=1)
    mean_selected = np.mean(selected_regret, axis=0)
    std_selected = np.std(selected_regret, axis=0) / np.sqrt(n_trails)
    random_regret = np.cumsum(best_reward - random_rewards, axis=1)
    mean_random = np.mean(random_regret, axis=0)
    std_random = np.std(random_regret, axis=0) / np.sqrt(n_trails)
    plt.plot(np.mean(selected_regret, axis=0), label='Using bits', color='r')
    plt.fill_between(range(n_episodes), mean_selected - std_selected, mean_selected + std_selected, alpha=0.5, color='r')
    plt.plot(np.mean(random_regret, axis=0), label='Random', color='y')
    plt.fill_between(range(n_episodes), mean_random - std_random, mean_random + std_random, alpha=0.5, color='y')
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.legend()
    plt.show()
    return None


def print_values(seed_list, n_episodes, n_trails, per, penalty=0.1):
    n_seeds = len(seed_list)
    large_rewards = np.zeros((n_seeds, n_trails, n_episodes))
    small_rewards = np.zeros((n_seeds, n_trails, n_episodes))
    small_mi = np.zeros((n_seeds, n_trails, n_episodes))
    for i, seed in enumerate(seed_list):
        large_filename = 'results/llm_mcqa_{ms}_{s}_{h}.json'.format(ms='large', s=seed, h=n_episodes)
        small_filename = 'results/llm_mcqa_{ms}_{s}_{h}.json'.format(ms='small', s=seed, h=n_episodes)
        with open(large_filename) as f:
            large_results = json.load(f)
        with open(small_filename) as g:
            small_results = json.load(g)
        for j in range(n_trails):
            large_rewards[i, j] = large_results[j]['rewards'][0:n_episodes]
            small_rewards[i, j] = small_results[j]['rewards'][0:n_episodes]
            small_mi[i, j] = small_results[j]['mutual_information'][0:n_episodes]
    large_rewards -= penalty
    mi_lower = 0
    mi_upper = np.percentile(small_mi, per * 100)
    selected_indices = np.where(np.logical_and(small_mi <= mi_upper, small_mi >= mi_lower), False, True)
    selected_rewards = np.copy(small_rewards)
    selected_rewards[selected_indices] = large_rewards[selected_indices]
    best_reward = np.where(small_rewards >= large_rewards, small_rewards, large_rewards)
    random_rewards = per * small_rewards + large_rewards * (1-per)
    selected_regret = np.sum(best_reward - selected_rewards, axis=2)
    random_regret = np.sum(best_reward - random_rewards, axis=2)
    mean_selected = np.mean(selected_regret)
    std_selected = np.std(selected_regret) / np.sqrt(n_seeds * n_trails)
    mean_random = np.mean(random_regret)
    std_random = np.std(random_regret) / np.sqrt(n_seeds * n_trails)
    return mean_selected, std_selected, mean_random, std_random


def arg_parse():
    parser = argparse.ArgumentParser(description='Run LLM Contextual Bandit')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--n_episodes', type=int, default=200, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=123, nargs='+',
                        help='Seed to plot or a list of seeds to print values')
    parser.add_argument('--percentile', type=float, default=0.5, help='Percentile to use')
    parser.add_argument('--penalty', type=float, default=0.1, help='Penalty to use')
    parser.add_argument('--print', action='store_true', help='Print values')
    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parse()
    matplotlib.rcParams.update({'font.size': 14})
    if args.print:
        if isinstance(args.seed, int):
            seed_list = [args.seed]
        else:
            seed_list = args.seed
        mean_selected, std_selected, mean_random, std_random = print_values(seed_list, args.n_episodes, args.n_trials, args.percentile, args.penalty)
        print('Selected:', mean_selected, std_selected)
        print('Random:', mean_random, std_random)
    else:
        seed_list = args.seed
        assert (len(seed_list) == 1), 'Please provide a single seed to plot'
        seed = seed_list[0]
        plot_results(seed, args.n_episodes, args.n_trials, args.percentile)

