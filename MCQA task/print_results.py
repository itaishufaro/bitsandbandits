import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import argparse


def plot_results(seed_list, n_episodes, n_trails, per, penalty=0.1, small='', large=''):
    n_seeds = len(seed_list)
    large_rewards = np.zeros((n_seeds, n_trails, n_episodes))
    small_rewards = np.zeros((n_seeds, n_trails, n_episodes))
    small_mi = np.zeros((n_seeds, n_trails, n_episodes))
    for i, seed in enumerate(seed_list):
        large_filename = 'results/llm_contextual_bandit_large_{ms}_{s}_{h}_noaccumulate.json'.format(ms=large, s=seed,
                                                                                                    h=args.n_episodes)
        small_filename = 'results/llm_contextual_bandit_small_{ms}_{s}_{h}_noaccumulate.json'.format(ms=small, s=seed,
                                                                                                    h=args.n_episodes)
        with open(large_filename) as f:
            large_results = json.load(f)
        with open(small_filename) as g:
            small_results = json.load(g)
        for j in range(n_trails):
            large_rewards[i, j] = large_results[j]['rewards'][0:n_episodes]
            small_rewards[i, j] = small_results[j]['rewards'][0:n_episodes]
            small_mi[i, j] = small_results[j]['mutual_information'][0:n_episodes]
    large_rewards -= penalty
    mi_lower = np.percentile(small_mi, (1 - per) * 100)
    mi_upper = np.percentile(small_mi, 100)
    selected_indices = np.where(np.logical_and(small_mi <= mi_upper, small_mi >= mi_lower), True, False)
    selected_rewards = np.copy(small_rewards)
    selected_rewards[selected_indices] = large_rewards[selected_indices]
    best_reward = np.where(small_rewards >= large_rewards, small_rewards, large_rewards)
    random_rewards = (1-per) * small_rewards + large_rewards * per
    selected_regret = np.cumsum(best_reward - selected_rewards, axis=2)
    random_regret = np.cumsum(best_reward - random_rewards, axis=2)
    std_selected = np.std(selected_regret, (0,1)) / np.sqrt(n_seeds * n_trails)
    std_random = np.std(random_regret, (0,1)) / np.sqrt(n_seeds * n_trails)
    plt.figure()
    plt.plot(np.mean(selected_regret, axis=(0,1)), label='Using bits', color='blue')
    plt.fill_between(np.arange(n_episodes), np.mean(selected_regret, axis=(0,1)) - std_selected,
                     np.mean(selected_regret, axis=(0,1)) + std_selected, alpha=0.2, color='blue')
    plt.plot(np.mean(random_regret, axis=(0,1)), label='Random', color='red')
    plt.fill_between(np.arange(n_episodes), np.mean(random_regret, axis=(0,1)) - std_random,
                     np.mean(random_regret, axis=(0,1)) + std_random, alpha=0.2, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid()
    plt.show()
    return None



def print_values(seed_list, n_episodes, n_trails, per, penalty=0.1, small='', large=''):
    if small != '':
        small = '_' + small
    if large != '':
        large = '_' + large
    n_seeds = len(seed_list)
    large_rewards = np.zeros((n_seeds, n_trails, n_episodes))
    small_rewards = np.zeros((n_seeds, n_trails, n_episodes))
    small_mi = np.zeros((n_seeds, n_trails, n_episodes))
    for i, seed in enumerate(seed_list):
        large_filename = 'results/llm_contextual_bandit_large{ms}_{s}_{h}_noaccumulate.json'.format(ms=large, s=seed, h=args.n_episodes)
        small_filename = 'results/llm_contextual_bandit_small{ms}_{s}_{h}_noaccumulate.json'.format(ms=small, s=seed, h=args.n_episodes)
        with open(large_filename) as f:
            large_results = json.load(f)
        with open(small_filename) as g:
            small_results = json.load(g)
        for j in range(n_trails):
            large_rewards[i, j] = large_results[j]['rewards'][0:n_episodes]
            small_rewards[i, j] = small_results[j]['rewards'][0:n_episodes]
            small_mi[i, j] = small_results[j]['mutual_information'][0:n_episodes]
    large_rewards -= penalty
    # mi_lower = 0
    # mi_upper = np.percentile(small_mi, per * 100)
    mi_lower = np.percentile(small_mi, (1-per) * 100)
    mi_upper = np.percentile(small_mi, 100)
    selected_indices = np.where(np.logical_and(small_mi <= mi_upper, small_mi >= mi_lower), True, False)
    selected_rewards = np.copy(small_rewards)
    selected_rewards[selected_indices] = large_rewards[selected_indices]
    best_reward = np.where(small_rewards >= large_rewards, small_rewards, large_rewards)
    random_rewards = (1-per) * small_rewards + large_rewards * per
    selected_regret = np.sum(best_reward - selected_rewards, axis=2)
    random_regret = np.sum(best_reward - random_rewards, axis=2)
    mean_selected = np.mean(selected_regret)
    std_selected = np.std(selected_regret) / np.sqrt(n_seeds * n_trails)
    mean_random = np.mean(random_regret)
    std_random = np.std(random_regret) / np.sqrt(n_seeds * n_trails)
    return mean_selected, std_selected, mean_random, std_random


def arg_parse():
    parser = argparse.ArgumentParser(description='Run LLM Contextual Bandit')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('--n_episodes', type=int, default=200, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=123, nargs='+',
                        help='Seed to plot or a list of seeds to print values')
    parser.add_argument('--percentile', type=float, default=0.5, help='Percentile to use')
    parser.add_argument('--penalty', type=float, default=0, help='Penalty to use')
    parser.add_argument('--print', action='store_true', help='Print values')
    parser.add_argument('--small', type=str, default='mistral', help='Type of model')
    parser.add_argument('--large', type=str, default='mistral', help='Type of model')
    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parse()
    matplotlib.rcParams.update({'font.size': 14})
    # args.large = 'gemma'
    # args.seed = [123,123456,123456789,667,7890,54678,75867,1738]
    # args.print = True
    if args.print:
        if isinstance(args.seed, int):
            seed_list = [args.seed]
        else:
            seed_list = args.seed
        mean_selected, std_selected, mean_random, std_random = print_values(seed_list, args.n_episodes, args.n_trials, args.percentile, args.penalty, args.small, args.large)
        print('Selected:', mean_selected, std_selected)
        print('Random:', mean_random, std_random)
    else:
        seed_list = args.seed
        # assert (len(seed_list) == 1), 'Please provide a single seed to plot'
        # seed = seed_list[0]
        if isinstance(args.seed, int):
            seed_list = [args.seed]
        else:
            seed_list = args.seed
        plot_results(seed_list, args.n_episodes, args.n_trials, args.percentile, args.penalty, args.small, args.large)

