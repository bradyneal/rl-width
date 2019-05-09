#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:15:18 2019

@author: brady
"""

import argparse
import difflib
import os
from pprint import pprint

import gym
from train import Trainer, STR_TO_ALGO

#from results_plotting import plot_all_widths


RL_BASELINES_ZOO_HYPER = 'rl-baselines-zoo'
RLLIB_HYPER = 'rllib'
DEFAULT_HYPER = 'default'
HYPERPARAM_CHOICES = [RL_BASELINES_ZOO_HYPER, RLLIB_HYPER, DEFAULT_HYPER]
ACT_FUN_CHOICES = ['relu', 'tanh']

RESULTS_DEF = os.getcwd()
LOG_DEF = 'logs'
EXP_DEF = 'general'
FIG_DEF = 'figures'
HYPERPARAM_DEF = RL_BASELINES_ZOO_HYPER
DEPTH_DEF = 2
LR_POW_DEF = -1
ACT_FUN_DEF = 'tanh'

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--results-dir', default=RESULTS_DEF, type=str, help='path to where all results are written')
PARSER.add_argument('--log-dir', default=LOG_DEF, type=str, help='log folder for CSV logs, saved models, and hyperparameter settings')
PARSER.add_argument('--name', default=EXP_DEF, type=str, help='name of experiment')
PARSER.add_argument('--figure-dir', default=FIG_DEF, type=str, help='folder for figures of experiments')
PARSER.add_argument('--env', nargs='+', default=['CartPole-v1'], type=str, help='environment ID(s)')
PARSER.add_argument('--algo', nargs='+', default=['ppo2'], type=str, choices=list(STR_TO_ALGO.keys()), help='RL Algorithm')
PARSER.add_argument('-s', '--start-end-seed', nargs=2, type=int, help='first and last (inclusive) random seeds')
PARSER.add_argument('--n-seeds', default=1, type=int, help='number of random seeds to run')
PARSER.add_argument('-w', '--widths', nargs='+', default=[64], type=int, help='network width(s)')
PARSER.add_argument('--hyperparam', default=HYPERPARAM_DEF, type=str, choices=HYPERPARAM_CHOICES, help='hyperparameter settings')
PARSER.add_argument('--act-fun', default=ACT_FUN_DEF, type=str, choices=ACT_FUN_CHOICES, help='network activation function')

    
PARSER.add_argument('-d', '--depth', default=DEPTH_DEF, type=int, help='number of hidden layers')
PARSER.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1, type=int)
#parser.add_argument('-tb', '--tensorboard-dir', default='tensorboard', type=str, help='Tensorboard log dir')
PARSER.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
PARSER.add_argument('--scale-lr', action='store_true', help='scale learning rate with width')
PARSER.add_argument('--lr-pow', default=LR_POW_DEF, type=float, help='exponent to scale learning rate with width by')
PARSER.add_argument('--no-tensorboard', action='store_true', help='do not log tensorboard events files (logged by default)')


def check_envs_valid(env_ids):
    registered_envs = set(gym.envs.registry.env_specs.keys())
    for env_id in env_ids:
        # If the environment is not found, suggest the closest match
        if env_id not in registered_envs:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            raise ValueError('{} not found in gym registry. Maybe you meant {}?'
                             .format(env_id, closest_match))


def build_log_dir(env_id, algo, width, seed, results_dir=RESULTS_DEF,
                  log_folder=LOG_DEF, exp_name=EXP_DEF, hyperparam_setting=HYPERPARAM_DEF,
                  scale_lr=False, depth=DEPTH_DEF, lr_pow=LR_POW_DEF):
    algo_dir = get_algo_fullname(algo, hyperparam_setting, scale_lr, lr_pow=lr_pow)
    log_dir = os.path.join(results_dir, log_folder, exp_name, env_id, algo_dir, 
                           'w{}_d{}'.format(width, depth), 'seed{}'.format(seed))
    return log_dir


def get_algo_fullname(algo, hyperparam_setting, scale_lr, lr_pow=LR_POW_DEF):
    if scale_lr:
        algo_fullname = '{}_{}_scale-lr{}'.format(algo, hyperparam_setting, lr_pow)
    else:
        algo_fullname = '{}_{}'.format(algo, hyperparam_setting)
    return algo_fullname


def main():
    args = PARSER.parse_args()
    args_dict = vars(args)
    print('Effective command line arguments:')
    pprint(args_dict)
    
    # Unpack some command line arguments
    exp_name = args.name
    env_ids = args.env
    algos = args.algo
    if args.start_end_seed is None:
        start_seed = 0
        end_seed = args.n_seeds - 1
    else:
        start_seed, end_seed = args.start_end_seed
    widths = args.widths
    hyperparam_setting = args.hyperparam
    results_dir = args.results_dir
    log_folder = args.log_dir
    figure_dir = os.path.join(results_dir, args.figure_dir)
    
    # Create dictionary for passing remaining named arguments to function
    remain_args_dict = {k: v for k, v in args_dict.items() if k not in
                        set(['env', 'algo', 'start_end_seed', 'widths',
                             'hyperparam', 'results_dir', 'log_dir', 'name',
                             'figure_dir', 'n_seeds'])}
    
    # Check if environments are valid and run experiments
    check_envs_valid(env_ids)
    for env_id in env_ids:
        for algo in algos:
            for seed in range(start_seed, end_seed + 1):
                for width in widths:
                    log_dir = build_log_dir(env_id, algo, width, seed, results_dir=results_dir, log_folder=log_folder, exp_name=exp_name, hyperparam_setting=hyperparam_setting, scale_lr=args.scale_lr, depth=args.depth, lr_pow=args.lr_pow)
                    trainer = Trainer(env_id, algo, seed, width, log_dir, args_dict, **remain_args_dict)
                    if hyperparam_setting == RL_BASELINES_ZOO_HYPER:
                        trainer.zoo_train()
                    elif hyperparam_setting == RLLIB_HYPER:
                        raise NotImplementedError("To be implemented")
                    elif hyperparam_setting == DEFAULT_HYPER:
                        trainer.default_train()
#            if start_seed == 0:
#                plot_all_widths(env_id, algo, widths, end_seed + 1, figure_dir)

       
if __name__ == '__main__':
    main()
