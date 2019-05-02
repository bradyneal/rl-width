#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:55:16 2019

@author: brady
"""

import os
import argparse
import shutil

from main import RESULTS_DEF, LOG_DEF, DEPTH_DEF


def get_algo_folder(path, algo):
    algo_folders = [folder for folder in os.listdir(path) if folder.startswith(algo)]
    if len(algo_folders) != 1:
        raise NotImplementedError("Multiple {} dirs in {}: {} ... only 1 is expected"
                                  .format(algo, path, algo_folders))
    return algo_folders[0]


WIDTHS_DEF = [2, 4, 8, 16, 32, 64]  # because 64 is the width learning rate was tuned for

parser = argparse.ArgumentParser()
parser.add_argument('from_exp', type=str, help='experiment name to copy from')
parser.add_argument('to_exp', type=str, help='experiment name to copy from')
parser.add_argument('-t', '--tasks', nargs='+', default=['CartPole-v1'], type=str, help='tasks to copy widths for')
parser.add_argument('--algos', nargs='+', default=['ppo2'], type=str, help='algos to copy widths for')
parser.add_argument('-w', '--widths', nargs='+', default=WIDTHS_DEF, type=int, help='network width(s) to copy')
parser.add_argument('--results-dir', default=RESULTS_DEF, type=str, help='path to where all results are written')
parser.add_argument('--log-dir', default=LOG_DEF, type=str, help='log folder for CSV logs, saved models, and hyperparameter settings')
parser.add_argument('-d', '--depth', default=DEPTH_DEF, type=int, help='number of hidden layers')

args = parser.parse_args()
print(args)
log_dir = os.path.join(args.results_dir, args.log_dir)
for task in args.tasks:
    from_dir = os.path.join(log_dir, args.from_exp, task)
    to_dir = os.path.join(log_dir, args.to_exp, task)
    
    for algo in args.algos:
        from_algo_dir = os.path.join(from_dir, get_algo_folder(from_dir, algo))
        to_algo_dir = os.path.join(to_dir, get_algo_folder(to_dir, algo))
        
        for width in args.widths:
            width_folder = 'w{}_d{}'.format(width, args.depth)
            from_path = os.path.join(from_algo_dir, width_folder)
            to_path = os.path.join(to_algo_dir, width_folder)
            print('copying {} to {}'.format(os.path.relpath(from_path), os.path.relpath(to_path)))
            shutil.copytree(from_path, to_path)
