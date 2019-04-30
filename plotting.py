#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:57:36 2019

@author: brady
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.bench.monitor import load_results
from stable_baselines.results_plotter import \
    ts2xy, window_func, X_TIMESTEPS, X_EPISODES, X_WALLTIME

from main import build_log_dir
from train import build_monitor_dir


XAXIS_DEF = X_EPISODES

# TODO: refactor into a class so that all the arguments are only necessary in
# a single function def

def get_x_y(monitor_dir, xaxis=XAXIS_DEF):
    timesteps_df = load_results(monitor_dir)
    x, y = ts2xy(timesteps_df, xaxis)
    return x, y

def get_seeds(env_id, algo, width, num_seeds, path_args={}, xaxis=XAXIS_DEF):
    pass

def plot_mean_std_single(env_id, algo, width, num_seeds, path_args={}, smooth_window=None, xaxis=XAXIS_DEF, alpha=0.65):
    # Calculate mean and std
    x_seeds = []
    y_seeds = []
    min_len = float('inf')
    for seed in range(num_seeds):
        log_dir = build_log_dir(env_id=env_id, algo=algo, width=width, seed=seed, **path_args)
        monitor_dir = build_monitor_dir(log_dir)
        x_seed, y_seed = ts2xy(load_results(monitor_dir), xaxis)
        min_len = min(min_len, len(x_seed))
        x_seeds.append(x_seed)
        y_seeds.append(y_seed)
    
    # Trim all to the same length
    x = x_seeds[0][:min_len]
    for i in range(num_seeds):
        np.testing.assert_array_equal(x, x_seeds[i][:min_len])
        y_seeds[i] = y_seeds[i][:min_len]

    # Calculate mean and std
    Y = np.vstack(y_seeds)
    y_mean = np.mean(Y, axis=0)
    y_std = np.std(Y, axis=0)
    
    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
    if smooth_window is not None and x.shape[0] >= smooth_window:
        # Compute and plot rolling mean with window of size EPISODE_WINDOW
        x, y_mean = window_func(x, y_mean, smooth_window, np.mean)
        y_std = y_std[smooth_window - 1:]
    
    # Plot
    plt.figure()
    plt.plot(x, y_mean, '-')
    plt.fill_between(x, y_mean + y_std, y_mean - y_std, alpha=alpha)
    plt.tight_layout()
#    plt.show()
    