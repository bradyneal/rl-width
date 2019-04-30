#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:57:36 2019

@author: brady
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from stable_baselines.bench.monitor import load_results
from stable_baselines.results_plotter import \
    ts2xy, window_func, X_TIMESTEPS, X_EPISODES, X_WALLTIME

from main import build_log_dir
from train import build_monitor_dir


# 0 corresponds to implementing np.vstack code, whereas
# 1 correpsonds to implementing np.hstack code
STACK_DIM = 0


def plot_all_widths(*args, **kwargs):
    plotter = ResultsPlotter(*args, **kwargs)
    plotter.plot_all_widths()


class ResultsPlotter:
    
    def __init__(self, env_id, algo, widths, num_seeds, path_args={},
                 smooth_window=50, smooth_seeds=True, smooth_mean=False,
                 smooth_std=False, xaxis=X_EPISODES, conf_int='mean',
                 alpha=0.5, trim_diff_widths=True):
        self.env_id = env_id
        self.algo = algo
        self.widths = widths
        self.num_seeds = num_seeds
        self.path_args = path_args
        self.smooth_window = smooth_window
        self.smooth_seeds = smooth_seeds
        self.smooth_mean = smooth_mean
        self.smooth_std = smooth_std
        self.xaxis = xaxis
        self.conf_int = conf_int
        self.alpha = alpha
        self.trim_diff_widths = trim_diff_widths
        
    def get_monitor_dir(self, width, seed):
        log_dir = build_log_dir(env_id=self.env_id, algo=self.algo,
                                width=width, seed=seed, **self.path_args)
        return build_monitor_dir(log_dir)
        
    def get_x_y(self, width, seed):
        timesteps_df = load_results(self.get_monitor_dir(width, seed))
        x, y = ts2xy(timesteps_df, self.xaxis)
        return x, y
    
    def get_seeds(self, width, seeds=None):
        if seeds is None:
            seeds = range(self.num_seeds)
            
        # Load the seeds
        x_seeds = []
        y_seeds = []
        min_len = float('inf')
        for seed in seeds:
            x_seed, y_seed = self.get_x_y(width, seed)
            
            if self.smooth_seeds:
                x_seed, y_seed = self.smooth(x_seed, y_seed)
                
            min_len = min(min_len, x_seed.shape[0])
            x_seeds.append(x_seed)
            y_seeds.append(y_seed)
        
        # Trim all to the same length and stack them
        x = x_seeds[0][:min_len]
        for i in range(len(seeds)):
            np.testing.assert_array_equal(x, x_seeds[i][:min_len])
            y_seeds[i] = y_seeds[i][:min_len]
        Y = np.stack(y_seeds, axis=STACK_DIM)
        
        return x, Y
    
    def smooth(self, x, y):
        if self.smooth_window is not None and x.shape[0] >= self.smooth_window:
            x_shift, y_smooth = window_func(x, y, self.smooth_window, np.mean)
        return x_shift, y_smooth
    
    def get_all_widths_mean_and_std(self):
        # Get mean and std for each width
        xs = []
        means = []
        stds = []
        min_len = float('inf')
        for width in self.widths:
            x, Y = self.get_seeds(width)
            min_len = min(min_len, x.shape[0])
            mean = np.mean(Y, axis=STACK_DIM)
            std = np.std(Y, axis=STACK_DIM)
            if self.conf_int == 'mean':
                std /= sqrt(self.num_seeds)
            elif self.conf_int == 'seed':
                pass
            else:
                raise ValueError('Invalid "conf_int" value: {}'.format(self.conf_int))
            xs.append(x)
            means.append(mean)
            stds.append(std)
        
        # Trim all seeds to the same length
        if self.trim_diff_widths:
            for i in range(len(self.widths)):
                xs[i] = xs[i][:min_len]
                means[i] = means[i][:min_len]
                stds[i] = stds[i][:min_len]
        
        # Smooth        
        for i in range(len(means)):
            if self.smooth_mean:
                x_shift, mean_smooth = self.smooth(xs[i], means[i])
                means[i] = mean_smooth
                if self.smooth_std:
                    _, std_smooth = self.smooth(xs[i], stds[i])
                    stds[i] = std_smooth
                else:
                    stds[i] = stds[i][self.smooth_window - 1:]
                xs[i] = x_shift
                
        return xs, means, stds
    
    def plot_all_widths(self):
        widths = self.widths
        xs, y_avgs, y_offsets = self.get_all_widths_mean_and_std()
        assert len(xs) == len(widths)
        
        plt.figure()
        for width, x, y, y_offset in zip(widths, xs, y_avgs, y_offsets):
            plt.plot(x, y, '-', label=str(width))
            plt.fill_between(x, y + y_offset, y - y_offset, alpha=self.alpha)
        plt.legend()
        plt.xlabel(self.xaxis)
        plt.ylabel('average return')
        plt.tight_layout()

    def get_all_widths_est_bootstrap(self):
        raise NotImplementedError("To be implemented")
