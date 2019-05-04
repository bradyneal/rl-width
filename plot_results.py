#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:57:36 2019

@author: brady
"""

import os
import numpy as np
from math import sqrt
from statistics import median
from scipy.stats import t
from collections import OrderedDict
import ast

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
#sns.set(style='darkgrid')

from stable_baselines.bench.monitor import load_results
from stable_baselines.results_plotter import \
    ts2xy, window_func, X_TIMESTEPS, X_EPISODES, X_WALLTIME

from main import build_log_dir, get_algo_fullname, PARSER, EXP_DEF, HYPERPARAM_DEF, LR_POW_DEF
from train import build_monitor_dir


tab10 = [c for i, c in enumerate(sns.color_palette("tab20", 20)) if i % 2 == 0]
tab10_light = [c for i, c in enumerate(sns.color_palette("tab20", 20)) if i % 2 == 1]
TAB20_DARK_LIGHT = tab10 + tab10_light
TAB20_CUSTOM_STR = 'tab20_custom'
COLOR_PALETTE_DEF = TAB20_CUSTOM_STR

# 0 corresponds to implementing np.vstack code, whereas
# 1 correpsonds to implementing np.hstack code
STACK_DIM = 0
LINESTYLES = ['-', '--', '-.', ':']
# https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html
FANCY_LINESTYLES = OrderedDict(
    [('solid',               (0, ())),
#     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

#     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

#     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
#     ('dashdotted',          (0, (3, 5, 1, 5))),
#     ('densely dashdotted',  (0, (3, 1, 1, 1))),

#     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
     ])
TRIM_WIDTHS_DEF = 'median'
XAXIS_DEF = X_EPISODES


def plot_all_widths(*args, **kwargs):
    plotter = ResultsPlotter(*args, **kwargs)
    plotter.plot_all_widths()


class ResultsPlotter:
    
    def __init__(self, env_id, algo, widths, n_seeds, figure_dir,
                 hyperparam_setting=HYPERPARAM_DEF, scale_lr=False, lr_pow=LR_POW_DEF,
                 path_args={}, smooth_window=50, smooth_seeds=True, smooth_mean=False,
                 smooth_std=False, xaxis=XAXIS_DEF, conf_int_type='mean', conf_int=0.75,
                 alpha=0.5, trim_widths_type=TRIM_WIDTHS_DEF, color_palettes=[COLOR_PALETTE_DEF], linestyle='-'):
        self.env_id = env_id
        self.algo = algo
        self.widths = widths
        self.n_seeds = n_seeds
        self.figure_dir = figure_dir
        self.hyperparam_setting = hyperparam_setting
        self.scale_lr = scale_lr
        self.lr_pow = lr_pow
        self.path_args = path_args
        self.smooth_window = smooth_window
        self.smooth_seeds = smooth_seeds
        self.smooth_mean = smooth_mean
        self.smooth_std = smooth_std
        self.xaxis = xaxis
        self.conf_int_type = conf_int_type
        self.conf_int = conf_int
        self.alpha = alpha
        self.trim_widths_type = trim_widths_type
        self.linestyle = linestyle
        self.color_palettes = color_palettes
        
    def get_monitor_dir(self, width, seed):
        log_dir = build_log_dir(env_id=self.env_id, algo=self.algo,
                                width=width, seed=seed, scale_lr=self.scale_lr,
                                hyperparam_setting=self.hyperparam_setting,
                                lr_pow=self.lr_pow, **self.path_args)
        return build_monitor_dir(log_dir)
        
    def get_x_y(self, width, seed):
        timesteps_df = load_results(self.get_monitor_dir(width, seed))
        if timesteps_df.shape[0] == 0:
            raise ValueError('Timesteps for seed {} of width {} is empty'.format(seed, width))
        x, y = ts2xy(timesteps_df, self.xaxis)
        return x, y
    
    def get_seeds(self, width, seeds=None):
        if seeds is None:
            seeds = range(self.n_seeds)
            
        # Load the seeds
        x_seeds = []
        y_seeds = []
        min_len = float('inf')
        for seed in seeds:
            x_seed, y_seed = self.get_seed(width, seed)                
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
    
    def get_seed(self, width, seed):
        x, y = self.get_x_y(width, seed)
        if self.smooth_seeds:
            x, y = self.smooth(x, y)
        
        if self.xaxis == X_TIMESTEPS:
            y_full = np.full(x[-1], np.nan, dtype=y.dtype)
            y_full[x - 1] = y
            assert y[0] and not np.isnan(y[0])
            last_return = y[0]
            for i, y_i in enumerate(y_full):                    
#                print(y_i)
                if np.isnan(y_i):
                    y_full[i] = last_return
                else:
                    last_return = y_i
            x = np.arange(1, x[-1] + 1)
            y = y_full
        elif self.xaxis == X_EPISODES:
            pass    # already in the right format
        elif self.xaxis == X_WALLTIME:
            raise NotImplementedError('To be implemented')
        else:
            raise ValueError('Invalid x-axis type: {}'.format(self.xaxis))
            
        return x, y
    
    def smooth(self, x, y):
        if self.smooth_window is not None and x.shape[0] >= self.smooth_window:
            x_shift, y_smooth = window_func(x, y, self.smooth_window, np.mean)
            return x_shift, y_smooth
        else:
            return x, y
    
    def get_all_widths_mean_and_std(self):
        # Get mean and std for each width
        xs = []
        means = []
        stds = []
        for width in self.widths:
            x, Y = self.get_seeds(width)
            mean = np.mean(Y, axis=STACK_DIM)
            std = np.std(Y, axis=STACK_DIM)
            if self.conf_int_type == 'mean':
                std = get_t(self.conf_int, df=self.n_seeds-1) * std / sqrt(self.n_seeds)
            elif self.conf_int_type == 'seed':
                pass
            else:
                raise ValueError('Invalid "conf_int_type" value: {}'.format(self.conf_int_type))
            xs.append(x)
            means.append(mean)
            stds.append(std)
        
        # Trim all seeds to the same length
        if self.trim_widths_type is not None:
            lengths = [len(x) for x in xs]
            if self.trim_widths_type == 'min':
                trim_len = min(lengths)
            elif self.trim_widths_type == 'median':
                print('Trimming to median length')
                trim_len = median(lengths)
            else:
                raise ValueError('Invalid trim type: {}'.format(self.trim_diff_widths_type))
                
            for i in range(len(self.widths)):
                xs[i] = xs[i][:trim_len]
                means[i] = means[i][:trim_len]
                stds[i] = stds[i][:trim_len]
        
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
        if self.linestyle == 'fancy':
            linestyles = list(FANCY_LINESTYLES.values())
        elif self.linestyle == 'normal':
            linestyles = LINESTYLES
        elif self.linestyle in LINESTYLES:
            linestyles = [self.linestyle]
        else:
            raise ValueError('Invalid linestyle:', self.linestyle)
        
        widths = self.widths
        xs, y_avgs, y_offsets = self.get_all_widths_mean_and_std()
        assert len(xs) == len(widths)
        
        n_colors = len(widths)
        for palette in self.color_palettes:
            print('palette:', palette)
            if palette == TAB20_CUSTOM_STR:
                sns.set_palette(TAB20_DARK_LIGHT)
            elif palette.startswith('hls'):
                kwargs = get_kwargs_after_delim(palette, delim='_', default={'l': 0.55})
                sns.set_palette(sns.hls_palette(n_colors, **kwargs))
            elif palette.startswith('husl'):
                kwargs = get_kwargs_after_delim(palette, delim='_')
                sns.set_palette(sns.husl_palette(n_colors, **kwargs))
#                sns.set_palette(sns.husl_palette(n_colors, h=.7, l=.6))
            elif palette in ['tab10', 'tab20', 'muted', 'deep', 'colorblind', 'Paired']:
                sns.set_palette(sns.color_palette(palette, n_colors))
            else:
                raise ValueError('Unsupported color palette: {}'.format(palette))

            fig = plt.figure()
            for i, (width, x, y, y_offset) in enumerate(zip(widths, xs, y_avgs, y_offsets)):
                linestyle = linestyles[i % len(linestyles)]
                plt.plot(x, y, '-', label=str(width), linestyle=linestyle)
                plt.fill_between(x, y + y_offset, y - y_offset, alpha=self.alpha)
            plt.legend()
            plt.xlabel(self.xaxis)
            plt.ylabel('average return')
            plt.tight_layout()
            
            os.makedirs(self.figure_dir, exist_ok=True)
            algo_fullname = get_algo_fullname(self.algo, self.hyperparam_setting,
                                              self.scale_lr, lr_pow=self.lr_pow)
            if len(self.color_palettes) == 1:
                filename = '{}_{}.pdf'.format(self.env_id, algo_fullname)
            else:
                filename = '{}_{}_{}.pdf'.format(self.env_id, algo_fullname, palette)
            path = os.path.join(self.figure_dir, filename)
            print('Saving figure to', path)
            fig.savefig(path, bbox_inches='tight')
            fig.show()
        
    def get_all_widths_est_bootstrap(self):
        raise NotImplementedError("To be implemented")
        

def get_kwargs_after_delim(s, delim='_', default={}):
    split = s.split(delim, 1)
    if len(split) == 1:
        kwargs = default
    else:
        kwargs = ast.literal_eval(split[1])
    return kwargs

def get_t(alpha, df, two_sided=True):
    if two_sided:
        alpha = (1 + alpha) / 2
    return t.ppf(alpha, 4)

        
if __name__ == '__main__':
    PARSER.add_argument('--palettes', nargs='+', default=[COLOR_PALETTE_DEF], type=str, help='color palettes to make plots in')
    PARSER.add_argument('--x-axis', default=XAXIS_DEF, type=str, choices=[X_TIMESTEPS, X_EPISODES, X_WALLTIME], help='what to plot on the x-axis')
    args = PARSER.parse_args()
    args = PARSER.parse_args()
    
    if args.start_end_seed is None:
        n_seeds = args.n_seeds
    else:
        start_seed, end_seed = args.start_end_seed
        n_seeds = end_seed - start_seed + 1
        
    for env_id in args.env:
        for algo in args.algo:
            figure_dir = os.path.join(args.results_dir, args.figure_dir, args.name)
            plot_all_widths(env_id, algo, args.widths, n_seeds=n_seeds, figure_dir=figure_dir,
                 hyperparam_setting=args.hyperparam, scale_lr=args.scale_lr, lr_pow=args.lr_pow,
                 xaxis=args.x_axis, color_palettes=args.palettes, path_args={'exp_name': args.name})
            
