#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:35:39 2019

@author: brady
"""

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, A2C, ACKTR, ACER
from stable_baselines.common import set_global_seeds

# multiprocess environment
TB_FOLDER = 'tensorboard'
DEFAULT_NUM_HIDDEN_LAYERS = 2
NUM_CPU = 4

NUM_STEPS = 500000
#LEARNERS = ['PPO2', 'ACER', 'A2C']
LEARNERS = ['PPO2']


START_SEED = 0
END_SEED = 4
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
# WIDTHS = [512, 1024]


def run_learner(learner_str, net_width, exp_name, seed,
                num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
                env_str='CartPole-v1', num_cpu=NUM_CPU):
    print('Training {} with network width {} and seed {}'.format(learner_str, net_width, seed))
    str_to_learner = {
            'PPO2': PPO2,
            'A2C': A2C,
            'ACKTR': ACKTR,
            'ACER': ACER,
    }
    # TODO: add check if key in dict
    learner = str_to_learner[learner_str]
#    for seed in range(num_seeds):
    set_global_seeds(seed)
    env = SubprocVecEnv([lambda: gym.make(env_str) for i in range(num_cpu)])
    
    #policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
    policy_kwargs = dict(net_arch=[net_width for _ in range(num_hidden_layers)])
    log_dir = '{}/{}_{}/{}/'.format(TB_FOLDER, exp_name, env_str, learner_str)
    model = learner(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=0,
                    tensorboard_log=log_dir)
    if num_hidden_layers == DEFAULT_NUM_HIDDEN_LAYERS:
        log_file = '{}_w{}_s{}'.format(learner_str, net_width, seed)
    else:
        log_file = '{}_w{}_d{}_s{}'.format(learner_str, net_width, num_hidden_layers, seed)
    model.learn(total_timesteps=NUM_STEPS, tb_log_name=log_file)
#    model.save("ppo2_cartpole")

if __name__ == '__main__':
    exp_name = 'width512-2048_5seeds_PPO'
    print('Running experiment:', exp_name)
    for learner_str in LEARNERS:
        for seed in range(START_SEED, END_SEED + 1):
            for width in WIDTHS:
                run_learner(learner_str, width, exp_name, seed)
    print('Done!')
