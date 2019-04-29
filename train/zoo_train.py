#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:00:56 2019

@author: brady
"""

import os
from collections import OrderedDict
from pprint import pprint
import numpy as np
import yaml

import gym
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, DDPG, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn
    
HYPERPARAMS_PARENT_FOLDER = 'hyperparams'
HYPERPARAMS_FOLDER = os.path.join(HYPERPARAMS_PARENT_FOLDER, 'rl-baselines-zoo')
TB_LOG_NAME = 'tb'
STR_TO_ALGO = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2
}


def zoo_train(env_id, algo, seed, width, log_dir, args_dict, depth, n_timesteps,
            log_interval, scale_lr, no_tensorboard):
    """
    Train an RL agent with the given specifications, using the Stable Baselines
    library and tuned hyperparametes from rl-baselines-zoo.
    Code largely adapted from rl-baselines-zoo's train.py:
        https://github.com/araffin/rl-baselines-zoo/blob/master/train.py
    """
    os.makedirs(log_dir, exist_ok=True)
    
    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)

    # Load hyperparameters from yaml file
    with open(os.path.join(HYPERPARAMS_FOLDER, algo) + '.yml', 'r') as f:
        if is_atari:
            hyperparams = yaml.safe_load(f)['atari']
        else:
            hyperparams = yaml.safe_load(f)[env_id]

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if algo in ["ppo2", "sac"]:
        for key in ['learning_rate', 'cliprange']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], float):
                hyperparams[key] = constfn(hyperparams[key])
            else:
                raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

    # Should we overwrite the number of timesteps?
    if n_timesteps <= 0:
        n_timesteps = int(hyperparams['n_timesteps'])

    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # Create the environment and wrap it if necessary
    if is_atari:
        raise NotImplementedError("To be implemented")
#        print("Using Atari wrapper")
#        env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
#        # Frame-stacking with 4 frames
#        env = VecFrameStack(env, n_stack=4)
    elif algo in ['dqn', 'ddpg']:
        raise NotImplementedError("To be implemented")
#        if hyperparams.get('normalize', False):
#            print("WARNING: normalization not supported yet for DDPG/DQN")
#        env = gym.make(env_id)
#        env.seed(args.seed)
    else:
        if n_envs == 1:
            env = DummyVecEnv([make_env(env_id, log_dir, seed, env_i=0, n_envs=1)])
        else:
            env = SubprocVecEnv([make_env(env_id, log_dir, seed, env_i=i, n_envs=n_envs) for i in range(n_envs)])
        if normalize:
            print("Normalizing input and return")
            env = VecNormalize(env, **normalize_kwargs)

    # Optional Frame-stacking
    n_stack = 1
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']

    # Parse noise string for DDPG
    if algo == 'ddpg' and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                            sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        del hyperparams['noise_std']

#    if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
#        # Continue training
#        print("Loading pretrained agent")
#        # Policy should not be changed
#        del hyperparams['policy']
#
#        model = ALGOS[args.algo].load(args.trained_agent, env=env,
#                                      tensorboard_log=tensorboard_log, verbose=1, **hyperparams)
#
#        exp_folder = args.trained_agent.split('.pkl')[0]
#        if normalize:
#            print("Loading saved running average")
#            env.load_running_average(exp_folder)
#    else:
#        # Train an agent from scratch
#        model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)
    
    # Train agent
    tensorboard_log = None if no_tensorboard else log_dir
    policy_kwargs = dict(net_arch=[width for _ in range(depth)])    # act_fun defaults to tanh
#    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[width for _ in range(depth)])
    model = STR_TO_ALGO[algo](env=env, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, verbose=0, **hyperparams)

    kwargs = {'tb_log_name': TB_LOG_NAME}
    if log_interval > -1:
        kwargs = {'log_interval': log_interval}

    model.learn(n_timesteps, **kwargs)

    # Save trained model
    print("Saving to {}".format(log_dir))
    model.save(os.path.join(log_dir, 'model'))
    
    # Save command line arguments
    with open(os.path.join(log_dir, 'cmd_args.yml'), 'w') as f:
        yaml.dump(args_dict, f)
    
    # Save hyperparams
    with open(os.path.join(log_dir, 'hyperparams.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    if normalize:
        # Unwrap
        if isinstance(env, VecFrameStack):
            env = env.venv
        # Important: save the running average, for testing the agent we need that normalization
        env.save_running_average(log_dir)
        
        
def make_env(env_id, log_dir, seed, env_i=0, n_envs=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    effective_seed = seed * n_envs + env_i
    set_global_seeds(effective_seed)
    def _init():
        env = Monitor(gym.make(env_id), log_dir)
        env.seed(effective_seed)
        return env
    return _init


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func