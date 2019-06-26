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
import math

import gym
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, DDPG, SAC, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf

HYPERPARAMS_PARENT_FOLDER = 'hyperparams'
HYPERPARAMS_FOLDER = os.path.join(HYPERPARAMS_PARENT_FOLDER, 'rl-baselines-zoo')
MONITOR_FOLDER = 'monitor'
TB_LOG_NAME = 'tb'
TUNED_WIDTH = 64
LR_KEY = 'learning_rate'
NSTEPS_KEY = 'n_steps'

STR_TO_ALGO = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
}

# Default learnings rates copied from Stable Baselines on May 1, 2019
ALGO_TO_DEF_LR = {
    'a2c': 7e-4,
    'acer': 7e-4,
    'acktr': 0.25,  # just the discrete one
    'dqn': 5e-4,
#    'ddpg': {'actor_lr': 1e-4, 'critic_lr': 1e-3},
    'sac': 3e-4,
    'ppo2': 2.5e-4,
}

# Default n_steps copied from Stable Baselines on June 26, 2019
ALGO_TO_DEF_NSTEPS = {
    'a2c': 5,
    'acer': 20,
    'acktr': 20,  # just the discrete one
    # 'dqn': 32, ("batch_size=32")
    # 'ddpg': 128, ("batch_size=128")
    # 'sac': 64, ("batch_size=64")
    'ppo2': 128,
}

STR_TO_ACT_FUN = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
}

def build_monitor_dir(log_dir):
    return os.path.join(log_dir, MONITOR_FOLDER)


class Trainer:

    def __init__(self, env_id, algo, seed, width, log_dir, args_dict, depth, n_timesteps,
            log_interval, scale_lr, no_tensorboard, lr_pow, act_fun, break_width, default_hyper=False):
        # Add all arguments as fields
        self.__dict__.update(locals())
        del self.__dict__['self']

        self.monitor_dir = build_monitor_dir(log_dir)
        self.tensorboard_log = None if no_tensorboard else log_dir
        if act_fun not in STR_TO_ACT_FUN:
            raise ValueError('Invalid activation function: {}'.format(act_fun))
        else:
            print('Using activation function:', act_fun)
        self.net_kwargs = dict(act_fun=STR_TO_ACT_FUN[act_fun], net_arch=[width for _ in range(depth)])
        learn_kwargs = {'tb_log_name': TB_LOG_NAME}
        if self.log_interval > -1:
            learn_kwargs = {'log_interval': self.log_interval}
        self.learn_kwargs = learn_kwargs

        # Make directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.monitor_dir, exist_ok=True)

    def save(self, model, saved_hyperparams):
        print("Saving to {}".format(self.log_dir))
        # Save trained model
        model.save(os.path.join(self.log_dir, 'model'))
        # Save command line arguments
        with open(os.path.join(self.log_dir, 'cmd_args.yml'), 'w') as f:
            yaml.dump(self.args_dict, f)
        # Save hyperparams
        with open(os.path.join(self.log_dir, 'hyperparams.yml'), 'w') as f:
            yaml.dump(saved_hyperparams, f)

    def default_train(self):
        print('Running with Stable Baselines default hyperparameters')

        # Set default
        if self.n_timesteps == -1:
            self.n_timesteps = 1e6

        # Only using single env (for TRPO) right now
        env = DummyVecEnv([make_env(self.env_id, self.monitor_dir, self.seed, env_i=0, n_envs=1)])
        model = STR_TO_ALGO[self.algo](MlpPolicy, env=env, policy_kwargs=self.net_kwargs,
                                  tensorboard_log=self.tensorboard_log, verbose=0)
        model.learn(self.n_timesteps, **self.learn_kwargs)

        hyperparams = 'default'
        self.save(model, hyperparams)
        env.close()

    def zoo_train(self):
        """
        Train an RL agent with the given specifications, using the Stable Baselines
        library and tuned hyperparametes from rl-baselines-zoo.
        Code largely adapted from rl-baselines-zoo's train.py:
            https://github.com/araffin/rl-baselines-zoo/blob/master/train.py
        """
        is_atari = False
        if 'NoFrameskip' in self.env_id:
            is_atari = True

        print("=" * 10, self.env_id, "=" * 10)

        # Load hyperparameters from yaml file
        with open(os.path.join(HYPERPARAMS_FOLDER, self.algo) + '.yml', 'r') as f:
            if is_atari:
                hyperparams = yaml.safe_load(f)['atari']
            else:
                hyperparams = yaml.safe_load(f)[self.env_id]

        # Scale learning rate and batch size with width
        if self.scale_lr:
            get_scaled = lambda x: min(x, x * math.pow(self.width / self.break_width, self.lr_pow))

            lr_before = hyperparams.get(LR_KEY, ALGO_TO_DEF_LR[self.algo])
            if LR_KEY not in hyperparams:
                hyperparams[LR_KEY] = get_scaled(ALGO_TO_DEF_LR[self.algo])
            elif isinstance(hyperparams[LR_KEY], str):
                schedule, initial_value = hyperparams[LR_KEY].split('_')
                initial_value = str(get_scaled(float(initial_value)))
                hyperparams[LR_KEY] = schedule + '_' + initial_value
            elif isinstance(hyperparams[LR_KEY], float):
                hyperparams[LR_KEY] = get_scaled(hyperparams[LR_KEY])
            else:
                raise ValueError('Invalid value for {}: {}'.format(LR_KEY, hyperparams[LR_KEY]))
            print('Scaled learning rate from {} to {}'.format(lr_before, hyperparams[LR_KEY]))

            # Batch size scaling currently only scaling n_steps (not in all learners)
            nsteps_before = hyperparams.get(NSTEPS_KEY, ALGO_TO_DEF_NSTEPS[self.algo])
            if NSTEPS_KEY not in hyperparams:
                hyperparams[NSTEPS_KEY] = int(get_scaled(ALGO_TO_DEF_NSTEPS[self.algo]))
            elif isinstance(hyperparams[NSTEPS_KEY], int):
                hyperparams[NSTEPS_KEY] = int(get_scaled(hyperparams[NSTEPS_KEY]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(NSTEPS_KEY, hyperparams[NSTEPS_KEY]))
            print('Scaled n_steps from {} to {}'.format(nsteps_before, hyperparams[NSTEPS_KEY]))

        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
        pprint(saved_hyperparams)

        n_envs = hyperparams.get('n_envs', 1)

        print("Using {} environments".format(n_envs))

        # Create learning rate schedules for ppo2 and sac
        if self.algo in ["ppo2", "sac"]:
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
                    raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

        # Should we overwrite the number of timesteps?
        if self.n_timesteps <= 0:
            self.n_timesteps = int(hyperparams['n_timesteps'])

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
    #        env = make_atari_env(self.env_id, num_env=n_envs, seed=self.vseed)
    #        # Frame-stacking with 4 frames
    #        env = VecFrameStack(self.env, n_stack=4)
        elif self.algo in ['dqn', 'ddpg']:
            raise NotImplementedError("To be implemented")
    #        if hyperparams.get('normalize', False):
    #            print("WARNING: normalization not supported yet for DDPG/DQN")
    #        env = gym.make(self.env_id)
    #        env.seed(self.seed)
        else:
            if n_envs == 1:
                env = DummyVecEnv([make_env(self.env_id, self.monitor_dir, self.seed, env_i=0, n_envs=1)])
            else:
                env = SubprocVecEnv([make_env(self.env_id, self.monitor_dir, self.seed, env_i=i, n_envs=n_envs) for i in range(n_envs)])
            if normalize:
                print("Normalizing input and return")
                env = VecNormalize(env, **normalize_kwargs)

        # Optional Frame-stacking
        if hyperparams.get('frame_stack', False):
            n_stack = 1
            n_stack = hyperparams['frame_stack']
            env = VecFrameStack(env, n_stack)
            print("Stacking {} frames".format(n_stack))
            del hyperparams['frame_stack']

        # Parse noise string for DDPG
        if self.algo == 'ddpg' and hyperparams.get('noise_type') is not None:
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

       # if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
       #     # Continue training
       #     print("Loading pretrained agent")
       #     # Policy should not be changed
       #     del hyperparams['policy']
       #
       #     model = ALGOS[args.algo].load(args.trained_agent, env=env,
       #                                   tensorboard_log=tensorboard_log, verbose=1, **hyperparams)
       #
       #     exp_folder = args.trained_agent.split('.pkl')[0]
       #     if normalize:
       #         print("Loading saved running average")
       #         env.load_running_average(exp_folder)
       # else:
       #     # Train an agent from scratch
       #     model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

        # Train agent
        model = STR_TO_ALGO[self.algo](env=env, policy_kwargs=self.net_kwargs, tensorboard_log=self.tensorboard_log, verbose=0, **hyperparams)
        model.learn(self.n_timesteps, **self.learn_kwargs)

        self.save(model, saved_hyperparams)

        if normalize:
            # Unwrap
            if isinstance(env, VecFrameStack):
                env = env.venv
            # Important: save the running average, for testing the agent we need that normalization
            env.save_running_average(self.log_dir)

        env.close()




def make_env(env_id, monitor_dir, seed, env_i=0, n_envs=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param monitor_dir (str): path to where to save monitor files
    :param seed: (int) the inital seed for RNG
    :param env_i: (int) index of the subprocess
    :param n_envs: (int) the number of environments you wish to have in subprocesses
    """
    def _init():
        effective_seed = seed * n_envs + env_i
        set_global_seeds(effective_seed)
        env = Monitor(gym.make(env_id),
                      os.path.join(monitor_dir, str(env_i)),
                      allow_early_resets=True)
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
