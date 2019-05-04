import argparse
import os
import random
import time

import gym
import imageio
import numpy as np
import torch

import SAC
import utils

from utils import Logger
from utils import create_folder


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy,
                    total_timesteps,
                    eval_episodes=10,
                    render=False,
                    skip_frame=10):
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()

        if render and i == 0:
            frames = [env.render(mode='rgb_array').copy()]

        done = False
        t = 0
        while not done:
            t += 1
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            if render and i == 0 and t % skip_frame == 0:
                frames.append(env.render(mode='rgb_array').copy())

    avg_reward /= eval_episodes


    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", default="HalfCheetah-v1")  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4,
        type=int)  # How many time steps purely random policy is run for
    parser.add_argument(
        "--eval_freq", default=5e3,
        type=float)  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=1e6,
        type=float)  # Max time steps to run environment for
    parser.add_argument(
        "--save_models",
        action="store_true")  # Whether or not models are saved
    parser.add_argument(
        "--save_videos",
        action="store_true")  # Whether or not evaluation vides are saved
    parser.add_argument(
        "--print_fps", action="store_true")  # Whether or not print fps
    parser.add_argument(
        "--batch_size", default=100,
        type=int)  # Batch size for both actor and critic
    parser.add_argument(
        "--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument(
        "--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--initial_temperature", default=0.2, type=float)  # SAC temperature
    parser.add_argument(
        "--learn_temperature",
        action="store_true")  # Whether or not learn the temperature
    parser.add_argument(
        "--policy_freq", default=2,
        type=int)  # Frequency of delayed policy updates
    parser.add_argument(
        "--normalize_returns", action="store_true")  # Normalize returns
    parser.add_argument("--linear_lr_decay", action="store_true")  # Decay lr

    parser.add_argument("--policy_name", default="SAC", help = "SAC")
    parser.add_argument("--folder", type=str, default='./results/') 
    parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')
    

    parser.add_argument("--width", default=256, type=int) ## only one width hparam - same width across layers 

    args = parser.parse_args()

    if args.normalize_returns and args.initial_temperature != 0.01:
        print("Please use temperature of 0.01 for normalized returns")

    if args.use_logger:
        file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
        logger = Logger(experiment_name = args.policy_name, environment_name = args.env_name, width_net=str(args.width), folder = args.folder)
        logger.save_args(args)

        print ('Saving to', logger.save_folder)

    env = gym.make(args.env_name)

    # Set seeds
    seed = np.random.randint(20)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)


    if args.use_logger: 
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("Seed : %s" % (seed))
        print ("---------------------------------------")



    if torch.cuda.is_available():
        torch.set_num_threads(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "SAC":
        policy = SAC.SAC(state_dim, action_dim, max_action, args.initial_temperature, args.width)



    replay_buffer = utils.ReplayBuffer(norm_ret=args.normalize_returns)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy, 0, render=args.save_videos)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    if args.print_fps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prev_time = time.time()
        prev_eval_timesteps = 0

    while total_timesteps < args.max_timesteps:

        if args.linear_lr_decay:
            policy.set_lr(1e-3 * (1 - float(total_timesteps) / args.max_timesteps))

        if done:
            if total_timesteps != 0:
                if args.print_fps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    fps = (total_timesteps - prev_eval_timesteps) / (
                        time.time() - prev_time)
                    print((
                        "Total T: %d FPS %d Episode Num: %d Episode T: %d Reward: %f"
                    ) % (total_timesteps, fps, episode_num, episode_timesteps,
                         episode_reward))
                else:
                    print(
                        ("Total T: %d Episode Num: %d Episode T: %d Reward: %f"
                         ) % (total_timesteps, episode_num, episode_timesteps,
                              episode_reward))

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy, total_timesteps, render=args.save_videos))

                if args.use_logger:
                    logger.record_reward(evaluations)
                    logger.save()
                    # np.save("./results/%s" % (file_name), evaluations)

                if args.print_fps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prev_time = time.time()
                    prev_eval_timesteps = total_timesteps

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.sample_action(np.array(obs))


        if total_timesteps > 1e3:
            policy.train(replay_buffer, total_timesteps, args.batch_size,
                         args.discount, args.tau, args.policy_freq,
                         -action_dim if args.learn_temperature else None)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # Final evaluation
    evaluations.append(evaluate_policy(policy, total_timesteps, render=args.save_videos))
    if args.use_logger:
        logger.record_reward(evaluations)
        logger.save()

