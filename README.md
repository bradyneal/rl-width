# rl-width
Repo to study the effect of network width on performance in reinforcement learning.

## Example usage
Test run on Cartpole environment with 1000 steps:
`python main.py --n-timesteps 1000`

Same test run as above with many default arguments specified:
`python main.py --n-timesteps 1000 --env CartPole-v1 --algo ppo2 --widths 64 --depth 2 --start-end-seed 0 0 --hyperparam rl-baselines-zoo --results-dir . --name general --log-dir=logs --log-interval -1`

Run full experiment on Cartpole and MountainCar-v0 on widths 4, 16, and 64 for both PPO and A2C with 5 random seeds each:
`python main.py --env CartPole-v1 MountainCar-v0 --algo ppo2 a2c --widths 4 16 64 --start-end-seed 0 4`
