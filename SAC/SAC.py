import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

# From https://github.com/openai/spinningup/blob/master/spinup/algos/sac/core.py


def gaussian_likelihood(noise, log_std):
    pre_sum = -0.5 * noise.pow(2) - log_std
    return pre_sum.sum(
        -1, keepdim=True) - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def apply_squashing_func(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 2 * action_dim)

        self.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu, log_std = self.l3(x).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_likelihood(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = apply_squashing_func(mu, pi, log_pi)

        return mu, pi, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, width):
        super(Critic, self).__init__()

        ## orginal width : 256 for all layers
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, width)
        self.l5 = nn.Linear(width, width)
        self.l6 = nn.Linear(width, 1)

        self.apply(weight_init)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, initial_temperature, width):
        self.actor = Actor(state_dim, action_dim, max_action, width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, width).to(device)
        self.critic_target = Critic(state_dim, action_dim, width).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])

        self.max_action = max_action

    def set_lr(self, lr):
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr

        for param_group in self.log_alpha_optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            mu, _, _ = self.actor(
                state, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            mu, pi, _ = self.actor(state, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def train(self,
              replay_buffer,
              total_timesteps,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              policy_freq=2,
              target_entropy=None):

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        def fit_critic():
            with torch.no_grad():
                _, policy_action, log_pi = self.actor(next_state)
                target_Q1, target_Q2 = self.critic_target(
                    next_state, policy_action)
                target_V = torch.min(target_Q1,
                                     target_Q2) - self.alpha.detach() * log_pi
                target_Q = reward + (done * discount * target_V)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        fit_critic()

        def fit_actor():
            # Compute actor loss
            _, pi, log_pi = self.actor(state)
            actor_Q1, actor_Q2 = self.critic(state, pi)

            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if target_entropy is not None:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (
                    self.alpha * (-log_pi - target_entropy).detach()).mean()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

        if total_timesteps % policy_freq == 0:
            fit_actor()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(),
                   '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(
            torch.load('%s/%s_critic.pth' % (directory, filename)))
