import os
import math
import time
import gymnasium as gym
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import ptan


class PPOActor(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64):
        super(PPOActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        return self.mu(x)


class PPOCritic(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=64):
        super(PPOCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.value(x)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.actor = net
        self.device = device

    def __call__(self, states, agent_states):
        states_tensor = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_tensor = self.actor(states_tensor)
        mu = mu_tensor.data.cpu().numpy()
        logstd = self.actor.logstd.data.cpu().numpy()
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class PPOTrainer:
    def __init__(self, hidden_dim=64, lr=3e-4, gamma=0.99, gae_lamda=0.95, clip_eps=0.2, epochs=10, batch_size=64):
        self.env = gym.make("MountainCarContinuous-v0")
        self.test_env = gym.make("MountainCarContinuous-v0")
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.gamma = gamma
        self.gae_lamda = gae_lamda
        self.clip_eps = clip_eps
        self.epochs = epochs 

        self.trajectory_size = 2049
        self.batch_size = batch_size
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.test_iters = 100000
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PPOActor(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic = PPOCritic(self.state_dim, hidden_dim).to(self.device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        self.agent = AgentA2C(self.actor, device=self.device)
        self.save_path = "/home/chenxiang.101/workspace/outputs/MPD-PPO"
        self.writer = SummaryWriter(self.save_path)
    
    # log(policy) 
    def policy_logprob(self, states, actions):
        mu = self.actor(states)
        logstd = self.actor.logstd
        p1 = - ((mu - actions) ** 2) / (2 * torch.exp(logstd).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd)))
        return p1 + p2

    # A, Q
    def calc_adv_ref(self, rewards, states, dones):
        values = self.critic(states)
        values = values.squeeze().data.cpu().numpy()
        gae = 0.0  
        advs = []
        refs = []
        for val, next_val, reward, done in zip(
            reversed(values[:-1]), reversed(values[1:]), reversed(rewards[:-1]), reversed(dones[:-1])
        ):
            if done:
                delta = reward - val
                gae = delta
            else:
                delta = reward + self.gamma * next_val - val
                gae = delta + self.gamma * self.gae_lamda * gae 
            advs.append(gae)
            refs.append(gae + val)

        adv = torch.FloatTensor(list(reversed(advs))).to(self.device)
        ref = torch.FloatTensor(list(reversed(refs))).to(self.device)
        return adv, ref

    def test(self):
        rewards = 0.0
        steps = 0
        counts = 10
        for _ in range(counts):
            obs, _ = self.test_env.reset()
            while True:
                states = torch.FloatTensor(np.array([obs])).to(self.device)
                mu = self.actor(states)  
                action = mu.squeeze(dim=0).data.cpu().numpy() 
                action = np.clip(action, -1, 1)  
                obs, reward, terminated, truncated, _ = self.test_env.step(action)
                rewards += reward
                steps += 1
                if terminated or truncated:
                    break
        return rewards / counts, steps / counts


    def train(self):
        exp_source = ptan.experience.ExperienceSource(self.env, self.agent, steps_count=1)
        trajectory = [] 
        best_reward = -1000
        
        with ptan.common.utils.RewardTracker(self.writer) as tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    self.writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % self.test_iters == 0:
                    ts = time.time()
                    rewards, steps = self.test()
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
                    self.writer.add_scalar("test_reward", rewards, step_idx)
                    self.writer.add_scalar("test_steps", steps, step_idx)
                    
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            torch.save(self.actor.state_dict(), os.path.join(self.save_path, 'best_' + str(rewards)))
                        if rewards > 90:
                            break
                        best_reward = rewards
                        
                trajectory.append(exp)
                if len(trajectory) < self.trajectory_size:
                    continue

                traj_states = torch.tensor(np.array([t[0].state for t in trajectory]), dtype=torch.float32).to(self.device)
                traj_actions = torch.tensor(np.array([t[0].action for t in trajectory]), dtype=torch.float32).to(self.device)
                traj_rewards = torch.tensor(np.array([t[0].reward for t in trajectory]), dtype=torch.float32).to(self.device)
                traj_dones = torch.tensor(np.array([t[0].done_trunc for t in trajectory]), dtype=torch.bool).to(self.device)
                traj_advs, traj_refs = self.calc_adv_ref(traj_rewards, traj_states, traj_dones)
                old_logprob = self.policy_logprob(traj_states, traj_actions)
                traj_advs = (traj_advs - torch.mean(traj_advs)) / torch.std(traj_advs)
                
                old_logprobs = old_logprob[:-1].detach()

                sum_loss_value = 0.0
                sum_loss_policy = 0.0
                count_steps = 0

                for epoch in range(self.epochs):
                    for batch_idx in range(0, len(trajectory)-1, self.batch_size):
                        batch_states = traj_states[batch_idx:batch_idx + self.batch_size]
                        batch_actions = traj_actions[batch_idx:batch_idx + self.batch_size]
                        batch_advs = traj_advs[batch_idx:batch_idx + self.batch_size].unsqueeze(-1)
                        batch_refs = traj_refs[batch_idx:batch_idx + self.batch_size]
                        batch_old_logprobs = old_logprobs[batch_idx:batch_idx + self.batch_size]

                        self.critic_optim.zero_grad()
                        batch_values = self.critic(batch_states)
                        loss_values = F.mse_loss(batch_values.squeeze(-1), batch_refs)
                        loss_values.backward()
                        self.critic_optim.step()

                        self.actor_optim.zero_grad()
                        batch_log_probs = self.policy_logprob(batch_states, batch_actions)
                        ratio = torch.exp(batch_log_probs - batch_old_logprobs)
                        surr_loss = batch_advs * ratio
                        clipped_surr = batch_advs * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        loss_policy = -torch.min(surr_loss, clipped_surr).mean()
                        loss_policy.backward()
                        self.actor_optim.step()

                        sum_loss_value += loss_values.item()
                        sum_loss_policy += loss_policy.item()
                        count_steps += 1

                trajectory.clear()
                self.writer.add_scalar("advantage", traj_advs.mean().item(), step_idx)
                self.writer.add_scalar("values", traj_refs.mean().item(), step_idx)
                self.writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
                self.writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)
        
        
if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
