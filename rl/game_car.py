import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import random
import math
from collections import deque, namedtuple
import numpy as np
import os
import ptan
import copy


# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)
# render_model = "human/None/rgb_array/xx", human for visualization, None for faster training

def test():
    # position from -1.2 to 0.6, velocity from -0.07 to 0.07
    # the target(flag) is at 0.45
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    print(env.action_space)

    observation, info = env.reset(seed=42)
    for steps in range(1000):
        # this is where you would insert your policy
        action = env.action_space.sample()
        print('action: ', action)
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        print('state: ', observation, 'reward: ', reward)
        if terminated or truncated:
            # observation, info = env.reset()
            print(f'Game over after {steps} steps')
            break

    env.close()


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class PPOActor(nn.Module):
    """Actor网络，输出连续动作的均值和标准差"""
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = F.tanh(self.mean_layer(x))  # [-1, 1]
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std
 
 
class PPOCritic(nn.Module):
    """Critic网络：输出状态价值"""
    def __init__(self, state_dim=2, hidden_dim=128):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_layer(x)
    
    
class PPOTrainer:
    def __init__(self, hidden_dim=64, gamma=0.99, lamda=0.95, clip_eps=0.2, batch_size=64, update_epochs=10):
        self.env = gym.make("MountainCarContinuous-v0")
        self.test_env = gym.make("MountainCarContinuous-v0")
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        
        self.actor = PPOActor(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic = PPOCritic(self.state_dim, hidden_dim).to(device)
        
        self.actor_lr = 1e-4
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        self.critic_lr = 1e-3
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.update_epochs = update_epochs
        self.save_path = "/home/chenxiang.101/workspace/outputs/PPO-Car"
    
    
    def compute_gae(self, rewards, values, next_values, dones):
        """计算GAE优势函数"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages
    
    def update_actor(self, states, actions, old_log_probs, advantages):
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
      
        # PPO裁剪目标
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages.detach()
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # entropy = dist.entropy().mean()
        # actor_loss = actor_loss - 0.03 * entropy  # 增加熵正则化系数，鼓励更多探索
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optim.step()
        return actor_loss.item()
    
    
    def update_critic(self, b_states, b_advantages, b_values):
        values = self.critic(b_states)
        b_returns = b_advantages + b_values
        critic_loss = F.mse_loss(values.squeeze(1), b_returns.squeeze(1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        # 梯度裁剪
        # orch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()
        return critic_loss.item()
    
    
    def action_sampler_v1(self, state):
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_clamped = torch.clamp(action, -1.0, 1.0)   # 限制动作在合法范围
        log_prob = dist.log_prob(action_clamped)
        return action_clamped[0].detach().cpu().numpy(), log_prob[0].detach().cpu().numpy()
    
    def action_sampler_v2(self, state):
        mu, std = self.actor(state)
        mu = mu.data.cpu().numpy()
        std = std.data.cpu().numpy()
        action = mu + std * np.random.normal(size=std.shape)
        action = np.clip(action, -1, 1)
        
        logstd = self.actor.log_std.data.cpu().numpy()
        p1 = - ((mu - action) ** 2) / (2 * np.exp(logstd).clip(min=1e-3))
        p2 = - np.log(np.sqrt(2 * math.pi * np.exp(logstd)))
        log_prob = p1 + p2
        return action[0], log_prob[0]
    

    @torch.no_grad()
    def trajectory(self, max_steps=2048):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        states, rewards, actions, next_states, log_probs, dones = [], [], [], [], [], []
        values, next_values = [], []
        step = 0
        episode_rewards = list()
        
        while step < max_steps:
            step += 1
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0)   # 限制动作在合法范围
            log_prob = dist.log_prob(action_clamped)         
            
            next_state, reward, terminated, truncated, _ = self.env.step(action_clamped[0].detach().cpu().numpy())    # 执行动作
            
            # 奖励塑形：基于距离目标的距离给予奖励
            # position, velocity = next_state
            # # 目标位置是0.45，计算距离目标的距离
            # target_position = 0.45
            # distance_to_target = abs(target_position - position)
            # # 距离目标越近，奖励越高，确保始终为正奖励
            # reward += 20 * (1.0 - distance_to_target / 1.5)  # 距离奖励，最大20
            # # 鼓励向右移动（正速度）
            # reward += 10 * velocity
            # # 如果到达目标，给予大额奖励
            # if position >= target_position:
            #     reward += 500  # 完成奖励
        
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action_clamped)
            rewards.append(torch.tensor([reward], device=device))
            log_probs.append(log_prob)
            next_states.append(next_state)
            values.append(self.critic(state))
            dones.append(done)
            
            episode_rewards.append(reward)
            if done:
                next_values.append(torch.tensor([[0.0]], device=device))
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                
                print(f"Episode steps: {len(episode_rewards)}, Total Reward: {sum(episode_rewards):.4f}")
                episode_rewards.clear()
            else:
                next_values.append(self.critic(next_state))
            
            state = next_state
        advantages = self.compute_gae(rewards, values, next_values, dones)    
        return {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'rewards': rewards,
            'log_probs': log_probs,
            'advantages': advantages,
            'values': values
        }


    def train(self, max_steps=10000):
        step = turn = 0
        best_reward = -1000.0
        while step < max_steps:
            turn += 1
            # 学习率线性衰减
            # progress = step / max_steps
            # for param_group in self.actor_optim.param_groups:
            #     param_group['lr'] = self.actor_lr * (1 - progress)
            # for param_group in self.critic_optim.param_groups:
            #     param_group['lr'] = self.critic_lr * (1 - progress)
        
            samples = self.trajectory()
            states = samples['states']
            actions = samples['actions']
            log_probs = samples['log_probs']
            advantages = samples['advantages']
            values = samples['values']
            
            step += len(states)
            if step % 20480 == 0:
                rewards, test_steps = self.test()
                print(f"  Test episode steps: {test_steps}, Test Total Reward: {rewards:.4f}")
                if rewards > best_reward:
                    best_reward = rewards
                    torch.save(self.actor.state_dict(), os.path.join(self.save_path, 'best_' + str(rewards)))
                if rewards >= 90.0:
                    break
                    
            indices = np.arange(len(states))
            for epoch in range(self.update_epochs):
                np.random.shuffle(indices)
                actor_loss = critic_loss = 0.0 
                
                for b_start in range(0, len(states), self.batch_size):
                    batch_indices = indices[b_start:b_start+self.batch_size]
                    b_states = torch.cat([states[i] for i in batch_indices])
                    b_actions = torch.cat([actions[i] for i in batch_indices])
                    b_log_probs = torch.cat([log_probs[i] for i in batch_indices])
                    b_advantages = torch.cat([advantages[i] for i in batch_indices]).float()
                    b_values = torch.cat([values[i] for i in batch_indices])   
                    critic_loss += self.update_critic(b_states, b_advantages, b_values)     
                        
                    adv_mean, adv_std = b_advantages.mean().item(), b_advantages.std().item()
                    b_adv_norm = (b_advantages - adv_mean) / (adv_std + 1e-8)    # 归一化优势
                    actor_loss += self.update_actor(b_states, b_actions, b_log_probs, b_adv_norm)  
                
                batch_steps = (len(states) // self.batch_size) + 1
                actor_loss, critic_loss = np.array([actor_loss, critic_loss]) / batch_steps
                if epoch == self.update_epochs - 1:
                    print(f'Round {turn}, Epoch {epoch+1}/{self.update_epochs}, steps: {step}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}')
              
    @torch.no_grad()   
    def test(self):
        rewards = 0.0 
        steps = 0
        counts = 10
        for _ in range(counts):
            state, _ = self.test_env.reset()
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mean, _ = self.actor(state_tensor)
                action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy()
                state, reward, terminated, truncated, _ = self.test_env.step(action)
                rewards += reward
                done = terminated or truncated
                steps += 1
                if done:
                    break
        return rewards / counts, steps / counts
        
        
class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mean, std = self.net(states_v)
        dist = Normal(mean, std)
        action = dist.sample()
        action_clamped = torch.clamp(action, -1.0, 1.0).cpu().numpy()  
        return action_clamped, agent_states


def ptan_test():
    env = gym.make("MountainCarContinuous-v0")
    net = PPOActor().to('cuda')
    agent = AgentA2C(net, device='cuda')
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)
    print(exp_source)
    
    for idx, exp in enumerate(exp_source):
        # print(idx, exp)
        # exp: (Experience(state=array([-0.16255173,  0.00907798], dtype=float32), action=array([0.34531724], dtype=float32), reward=-0.01192439993696013, done_trunc=False),)
        # pop_rewards_steps 一次性pop一个回合的数据，总奖励和步长， [(-33.0852099433644, 999)]
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            print(rewards_steps)
            rewards, steps = zip(*rewards_steps)
            print(rewards)
            print(steps) 
        
        if idx > 1000:
            break
    

     
if __name__ == '__main__':
    # test()
    trainer = PPOTrainer()
    trainer.train(max_steps=1000000)
    
    # state = torch.tensor([-0.16255173,  0.00907798], dtype=torch.float32, device=device).unsqueeze(0)
    # action, log_prob = trainer.action_sampler_v1(state)
    # print(action, log_prob)
    # action, log_prob = trainer.action_sampler_v2(state)
    # print(action, log_prob)
    
    # trainer.trajectory(max_steps=2048)  
