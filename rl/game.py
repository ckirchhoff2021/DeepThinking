import gymnasium as gym
# https://gymnasium.farama.org/introduction/basic_usage/
# https://gymnasium.farama.org/environments/classic_control/cart_pole/

import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialise the environment
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("CartPole-v1", render_mode="human")
print(env.action_space)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def rule_policy(observation):
    angle = observation[2]
    action = 0 if angle < 0 else 1
    return action


dqn_model = DQN(n_observations=4, n_actions=2)
dqn_model.load_state_dict(torch.load('ckpts/policy_net.pth', map_location=torch.device('cpu')))
dqn_model.eval()

def dqn_policy(observation):
    state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn_model(state)
    action = torch.argmax(q_values, dim=1).item()
    return action

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for steps in range(1000):
    # this is where you would insert your policy
    # action = env.action_space.sample()
    # action = rule_policy(observation)
    action = dqn_policy(observation)
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print('state: ', observation)
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        # observation, info = env.reset()
        print(f'Game over after {steps} steps')
        break

env.close()
