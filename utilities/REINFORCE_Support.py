# Standard Library Imports

import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.append('../')  # Adjust the path to import custom modules
# Third-Party Imports
import numpy as np
import pandas as pd
from tqdm import trange
import yaml
import mlflow
import mlflow.pytorch
import mlflow.exceptions
from mlflow.models.signature import infer_signature

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Custom Modules
from environments import GeneralEnvironment
from replay_buffer import *
from utility_functions import (
    load_initial_state,
    setup_logger
)
import warnings
warnings.filterwarnings("ignore")

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)
def setup_environment_and_logger(config_path: str, model_path: str):
    """Load configuration, setup environment, and logger."""
    # Load configuration
    config = yaml.safe_load(open(config_path, 'r'))
    training_parameters = config.get("model")['training_parameters']

    # Initialize logger
    logger, log_file_name = setup_logger(config['mlflow']['run_name'])

    # Load initial state
    initial_state, feature_names = load_initial_state(config)
    logger.info(f"Initial state: {initial_state}")
    logger.info(f"Feature names: {feature_names}")

    # Initialize the environment
    env = GeneralEnvironment(
        initial_state=initial_state,
        config=config,
        model=None,  # No pre-trained model for SAC
        input_dim=len(initial_state) + 1,
        max_steps=training_parameters['trajectory_length'],
        model_path=model_path
    )
    return config, env, logger, training_parameters, initial_state, feature_names
# Define the Actor Network

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, scale_factor: float = 1):
        """
        Actor network to output action distribution parameters (mean and std).

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in each layer.
            scale_factor (float): Factor to scale the mean and standard deviation for percentage changes.
        """
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.std_layer = nn.Linear(hidden_dim, action_dim)
        self.scale_factor = scale_factor

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the action distribution.
        """
        x = self.net(state)
        mean = torch.tanh(self.mean_layer(x))*15.0
        std = torch.sigmoid(self.std_layer(x)) * (10 - 3) + 3  # Ensure std is between 3 and 10
        return mean, std

import torch
import torch.optim as optim
from typing import List, Tuple, Optional

class ReinforceAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-4):
        # Exclude time-step from state_dim when initializing network
        self.policy_net = ActorNetwork(state_dim - 1, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        # Remove time-step from state input
        state_tensor = torch.FloatTensor(state[1:]).unsqueeze(0)
        mean, std = self.policy_net(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze().detach().numpy(), log_prob

    def update(self, trajectory_batch):
        policy_loss = 0
        for trajectory in trajectory_batch:
            states = torch.FloatTensor([s[1:] for s, _, _, _ in trajectory])
            rewards = torch.FloatTensor([r for _, _, r, _ in trajectory])
            log_probs = torch.stack([lp for _, _, _, lp in trajectory])
            
            # Calculate returns for each timestep
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + G  # No discount factor for episodic task
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate loss
            policy_loss += -(log_probs * returns).mean()

        policy_loss /= len(trajectory_batch)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

def main_reinforce_training(env, state_dim, action_dim, training_episodes=1000, max_steps=10):
    """
    Train the REINFORCE agent with step-wise rewards.
    """
    reinforce_agent = ReinforceAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(training_episodes):
        trajectories = []  # Store trajectory for training
        total_episode_reward = 0

        state = env.reset()
        trajectory = []

        for step in range(max_steps):
            # Select action and get log probability
            action, log_prob = reinforce_agent.select_action(state)

            # Take step in the environment
            next_state, done = env.step(action)

            # Accumulate trajectory
            trajectory.append((state, action, 0, log_prob))  # Temporary reward placeholder
            state = next_state

            if done:
                break

        # Calculate rewards for each step
        state_trajectory = np.array([step[0] for step in trajectory])
        final_features, _ = env.generate_full_features(state_trajectory)
        step_rewards, _ = env.calculate_final_reward_ml(final_features)

        # Create trajectories with actual step rewards
        for i, (state, action, _, log_prob) in enumerate(trajectory):
            trajectories.append((state, action, step_rewards[i], log_prob))

        # Update policy network
        reinforce_agent.update([trajectories])  # Pass a batch of one trajectory
        episode_rewards.append(step_rewards[-1])

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{training_episodes}: Avg Reward (last 100 episodes): {avg_reward:.2f}")

    return reinforce_agent, episode_rewards
#     """
#     Train the REINFORCE agent with step-wise rewards.
#     """
#     reinforce_agent = ReinforceAgent(state_dim, action_dim)
#     episode_rewards = []

#     for episode in range(training_episodes):
#         trajectories = []  # Store trajectory for training
#         total_episode_reward = 0

#         state = env.reset()
#         trajectory = []

#         for step in range(max_steps):
#             # Select action and get log probability
#             action, log_prob = reinforce_agent.select_action(state)

#             # Take step in the environment
#             next_state, done = env.step(action)

#             # Accumulate trajectory
#             trajectory.append((state, action, 0, log_prob))  # Temporary reward placeholder
#             state = next_state

#             if done:
#                 break

#         # Calculate rewards for each step
#         state_trajectory = np.array([step[0] for step in trajectory])
#         final_features, _ = env.generate_full_features(state_trajectory)
#         step_rewards, _ = env.calculate_final_reward_ml(final_features)

#         # Create trajectories with actual step rewards
#         for i, (state, action, _, log_prob) in enumerate(trajectory):
#             trajectories.append((state, action, step_rewards[i], log_prob))

#         # Update policy network
#         reinforce_agent.update([trajectories])  # Pass a batch of one trajectory
#         episode_rewards.append(step_rewards[-1])

#         if (episode + 1) % 10 == 0:
#             avg_reward = np.mean(episode_rewards[-100:])
#             print(f"Episode {episode + 1}/{training_episodes}: Avg Reward (last 100 episodes): {avg_reward:.2f}")

#     return reinforce_agent, episode_rewards


def train_reinforce(env, state_dim, action_dim,
                    episodes=1000, max_steps=10, gamma=0.99):
    agent = ReinforceAgent(state_dim, action_dim)
    all_episode_rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        trajectory = []  # list of (log_prob, reward)
        ep_rewards = []

        # 1) Roll out one episode, collect log-probs and immediate rewards
        for t in range(max_steps):
            action, logp = agent.select_action(state)
            next_state, done = env.step(action)
            
            # get immediate reward for this new state
            # (assumes calculate_final_reward_ml can be called one-by-one)
            features, _ = env.generate_full_features(np.array([state]))
            r_t, _ = env.calculate_final_reward_ml(features)
            r_t = float(r_t[0])  # if returned as array
            
            trajectory.append((logp, r_t))
            ep_rewards.append(r_t)

            state = next_state
            if done:
                break

        # 2) Compute reward-to-go for each step
        returns = []
        G = 0.0
        for (_, r_t) in reversed(trajectory):
            G = r_t + gamma * G
            returns.insert(0, G)

        # 3) Policy update: one gradient step with the collected data
        agent.optimizer.zero_grad()
        policy_loss = 0.0
        for (logp, _), Gt in zip(trajectory, returns):
            policy_loss -= logp * Gt   # gradient ascend
        policy_loss = policy_loss / len(trajectory)
        policy_loss.backward()
        agent.optimizer.step()

        # logging
        episode_return = sum(ep_rewards)
        all_episode_rewards.append(episode_return)
        if (ep+1) % 10 == 0:
            avg = np.mean(all_episode_rewards[-100:])
            print(f"Episode {ep+1}/{episodes}  AvgReturn(last100)={avg:.2f}")

    return agent, all_episode_rewards


# Standard Library Imports

import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.append('../')  # Adjust the path to import custom modules
# Third-Party Imports
import numpy as np
import pandas as pd
from tqdm import trange
import yaml
import mlflow
import mlflow.pytorch
import mlflow.exceptions
from mlflow.models.signature import infer_signature

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim


# Custom Modules
from environments import GeneralEnvironment
from replay_buffer import *
from utility_functions import (
    load_initial_state,
    setup_logger
)
import warnings
warnings.filterwarnings("ignore")

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)
# def setup_environment_and_logger(config_path: str, model_path: str):
#     """Load configuration, setup environment, and logger."""
#     # Load configuration
#     config = yaml.safe_load(open(config_path, 'r'))
#     training_parameters = config.get("model")['training_parameters']

#     # Initialize logger
#     logger, log_file_name = setup_logger(config['mlflow']['run_name'])

#     # Load initial state
#     initial_state, feature_names = load_initial_state(config)
#     logger.info(f"Initial state: {initial_state}")
#     logger.info(f"Feature names: {feature_names}")

#     # Initialize the environment
#     env = GeneralEnvironment(
#         initial_state=initial_state,
#         config=config,
#         model=None,  # No pre-trained model for SAC
#         input_dim=len(initial_state) + 1,
#         max_steps=training_parameters['trajectory_length'],
#         model_path=model_path
#     )
#     return config, env, logger, training_parameters, initial_state, feature_names
# Define the Actor Network
import torch
import numpy as np
import random
from IPython.display import clear_output

# Define the Value Network for baseline
import torch.nn as nn
import torch.optim as optim
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# Define Q-Network for SAC
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q(x)

# SAC Policy Network
class SACPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, log_std_min=0.01, log_std_max=10):
        super(SACPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        std = self.log_std_linear(x)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        std = torch.sigmoid(std) * (self.log_std_max - self.log_std_min) + self.log_std_min
        mean = torch.tanh(mean) * 15.0  # Scale to match REINFORCE action range
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # reparameterization trick
        # match REINFORCE action range
        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean, std

class ReinforceWithBaselineAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-4):
        # Exclude time-step from state_dim when initializing networks
        self.policy_net = ActorNetwork(state_dim - 1, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim - 1)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state[1:]).unsqueeze(0)  # Exclude time-step
        mean, std = self.policy_net(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = action.squeeze().detach().numpy()
        return action, log_prob

    def update(self, trajectories):
        states = torch.FloatTensor([s[1:] for s, _, _, _ in trajectories])  # Exclude time-step
        rewards = torch.FloatTensor([r for _, _, r, _ in trajectories])
        log_probs = torch.stack([lp for _, _, _, lp in trajectories])
        
        # Calculate baseline values
        baseline_values = self.value_net(states).squeeze()
        
        # Calculate advantages (returns - baseline)
        advantages = rewards - baseline_values.detach()
        
        # Policy loss using advantages
        policy_loss = (-log_probs * advantages).mean()
        
        # Value loss (MSE between baseline and actual returns)
        value_loss = nn.functional.mse_loss(baseline_values, rewards)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, alpha=None, gamma=0.99, tau=0.005):
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft target update rate
        self.target_entropy = -action_dim  # SAC default
        self.log_alpha = torch.tensor(np.log(0.2) if alpha is None else np.log(alpha), requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self._auto_alpha = True

        # Exclude time-step from state_dim when initializing networks
        self.policy = SACPolicy(state_dim - 1, action_dim, hidden_dim)
        self.q_critic1 = QNetwork(state_dim - 1, action_dim, hidden_dim)
        self.q_critic2 = QNetwork(state_dim - 1, action_dim, hidden_dim)
        self.target_q_critic1 = QNetwork(state_dim - 1, action_dim, hidden_dim)
        self.target_q_critic2 = QNetwork(state_dim - 1, action_dim, hidden_dim)

        # Initialize target networks
        for target_param, param in zip(self.target_q_critic1.parameters(), self.q_critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_q_critic2.parameters(), self.q_critic2.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_critic1_optimizer = optim.Adam(self.q_critic1.parameters(), lr=lr)
        self.q_critic2_optimizer = optim.Adam(self.q_critic2.parameters(), lr=lr)


    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state[1:]).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.policy.sample(state_tensor)
        return action.cpu().detach().numpy().flatten()

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return 0, 0, 0

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch)[:, 1:]
        next_state_batch = torch.FloatTensor(next_state_batch)[:, 1:]
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        # Q targets
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.policy.sample(next_state_batch)
            target_q1 = self.target_q_critic1(next_state_batch, next_action)
            target_q2 = self.target_q_critic2(next_state_batch, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_q = min_target_q - alpha * next_log_prob
            target_value = reward_batch + (1.0 - done_batch) * self.gamma * target_q

        # Q-network loss
        current_q1 = self.q_critic1(state_batch, action_batch)
        current_q2 = self.q_critic2(state_batch, action_batch)
        q1_loss = torch.nn.functional.mse_loss(current_q1, target_value)
        q2_loss = torch.nn.functional.mse_loss(current_q2, target_value)

        self.q_critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.q_critic1_optimizer.step()

        self.q_critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.q_critic2_optimizer.step()

        # Policy loss (maximize Q - alpha * entropy)
        new_action, log_prob, _, _ = self.policy.sample(state_batch)
        q1 = self.q_critic1(state_batch, new_action)
        q2 = self.q_critic2(state_batch, new_action)
        q = torch.min(q1, q2)
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_prob - q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Automatic entropy tuning
        if self._auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            print(f"Alpha: {self.alpha:.4f}, Alpha Loss: {alpha_loss.item():.4f}")
        # Soft update target networks
        for target_param, param in zip(self.target_q_critic1.parameters(), self.q_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_critic2.parameters(), self.q_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return policy_loss.item(), q1_loss.item(), q2_loss.item()

# Replay buffer for SAC
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def main_reinforce_with_baseline_training(env, state_dim, action_dim, training_episodes=1000, max_steps=10):
    """Train the REINFORCE with Baseline agent."""
    reinforce_baseline_agent = ReinforceWithBaselineAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(training_episodes):
        state = env.reset()
        trajectory = []

        for step in range(max_steps):
            action, log_prob = reinforce_baseline_agent.select_action(state)
            next_state, done = env.step(action)
            trajectory.append((state, action, 0, log_prob))  # Reward placeholder
            state = next_state
            if done:
                break

        # Calculate step-wise rewards
        state_trajectory = np.array([step[0] for step in trajectory])
        final_features, _ = env.generate_full_features(state_trajectory)
        step_rewards, _ = env.calculate_final_reward_ml(final_features)

        # Update trajectory with actual step rewards
        for i in range(len(trajectory)):
            trajectory[i] = list(trajectory[i])
            trajectory[i][2] = step_rewards[i]  # Assign the specific reward for each step
            trajectory[i] = tuple(trajectory[i])

        # Update agent
        reinforce_baseline_agent.update(trajectory)
        episode_rewards.append(step_rewards[-1])

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{training_episodes}: Avg Reward (last 100 episodes): {avg_reward:.2f}")

    return reinforce_baseline_agent, episode_rewards

import matplotlib.pyplot as plt

def main_sac_training(
    env, state_dim, action_dim,
    training_episodes=1000,
    max_steps=12,
    batch_size=64,
    buffer_size=10000,
    exploration_steps=500,
    alpha=None,
    gamma=0.99,
    tau=0.005,
    lr=3e-4
):
    """
    Train the SAC agent, tracking rewards and losses, and plot at the end.
    """
    sac_agent = SACAgent(state_dim, action_dim, alpha=alpha, gamma=gamma, tau=tau, lr=lr)
    print(f"SAC Agent initialized with alpha: {sac_agent.alpha:.4f}, gamma: {gamma}, tau: {tau}")
    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
    episode_rewards = []
    avg_rewards = []
    policy_losses = []
    q1_losses = []
    q2_losses = []

    for episode in range(training_episodes):
        state = env.reset()
        episode_reward = 0
        trajectory = []

        for step in range(max_steps):
            action = sac_agent.select_action(state)
            next_state, done = env.step(action)
            trajectory.append((state, action, next_state, done))
            state = next_state
            if done:
                break

        # Calculate final reward and assign proxy rewards to each step
        if trajectory:
            final_trajectory = np.array([t[0] for t in trajectory])
            final_features, _ = env.generate_full_features(final_trajectory)
            final_rewards, _ = env.calculate_final_reward_ml(final_features)
            
            for i, (s, a, ns, d) in enumerate(trajectory):
                replay_buffer.add(s, a, final_rewards[i], ns, d)
                episode_reward += final_rewards[i]

            # Update agent if enough samples in buffer
            if len(replay_buffer) >= batch_size:
                policy_loss, q1_loss, q2_loss = sac_agent.update(replay_buffer, batch_size)
                policy_losses.append(policy_loss)
                q1_losses.append(q1_loss)
                q2_losses.append(q2_loss)

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0 or (episode + 1) == training_episodes:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_rewards.append(avg_reward)
            print(f"Episode {episode + 1}/{training_episodes}: Avg Reward (last 100 episodes): {avg_reward:.2f}, current alpha: {sac_agent.alpha:.4f}")

    # Plot rewards and losses at the end
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(episode_rewards, color=color, label='Episode Reward')
    ax1.plot(
        [np.mean(episode_rewards[max(0, i-100):i+1]) for i in range(len(episode_rewards))],
        color='tab:cyan', linestyle='--', label='Avg Reward (last 100)'
    )
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(policy_losses, color='tab:red', alpha=0.5, label='Policy Loss')
    ax2.plot(q1_losses, color='tab:orange', alpha=0.5, label='Q1 Loss')
    ax2.plot(q2_losses, color='tab:green', alpha=0.5, label='Q2 Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('SAC Training: Rewards and Losses')
    plt.tight_layout()
    plt.show()

    return sac_agent, episode_rewards

def train_rl_agent(env, state_dim, action_dim, algorithm='reinforce', training_episodes=1000, max_steps=10, **kwargs):
    """
    Train a reinforcement learning agent using the specified algorithm.

    Args:
        env: The environment
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        algorithm: 'reinforce', 'reinforce_baseline', or 'sac'
        training_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        **kwargs: Additional algorithm-specific parameters

    Returns:
        The trained agent and episode rewards
    """
    # Set random seed for reproducibility
    seed = kwargs.get('seed', 42)
    set_global_seed(seed)
    print(f"Random seed set to: {seed}")
    # Extract algorithm-specific parameters
    batch_size = kwargs.get('batch_size', 64)
    buffer_size = kwargs.get('buffer_size', 25000)
    exploration_steps = kwargs.get('exploration_steps', 500)
    sac_params = kwargs.get('sac_params', {})
    alpha = sac_params.get('alpha', None)
    gamma = sac_params.get('gamma', 0.99)
    tau = sac_params.get('tau', 0.005)
    lr = sac_params.get('lr', 1e-4)
    if algorithm.lower() == 'reinforce':
        return main_reinforce_training(env, state_dim, action_dim, training_episodes, max_steps)
    elif algorithm.lower() == 'reinforce_baseline':
        return main_reinforce_with_baseline_training(env, state_dim, action_dim, training_episodes, max_steps)
    elif algorithm.lower() == 'sac':
        return main_sac_training(env, state_dim, action_dim, training_episodes,
                                  max_steps, batch_size, buffer_size, exploration_steps,
                                    alpha, gamma, tau,lr)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'reinforce', 'reinforce_baseline', or 'sac'.")

