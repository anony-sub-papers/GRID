import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utility_functions import seed_all

class SequentialMCMC:
    def __init__(self, env, num_steps: int, proposal_std: float = 0.5, temperature: float = 1.0):
        """
        Sequential MCMC sampler for trajectory generation.

        Args:
            env: The environment with step/reset/generate_full_features/reward.
            num_steps: Number of steps in the trajectory.
            proposal_std: Std deviation of Gaussian proposal.
            temperature: Temperature parameter for acceptance probability.(Higher = more exploration), 0.0 = no exploration
        """
        self.env = env
        self.num_steps = num_steps
        self.proposal_std = proposal_std
        self.temperature = temperature

    def log_reward(self, reward: float) -> float:
        """Compute log-proportional reward with stability."""
        return np.log(np.clip(reward, 1e-5, None))

    def sample_trajectory(self, initial_state=None):
        """
        Generate trajectory using sequential MCMC sampling.

        Returns:
            trajectory: Array of states
            final_reward: Reward of final state
            full_reward: Total trajectory reward
        """
        current_state = self.env.reset() if initial_state is None else initial_state
        trajectory = [current_state.copy()]
        current_log_p = 0.0

        for t in range(1, self.num_steps):
            # Propose new action
            proposal = np.random.normal(scale=self.proposal_std, size=current_state.shape[0] - 1)
            temp_env = self.env.copy_self()
            temp_env.set_state(current_state.copy())
            next_state, _ = temp_env.step(proposal)

            # Calculate acceptance probability
            full_traj = np.array(trajectory + [next_state])
            feats, _ = temp_env.generate_full_features(full_traj)
            reward, _ = temp_env.calculate_final_reward_ml(feats.iloc[-1:])
            log_p = self.log_reward(reward)

            # Accept or reject based on Metropolis criterion
            accept_prob = np.exp((log_p - current_log_p) / self.temperature)
            if np.random.rand() < accept_prob:
                current_state = next_state
                current_log_p = log_p

            trajectory.append(current_state.copy())

        traj_array = np.array(trajectory)
        final_feats, _ = self.env.generate_full_features(traj_array)
        final_reward, _ = self.env.calculate_final_reward_ml(final_feats.iloc[-1:])
        full_reward = self.env.calculate_final_reward_ml(final_feats)
        return traj_array, final_reward, full_reward
    
# class SequentialMCMC:
#     def __init__(self, env, num_steps, proposal_std=0.5):
#         """
#         MCMC sampler for generating trajectories in an environment.

#         Args:
#             env (GeneralEnvironment): Environment to simulate trajectories.
#             num_steps (int): Number of timesteps in the trajectory.
#             proposal_std (float): Standard deviation for the proposal distribution.
#         """
#         self.env = env
#         self.num_steps = num_steps
#         self.proposal_std = proposal_std

#     def sample_trajectory(self, initial_state=None):
#         """
#         Generate a trajectory using MCMC within the environment.

#         Args:
#             initial_state (np.array): Initial state vector (optional).

#         Returns:
#             trajectory (np.array): Array of sampled states (shape: [num_steps, state_dim]).
#             final_reward (float): The reward of the final state in the trajectory.
#         """
#         state = self.env.reset() if initial_state is None else initial_state
#         trajectory = [state.copy()]
#         current_state = state.copy()

#         for _ in range(1, self.num_steps):
#             # Propose a new action
#             proposed_action = np.random.normal(scale=self.proposal_std, size=self.env.state.shape[0] - 1)
            
#             # Apply action to the environment
#             next_state, _ = self.env.step(proposed_action)
            
#             # Save the state
#             trajectory.append(next_state.copy())
#             current_state = next_state

#         # Compute reward for the final state
#         final_trajectory = np.array(trajectory)
#         final_features, _ = self.env.generate_full_features(final_trajectory)
#         final_reward, _ = self.env.calculate_final_reward_ml(final_features.iloc[-1:])
#         full_reward = self.env.calculate_final_reward_ml(final_features)
#         return np.array(trajectory), final_reward,full_reward
from tqdm import tqdm
def generate_top_mcmc_results(env, num_trajectories, num_steps, proposal_std, feature_names,temperature=1.0, seed=None):
    """
    Generate trajectories using MCMC and extract results sorted by final rewards.

    Args:
        env (GeneralEnvironment): Environment to simulate trajectories.
        num_trajectories (int): Number of trajectories to generate.
        num_steps (int): Number of timesteps in each trajectory.
        proposal_std (float): Standard deviation for the MCMC proposal distribution.
        feature_names (List[str]): List of feature names for the states.

    Returns:
        np.ndarray: Array of sorted trajectories
        np.ndarray: Array of sorted full rewards
        np.ndarray: Array of sorted actions
        List[str]: Feature names
    """
    mcmc_sampler = SequentialMCMC(env, num_steps, proposal_std,temperature)
    all_trajectories = []
    rewards = []
    final_rewards = []
    all_actions = []

    for _ in tqdm(range(num_trajectories)):
        trajectory, final_reward, full_reward = mcmc_sampler.sample_trajectory()
        all_trajectories.append(trajectory)
        rewards.append(full_reward[0])
        final_rewards.append(final_reward)
        # Extract actions from trajectory differences
        actions = np.diff(trajectory, axis=0)
        all_actions.append(actions)
        

    # Convert lists to numpy arrays
    all_trajectories = np.array(all_trajectories, dtype=object)
    rewards = np.array(rewards, dtype=object)
    final_rewards = np.array(final_rewards, dtype=object)
    all_actions = np.array(all_actions, dtype=object)
    # Sort trajectories based on final rewards
    print(f"Final rewards: {final_rewards}")
    final_rewards = final_rewards.ravel()
    sorted_indices = np.argsort(final_rewards)[::-1]
    sorted_trajectories = [all_trajectories[i] for i in sorted_indices]
    sorted_rewards = [rewards[i] for i in sorted_indices]
    sorted_actions = [all_actions[i] for i in sorted_indices]

    return (np.array(sorted_trajectories, dtype=object), 
            np.array(sorted_rewards, dtype=object),
            np.array(sorted_actions, dtype=object), 
            feature_names,
            np.array(final_rewards, dtype=object))


# import numpy as np
# from typing import List, Tuple


# class SequentialMCMC:
#     def __init__(self, env, num_steps: int, proposal_std: float = 0.5, temperature: float = 1.0):
#         """
#         Sequential Metropolis-Hastings MCMC sampler for trajectory generation.

#         Args:
#             env: The environment with step/reset/generate_full_features/reward.
#             num_steps: Number of steps in the trajectory.
#             proposal_std: Std deviation of Gaussian proposal.
#             temperature: Exploration temperature (for MH acceptance).
#         """
#         self.env = env
#         self.num_steps = num_steps
#         self.proposal_std = proposal_std
#         self.temperature = temperature

#     def log_reward(self, reward: float) -> float:
#         """Compute log-proportional reward with stability."""
#         return np.log(np.clip(reward, 1e-5, None))

#     def sample_trajectory(self, initial_state=None) -> Tuple[np.ndarray, float, float]:
#         """
#         Generate trajectory via sequential MH sampling.

#         Returns:
#             trajectory: np.array of visited states.
#             final_reward: float reward of final state.
#             full_reward: total reward trajectory.
#         """
#         current_state = self.env.reset() if initial_state is None else initial_state
#         trajectory = [current_state.copy()]
#         current_log_p = 0.0

#         for t in range(1, self.num_steps):
#             best_state = current_state.copy()
#             best_logp = current_log_p

#             for _ in range(3):  # Try 3 proposals per step
#                 proposal = np.random.normal(scale=self.proposal_std, size=current_state.shape[0] - 1)
#                 temp_env = self.env.copy()
#                 temp_env.set_state(current_state.copy())
#                 next_state, _ = temp_env.step(proposal)

#                 full_traj = np.array(trajectory + [next_state])
#                 feats, _ = temp_env.generate_full_features(full_traj)
#                 reward, _ = temp_env.calculate_final_reward_ml(feats.iloc[-1:])
#                 log_p = self.log_reward(reward)

#                 accept_prob = np.exp((log_p - current_log_p) / self.temperature)
#                 if np.random.rand() < accept_prob:
#                     best_state = next_state
#                     best_logp = log_p
#                     break

#             current_state = best_state
#             current_log_p = best_logp
#             trajectory.append(current_state.copy())

#         traj_array = np.array(trajectory)
#         final_feats, _ = self.env.generate_full_features(traj_array)
#         final_reward, _ = self.env.calculate_final_reward_ml(final_feats.iloc[-1:])
#         full_reward = self.env.calculate_final_reward_ml(final_feats)
#         return traj_array, final_reward, full_reward

# def generate_top_mcmc_results(env, num_trajectories, num_steps, proposal_std, feature_names):
#     """
#     Generate trajectories using MCMC and extract results sorted by final rewards.

#     Args:
#         env (GeneralEnvironment): Environment to simulate trajectories.
#         num_trajectories (int): Number of trajectories to generate.
#         num_steps (int): Number of timesteps in each trajectory.
#         proposal_std (float): Standard deviation for the MCMC proposal distribution.
#         feature_names (List[str]): List of feature names for the states.

#     Returns:
#         np.ndarray: Array of sorted trajectories
#         np.ndarray: Array of sorted full rewards
#         np.ndarray: Array of sorted actions
#         List[str]: Feature names
#     """
#     mcmc_sampler = SequentialMCMC(env, num_steps, proposal_std)
#     all_trajectories = []
#     rewards = []
#     final_rewards = []
#     all_actions = []

#     for _ in range(num_trajectories):
#         trajectory, final_reward, full_reward = mcmc_sampler.sample_trajectory()
#         all_trajectories.append(trajectory)
#         rewards.append(full_reward[0])
#         final_rewards.append(final_reward)
#         # Extract actions from trajectory differences
#         actions = np.diff(trajectory, axis=0)
#         all_actions.append(actions)

#     # Convert lists to numpy arrays
#     all_trajectories = np.array(all_trajectories, dtype=object)
#     rewards = np.array(rewards, dtype=object)
#     final_rewards = np.array(final_rewards, dtype=object)
#     all_actions = np.array(all_actions, dtype=object)
#     # Sort trajectories based on final rewards
#     final_rewards = final_rewards.ravel()
#     sorted_indices = np.argsort(final_rewards)[::-1]
#     sorted_trajectories = [all_trajectories[i] for i in sorted_indices]
#     sorted_rewards = [rewards[i] for i in sorted_indices]
#     sorted_actions = [all_actions[i] for i in sorted_indices]

#     return (np.array(sorted_trajectories, dtype=object), 
#             np.array(sorted_rewards, dtype=object),
#             np.array(sorted_actions, dtype=object), 
#             feature_names)