## Helper Functions
import copy
import json
import logging
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from reward_calculator import RewardCalculator
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions import Distribution, constraints

from environments import GeneralEnvironment
from mlflow_logger import MLflowLogger
import yaml

# def seed_all(seed):
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.manual_seed(seed)

    
def seed_all(seed: int):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def calculate_statistics(series: pd.Series) -> dict:
    """Calculate the required statistics for a given series.

    Args:
        series (pd.Series): The input time series data.

    Returns:
        dict: A dictionary containing the statistics.
    """
    stats = {}
    stats['mean'] = series.mean()
    stats['max'] = series.max()
    stats['min'] = series.min()
    stats['quantiles'] = {q: series.quantile(q/100) for q in range(5, 100, 5)}
    
    # Differenced series statistics
    diff_series = series.diff().dropna()
    stats['diff_mean'] = diff_series.mean()
    stats['diff_max'] = diff_series.max()
    stats['diff_min'] = diff_series.min()
    stats['diff_quantiles'] = {q: diff_series.quantile(q/100) for q in range(5, 100, 5)}
    return stats

def process_variable_data(df: pd.DataFrame, config: dict, save_path: str):
    """Process each component and variable to extract statistics and save them in a JSON file.

    Args:
        df (pd.DataFrame): The DataFrame containing all the time series data.
        config (dict): The YAML configuration that contains the variables and components.
        save_path (str): The path where the JSON file will be saved.
    """
    # logging.info("Starting to process variable data for statistics extraction.")
    all_stats = {}
    
    for var_name, var_config in config['variables'].items():
        # logging.info(f"Processing variable: {var_name}")
        var_stats = {}
        
        for component_name, component_config in var_config['components'].items():
            # logging.info(f"Processing component: {component_name} for variable: {var_name}")
            source_name = component_config.get('source', component_name)
            
            try:
                series = df[source_name]
                var_stats[component_name] = calculate_statistics(series)
            except KeyError:
                # logging.warning(f"Missing data for component: {source_name}. Skipping.")
                var_stats[component_name] = None  # Indicate missing data
        
        # Calculate statistics for the variable itself
        try:
            # logging.info(f"Calculating statistics for the entire variable: {var_name}")
            variable_series = RewardCalculator('').calculate_variable(df, var_config)
            var_stats[var_name] = calculate_statistics(variable_series)
        except Exception as e:
            # logging.warning(f"Error calculating variable {var_name}: {e}. Skipping.")
            var_stats[var_name] = None  # Indicate calculation error
        
        all_stats[var_name] = var_stats
        # logging.info(f"Finished processing variable: {var_name}")
    
    # Save the statistics to a JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    # logging.info(f"Statistics saved to {save_path}")

def load_model(model_path: str, model: nn.Module, device: torch.device) -> nn.Module:
    """Load a PyTorch model from a file."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_entire_model(model_path: str, device: torch.device) -> nn.Module:
    """Load an entire PyTorch model from a file."""
    model = torch.load(model_path, map_location=device,weights_only = False)
    model.to(device)
    model.eval()
    return model

def generate_synthetic_data_for_scaling(initial_states, factor=3):
    """Generate synthetic data for scaling based on initial states."""
    synthetic_data = []
    
    for state in initial_states:
        synthetic_state = []
        for value in state:
            # Generate a range from 0 to `factor` times the value
            synthetic_feature_data = np.linspace(0, value * factor, num=100)
            synthetic_state.append(synthetic_feature_data)
        synthetic_data.append(np.array(synthetic_state).T)
    
    # Flatten the synthetic data into a 2D array
    synthetic_data = np.vstack(synthetic_data)
    return synthetic_data

def load_initial_state(config) -> np.ndarray:
    # Extract the initial state from the YAML configuration
    initial_state_dict = config.get('initial_state', {})
    sorted_keys = list(initial_state_dict.keys())

    # Convert the initial state dictionary to a NumPy array
    initial_state = np.array([initial_state_dict[key] for key in sorted_keys])

    return initial_state,sorted_keys

def setup_logger(run_name):
    """Setup a logger that writes only to a file."""
    # Create a logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Define the log file name
    log_file_name = f'logs/{run_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Setup logger
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler to log to a file
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    
    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False
    return logger, log_file_name

def plot_trajectories(trajectories: List[np.ndarray], rewards: List[float], plot_title: str = "Simulated Trajectories"):
    """Plot the trajectories and the rewards."""
    plt.figure(figsize=(12, 6))

    # Plot each trajectory
    for trajectory in trajectories:
        plt.plot(trajectory[:, :-1], alpha=0.2)  # Exclude the time step from the plot

    # Plot the average trajectory
    average_trajectory = np.mean(np.stack(trajectories), axis=0)
    plt.plot(average_trajectory[:, :-1], color='black', linewidth=2, label='Average Trajectory')
    
    # Plot the median trajectory
    median_trajectory = np.median(np.stack(trajectories), axis=0)
    plt.plot(median_trajectory[:, :-1], color='red', linewidth=2, label='Median Trajectory')
    
    # Plot rewards
    plt.figure(figsize=(24, 12))
    plt.plot(rewards, label='Rewards', marker='o')
    plt.xlabel('Trajectory Index', fontdict={'fontsize': 20})
    plt.ylabel('Reward', fontdict={'fontsize': 20})
    plt.title('Rewards for Simulated Trajectories', fontdict={'fontsize': 20})
    plt.legend(prop={'size': 20})
    plt.grid(True)
    plt.show()

def get_latest_model_path(artifact_uri: str, model_name: str) -> str:
    """
    Get the path to the latest saved model from the MLflow artifact directory.

    Args:
        artifact_uri (str): Base URI where artifacts are stored.
        model_name (str): The name of the model artifact directory.

    Returns:
        str: Path to the latest model file.
    """
    model_dir = os.path.join(artifact_uri, model_name)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    # Get the list of directories (iterations)
    iterations = sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[-1]), reverse=True)

    if not iterations:
        raise FileNotFoundError(f"No model iterations found in {model_dir}.")

    latest_iteration = iterations[0]
    latest_model_path = os.path.join(model_dir, latest_iteration, "data", "model.pth")
    
    return latest_model_path

def get_model_paths(mlflow_logger: MLflowLogger) -> Tuple[str, str]:
    """
    Get the paths to the latest saved forward and backward models from MLflow.

    Args:
        mlflow_logger (MLflowLogger): Instance of the MLflowLogger to handle logging.

    Returns:
        Tuple[str, str]: Paths to the forward and backward model files.
    """
    parent_dir = mlflow.get_artifact_uri()
    # parent_dir = os.path.dirname(artifact_uri)
    # Find here all directories with forward_model and backward_model
    latest_iter_fwd  = 0
    latest_iter_bwd = 0

    directories = os.listdir(parent_dir)

    for dir in directories:
        if "forward_model" in dir:
            if int(dir.split('_')[-1]) > latest_iter_fwd:
                latest_iter_fwd = int(dir.split('_')[-1])
        if "backward_model" in dir:
            if int(dir.split('_')[-1]) > latest_iter_bwd:
                latest_iter_bwd = int(dir.split('_')[-1])

    fwd_model_path = os.path.join(parent_dir, f"forward_model_iteration_{latest_iter_fwd}", "data", "model.pth")
    bwd_model_path = os.path.join(parent_dir, f"backward_model_iteration_{latest_iter_bwd}", "data", "model.pth")
    return fwd_model_path, bwd_model_path


def plot_and_log_metrics(losses, rewards, logZ, mlflow_logger):
    # Plot Losses
    plt.figure()
    plt.plot(losses, label='Loss')
    # Add rolling mean
    rolling_mean = pd.Series(losses).rolling(window=10).mean()
    plt.plot(rolling_mean, label='Rolling Mean Loss', color='red')
    rolling_mean_2 = pd.Series(losses).rolling(window=100).mean()
    plt.plot(rolling_mean_2, label='Rolling Mean Loss 100', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.grid(True)
    loss_plot_path = "loss_plot.png"
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot Log Losses
    log_losses = np.log1p(losses)  # Use log1p to avoid log(0)
    plt.figure()
    plt.plot(log_losses, label='Log Loss')
    # Add rolling mean
    rolling_mean_log = pd.Series(log_losses).rolling(window=10).mean()
    plt.plot(rolling_mean_log, label='Rolling Mean Log Loss', color='red')
    rolling_mean_log_2 = pd.Series(log_losses).rolling(window=100).mean()
    plt.plot(rolling_mean_log_2, label='Rolling Mean Log Loss 100', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Training Log Loss over Iterations')
    plt.legend()
    plt.grid(True)
    log_loss_plot_path = "log_loss_plot.png"
    plt.savefig(log_loss_plot_path)
    plt.close()


    plt.figure()
    plt.plot(rewards, label='Reward')
    # Add rolling mean
    rolling_mean = pd.Series(rewards).rolling(window=10).mean()
    plt.plot(rolling_mean, label='Rolling Mean Reward', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('Training Reward over Iterations')
    plt.legend()
    plt.grid(True)
    reward_plot_path = "reward_plot.png"
    plt.savefig(reward_plot_path)
    plt.close()

    # Log the plots as artifacts
    mlflow_logger.log_artifact(loss_plot_path)
    mlflow_logger.log_artifact(reward_plot_path)
    mlflow_logger.log_artifact(log_loss_plot_path)

    # Log logZ as a parameter
    mlflow_logger.log_params({"logZ": logZ.item()})

    # Delete the plots after uploading to MLflow
    os.remove(loss_plot_path)
    os.remove(reward_plot_path)
    os.remove(log_loss_plot_path)

def plot_distributions_per_feature(all_actions, n_features,feature_names, n_timesteps, mlflow_logger):
    """
    Plots a distribution matrix for each feature, where each matrix contains the 
    distribution of actions for that feature across timesteps.
    
    Args:
        all_actions: A 3D array of shape [num_trajectories, num_timesteps, num_features].
        n_features: The number of features (variables) being analyzed.
        n_timesteps: The number of timesteps.
        mlflow_logger: MLflow logger to log artifacts (distribution plots).
    """
    
    # Iterate over each feature
    for feature_idx in range(n_features):
        # Create a figure for the current feature
        fig, axs = plt.subplots(3, 4, figsize=(16, 12))  # Assuming 12 timesteps (3x4 grid)
        axs = axs.flatten()  # Flatten the 2D grid to iterate easily
        
        # Iterate over each timestep for this feature
        for timestep in range(n_timesteps):
            # Get the actions for the current feature at the current timestep
            actions_per_timestep = all_actions[:, timestep, feature_idx]
            
            # Plot the distribution of actions for this timestep
            sns.histplot(actions_per_timestep/100, kde=True, ax=axs[timestep], stat="density", bins=50)
            
            # Set labels and title for each subplot
            axs[timestep].set_title(f"Timestep {timestep + 1}", fontsize=12)
            axs[timestep].set_xlabel(f"Action Value", fontsize=10)
            axs[timestep].set_ylabel(f"Density", fontsize=10)
        
        # Adjust layout and set a super title for the current feature
        fig.suptitle(f"Distribution of Sampled Actions - Feature {feature_names[feature_idx+1]}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the title
        
        # Save the figure
        distribution_plot_path = f"distribution_sampled_actions_feature_per_ts_{feature_names[feature_idx+1]}.png"
        plt.savefig(distribution_plot_path)
        
        # Log the distribution plot to MLflow
        mlflow_logger.log_artifact(distribution_plot_path)
        
        # Optionally, remove the plot file after logging to avoid cluttering
        os.remove(distribution_plot_path)
        
        # Close the plot to free up memory
        plt.close()

from scipy.spatial.distance import pdist
def calculate_euclidean_diversity(final_states: np.ndarray, rewards):
    distances = pdist(final_states, metric='euclidean')
    avg_div = distances.mean()
    avg_div_normalized = (avg_div / (max(rewards)-min(rewards))) * max(rewards)
    # Convert to square form for better visualization
    return avg_div, avg_div_normalized,distances

def calculate_cosine_diversity(final_states: np.ndarray,rewards):
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(final_states)
    
    # Convert similarity to distance (diversity metric)
    cosine_distances = similarity_matrix
    
    # Calculate the average diversity (off-diagonal only)
    n_samples = final_states.shape[0]
    avg_cosine_distance = np.sum(cosine_distances) / (n_samples * (n_samples - 1))  # Exclude diagonal
    avg_cosine_diversity_normalized = (avg_cosine_distance / (max(rewards)-min(rewards)))*max(rewards)
    return avg_cosine_distance, avg_cosine_diversity_normalized,cosine_distances

def calculate_quality_diversity(states: np.ndarray, rewards: np.ndarray) -> dict:
    """
    Compute diversity metrics adjusted for reward quality.

    Args:
        states: Array of shape (N, D) representing final states or trajectory summaries.
        rewards: Array of shape (N,) of corresponding rewards.

    Returns:
        A dict with:
        - avg_div: mean pairwise Euclidean distance
        - norm_div: avg_div normalized by the maximum pairwise distance
        - avg_reward: mean reward
        - reward_norm: (avg_reward - min_reward) / (max_reward - min_reward)
        - quality_diversity: avg_div * reward_norm (high only when diversity and reward quality are high)
        - distances: raw pairwise distances
    """
    distances = pdist(states, metric='euclidean')
    avg_div = float(distances.mean())
    max_div = float(distances.max())
    
    avg_reward = float(np.mean(rewards))
    min_reward, max_reward = float(np.min(rewards)), float(np.max(rewards))
    if max_reward > min_reward:
        reward_norm = (avg_reward - min_reward) / (max_reward - min_reward)
    else:
        reward_norm = 0.0

    norm_div = avg_div / max_div if max_div > 0 else np.nan
    quality_diversity = avg_div * reward_norm

    return {
        "avg_div": avg_div,
        "norm_div": norm_div,
        "avg_reward": avg_reward,
        "reward_norm": reward_norm,
        "quality_diversity": quality_diversity,
        "distances": distances
    }

def visualize_and_log_simulated_trajectories(
    mlflow_logger: MLflowLogger,
    trajectory_length: int = 30,
    n_trajectories: int = 1000,
    plot_title: str = "Simulated Trajectories",
    logger = None,
    model_path = None,
    distribution_type = 'normal',
    extra_parameters = None,
    config = None
):
    """
    Simulates trajectories using the trained models, visualizes them, and logs the artifacts to MLflow.

    Args:
        mlflow_logger (MLflowLogger): Instance of the MLflowLogger to handle logging.
        trajectory_length (int, optional): Length of the trajectories to simulate. Default is 30.
        n_trajectories (int, optional): Number of trajectories to simulate. Default is 1000.
        plot_title (str, optional): Title of the plot. Default is "Simulated Trajectories".
    """
    # Get the paths to the latest forward and backward models
    fwd_model_path, bwd_model_path = get_model_paths(mlflow_logger)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    if config is None:
        raise ValueError("Config dictionary must be provided.")

    # Load the trained models
    forward_model = load_entire_model(fwd_model_path, device)
    backward_model = load_entire_model(bwd_model_path, device)

    top_50_cases_colors = sns.color_palette("husl", 50)
    initial_state,feature_names = load_initial_state(config)

    trajectories, rewards,all_actions,_ = simulate_trajectories(
        env_class=GeneralEnvironment,
        forward_model=forward_model,
        backward_model=backward_model,
        initial_state=initial_state,
        config=config,
        trajectory_length=trajectory_length,
        n_trajectories=n_trajectories,
        device=device,
        logger = logger,
        model_path = model_path,
        distribution=distribution_type,
        extra_parameters=extra_parameters

    )
    
    # indices of the 5 top rewards
    last_step_rewards = [reward[-1] for reward in rewards]
    top_50_rewards_indices = np.argsort(last_step_rewards)[-50:][::-1]
    feature_names = ['TimeStep'] + feature_names


    logger.info(f"####### Now Simulating Trajectories #######")
    logger.info(f"Trajectories Shape: {trajectories[0].shape}")
    logger.info(f"Rewards Shape: {rewards[0].shape}")

    # Stack trajectories and rewards for easier manipulation
    trajectories = np.stack(trajectories)  # Shape: (n_trajectories, trajectory_length, n_features)
    # Clamp rewards to avoid negative values
    rewards = np.maximum(rewards, 0)
    rewards = np.stack(rewards)  # Shape: (n_trajectories, trajectory_length)
    all_actions = np.stack(all_actions)  # Shape: (n_trajectories, trajectory_length, n_features)

    n_features = trajectories.shape[2]
    n_actions = all_actions.shape[2]
    # Plot each feature individually
    for feature_idx in range(n_features):
        plt.figure(figsize=(10, 5))
        for traj_idx, trajectory in enumerate(trajectories):
            if traj_idx in top_50_rewards_indices:  # If this trajectory is in the top 5 rewards
                color_idx = np.where(top_50_rewards_indices == traj_idx)[0][0]  # Get the color index for the top case
                # Mark with Stars the top 5 cases
                plt.plot(trajectory[:, feature_idx], alpha=1,
                        color=top_50_cases_colors[color_idx],linewidth=1, linestyle='dashed')
            else:
                plt.plot(trajectory[:, feature_idx], alpha=0.05, color='blue')  # Plot the non-top cases
        # Add mean and median
        mean_feature = np.mean(trajectories[:, :, feature_idx], axis=0)
        median_feature = np.median(trajectories[:, :, feature_idx], axis=0)
        plt.plot(mean_feature, color='black', linewidth=2, label='Mean Trajectory')
        plt.plot(median_feature, color='red', linewidth=2, label='Median Trajectory')
        plt.legend(prop={'size': 10}, loc='upper left')
        plt.xlabel("Time Step", fontdict={'fontsize': 10})
        plt.ylabel(f"Feature {feature_names[feature_idx]} Value", fontdict={'fontsize': 10})
        plt.title(f"{plot_title} - Feature {feature_names[feature_idx]}", fontdict={'fontsize': 10})
        plt.grid(True)

        # Save the feature plot to a file
        feature_plot_path = f"simulated_trajectories_feature_{feature_names[feature_idx]}.png"
        plt.savefig(feature_plot_path)
        plt.close()

        # Log the feature plot to MLflow
        mlflow_logger.log_artifact(feature_plot_path)

        # Optionally, remove the plot file after logging
        os.remove(feature_plot_path)

    n_timesteps = trajectories.shape[1]
    n_features = trajectories.shape[2]
    plot_distributions_per_feature(all_actions, n_features=n_features-1,feature_names=feature_names, n_timesteps=n_timesteps-1, mlflow_logger=mlflow_logger)

    for feature_idx in range(n_actions):
        # Generate distribution plot for the final actions of this feature
        final_actions = all_actions[:, :, feature_idx].flatten()
        plt.figure(figsize=(10, 5))
        sns.histplot(final_actions/100, kde=True, stat="density", bins=30)
        plt.xlabel(f"Action {feature_idx + 1} Value", fontdict={'fontsize': 14})
        plt.ylabel("Density", fontdict={'fontsize': 14})
        plt.title(f"Distribution of Sampled Actions - Feature {feature_names[feature_idx + 1]}", fontdict={'fontsize': 16})
        plt.grid(True)

        # Save the distribution plot to a file
        distribution_plot_path = f"distribution_sampled_actions_feature_{feature_names[feature_idx + 1]}.png"
        plt.savefig(distribution_plot_path)
        plt.close()

        # Log the distribution plot to MLflow
        mlflow_logger.log_artifact(distribution_plot_path)

        # Optionally, remove the plot file after logging
        os.remove(distribution_plot_path)

    # Calculate mean and median of rewards across trajectories at each time step
    # Add Reward = 0 for the initial time step
    rewards = np.concatenate([np.zeros((n_trajectories, 1)), rewards], axis=1)

    mean_rewards = np.mean(rewards, axis=0)
    median_rewards = np.median(rewards, axis=0)
    # Calculate Q1, Q3, and IQR for each time step
    q1_rewards = np.percentile(rewards, 25, axis=0)
    q3_rewards = np.percentile(rewards, 75, axis=0)
    iqr_rewards = q3_rewards - q1_rewards

    # Identify outliers (values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR)
    lower_bound = q1_rewards - 1.5 * iqr_rewards
    upper_bound = q3_rewards + 1.5 * iqr_rewards

    outliers = (rewards < lower_bound) | (rewards > upper_bound)

    # Plot the mean and median rewards over time
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, color='black', linewidth=2, label='Mean Reward')
    plt.plot(median_rewards, color='red', linewidth=2, label='Median Reward')

    # Plot IQR candles for each time step
    for t in range(rewards.shape[1]):
        plt.vlines(t, q1_rewards[t], q3_rewards[t], color='blue', lw=5, alpha=0.5)

    # Scatter plot for outliers
    for t in range(rewards.shape[1]):
        outlier_points = rewards[outliers[:, t], t]
        plt.scatter([t] * len(outlier_points), outlier_points, color='orange', edgecolor='black', zorder=3,alpha=0.25)

    plt.ylim(0, 1.1 * np.max(rewards))
    plt.xlabel("Time Step", fontdict={'fontsize': 14})
    plt.ylabel("Reward", fontdict={'fontsize': 14})
    plt.title("Mean and Median Rewards Over Time", fontdict={'fontsize': 16})
    plt.legend(prop={'size': 14})
    plt.grid(True)

    # Save the rewards plot to a file
    rewards_plot_path = "mean_median_rewards.png"
    plt.savefig(rewards_plot_path)
    plt.close()

    mlflow_logger.log_artifact(rewards_plot_path)
    # Optionally, remove the plot file after logging
    os.remove(rewards_plot_path)

    # Save rewards csv
    rewards_df = pd.DataFrame(rewards)
    rewards_df.to_csv("rewards.csv",index=False)
    # log the rewards csv
    mlflow_logger.log_artifact("rewards.csv")
    os.remove("rewards.csv")

    # Plot rewards_df
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_df.T, alpha=0.1, color='blue')
    plt.plot(mean_rewards, color='black', linewidth=2, label='Mean Reward')
    plt.plot(median_rewards, color='red', linewidth=2, label='Median Reward')
    plt.xlabel("Time Step", fontdict={'fontsize': 14})
    plt.ylabel("Reward", fontdict={'fontsize': 14})
    plt.title("Rewards Over Time", fontdict={'fontsize': 16})
    plt.legend(prop={'size': 14})
    plt.grid(True)

    # Save the rewards plot to a file
    rewards_plot_path = "rewards_over_time.png"
    plt.savefig(rewards_plot_path)
    plt.close()

    mlflow_logger.log_artifact(rewards_plot_path)
    # Optionally, remove the plot file after logging
    os.remove(rewards_plot_path)


    # Plot Rewards Distribution with KDE
    plt.figure(figsize=(10, 5))
    final_step_rewards = rewards[:, -1]  # Extract rewards at the final step
    sns.histplot(final_step_rewards, kde=True, stat="density", bins=50)
    plt.xlabel("Final Step Reward Value", fontdict={'fontsize': 14})
    plt.ylabel("Density", fontdict={'fontsize': 14})
    plt.title("Distribution of Final Step Rewards", fontdict={'fontsize': 16})
    plt.grid(True)

    # Save the rewards plot to a file
    rewards_plot_path = "rewards_distribution.png"
    plt.savefig(rewards_plot_path)
    plt.close()
    mlflow_logger.log_artifact(rewards_plot_path)
    # Optionally, remove the plot file after logging
    os.remove(rewards_plot_path)



    ############
    # 8. -------------------------------------------------
    # Build a table summarizing final states and rewards for top 10
    # ----------------------------------------------------
    summary_rows = []
    for rank, idx in enumerate(top_50_rewards_indices):
        row_dict = {
            "Rank": rank + 1,
            "CaseIndex": idx,
            "FinalReward": last_step_rewards[idx]
        }
        # Each feature’s final value
        final_values = trajectories[idx, -1, :]  # shape: (n_features,)
        for f_i, f_name in enumerate(feature_names):
            row_dict[f_name] = final_values[f_i]
        summary_rows.append(row_dict)

    top10_df = pd.DataFrame(summary_rows)
    top10_csv_path = "top10_cases.csv"
    top10_df.to_csv(top10_csv_path, index=False)
    mlflow_logger.log_artifact(top10_csv_path)
    os.remove(top10_csv_path)

    # If you want a quick styled HTML table:
    top10_html_path = "top10_cases.html"
    top10_df.style.background_gradient(cmap="Blues").to_html(top10_html_path)
    mlflow_logger.log_artifact(top10_html_path)
    os.remove(top10_html_path)
    finalrewards = top10_df['FinalReward']
    # Add similarity matrix for top 10
    avg_diversity, avg_div_normalized, distances = calculate_euclidean_diversity(top10_df[feature_names].drop(columns=['TimeStep']).values, finalrewards)
    # Convert distance vector to square distance matrix
    from scipy.spatial.distance import squareform
    distance_matrix = squareform(distances)
    # Upload it as a CARD plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(distance_matrix, cmap='viridis', fmt=".2f", cbar_kws={'label': 'Euclidean Distance'})
    plt.title(f"Euclidean Distance Matrix for Top 10 Trajectories Diversity Score: {avg_diversity}, Normalized: {avg_div_normalized:.4f}")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Trajectory Index")
    plt.tight_layout()
    euclidean_distance_plot_path = "euclidean_distance_matrix.png"
    plt.savefig(euclidean_distance_plot_path)
    mlflow_logger.log_artifact(euclidean_distance_plot_path)
    plt.close()

    
    ############
    # 8. Perform SHAP analysis on the forward model
    #    We'll pick some random subset of trajectories as "sample_data"
    #    For instance, the final states or partial states. This is up to you.
    #    Example: use the final state from the top 5 trajectories:
    sample_data_size = config['results']['n_simulations']  # or any suitable number
    sample_data_indices = np.random.choice(range(trajectories.shape[0]), size=sample_data_size, replace=False)
    # Suppose we want to explain the final states. shape: (sample_data_size, n_features)
    final_states = []
    for idx in sample_data_indices:
        final_states.append(trajectories[idx, -1, :])  # last time step
    sample_data = np.stack(final_states, axis=0)  # (sample_data_size, n_features)

    do_shap_analysis(
        model=forward_model,
        sample_data=sample_data,
        feature_names=feature_names,
        mlflow_logger=mlflow_logger,
        logger=logger,
        config=config
    )

    # Analysis for 10 top cases

    if logger:
        logger.info("Completed visualization, logging, and SHAP analysis.")


def do_shap_analysis(
    model: torch.nn.Module,
    sample_data: np.ndarray,
    feature_names: List[str],
    mlflow_logger: MLflowLogger,
    logger=None,
    config = None
) -> None:
    """
    Performs a SHAP analysis on the given model and sample data.
    Logs the SHAP summary plot to MLflow.

    Args:
        model (torch.nn.Module): The forward model to explain.
        sample_data (np.ndarray): Input data for computing SHAP values (shape: (num_samples, input_dim)).
        feature_names (List[str]): Names of each input dimension.
        mlflow_logger (MLflowLogger): For logging artifacts.
        logger: Python logger.
    """
    # Convert your PyTorch model to a SHAP-friendly callable
    # Typically, for a regression model: shap.Explainer(model, ...)
    # But for custom models, you might define a wrapper. Example approach:
    model.eval()
    model_cpu = copy.deepcopy(model).cpu()  # move to CPU for shap
    sample_data_torch = torch.tensor(sample_data, dtype=torch.float32)

    # Create a PyTorch Deep Explainer or Gradient Explainer (depending on model type)
    # E.g., for an MLP:
    explainer = shap.DeepExplainer(model_cpu, sample_data_torch[:config['results']['n_simulations']//10])  # reference baseline
    shap_values = explainer.shap_values(sample_data_torch,check_additivity=False)
    sample_data_np = sample_data_torch.numpy()
    shap_values = shap_values.mean(axis=-1)  

    # 4. Global Visualizations
    ## 4a. SHAP Summary (bar)
    shap_summary_plot(
        shap_values=shap_values,
        sample_data=sample_data_np,
        feature_names=feature_names,
        mlflow_logger=mlflow_logger,
        plot_name="shap_summary_bar"
    )

    ## 4b. SHAP Beeswarm (dot)
    shap_beeswarm_plot(
        shap_values=shap_values,
        sample_data=sample_data_np,
        feature_names=feature_names,
        mlflow_logger=mlflow_logger,
        plot_name="shap_beeswarm"
    )

    ## 4c. Dependence plots for all or a subset of features
    shap_dependence_plots(
        shap_values=shap_values,
        sample_data=sample_data_np,
        feature_names=feature_names,
        mlflow_logger=mlflow_logger,
        features_to_plot=None,            # e.g., plot them all
        interaction_index='auto'
    )
def shap_summary_plot(
    shap_values: np.ndarray,
    sample_data: np.ndarray,
    feature_names: list,
    mlflow_logger,
    plot_name: str = "shap_summary",
    cleanup: bool = True
):
    """
    Creates and logs a SHAP summary plot (bar/beeswarm style).
    
    Args:
        shap_values (np.ndarray): SHAP values array from the explainer.
        sample_data (np.ndarray): Original input data used for SHAP.
        feature_names (list): Names of the input features.
        mlflow_logger: Instance of MLflow logger for logging artifacts.
        plot_name (str): Base filename for the artifact.
        cleanup (bool): If True, remove file after logging to MLflow.
    """
    plt.figure()
    # The default shap.summary_plot with show=False for script usage
    shap.summary_plot(
        shap_values,
        sample_data,
        feature_names=feature_names,
        show=False,
        plot_type="bar"  # use "bar" or "dot" for beeswarm
    )
    summary_plot_path = f"{plot_name}.png"
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()

    mlflow_logger.log_artifact(summary_plot_path)
    if cleanup:
        os.remove(summary_plot_path)


def shap_beeswarm_plot(
    shap_values: np.ndarray,
    sample_data: np.ndarray,
    feature_names: list,
    mlflow_logger,
    plot_name: str = "shap_beeswarm",
    cleanup: bool = True
):
    """
    Creates and logs a SHAP beeswarm plot for global interpretability.
    """
    plt.figure()
    shap.summary_plot(
        shap_values,
        sample_data,
        feature_names=feature_names,
        show=False,
        plot_type="dot"  # "dot" => beeswarm
    )
    beeswarm_plot_path = f"{plot_name}.png"
    plt.savefig(beeswarm_plot_path, bbox_inches="tight")
    plt.close()

    mlflow_logger.log_artifact(beeswarm_plot_path)
    if cleanup:
        os.remove(beeswarm_plot_path)


def shap_dependence_plots(
    shap_values: np.ndarray,
    sample_data: np.ndarray,
    feature_names: list,
    mlflow_logger,
    features_to_plot: list = None,
    interaction_index: str = None,
    cleanup: bool = True
):
    """
    Creates and logs SHAP dependence plots for one or more features.
    These show how SHAP values for one feature vary with that feature's value.

    Args:
        shap_values (np.ndarray): SHAP values array.
        sample_data (np.ndarray): Input data (numpy array).
        feature_names (list): Names of each feature dimension.
        mlflow_logger: MLflow logger to log artifacts.
        features_to_plot (list): Which feature indices to plot. If None, plot all.
        interaction_index (str): Feature index for color-coding interactions. 
                                 e.g., pass 'auto' or an integer index.
        cleanup (bool): Remove the generated files after logging.
    """
    num_features = len(feature_names)
    if features_to_plot is None:
        features_to_plot = range(num_features)

    for feature_idx in features_to_plot:
        plt.figure()
        shap.dependence_plot(
            ind=feature_idx,
            shap_values=shap_values,
            features=sample_data,
            feature_names=feature_names,
            interaction_index=interaction_index,  # e.g. "auto" or a feature index
            show=False
        )
        dep_plot_path = f"shap_dependence_{feature_names[feature_idx]}.png"
        plt.savefig(dep_plot_path, bbox_inches="tight")
        plt.close()

        mlflow_logger.log_artifact(dep_plot_path)
        if cleanup:
            os.remove(dep_plot_path)


def shap_force_plot_individual(
    shap_values: np.ndarray,
    sample_data: np.ndarray,
    feature_names: list,
    mlflow_logger,
    sample_index: int,
    plot_name: str = None,
    cleanup: bool = True
):
    """
    Creates and logs a SHAP force plot for a single sample.
    
    Args:
        shap_values (np.ndarray): SHAP values for the sample batch. shape: (num_samples, num_features).
        sample_data (np.ndarray): The input data used for SHAP. shape: (num_samples, num_features).
        feature_names (list): Names of each input dimension.
        mlflow_logger: MLflow logger.
        sample_index (int): Index of the sample in the batch to visualize.
        plot_name (str): Base name of the output file.
        cleanup (bool): If True, remove file after logging.
    """
    # For force plots, use shap.plots.force or shap.force_plot
    # Note: shap.force_plot returns an HTML object by default, so we'll save it as HTML
    if plot_name is None:
        plot_name = f"shap_force_sample_{sample_index}"

    # We can create a matplotlib figure from the force plot using the following approach:
    force_plot_html = shap.force_plot(
        base_value=0,  # or the expected_value if your explainer provides one
        shap_values=shap_values[sample_index, :],
        features=sample_data[sample_index, :],
        feature_names=feature_names,
        matplotlib=False
    )

    # Save as HTML
    force_plot_path = f"{plot_name}.html"
    shap.save_html(force_plot_path, force_plot_html)

    mlflow_logger.log_artifact(force_plot_path)
    if cleanup:
        os.remove(force_plot_path)

def simulate_trajectories(
        env_class,
        forward_model: nn.Module,
        backward_model: nn.Module,
        initial_state: np.ndarray,
        config: dict,
        trajectory_length: int,
        n_trajectories: int,
        device: torch.device,
        logger = None,
        model_path = None,
        distribution: str = "normal",
        extra_parameters: dict = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
    """Simulate trajectories using the trained forward model."""
    trajectories = []
    rewards = []
    all_actions = []

    envs = [env_class(initial_state=initial_state, config=config, model=forward_model,
                       input_dim=initial_state.shape[0]+1, max_steps=trajectory_length,model_path=model_path) for _ in range(n_trajectories)]
    
    for env in envs:
        trajectory = []
        actions = []
        state = env.reset()
        # Log the env parameters
        logger.info(f"Initial State: {state}")
        logger.info(f"Initial State Shape: {state.shape}")
        logger.info(f"Initial State Type: {type(state)}")
        logger.info(f"Done Flag: {env.done}")

        trajectory.append(state.copy())

        for _ in range(1,trajectory_length+1):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            policy_dist, _ = get_policy_dist(forward_model, state_tensor,logger=logger,
                                             dist_type=distribution,
                                             extra_params=extra_parameters)
            action = policy_dist.sample().cpu().detach().numpy().squeeze()
            state, done = env.step(action)
            trajectory.append(state.copy())
            actions.append(action.copy())
            if done:
                break
        trajectories.append(np.array(trajectory))
        all_actions.append(np.array(actions))

        final_features,feature_names = env.generate_full_features(trajectory)
        if model_path:
            raw_rewards,f_names = env.calculate_final_reward_ml(final_features)
        else:
            raw_rewards,f_names = env.calculate_final_reward(final_features)

        # softplus_reward = F.softplus(torch.tensor(reward, dtype=torch.float32, device=device))
        raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        log_rewards = torch.log(1 + raw_rewards.clamp(min=0))
        rewards.append(raw_rewards.cpu().numpy())
        # rewards.append(reward.sum().item())
        # final_features.to_csv("final_features.csv",index=False)
    
    return trajectories, rewards, all_actions,f_names



class StudentT(Distribution):
    arg_constraints = {
        'df': constraints.positive,
        'loc': constraints.real,
        'scale': constraints.positive
    }
    support = constraints.real
    has_rsample = False  # not implementing reparameterized samples

    def __init__(
        self,
        df: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool = None
    ):
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df, dtype=loc.dtype, device=loc.device)
        self.df = df
        self.loc = loc
        self.scale = scale

        # Let PyTorch figure out the broadcasted batch shape
        broadcast_shape = torch.broadcast_shapes(self.df.shape, self.loc.shape, self.scale.shape)
        super().__init__(batch_shape=broadcast_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(StudentT, new).__init__(batch_shape=batch_shape, validate_args=False)
        return new

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        log_prob(x) = lgamma((ν+1)/2) - lgamma(ν/2)
                      - 0.5*(log(ν) + log(pi)) - log(scale)
                      - ((ν+1)/2)*log(1 + ((x - loc)^2 / (ν*scale^2)))
        """
        df = self.df
        loc = self.loc
        scale = self.scale
        x = value

        half_nu = df / 2
        half_nu_plus_one = (df + 1) / 2

        lgamma_half_nu_plus_one = torch.lgamma(half_nu_plus_one)
        lgamma_half_nu = torch.lgamma(half_nu)

        # Keep everything in Torch space (avoid mixing math.log(...) with Torch expressions in one line)
        log_factor = lgamma_half_nu_plus_one - lgamma_half_nu
        log_factor = log_factor - 0.5 * torch.log(df)
        log_factor = log_factor - 0.5 * torch.log(torch.tensor(math.pi, device=scale.device))
        log_factor = log_factor - torch.log(scale)

        z = ((x - loc) / scale) ** 2
        log_quad = -half_nu_plus_one * torch.log1p(z / df)

        return log_factor + log_quad

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        normal = torch.distributions.Normal(
            torch.zeros(shape, device=self.loc.device),
            torch.ones(shape, device=self.loc.device)
        )
        X = normal.sample()

        gamma = torch.distributions.Gamma(self.df / 2, torch.tensor(2.0, device=self.loc.device))
        Y = gamma.sample(shape)

        # Let PyTorch broadcast automatically; no manual expand
        T = self.loc + self.scale * X * torch.sqrt(self.df / Y)
        return T

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if not hasattr(torch.special, 'stdtr'):
            raise NotImplementedError("CDF requires torch.special.stdtr (PyTorch >= 1.10).")
        standardized_value = (value - self.loc) / self.scale
        return torch.special.stdtr(self.df, standardized_value)

    @property
    def mean(self) -> torch.Tensor:
        return torch.where(
            self.df > 1,
            self.loc,
            torch.tensor(float('nan'), device=self.loc.device)
        )

    @property
    def variance(self) -> torch.Tensor:
        return torch.where(
            self.df > 2,
            self.scale ** 2 * (self.df / (self.df - 2)),
            torch.tensor(float('nan'), device=self.loc.device)
        )

    
import torch
import torch.distributions as dist


from torch.distributions import TransformedDistribution, Beta, AffineTransform, Independent

def get_policy_dist(
    model: torch.nn.Module,
    state: torch.Tensor,
    off_policy_noise: float = 0.0,
    logger: Optional[Any] = None,
    dist_type: str = "normal",
    extra_params: Optional[Dict[str, Any]] = None
) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution]:
    """
    Returns policy and exploration distributions based on the specified distribution type.
    
    Args:
        model: A torch.nn.Module that takes `state` as input and outputs parameters of the chosen distribution.
        state: Input state tensor of shape [batch_size, input_dim].
        off_policy_noise: Magnitude of optional off-policy noise for exploration.
        logger: Optional logger for debugging.
        dist_type: Type of distribution to use. Supported: 'normal', 'lognormal', 'beta', 'uniform'.
                   'studentT' is included as a template but not fully implemented.
        extra_params: Additional parameters required by certain distributions (e.g., degrees_of_freedom for Student-T).
    
    Returns:
        policy_dist: The main policy distribution.
        exploration_dist: The exploration distribution (may differ if off_policy_noise > 0).
    """

    # Extract raw parameters from the model
    # print(f"State Shape {state.shape}")
    raw_params = model(state)
    if logger:
        logger.debug(f"Raw model output for distribution parameters: {raw_params}")

    # Choose distribution type
    if dist_type == 'normal':
        # raw_params: mean and std_dev concatenated
        half = raw_params.shape[-1] // 2
        mean = raw_params[..., :half]
        std_dev = raw_params[..., half:]
        max_policy_std = 10 # used to be 10
        min_policy_std = 3 # used to be 3
        std_dev = torch.sigmoid(std_dev) * (max_policy_std - min_policy_std) + min_policy_std
        max_abs_policy_mean = 15.0 # used to be 15 
        mean = torch.tanh(mean) * max_abs_policy_mean
        base_dist = torch.distributions.Normal(mean, std_dev)
        # Make it a multivariate distribution (assuming each dimension independent)
        policy_dist = torch.distributions.Independent(base_dist, 1)

        if off_policy_noise > 0:
            noise_dist = torch.distributions.Normal(0, off_policy_noise)
            off_policy_noise_sample = noise_dist.rsample(mean.shape)
            exploration_mean = mean + off_policy_noise_sample
            exploration_base_dist = torch.distributions.Normal(exploration_mean, std_dev)
            exploration_dist = torch.distributions.Independent(exploration_base_dist, 1)
        else:
            exploration_dist = policy_dist


    elif dist_type == 'mixture_gaussian':
        num_components = extra_params.get('mixture_components', 2)
        num_vars = extra_params.get('num_variables', 1)
        temperature = extra_params.get('temperature', 1.0)

        # Parameters for each variable: weights, means, stds for each component
        params_per_var = 3 * num_components
        total_params = params_per_var * num_vars

        if raw_params.shape[-1] != total_params:
            raise ValueError(f"Expected {total_params} parameters but got {raw_params.shape[-1]}")

        # Create a normal distribution for each variable
        distributions = []
        for var_idx in range(num_vars):
            start_idx = var_idx * params_per_var
            
            # Extract and process parameters
            weights_logits = raw_params[..., start_idx:start_idx + num_components]
            means_raw = raw_params[..., start_idx + num_components:start_idx + 2*num_components]
            stds_raw = raw_params[..., start_idx + 2*num_components:start_idx + 3*num_components]

            # Process parameters
            weights = F.gumbel_softmax(weights_logits, tau=temperature, hard=False)
            means = torch.tanh(means_raw) * 15.0  # max_abs_policy_mean
            stds = torch.sigmoid(stds_raw) * 7 + 3  # scale to [3, 10]

            # Create the mixed distribution for this variable
            dist = torch.distributions.Normal(
                loc=(weights * means).sum(dim=-1),
                scale=(weights * stds).sum(dim=-1)
            )
            distributions.append(dist)

        # Combine into a multivariate distribution matching expected shape
        # Reshape the locations and scales to match the expected dimensions
        loc = torch.stack([d.loc for d in distributions], dim=-1)  # Shape: [..., num_vars]
        scale = torch.stack([d.scale for d in distributions], dim=-1)  # Shape: [..., num_vars]
        
        policy_dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=loc, scale=scale),
            1  # reduce by 0 dimensions since we already have the correct shape
        )
        exploration_dist = policy_dist

    elif dist_type == 'mixture_beta':
        num_components = extra_params.get('mixture_components', 2)
        num_vars = extra_params.get('num_variables', 1)
        temperature = extra_params.get('temperature', 1.0)
        
        # Parameters for each variable: weights, alpha, beta for each component
        params_per_var = 3 * num_components
        total_params = params_per_var * num_vars

        if raw_params.shape[-1] != total_params:
            raise ValueError(f"Expected {total_params} parameters but got {raw_params.shape[-1]}")

        # Create a beta distribution for each variable
        distributions = []
        for var_idx in range(num_vars):
            start_idx = var_idx * params_per_var
            
            # Extract and process parameters
            weights_logits = raw_params[..., start_idx:start_idx + num_components]
            alpha_raw = raw_params[..., start_idx + num_components:start_idx + 2*num_components]
            beta_raw = raw_params[..., start_idx + 2*num_components:start_idx + 3*num_components]

            # Process parameters
            weights = F.gumbel_softmax(weights_logits, tau=temperature, hard=False)
            alpha = F.softplus(alpha_raw) + 1.0  # ensure alpha > 0
            beta = F.softplus(beta_raw) + 1.0  # ensure beta > 0

            # Create the mixed beta distribution for this variable
            # Calculate weighted parameters (mixture of betas)
            mixed_alpha = (weights * alpha).sum(dim=-1)
            mixed_beta = (weights * beta).sum(dim=-1)
            
            # Create beta distribution with the mixed parameters
            dist = torch.distributions.Beta(
                concentration1=mixed_alpha,
                concentration0=mixed_beta
            )
            
            distributions.append(dist)

        # Combine into a multivariate distribution
        alpha = torch.stack([d.concentration1 for d in distributions], dim=-1)
        beta = torch.stack([d.concentration0 for d in distributions], dim=-1)
        
        # Create a multivariate Beta distribution
        base_dist = torch.distributions.Beta(alpha, beta)
        # Transform from (0,1) to desired action range (-30, 30)
        transform = AffineTransform(loc=-30.0, scale=60.0)
        # transform = AffineTransform(loc=0,scale=20)
        transformed_dist = TransformedDistribution(base_dist, [transform])
        
        policy_dist = torch.distributions.Independent(transformed_dist, 1)
        exploration_dist = policy_dist

    elif dist_type == 'lognormal':
        # raw_params: underlying mean and std for lognormal
        half = raw_params.shape[-1] // 2
        underlying_mean = raw_params[..., :half]
        underlying_std = raw_params[..., half:]
        # Ensure underlying_std is positive
        underlying_std = F.softplus(underlying_std) + 1
        base_dist = torch.distributions.LogNormal(underlying_mean, underlying_std)
        policy_dist = torch.distributions.Independent(base_dist, 1)
        exploration_dist = policy_dist  # Off-policy noise not implemented for lognormal

    elif dist_type == 'beta_soft':
        # raw_params: alpha and beta
        half = raw_params.shape[-1] // 2
        alpha_raw = raw_params[..., :half]
        beta_raw = raw_params[..., half:]

        # Ensure alpha, beta > 0
        alpha = F.softplus(alpha_raw) + 1e-4
        beta = F.softplus(beta_raw) + 1e-4

        base_beta = torch.distributions.Beta(alpha, beta)
        
        # First wrap in Independent to handle the multivariate nature
        independent_beta = torch.distributions.Independent(base_beta, 1)
        
        # Then transform from (0, 1) to (-30, 30)
        transform = AffineTransform(loc=-30.0, scale=60.0)
        policy_dist = TransformedDistribution(independent_beta, [transform])
        exploration_dist = policy_dist


    elif dist_type == 'beta_sig':
        num_vars = extra_params.get("num_variables", 1)

        if raw_params.shape[-1] != 2 * num_vars:
            raise ValueError(f"Expected {2 * num_vars} parameters but got {raw_params.shape[-1]}")

        half = raw_params.shape[-1] // 2
        alpha_raw = raw_params[..., :half]
        beta_raw = raw_params[..., half:]

        # Stretch values into (1, 10) using sigmoid
        alpha = torch.sigmoid(alpha_raw) * 99.0 + 1.0
        beta = torch.sigmoid(beta_raw) * 99.0 + 1.0

        base_dist = torch.distributions.Beta(alpha, beta)
        policy_dist = torch.distributions.Independent(base_dist, 1)
        exploration_dist = policy_dist  # Could add noise by perturbing alpha/beta if desired


    elif dist_type == 'uniform':
        # raw_params: low_raw, high_raw
        half = raw_params.shape[-1] // 2
        low_raw = raw_params[..., :half]
        high_raw = raw_params[..., half:]
        # Ensure low < high by sorting
        low = torch.min(low_raw, high_raw)
        high = torch.max(low_raw, high_raw)
        base_dist = torch.distributions.Uniform(low, high)
        policy_dist = torch.distributions.Independent(base_dist, 1)
        exploration_dist = policy_dist  # Off-policy noise not implemented for uniform

    elif dist_type == 'cauchy':
        half = raw_params.shape[-1] // 2
        x0 = raw_params[..., :half]
        gamma = F.softplus(raw_params[..., half:]) + 1e-4  # ensure gamma > 0
        base_dist = torch.distributions.Cauchy(x0, gamma)
        policy_dist = torch.distributions.Independent(base_dist, 1)
        # Off-policy noise for Cauchy is not implemented, but you could do something similar to normal.
        exploration_dist = policy_dist

    elif dist_type == 'studentT':
        if extra_params is None or 'df' not in extra_params:
            raise ValueError("For StudentT distribution, 'df' must be provided in extra_params.")
        df = extra_params['df']  # degrees of freedom tensor
        # raw_params: mean and scale
        half = raw_params.shape[-1] // 2
        loc = raw_params[..., :half]
        scale = F.softplus(raw_params[..., half:]) + 1e-4
        df = torch.tensor(df, dtype=loc.dtype, device=loc.device)
        base_dist = StudentT(df, loc, scale)
        # Not strictly required to wrap StudentT with Independent if it's already multi-dimensional in a single dimension,
        # but if you have multiple action dims, do Independent:
        policy_dist = torch.distributions.Independent(base_dist, 1)
        exploration_dist = policy_dist

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    if logger:
        # logger.debug(f"Created policy distribution: {policy_dist}")
        # logger.debug(f"Created exploration distribution: {exploration_dist}")
        pass

    return policy_dist, exploration_dist
