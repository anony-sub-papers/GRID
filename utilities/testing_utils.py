import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import seaborn as sns

def generate_high_reward_trajectories(env, trained_agent, num_trajectories=100, max_steps=10):
    """
    Generate trajectories using the trained agent and store only states.

    Args:
        env (GeneralEnvironment): The environment.
        trained_agent (ReinforceAgent): The trained REINFORCE agent.
        num_trajectories (int): Number of trajectories to generate.
        max_steps (int): Maximum steps per trajectory.

    Returns:
        List[np.ndarray]: List of state trajectories
        List[float]: List of rewards for each trajectory
        List[np.ndarray]: List of action sequences for each trajectory
    """
    all_trajectories = []
    rewards = []
    all_actions = []

    for _ in range(num_trajectories):
        state = env.reset()
        trajectory = [state.copy()]  # Start with initial state
        actions = []

        for step in range(max_steps):
            # Support agents that return (action, log_prob) or just action
            action_result = trained_agent.select_action(state)
            if isinstance(action_result, tuple):
                action = action_result[0]
            else:
                action = action_result
            
            next_state, done = env.step(action)
            trajectory.append(next_state.copy())  # Store only states
            actions.append(action.copy())
            state = next_state

            if done:
                break

        # Compute final reward for the trajectory
        trajectory = np.array(trajectory)
        final_features, feature_names = env.generate_full_features(trajectory)
        final_reward, f_names = env.calculate_final_reward_ml(final_features.iloc[-1:])

        rewards.append(final_reward)
        all_actions.append(np.array(actions))
        all_trajectories.append(trajectory)

    # Sort trajectories based on rewards
    rewards = np.array(rewards)
    # Squeeze
    if rewards.ndim > 1:
        rewards = rewards.squeeze()
    print(f"Rewards: {rewards.shape}")
    sorted_indices = np.argsort(rewards)[::-1].tolist()
    sorted_trajectories = [all_trajectories[i] for i in sorted_indices]
    sorted_rewards = [rewards[i] for i in sorted_indices]
    sorted_actions = [all_actions[i] for i in sorted_indices]

    return np.array(sorted_trajectories, dtype=object), np.array(sorted_rewards), np.array(sorted_actions, dtype=object), f_names

def plot_cosine_similarity_heatmap(cosine_distances, labels=None,output_path=None):
    """
    Plot a heatmap of pairwise cosine distances.

    Args:
        cosine_distances (np.ndarray): Pairwise cosine distance matrix.
        labels (List[str]): Optional labels for rows/columns.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_distances, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.title("Cosine Distance Heatmap")
    plt.xlabel("State Index")
    plt.ylabel("State Index")
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def plot_trajectory(trajectory, feature_names):
    """
    Plot the state and action evolution of a trajectory.

    Args:
        trajectory (List[Tuple]): A list of (state, action) pairs.
        feature_names (List[str]): Names of the state features.
    """
    states = np.array([step[0] for step in trajectory])  # Extract states
    actions = np.array([step[1] for step in trajectory]) / 100  # Scale actions as percentages

    # Debug state and action shapes
    print(f"States Shape: {states.shape}, Actions Shape: {actions.shape}")

    # Plot states
    plt.figure(figsize=(10, 5))
    for i in range(1, states.shape[1]):  # Skip time-step (index 0)
        plt.plot(range(len(states)), states[:, i], label=f"State: {feature_names[i-1]}")
    plt.title("State Evolution Over Trajectory")
    plt.xlabel("Step")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot actions
    plt.figure(figsize=(10, 5))
    for i in range(actions.shape[1]):
        plt.plot(range(len(actions)), actions[:, i], label=f"Action {i+1}")
    plt.title("Action Evolution Over Trajectory")
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def create_top10_summary_table(high_reward_trajectories, feature_names):
    """
    Create and style a summary table for the top 10 high-reward trajectories.

    Args:
        high_reward_trajectories (List[Dict]): List of dictionaries with high-reward trajectories.
        feature_names (List[str]): Names of the state features.

    Returns:
        pd.DataFrame: DataFrame containing the summary table.
    """
    summary_rows = []

    for rank, traj_data in enumerate(high_reward_trajectories):
        trajectory = traj_data["trajectory"]
        final_state = trajectory[-1][0]  # Last state in the trajectory
        final_reward = traj_data["final_reward"]

        row_dict = {
            "Rank": rank + 1,
            "FinalReward": final_reward
        }

        # Add each feature's final value
        for f_i, f_name in enumerate(feature_names):
            row_dict[f_name] = final_state[f_i]

        summary_rows.append(row_dict)

    # Create a DataFrame
    top10_df = pd.DataFrame(summary_rows)

    # Save to CSV and HTML for logging/styling purposes
    top10_csv_path = "top10_cases.csv"
    top10_df.to_csv(top10_csv_path, index=False)

    # Style the table with a blue gradient
    top10_html_path = "top10_cases.html"
    styled_df = top10_df.style.background_gradient(cmap="Blues")
    styled_df.to_html(top10_html_path)

    return top10_df, top10_csv_path, top10_html_path


def calculate_cosine_diversity(final_states: np.ndarray,rewards):
    """
    Calculate diversity using cosine similarity among final states.

    Args:
        final_states (np.ndarray): Array of final states (shape: [n_samples, n_features]).

    Returns:
        float: Average cosine distance (diversity score).
        np.ndarray: Pairwise cosine similarity matrix.
    """
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(final_states)
    
    # Convert similarity to distance (diversity metric)
    cosine_distances = similarity_matrix
    
    # Calculate the average diversity (off-diagonal only)
    n_samples = final_states.shape[0]
    avg_cosine_distance = np.sum(cosine_distances) / (n_samples * (n_samples - 1))  # Exclude diagonal
    avg_cosine_diversity_normalized = (avg_cosine_distance / (max(rewards)-min(rewards)))*max(rewards)
    return avg_cosine_distance, avg_cosine_diversity_normalized,cosine_distances

from scipy.spatial.distance import pdist, squareform

def calculate_euclidean_diversity(final_states: np.ndarray, rewards):
    distances = pdist(final_states, metric='euclidean')
    avg_div = distances.mean()
    avg_div_normalized = (avg_div / (max(rewards)-min(rewards))) * max(rewards)
    # Convert to square form for better visualization
    return avg_div, avg_div_normalized,distances

def plot_mcmc_rewards_distribution(rewards_df, figsize=(10, 5)):
    """
    Plot the distribution of rewards over time steps from MCMC results.
    
    Parameters:
    -----------
    rewards_df : pandas.DataFrame
        DataFrame containing rewards for each trajectory and timestep
    figsize : tuple, optional
        Figure size (width, height) in inches, default is (10, 5)
    """
    plt.figure(figsize=figsize)
    
    # Plot individual trajectories with low opacity
    plt.plot(rewards_df.T, alpha=0.1, color='blue')
    
    # Calculate and plot mean and median
    mean_rewards = rewards_df.mean(axis=0)
    median_rewards = rewards_df.median(axis=0)
    
    # Plot statistics
    plt.plot(mean_rewards, color='black', linewidth=2, label='Mean Reward')
    plt.plot(median_rewards, color='red', linewidth=2, label='Median Reward')
    
    # Customize plot
    plt.xlabel("Time Step", fontdict={'fontsize': 14})
    plt.ylabel("Reward", fontdict={'fontsize': 14})
    plt.title("Rewards Over Time", fontdict={'fontsize': 16})
    plt.legend(prop={'size': 14})
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()