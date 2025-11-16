import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

def compare_reward_distributions(results_dict, title="Comparison of Reward Distributions"):
    """
    Compare reward distributions across different methods.
    
    Args:
        results_dict: Dictionary mapping method names to arrays of final rewards
    """
    plt.figure(figsize=(12, 8))
    
    # Create violin plots
    positions = range(len(results_dict))
    violins = plt.violinplot(
        [results_dict[method] for method in results_dict.keys()],
        positions=positions,
        vert=True,
        widths=0.8,
        showmeans=True,
        showextrema=True
    )
    
    # Customize violin colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add box plots for additional statistics
    bp = plt.boxplot(
        [results_dict[method] for method in results_dict.keys()],
        positions=positions,
        vert=True,
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )
    
    # Customize box plots
    for i, box in enumerate(bp['boxes']):
        box.set(color='black', linewidth=1.5)
        box.set(facecolor='white', alpha=0.7)
    
    # Add scatter points for top-10 values
    for i, method in enumerate(results_dict.keys()):
        rewards = np.sort(results_dict[method])[-10:]  # Top 10 values
        plt.scatter([i] * len(rewards), rewards, color='red', s=30, zorder=3, alpha=0.7)
    
    # Add statistics
    for i, method in enumerate(results_dict.keys()):
        rewards = results_dict[method]
        plt.annotate(f"Mean: {np.mean(rewards):.2f}\nMax: {np.max(rewards):.2f}", 
                     xy=(i, np.min(rewards)), xytext=(i-0.4, np.min(rewards)-0.5),
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    # Configure plot
    plt.xticks(positions, list(results_dict.keys()))
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Final Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()

def visualize_state_space_exploration(trajectories_dict, feature_names, method_colors=None):
    """
    Visualize how different methods explore the state space using PCA.
    
    Args:
        trajectories_dict: Dictionary mapping method names to lists of trajectories
        feature_names: List of feature names
    """
    plt.figure(figsize=(15, 12))
    
    # Create PCA projection of state space
    all_states = np.vstack([traj.reshape(-1, traj.shape[-1]) for method in trajectories_dict.keys() 
                           for traj in trajectories_dict[method]])
    
    pca = PCA(n_components=2)
    pca_states = pca.fit_transform(all_states)
    
    # Get variance explained
    var_explained = pca.explained_variance_ratio_
    
    # Split PCA states by method
    pca_by_method = {}
    start_idx = 0
    
    for method, trajs in trajectories_dict.items():
        num_states = sum(traj.shape[0] * traj.shape[1] for traj in trajs)
        end_idx = start_idx + num_states
        pca_by_method[method] = pca_states[start_idx:end_idx]
        start_idx = end_idx
    
    # Plot results with KDE background
    plt.figure(figsize=(15, 12))
    
    # Create color scheme
    if method_colors is None:
        method_colors = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories_dict)))
        for i, method in enumerate(trajectories_dict.keys()):
            method_colors[method] = colors[i]
    
    # Plot KDE for each method
    for method, pca_data in pca_by_method.items():
        sns.kdeplot(
            x=pca_data[:, 0], y=pca_data[:, 1],
            cmap=LinearSegmentedColormap.from_list('custom', ['white', method_colors[method]]),
            fill=True, alpha=0.4, levels=5
        )
    
    # Plot final states for each method
    for method, trajs in trajectories_dict.items():
        final_states = np.array([traj[-1] for traj in trajs])
        final_pca = pca.transform(final_states)
        plt.scatter(
            final_pca[:, 0], final_pca[:, 1],
            label=f"{method} Final States",
            color=method_colors[method],
            edgecolor='black',
            s=80,
            alpha=0.9
        )
    
    # Plot component loadings
    loadings = pca.components_.T
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='black', alpha=0.5)
        plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature, color='black', fontsize=10)
    
    plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1%} variance)", fontsize=12)
    plt.title("State Space Exploration by Different Methods", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    return plt.gcf()

def feature_evolution_comparison(trajectories_dict, feature_idx, feature_name, timesteps=None):
    """
    Compare how a specific feature evolves over time across different methods.
    
    Args:
        trajectories_dict: Dictionary mapping method names to lists of trajectories
        feature_idx: Index of the feature to analyze
        feature_name: Name of the feature
        timesteps: List of timesteps to include (None = all)
    """
    plt.figure(figsize=(12, 8))
    
    # Create color scheme
    methods = list(trajectories_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        trajectories = trajectories_dict[method]
        
        # Extract feature values at each timestep
        max_length = max(traj.shape[0] for traj in trajectories)
        if timesteps is None:
            timesteps = list(range(max_length))
        
        # Calculate statistics per timestep
        means = []
        p25 = []
        p75 = []
        
        for t in timesteps:
            values = [traj[t, feature_idx] for traj in trajectories if t < traj.shape[0]]
            means.append(np.mean(values))
            p25.append(np.percentile(values, 25))
            p75.append(np.percentile(values, 75))
        
        # Plot mean with confidence band
        plt.plot(timesteps, means, label=method, color=colors[i], linewidth=2)
        plt.fill_between(timesteps, p25, p75, color=colors[i], alpha=0.2)
    
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel(f"{feature_name} Value", fontsize=12)
    plt.title(f"Evolution of {feature_name} Across Methods", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def performance_radar_chart(methods_data, metrics):
    """
    Create radar chart comparing methods across different metrics.
    
    Args:
        methods_data: Dictionary {method_name: {metric_name: value}}
        metrics: List of metric names to include
    """
    num_metrics = len(metrics)
    methods = list(methods_data.keys())
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Normalize data for radar chart (0-1 scale)
    norm_data = {}
    for metric in metrics:
        all_values = [methods_data[method][metric] for method in methods]
        min_val = min(all_values)
        max_val = max(all_values)
        # Avoid division by zero
        if max_val == min_val:
            norm_data[metric] = {method: 0.5 for method in methods}
        else:
            norm_data[metric] = {
                method: (methods_data[method][metric] - min_val) / (max_val - min_val)
                for method in methods
            }
    
    # Plot each method
    for i, method in enumerate(methods):
        values = [norm_data[metric][method] for metric in metrics]
        values += values[:1]  # Close the polygon
        
        color = plt.cm.tab10(i/len(methods))
        ax.plot(angles, values, color=color, linewidth=2, label=method)
        ax.fill(angles, values, color=color, alpha=0.25)
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add performance value annotations
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            angle = angles[i]
            radius = norm_data[metric][method]
            ha = 'left' if -0.5 <= np.sin(angle) <= 0.5 else 'right'
            va = 'bottom' if 0 <= np.cos(angle) <= 1 else 'top'
            
            # Add slight offset based on method index
            offset_angle = angle + (j - len(methods)/2) * 0.05
            x = (radius + 0.1) * np.cos(offset_angle)
            y = (radius + 0.1) * np.sin(offset_angle)
            
            ax.annotate(f"{methods_data[method][metric]:.2f}", 
                       (x, y), color=plt.cm.tab10(j/len(methods)))
    
    ax.set_title("Method Performance Comparison", size=15, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return fig

def plot_ppo_implementation():
    """
    Implementation of PPO algorithm (simplified version) for paper comparison
    """
    # Code not provided since you asked for BO first, but can implement PPO if needed
    pass