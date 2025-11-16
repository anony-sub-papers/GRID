#!/usr/bin/env python3
"""
Evaluation script for pre-trained GFlowNet models.

This script allows loading trained GFlowNet models to generate trajectories,
analyze rewards, and create visualizations for research papers/theses.
"""

import argparse
import copy
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Import project modules
from environments import GeneralEnvironment
from mlflow_logger import MLflowLogger
from utility_functions import (
    calculate_cosine_diversity, get_policy_dist, load_initial_state,
    load_entire_model, plot_distributions_per_feature,
    setup_logger, simulate_trajectories
)


class GFlowNetEvaluator:
    """Class for evaluating a trained GFlowNet model."""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        output_dir: str = "evaluation_results",
        logger: Optional[logging.Logger] = None,
        device: str = "cpu",
    ):
        """
        Initialize the evaluator with a trained model and configuration.
        
        Args:
            config_path: Path to the YAML configuration file.
            model_path: Path to the trained model directory.
            output_dir: Directory to save evaluation results.
            logger: Optional logger for messages.
            device: Computation device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        
        # Setup logger if not provided
        if logger is None:
            self.logger, self.log_file = setup_logger("gflownet_eval")
        else:
            self.logger = logger
            self.log_file = None
        
        # Load configuration
        self.logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.model_path = model_path
        self.forward_model = self._load_model("forward")
        self.backward_model = self._load_model("backward")
        
        # Extract config parameters
        self.initial_state, self.feature_names = load_initial_state(self.config)
        self.input_dim = len(self.initial_state) + 1  # +1 for time dimension

        # Get distribution info
        training_params = self.config.get("model", {}).get("training_parameters", {})
        self.distribution_type = training_params.get("distribution_type", "normal")
        self.is_mixture = training_params.get("mixture", False)
        self.mixture_components = training_params.get("mixture_components", 2) if self.is_mixture else None
    
    def _load_model(self, model_type: str = "forward") -> torch.nn.Module:
        """
        Load a trained model from file.
        
        Args:
            model_type: Type of model to load ('forward' or 'backward').
            
        Returns:
            Loaded PyTorch model.
        """
        # Check if path is a directory (from MLflow) or direct path to model file
        if os.path.isdir(self.model_path):
            # Try to find the most recent model of requested type in the directory
            prefix = f"{model_type}_model_iteration_"
            
            # Find all subdirectories matching the pattern
            candidates = [d for d in os.listdir(self.model_path) if d.startswith(prefix)]
            
            if not candidates:
                raise FileNotFoundError(f"No {model_type} model found in {self.model_path}")
            
            # Sort by iteration number
            candidates.sort(key=lambda x: int(x.replace(prefix, "")), reverse=True)
            model_file = os.path.join(self.model_path, candidates[0], "data", "model.pth")
            
            self.logger.info(f"Loading {model_type} model from {model_file}")
            
        else:
            # Direct path to model file
            model_file = self.model_path
            self.logger.info(f"Loading model from {model_file}")
        
        # Load the model
        try:
            model = torch.load(model_file, map_location=self.device)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_trajectories(
        self,
        n_trajectories: int = 1000,
        trajectory_length: int = 30,
        save_trajectories: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectories using the trained forward model.
        
        Args:
            n_trajectories: Number of trajectories to generate.
            trajectory_length: Number of steps in each trajectory.
            save_trajectories: Whether to save trajectories to disk.
            
        Returns:
            Tuple containing:
            - trajectories: shape (n_trajectories, trajectory_length+1, state_dim)
            - rewards: shape (n_trajectories, trajectory_length)
            - actions: shape (n_trajectories, trajectory_length, action_dim)
        """
        self.logger.info(f"Generating {n_trajectories} trajectories of length {trajectory_length}")
        
        # Get oracle model path if specified
        oracle_path = self.config.get("oracle", {}).get("model_path", None)
        
        # Get distribution parameters
        extra_params = None
        if self.is_mixture:
            extra_params = {
                'mixture_components': self.mixture_components,
                'num_variables': self.input_dim - 1,
            }
        
        # Generate trajectories
        trajectories, rewards, actions, _ = simulate_trajectories(
            env_class=GeneralEnvironment,
            forward_model=self.forward_model,
            backward_model=self.backward_model,
            initial_state=self.initial_state,
            config=self.config,
            trajectory_length=trajectory_length,
            n_trajectories=n_trajectories,
            device=self.device,
            logger=self.logger,
            model_path=oracle_path,
            distribution=self.distribution_type,
            extra_parameters=extra_params,
        )
        
        # Convert to numpy arrays
        trajectories_np = np.stack(trajectories)  # (n_trajectories, trajectory_length+1, state_dim)
        rewards_np = np.stack(rewards)  # (n_trajectories, trajectory_length)
        actions_np = np.stack(actions)  # (n_trajectories, trajectory_length, action_dim)
        
        # Save trajectories if requested
        if save_trajectories:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save trajectories
            traj_path = self.output_dir / f"trajectories_{timestamp}.npz"
            np.savez_compressed(
                traj_path,
                trajectories=trajectories_np,
                rewards=rewards_np,
                actions=actions_np,
                feature_names=self.feature_names,
            )
            self.logger.info(f"Saved trajectories to {traj_path}")
        
        return trajectories_np, rewards_np, actions_np
    
    def analyze_trajectories(
        self,
        trajectories: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
    ) -> Dict:
        """
        Analyze the generated trajectories and compute statistics.
        
        Args:
            trajectories: Array of trajectories with shape (n_trajectories, trajectory_length+1, state_dim).
            rewards: Array of rewards with shape (n_trajectories, trajectory_length).
            actions: Array of actions with shape (n_trajectories, trajectory_length, action_dim).
            
        Returns:
            Dictionary with trajectory statistics.
        """
        self.logger.info("Analyzing trajectories")
        
        # Prepare container for statistics
        stats = {}
        
        # Basic shape information
        n_trajectories = trajectories.shape[0]
        trajectory_length = trajectories.shape[1] - 1  # -1 because states include initial state
        n_features = trajectories.shape[2]
        
        # Final rewards statistics
        final_rewards = rewards[:, -1]
        stats["rewards"] = {
            "mean": float(np.mean(final_rewards)),
            "median": float(np.median(final_rewards)),
            "min": float(np.min(final_rewards)),
            "max": float(np.max(final_rewards)),
            "std": float(np.std(final_rewards)),
            "percentiles": {
                f"p{p}": float(np.percentile(final_rewards, p))
                for p in [5, 25, 50, 75, 95]
            },
        }
        
        # Action statistics
        action_means = np.mean(actions, axis=(0, 1))  # Mean across trajectories and time steps
        action_stds = np.std(actions, axis=(0, 1))
        stats["actions"] = {
            "mean": action_means.tolist(),
            "std": action_stds.tolist(),
        }
        
        # Diversity metrics
        # 1. Compute cosine diversity of final states
        final_states = trajectories[:, -1, 1:]  # Exclude time dimension
        cosine_dist, cosine_normalized, _ = calculate_cosine_diversity(final_states, final_rewards)
        stats["diversity"] = {
            "cosine_distance": float(cosine_dist),
            "cosine_normalized": float(cosine_normalized),
        }
        
        # 2. Entropy of final state distributions per feature
        state_entropies = []
        for feature_idx in range(1, n_features):  # Skip time dimension
            # Bin the feature values to compute entropy
            hist, _ = np.histogram(trajectories[:, -1, feature_idx], bins=20, density=True)
            if np.sum(hist) > 0:  # Check if histogram isn't empty
                feature_entropy = entropy(hist)
                state_entropies.append(feature_entropy)
        
        stats["diversity"]["state_entropy"] = {
            "mean": float(np.mean(state_entropies)),
            "per_feature": [float(e) for e in state_entropies],
        }
        
        # 3. Count unique clusters of final states
        # Standardize features
        scaler = StandardScaler()
        final_states_scaled = scaler.fit_transform(final_states)
        
        # Use distance-based clustering
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.5, min_samples=5).fit(final_states_scaled)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        stats["diversity"]["n_clusters"] = int(n_clusters)
        
        # Top K trajectories by reward
        k = min(10, n_trajectories)
        top_indices = np.argsort(final_rewards)[-k:][::-1]
        
        stats["top_trajectories"] = []
        for i, idx in enumerate(top_indices):
            traj_dict = {
                "rank": i + 1,
                "index": int(idx),
                "final_reward": float(final_rewards[idx]),
                "final_state": trajectories[idx, -1, 1:].tolist(),  # Exclude time dimension
            }
            stats["top_trajectories"].append(traj_dict)
        
        return stats
    
    def visualize_trajectories(
        self,
        trajectories: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
        save_plots: bool = True,
    ):
        """
        Create various visualizations of the generated trajectories.
        
        Args:
            trajectories: Array of trajectories with shape (n_trajectories, trajectory_length+1, state_dim).
            rewards: Array of rewards with shape (n_trajectories, trajectory_length).
            actions: Array of actions with shape (n_trajectories, trajectory_length, action_dim).
            save_plots: Whether to save plots to disk.
        """
        self.logger.info("Creating visualizations")
        
        # Prepare output directory for plots
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Plot trajectories for each feature
        n_trajectories = trajectories.shape[0]
        n_features = trajectories.shape[2]
        
        # Find top 20 trajectories by final reward
        top_n = min(20, n_trajectories)
        final_rewards = rewards[:, -1]
        top_indices = np.argsort(final_rewards)[-top_n:][::-1]
        
        top_colors = sns.color_palette("husl", top_n)
        
        for feature_idx in range(1, n_features):  # Skip time dimension
            feature_name = self.feature_names[feature_idx - 1]  # Adjust index for feature names
            
            plt.figure(figsize=(10, 6))
            
            # Plot all trajectories with low alpha
            for i in range(min(1000, n_trajectories)):  # Limit to 1000 for performance
                if i not in top_indices:
                    plt.plot(trajectories[i, :, feature_idx], alpha=0.05, color='blue')
            
            # Plot top trajectories with distinct colors and higher alpha
            for rank, traj_idx in enumerate(top_indices):
                plt.plot(trajectories[traj_idx, :, feature_idx], alpha=0.8, 
                        color=top_colors[rank], linewidth=2,
                        label=f"#{rank+1}: R={final_rewards[traj_idx]:.2f}")
            
            # Plot mean and median
            mean_feature = np.mean(trajectories[:, :, feature_idx], axis=0)
            median_feature = np.median(trajectories[:, :, feature_idx], axis=0)
            plt.plot(mean_feature, color='black', linewidth=2, label='Mean')
            plt.plot(median_feature, color='red', linewidth=2, label='Median')
            
            plt.xlabel("Time Step")
            plt.ylabel(f"{feature_name} Value")
            plt.title(f"Trajectories of {feature_name}")
            plt.grid(True)
            
            # Add legend for top 5 and statistics
            if top_n > 5:
                # For readability, show only top 5 in legend
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(handles[:7], labels[:7], loc='best')
            else:
                plt.legend(loc='best')
            
            if save_plots:
                plt.tight_layout()
                plt.savefig(plots_dir / f"trajectories_{feature_name}.png", dpi=300)
                plt.close()
            else:
                plt.show()
        
        # 2. Plot reward distributions
        plt.figure(figsize=(10, 6))
        
        # Ensure rewards has initial zero rewards
        if rewards.shape[1] < trajectories.shape[1]:
            rewards_with_init = np.concatenate(
                [np.zeros((n_trajectories, 1)), rewards], axis=1
            )
        else:
            rewards_with_init = rewards
        
        # Plot individual reward trajectories with low alpha
        for i in range(min(1000, n_trajectories)):
            if i not in top_indices:
                plt.plot(rewards_with_init[i], alpha=0.05, color='blue')
        
        # Plot top reward trajectories
        for rank, traj_idx in enumerate(top_indices):
            plt.plot(rewards_with_init[traj_idx], alpha=0.8,
                    color=top_colors[rank], linewidth=2,
                    label=f"#{rank+1}: R={final_rewards[traj_idx]:.2f}")
        
        # Plot mean and median rewards
        mean_rewards = np.mean(rewards_with_init, axis=0)
        median_rewards = np.median(rewards_with_init, axis=0)
        plt.plot(mean_rewards, color='black', linewidth=2, label='Mean')
        plt.plot(median_rewards, color='red', linewidth=2, label='Median')
        
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.title("Reward Trajectories")
        plt.grid(True)
        
        # Add legend for top 5 and statistics
        if top_n > 5:
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[:7], labels[:7], loc='best')
        else:
            plt.legend(loc='best')
        
        if save_plots:
            plt.tight_layout()
            plt.savefig(plots_dir / "reward_trajectories.png", dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 3. Plot final reward distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(final_rewards, kde=True, bins=30)
        plt.axvline(np.median(final_rewards), color='red', linestyle='--', 
                   label=f'Median: {np.median(final_rewards):.2f}')
        plt.axvline(np.mean(final_rewards), color='black', linestyle='--',
                   label=f'Mean: {np.mean(final_rewards):.2f}')
        
        # Mark top 5 rewards
        for rank, traj_idx in enumerate(top_indices[:5]):
            plt.axvline(final_rewards[traj_idx], color=top_colors[rank], linestyle=':',
                       label=f'#{rank+1}: {final_rewards[traj_idx]:.2f}')
        
        plt.xlabel("Final Reward")
        plt.ylabel("Count")
        plt.title("Distribution of Final Rewards")
        plt.legend()
        plt.grid(True)
        
        if save_plots:
            plt.tight_layout()
            plt.savefig(plots_dir / "final_reward_distribution.png", dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 4. Plot action distributions per feature and timestep
        plot_distributions_per_feature(
            all_actions=actions,
            n_features=n_features-1,  # -1 to account for time dimension
            feature_names=self.feature_names,
            n_timesteps=actions.shape[1],
            mlflow_logger=None,  # We'll handle saving directly
        )
        
        for feature_idx in range(n_features - 1):  # Skip time dimension
            feature_name = self.feature_names[feature_idx]
            
            plt.figure(figsize=(10, 6))
            
            # Flatten actions across time steps for this feature
            feature_actions = actions[:, :, feature_idx].flatten()
            
            # Plot histogram with KDE
            sns.histplot(feature_actions, kde=True, bins=30)
            
            plt.xlabel(f"Action Value for {feature_name}")
            plt.ylabel("Density")
            plt.title(f"Distribution of Actions for {feature_name}")
            plt.grid(True)
            
            if save_plots:
                plt.tight_layout()
                plt.savefig(plots_dir / f"actions_distribution_{feature_name}.png", dpi=300)
                plt.close()
            else:
                plt.show()
    
    def export_results(
        self,
        stats: Dict,
        format_type: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export analysis results in the specified format.
        
        Args:
            stats: Dictionary of statistics from analyze_trajectories.
            format_type: Type of export format ('json', 'csv', 'latex').
            filename: Optional filename. If None, a default name is generated.
            
        Returns:
            Path to the exported file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gflownet_results_{timestamp}.{format_type}"
        
        output_path = self.output_dir / filename
        
        if format_type == "json":
            # Export as JSON
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        elif format_type == "csv":
            # Export key statistics as CSV
            # For nested dictionaries, flatten them
            flat_stats = {}
            
            # Flatten rewards stats
            for k, v in stats["rewards"].items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_stats[f"rewards_{k}_{sub_k}"] = sub_v
                else:
                    flat_stats[f"rewards_{k}"] = v
            
            # Flatten diversity stats
            for k, v in stats["diversity"].items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, list):
                            # For per-feature entropy, take the mean
                            flat_stats[f"diversity_{k}_{sub_k}"] = np.mean(sub_v)
                        else:
                            flat_stats[f"diversity_{k}_{sub_k}"] = sub_v
                else:
                    flat_stats[f"diversity_{k}"] = v
            
            # Create DataFrame and export
            df = pd.DataFrame([flat_stats])
            df.to_csv(output_path, index=False)
        
        elif format_type == "latex":
            # Export summary statistics as LaTeX tables
            with open(output_path, 'w') as f:
                # Rewards table
                f.write("% Rewards statistics\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{Summary of Reward Statistics}\n")
                f.write("\\begin{tabular}{lr}\n")
                f.write("\\toprule\n")
                f.write("Metric & Value \\\\\n")
                f.write("\\midrule\n")
                
                for k, v in stats["rewards"].items():
                    if k != "percentiles":
                        f.write(f"{k.capitalize()} & {v:.4f} \\\\\n")
                
                f.write("\\midrule\n")
                for p, v in stats["rewards"]["percentiles"].items():
                    f.write(f"{p.capitalize()} & {v:.4f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
                
                # Diversity table
                f.write("% Diversity statistics\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{Diversity Metrics}\n")
                f.write("\\begin{tabular}{lr}\n")
                f.write("\\toprule\n")
                f.write("Metric & Value \\\\\n")
                f.write("\\midrule\n")
                
                f.write(f"Cosine Distance & {stats['diversity']['cosine_distance']:.4f} \\\\\n")
                f.write(f"Normalized Cosine & {stats['diversity']['cosine_normalized']:.4f} \\\\\n")
                f.write(f"Mean State Entropy & {stats['diversity']['state_entropy']['mean']:.4f} \\\\\n")
                f.write(f"Number of Clusters & {stats['diversity']['n_clusters']} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
                
                # Top trajectories table
                f.write("% Top trajectories\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{Top 5 Trajectories by Reward}\n")
                f.write("\\begin{tabular}{rr")
                
                # Add columns for each feature
                for _ in range(len(stats["top_trajectories"][0]["final_state"])):
                    f.write("r")
                f.write("}\n")
                
                f.write("\\toprule\n")
                f.write("Rank & Reward")
                
                # Feature names in header
                for feature_idx, name in enumerate(self.feature_names):
                    if feature_idx < len(stats["top_trajectories"][0]["final_state"]):
                        f.write(f" & {name}")
                f.write(" \\\\\n")
                
                f.write("\\midrule\n")
                
                # Top 5 trajectories
                for traj in stats["top_trajectories"][:5]:
                    f.write(f"{traj['rank']} & {traj['final_reward']:.4f}")
                    
                    for val in traj["final_state"]:
                        f.write(f" & {val:.2f}")
                    
                    f.write(" \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        self.logger.info(f"Exported results to {output_path}")
        return str(output_path)
    
    def run_evaluation(
        self,
        n_trajectories: int = 1000,
        trajectory_length: int = 30,
        export_formats: List[str] = ["json", "csv"],
    ) -> Dict:
        """
        Run a complete evaluation pipeline.
        
        Args:
            n_trajectories: Number of trajectories to generate.
            trajectory_length: Length of each trajectory.
            export_formats: List of formats to export results in.
            
        Returns:
            Dictionary with analysis results.
        """
        # Track execution time
        start_time = time.time()
        self.logger.info(f"Starting evaluation with {n_trajectories} trajectories")
        
        # Generate trajectories
        trajectories, rewards, actions = self.generate_trajectories(
            n_trajectories=n_trajectories,
            trajectory_length=trajectory_length,
        )
        
        # Analyze trajectories
        stats = self.analyze_trajectories(trajectories, rewards, actions)
        
        # Create visualizations
        self.visualize_trajectories(trajectories, rewards, actions)
        
        # Export results
        for format_type in export_formats:
            self.export_results(stats, format_type)
        
        # Report execution time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
        
        return stats


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GFlowNet model.")
    
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained model directory or file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=1000,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--trajectory-length", type=int, default=30,
        help="Length of each trajectory.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device to use.",
    )
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = GFlowNetEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    evaluator.run_evaluation(
        n_trajectories=args.n_trajectories,
        trajectory_length=args.trajectory_length,
    )


if __name__ == "__main__":
    main()


# 
"""
python evaluate_gflownet.py \
    --config mlruns/4/a62d891737604b538733234860c17656/artifacts/run_params_config.yaml \
    --model-path mlruns/4/a62d891737604b538733234860c17656/artifacts \
    --output-dir evaluation_results \
    --n-trajectories 2000 \
    --trajectory-length 12 \
    --device cpu  
"""
