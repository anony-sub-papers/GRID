import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from utility_functions import simulate_trajectories
import torch

def run_multiple_simulations(
    env_class,
    forward_model: torch.nn.Module,
    backward_model: torch.nn.Module,
    initial_state: np.ndarray,
    config: dict,
    trajectory_length: int,
    n_trajectories: int,
    device: torch.device,
    n_runs: int = 10,
    confidence_level: float = 0.95,
    logger=None,
    model_path=None,
    distribution_type='normal',
    extra_parameters=None
) -> Dict:
    """
    Run simulate_trajectories multiple times to compute confidence intervals.
    
    Args:
        n_runs: Number of independent simulation runs
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        
    Returns:
        Dictionary containing aggregated statistics and confidence intervals
    """
    
    all_run_results = {
        'final_rewards': [],
        'trajectory_diversities': [],
        'mean_rewards_over_time': [],
        'final_states': [],
        'run_statistics': []
    }
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    for run_idx in range(n_runs):
        if logger:
            logger.info(f"Running simulation {run_idx + 1}/{n_runs}")
        
        # Run single simulation
        trajectories, rewards, all_actions, feature_names = simulate_trajectories(
            env_class=env_class,
            forward_model=forward_model,
            backward_model=backward_model,
            initial_state=initial_state,
            config=config,
            trajectory_length=trajectory_length,
            n_trajectories=n_trajectories,
            device=device,
            logger=logger,
            model_path=model_path,
            distribution=distribution_type,
            extra_parameters=extra_parameters
        )
        
        # Extract statistics for this run
        trajectories = np.stack(trajectories)
        rewards = np.maximum(rewards, 0)
        rewards = np.stack(rewards)
        
        # Final rewards
        final_rewards = rewards[:, -1] if rewards.ndim > 1 else rewards
        all_run_results['final_rewards'].append(final_rewards)
        
        # Final states for diversity calculation
        final_states = trajectories[:, -1, :]
        all_run_results['final_states'].append(final_states)
        
        # Mean rewards over time
        mean_rewards_over_time = np.mean(rewards, axis=0)
        all_run_results['mean_rewards_over_time'].append(mean_rewards_over_time)
        
        # Calculate diversity for this run
        from utility_functions import calculate_quality_diversity
        diversity_metrics = calculate_quality_diversity(final_states, final_rewards)
        all_run_results['trajectory_diversities'].append(diversity_metrics)
        
        # Store run-level statistics
        run_stats = {
            'run_idx': run_idx,
            'mean_final_reward': np.mean(final_rewards),
            'std_final_reward': np.std(final_rewards),
            'max_final_reward': np.max(final_rewards),
            'min_final_reward': np.min(final_rewards),
            'diversity_score': diversity_metrics['avg_div'],
            'quality_diversity': diversity_metrics['quality_diversity']
        }
        all_run_results['run_statistics'].append(run_stats)
    
    # Compute aggregate statistics and confidence intervals
    results = compute_confidence_intervals(all_run_results, lower_percentile, upper_percentile)
    
    return results

def compute_confidence_intervals(all_run_results: Dict, lower_percentile: float, upper_percentile: float) -> Dict:
    """Compute confidence intervals from multiple runs."""
    
    # Final rewards statistics
    all_final_rewards = np.concatenate(all_run_results['final_rewards'])
    final_reward_stats = {
        'mean': np.mean(all_final_rewards),
        'std': np.std(all_final_rewards),
        'ci_lower': np.percentile(all_final_rewards, lower_percentile),
        'ci_upper': np.percentile(all_final_rewards, upper_percentile),
        'median': np.median(all_final_rewards)
    }
    
    # Diversity statistics across runs
    diversity_scores = [run['diversity_score'] for run in all_run_results['run_statistics']]
    diversity_stats = {
        'mean': np.mean(diversity_scores),
        'std': np.std(diversity_scores),
        'ci_lower': np.percentile(diversity_scores, lower_percentile),
        'ci_upper': np.percentile(diversity_scores, upper_percentile)
    }
    
    # Quality-diversity statistics
    qd_scores = [run['quality_diversity'] for run in all_run_results['run_statistics']]
    qd_stats = {
        'mean': np.mean(qd_scores),
        'std': np.std(qd_scores),
        'ci_lower': np.percentile(qd_scores, lower_percentile),
        'ci_upper': np.percentile(qd_scores, upper_percentile)
    }
    
    # Mean rewards over time with confidence intervals
    mean_rewards_matrix = np.array(all_run_results['mean_rewards_over_time'])
    timestep_stats = {
        'mean': np.mean(mean_rewards_matrix, axis=0),
        'std': np.std(mean_rewards_matrix, axis=0),
        'ci_lower': np.percentile(mean_rewards_matrix, lower_percentile, axis=0),
        'ci_upper': np.percentile(mean_rewards_matrix, upper_percentile, axis=0)
    }
    
    # Run-to-run variability
    run_means = [run['mean_final_reward'] for run in all_run_results['run_statistics']]
    run_variability = {
        'mean_across_runs': np.mean(run_means),
        'std_across_runs': np.std(run_means),
        'ci_lower': np.percentile(run_means, lower_percentile),
        'ci_upper': np.percentile(run_means, upper_percentile)
    }
    
    return {
        'final_reward_stats': final_reward_stats,
        'diversity_stats': diversity_stats,
        'quality_diversity_stats': qd_stats,
        'timestep_stats': timestep_stats,
        'run_variability': run_variability,
        'raw_data': all_run_results,
        'run_statistics_df': pd.DataFrame(all_run_results['run_statistics'])
    }

def plot_confidence_intervals(results: Dict, mlflow_logger=None, save_plots: bool = True):
    """Create comprehensive plots with confidence intervals."""
    
    # 1. Final rewards distribution with CI
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Final rewards distribution
    all_final_rewards = np.concatenate(results['raw_data']['final_rewards'])
    axes[0, 0].hist(all_final_rewards, bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(results['final_reward_stats']['mean'], color='red', linestyle='--', 
                       label=f"Mean: {results['final_reward_stats']['mean']:.3f}")
    axes[0, 0].axvline(results['final_reward_stats']['ci_lower'], color='orange', linestyle=':', 
                       label=f"95% CI: [{results['final_reward_stats']['ci_lower']:.3f}, {results['final_reward_stats']['ci_upper']:.3f}]")
    axes[0, 0].axvline(results['final_reward_stats']['ci_upper'], color='orange', linestyle=':')
    axes[0, 0].set_title('Final Rewards Distribution with 95% CI')
    axes[0, 0].set_xlabel('Final Reward')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # Plot 2: Rewards over time with confidence bands
    timestep_stats = results['timestep_stats']
    time_steps = range(len(timestep_stats['mean']))
    axes[0, 1].plot(time_steps, timestep_stats['mean'], color='blue', linewidth=2, label='Mean')
    axes[0, 1].fill_between(time_steps, timestep_stats['ci_lower'], timestep_stats['ci_upper'], 
                           alpha=0.3, color='blue', label='95% CI')
    axes[0, 1].set_title('Mean Rewards Over Time with Confidence Intervals')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Run-to-run variability
    run_stats_df = results['run_statistics_df']
    axes[1, 0].boxplot([run_stats_df['mean_final_reward'], run_stats_df['diversity_score'], 
                        run_stats_df['quality_diversity']], 
                       labels=['Mean Final\nReward', 'Diversity\nScore', 'Quality\nDiversity'])
    axes[1, 0].set_title('Run-to-Run Variability')
    axes[1, 0].set_ylabel('Value')
    
    # Plot 4: Diversity vs Quality scatter
    axes[1, 1].scatter(run_stats_df['diversity_score'], run_stats_df['mean_final_reward'], alpha=0.7)
    axes[1, 1].set_xlabel('Diversity Score')
    axes[1, 1].set_ylabel('Mean Final Reward')
    axes[1, 1].set_title('Diversity vs Quality Trade-off')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        confidence_plot_path = "confidence_intervals_analysis.png"
        plt.savefig(confidence_plot_path, dpi=300, bbox_inches='tight')
        if mlflow_logger:
            mlflow_logger.log_artifact(confidence_plot_path)
        plt.close()
    else:
        plt.show()

def create_summary_report(results: Dict) -> str:
    """Create a text summary of the statistical analysis."""
    
    report = f"""
STATISTICAL ANALYSIS SUMMARY
============================

Final Reward Statistics:
- Mean: {results['final_reward_stats']['mean']:.4f} ± {results['final_reward_stats']['std']:.4f}
- 95% CI: [{results['final_reward_stats']['ci_lower']:.4f}, {results['final_reward_stats']['ci_upper']:.4f}]
- Median: {results['final_reward_stats']['median']:.4f}

Diversity Statistics:
- Mean Diversity: {results['diversity_stats']['mean']:.4f} ± {results['diversity_stats']['std']:.4f}
- 95% CI: [{results['diversity_stats']['ci_lower']:.4f}, {results['diversity_stats']['ci_upper']:.4f}]

Quality-Diversity Trade-off:
- Mean QD Score: {results['quality_diversity_stats']['mean']:.4f} ± {results['quality_diversity_stats']['std']:.4f}
- 95% CI: [{results['quality_diversity_stats']['ci_lower']:.4f}, {results['quality_diversity_stats']['ci_upper']:.4f}]

Run-to-Run Variability:
- Mean across runs: {results['run_variability']['mean_across_runs']:.4f}
- Std across runs: {results['run_variability']['std_across_runs']:.4f}
- 95% CI: [{results['run_variability']['ci_lower']:.4f}, {results['run_variability']['ci_upper']:.4f}]

Number of runs: {len(results['run_statistics_df'])}
Total trajectories analyzed: {len(results['run_statistics_df']) * len(results['raw_data']['final_rewards'][0])}
"""
    
    return report
