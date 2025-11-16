import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Dict
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns

class BayesianOptimizer:
    def __init__(self, env, num_steps: int, pbounds: Dict, random_state: int = 42):
        """
        Bayesian Optimization for trajectory generation.
        
        Args:
            env: Environment to simulate trajectories
            num_steps: Number of timesteps in the trajectory
            pbounds: Dictionary of parameter bounds {param_name: (lower_bound, upper_bound)}
            random_state: Random seed
        """
        self.env = env
        self.num_steps = num_steps
        self.pbounds = pbounds
        self.random_state = random_state
        
        # Initialize optimizer
        self.optimizer = BayesianOptimization(
            f=self._black_box_function,
            pbounds=self.pbounds,
            random_state=random_state,
            verbose=1
        )
        
        # Track all evaluated points
        self.all_evaluations = []
    
    def _black_box_function(self, **params):
        """Convert parameters to trajectory and evaluate reward"""
        # Convert dict parameters to action sequence
        actions = []
        for step in range(self.num_steps - 1):  # -1 because initial state is given
            step_actions = []
            for i in range(len(self.pbounds) // (self.num_steps - 1)):
                param_name = f"a{step}_{i}"
                if param_name in params:
                    step_actions.append(params[param_name])
            actions.append(step_actions)
        
        # Reset environment and apply actions
        state = self.env.reset()
        trajectory = [state.copy()]
        
        for action in actions:
            next_state, _ = self.env.step(action)
            trajectory.append(next_state.copy())
        
        # Compute reward for the final state
        final_trajectory = np.array(trajectory)
        final_features, _ = self.env.generate_full_features(final_trajectory)
        final_reward, _ = self.env.calculate_final_reward_ml(final_features.iloc[-1:])
        
        # Store evaluation
        self.all_evaluations.append({
            "params": params,
            "trajectory": final_trajectory,
            "reward": final_reward[0]
        })
        
        return final_reward[0]
    
    def optimize(self, init_points: int = 5, n_iter: int = 25, acq: str = "ei", kappa: float = 2.5):
        """Run optimization"""
        acq_function = BayesianOptimization.acquisition_function(kind=acq, kappa=kappa)
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, acq=acq_function)
        return self.optimizer.max
    
    def get_top_trajectories(self, n: int = 10):
        """Return the top n trajectories based on reward"""
        sorted_evals = sorted(self.all_evaluations, key=lambda x: x["reward"], reverse=True)
        top_n = sorted_evals[:n]
        
        # Create dataframe for visualization
        summary_rows = []
        for rank, result in enumerate(top_n):
            final_state = result["trajectory"][-1]
            row = {"Rank": rank + 1, "FinalReward": result["reward"]}
            
            # Add feature values
            for i in range(len(final_state)):
                row[f"Feature_{i}"] = final_state[i]
            
            summary_rows.append(row)
            
        return pd.DataFrame(summary_rows), sorted_evals


def setup_bayesopt_benchmark(env, num_steps, action_dim):
    """Setup Bayesian Optimization benchmark"""
    # Create parameter bounds for each action dimension across all steps
    pbounds = {}
    for step in range(num_steps - 1):  # -1 because initial state is given
        for dim in range(action_dim):
            pbounds[f"a{step}_{dim}"] = (-1.0, 1.0)  # Adjust bounds as needed
    
    return BayesianOptimizer(env, num_steps, pbounds)

__all__ = ['BayesianOptimizer', 'setup_bayesopt_benchmark']