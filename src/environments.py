import numpy as np
import pandas as pd
from typing import Tuple
from reward_calculator import RewardCalculator
from feature_creator import FeatureCreator
import numpy as np
import random
import torch
import copy
class GeneralEnvironment:
    def __init__(self, initial_state: np.ndarray, config: dict, model,
                  input_dim: int = 3, max_steps: int = 100,model_path: str = None):
        """
        Initialize the General Environment.
        
        Args:
            initial_state (np.ndarray): The initial state of the environment, excluding the time step.
            config (dict): The configuration dictionary.
            model: The model used to predict actions.
            input_dim (int): The dimension of the input features, including the time step.
            max_steps (int): The maximum number of time steps for the environment.
        """
        self.initial_state = initial_state
        self.state = np.concatenate(([0], initial_state))  # Include time step in the state
        self.config = config
        self.model = model
        self.input_dim = input_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        
        # Initialize the reward calculator
        if model_path:
            self.reward_calculator = RewardCalculator(config, model_path)
        else:
            self.reward_calculator = RewardCalculator(config)
        self.feature_creator = FeatureCreator(config)

    def set_state(self, state: np.ndarray):
        """Set the state of the environment."""
        if len(state) != self.input_dim:
            raise ValueError("State size must match the input dimension. \
                             Current sizes are: state={}, input_dim={}".format(len(state), self.input_dim))
        self.state = state
    
    def copy_self(self):
        new_env = copy.deepcopy(self)
        return new_env
    
    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state."""
        self.state = np.concatenate(([0], self.initial_state.copy()))  # Reset to initial state with time step 0
        self.current_step = 0
        self.done = False
        return self.state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Perform a step in the environment."""
        if self.done:
            raise ValueError("Cannot call step() on a done environment. Please reset first.")

        # Ensure the action only affects the non-time-step part of the state
        if len(action) != len(self.state) - 1:
            raise ValueError("Action size must match the non-time-step part of the state. \
                             Current sizes are: action={}, state={}".format(len(action), len(self.state)))

        # Update the state excluding the time step
        # self.state[1:] = self.state[1:] * (1 + action/100)
        # Normal instead of percentage change
        self.state[1:] = self.state[1:] + action/1 # used to be 100
        self.state[1:] = np.maximum(self.state[1:],-200)  # Clamp to 
        self.state[1:] = np.minimum(self.state[1:],200)  # Clamp to

        # Increment the time step
        self.current_step += 1
        self.state[0] = self.current_step  # Update the time step in the state

        if self.current_step >= self.max_steps:
            self.done = True
        return self.state, self.done

    def generate_full_features(self, trajectory: np.ndarray) -> pd.DataFrame:
        cols = ['TimeStep'] + self.reward_calculator.base_features.tolist()
        # print(f"The cols are {cols}")
        df_traj = pd.DataFrame(trajectory, columns=cols)
        df_full_features,feature_names = self.feature_creator.generate_features(df_traj)
        return df_full_features,feature_names


    def reseed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def generate_features_special_case(self, trajectory: np.ndarray) -> pd.DataFrame:
        # Trajectory , Features
        cols = ['TimeStep'] + self.reward_calculator.base_features.tolist()
        df_traj = pd.DataFrame(trajectory, columns=cols)
        for i in df_traj.columns[1:]:
            for lag in range(1, 13):
                df_traj[f'{i}_lag_{lag}'] = df_traj[i].shift(lag)
        return df_traj.dropna()
    def get_state(self) -> np.ndarray:
        return self.state
    
    def calculate_final_reward(self, final_features: pd.DataFrame) -> pd.Series:
        reward_series,features_names = self.reward_calculator.calculate_reward(final_features)
        reward_series = 1400 - reward_series # If SPX Grew above 1438, the reward is negative == 0 , else it is positive
        return (reward_series, features_names)
    
    def calculate_final_reward_ml(self, final_features: pd.DataFrame) -> pd.Series:
        reward_series,features_names = self.reward_calculator.calculate_reward_ml(final_features)
        # reward_series = 1485 - reward_series
        # print(f"Final reward in the reward series is {-reward_series[-1]}")
        reward_series = -reward_series
        # print(f"Reward Series is {reward_series}")
        return (reward_series, features_names)

    def custom_reward(self, final_features: pd.DataFrame) -> pd.Series:
        _,features_names = self.reward_calculator.calculate_reward_ml(final_features)

        # Define multiple target modes for features
        feature_modes = [
            [200, 180, 160, 140, 120, 100],      # Mode 1 - Strong decreasing pattern
            [-200, -180, -160, -140, -120, -100], # Mode 2 - Strong negative decreasing pattern
            # [100, 120, 140, 160, 180, 200],      # Mode 3 - Strong increasing pattern
            # [-100, -120, -140, -160, -180, -200], # Mode 4 - Strong negative increasing pattern
            # [0, 0, 0, 0, 0, 0]                   # Mode 5 - Flat neutral pattern
        ]
        
        # Get final feature values (excluding time step)
        reward_series = []
        for row in final_features.iterrows():
            feat_val = row[1][1:]
            reward_total = 0
            for m in feature_modes:
                if len(m) != len(feat_val):
                    raise ValueError("Feature mode length does not match feature values length.")
                else:
                    feat_val = np.array(feat_val)
                    distance = np.abs(feat_val - m)
                    # std = np.minimum(5, np.abs(feat_val / 5))
                    reward = 10*(1 / (1 + (distance )))#/ std))
                    reward_total += np.sum(reward)
            reward_series.append(reward_total)
        reward_series = pd.Series(reward_series)
        reward_series = reward_series.astype(float)
        return (reward_series, features_names)
    
    def is_done(self) -> bool:
        return self.done