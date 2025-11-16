import re
from typing import List
import pandas as pd
import numpy as np
import joblib
class RewardCalculator:
    # Optionally add path to model for calculating reward using ML
    def __init__(self, config: dict, model_path: str = None):
        """
        Initialize the RewardCalculator by reading the config dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        self.variables = config['variables']
        self.reward_formula = config['reward']['formula']
        if model_path:
            self.model = joblib.load(model_path)
        self.base_features = self.extract_base_feature_names()

    def extract_base_feature_names(self) -> List[str]:
        base_feature_names = self.model.feature_names_in_
        # base_feature_names = set()

        # for var_name, var_config in self.variables.items():
        #     components = var_config['components']
        #     for component_name, component_config in components.items():
        #         source_name = component_config.get('source', component_name)
        #         base_feature_names.add(source_name)

        # return list(base_feature_names)
        return base_feature_names

    def calculate_variable(self, df: pd.DataFrame, variable_config: dict) -> pd.Series:
        """
        Calculate the value of a variable based on its formula and components.

        Args:
            df (pd.DataFrame): The DataFrame containing input data.
            variable_config (dict): Configuration for the variable.

        Returns:
            pd.Series: The calculated series for the variable.
        """
        formula = variable_config['formula']
        for component_name in variable_config['components'].keys():
            # Replace only the exact match of the component name
            new_name = f'df[\"{component_name}\"]'
            formula = re.sub(rf'\b{component_name}\b', new_name, formula)

        # Evaluate the formula to compute the variable
        try:
            result = eval(formula)
        except Exception as e:
            print(f"Error evaluating formula: {formula}")
            raise e

        return result
    
    def calculate_reward(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the reward based on the configured formula and variables.

        Args:
            df (pd.DataFrame): The DataFrame containing input data.

        Returns:
            pd.Series: The calculated reward series.
        """

        calculated_vars = {}
        for var_name, var_config in self.variables.items():
            calculated_vars[var_name] = self.calculate_variable(df, var_config)
            if var_config.get('transform', False):
                calculated_vars[var_name] = np.log(calculated_vars[var_name])
            if var_config.get('difference', False):
                calculated_vars[var_name] = calculated_vars[var_name].diff().fillna(method='bfill')
            if var_config.get('lag', 0) > 0:
                calculated_vars[var_name] = calculated_vars[var_name].shift(var_config['lag']).fillna(method='bfill')
            df[var_name] = calculated_vars[var_name]

        # Replace variable names in the reward formula with their calculated values
        reward_formula = self.reward_formula
        for var_name in calculated_vars.keys():
            reward_formula = reward_formula.replace(var_name, f"df['{var_name}']")
        
        # Calculate the reward
        try:
            reward_series = eval(reward_formula)
        except Exception as e:
            print(f"Error evaluating reward formula: {reward_formula}")
            raise e
        # Extract names of features used for calculating the reward and return them
        feature_names = list(self.base_features)
        return reward_series, feature_names
    
    def calculate_reward_ml(self, df: pd.DataFrame) -> pd.Series:
        cols = self.model.feature_names_in_
        df = df[cols]
        reward_series = self.model.predict(df)
        return reward_series, cols