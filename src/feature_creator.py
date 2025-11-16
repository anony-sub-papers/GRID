import pandas as pd
import numpy as np

class FeatureCreator:
    def __init__(self, config: dict):
        self.config = config

    def apply_transformation(self, series: pd.Series, method: str) -> pd.Series:
        # Handle zeros and negatives before log transformation
        series = series.replace(0, np.nan)
        series = series.where(series > 0, np.nan)
        return np.log(series)

    def apply_difference(self, series: pd.Series) -> pd.Series:
        return series.diff()

    def apply_lag(self, series: pd.Series, lag: int) -> pd.Series:
        return series.shift(lag)
    
    # def special_case(df):
    #     for col in df.columns:
    #         for lag in range(1, 13):
    #             df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    #     return df.dropna()

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features = df.copy()
        is_ml = self.config.get('ML', False)
        # Iterate through the components in each variable
        if is_ml:
            return df_features, df_features.columns.tolist()
        else:
            for variable_config in self.config['variables'].values():
                for component_name, component_config in variable_config['components'].items():
                    # Determine the actual component name (in case of source specified)
                    actual_component_name = component_config.get('source', component_name)
                    # Start with the original series
                    series = df[actual_component_name]

                    # Apply transformation if required
                    if component_config.get('transform', False):
                        series = self.apply_transformation(series, component_config['method'])

                    # Apply lagging if required
                    if component_config.get('lag', 0) > 0:
                        series = self.apply_lag(series, component_config['lag'])

                    # Apply differencing if required
                    if component_config.get('difference', False):
                        series = self.apply_difference(series)

                    # Handle NaN and Inf values
                    series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    series.fillna(method='ffill', inplace=True)
                    series.fillna(method='bfill', inplace=True)
                    series.fillna(0, inplace=True)

                    # Store the resulting series back in the DataFrame with the appropriate name
                    df_features[f"{component_name}"] = series
            # print("Generated Features are: ", df_features.columns.tolist(),"\n", "Values are: ", df_features.values)
        return df_features, df_features.columns.tolist()
