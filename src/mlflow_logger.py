import copy
from typing import Any, Dict
import mlflow
import torch.nn as nn
import os 
import torch
class MLflowLogger:
    def __init__(self, config: Dict[str, Any],model_signature = None,logger=None):
        self.run_params = copy.copy(config)
        self.config = config['mlflow']
        self.tracking_uri = self.config['tracking_uri']
        self.experiment_name = self.config['experiment_name']
        self.run_name = self.config['run_name']
        self.save_artifacts = self.config.get('save_artifacts', True)
        self.log_every_n_steps = self.config.get('log_every_n_steps', 10)
        self.signature = model_signature
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            mlflow.set_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException as e:
            logger.info(f"MLflow exception: {e}")
            mlflow.create_experiment(self.experiment_name)

        self.run = mlflow.start_run(run_name=self.run_name)
        self.log_dict("run_params_config",self.run_params)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if self.config.get('log_params', True):
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        if self.config.get('log_metrics', True):
            if step:
                mlflow.log_metrics(metrics, step=step)
            else:
                mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: str):
        """Log a file as an artifact to MLflow."""
        if self.save_artifacts and self.config.get('log_artifacts', True):
            mlflow.log_artifact(file_path, artifact_path=self.config['artifact_path'])

    # def log_model(self, model: nn.Module, artifact_path: str):
    #     """Log a PyTorch model as an artifact to MLflow."""
    #     if self.save_artifacts and self.config.get('log_artifacts', True):
    #         mlflow.pytorch.log_model(model, artifact_path,signature=self.signature)
    
    def log_model(self, model: nn.Module, artifact_path: str):
        """
        Log a PyTorch model as a simple .pth artifact under the current run.

        This creates: <run_artifacts>/<artifact_path>/data/model.pth
        """
        if not (self.save_artifacts and self.config.get("log_artifacts", True)):
            return

        # Build a local temp path mirroring the desired artifact path
        local_dir = os.path.join("artifacts", artifact_path, "data")
        os.makedirs(local_dir, exist_ok=True)
        local_model_file = os.path.join(local_dir, "model.pth")

        # Save the entire model
        torch.save(model, local_model_file)

        # Log as an artifact under the run
        # artifact_path in MLflow == "forward_model_iteration_0/data", etc.
        mlflow.log_artifact(
            local_model_file,
            artifact_path=os.path.join(artifact_path, "data"),
        )

    def log_dict(self,name:str,dictionary:Dict):
        if self.save_artifacts:
            mlflow.log_dict(dictionary,f"{name}.yaml")

    def end_run(self):
        """End the MLflow run."""
        mlflow.end_run()