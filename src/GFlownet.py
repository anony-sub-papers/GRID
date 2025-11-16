# Imports
import copy
import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
from tqdm import trange
from typing import Any, Dict, List, Optional, Tuple
import warnings
import yaml
from mlflow.models.signature import infer_signature
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import MLPModel
from environments import GeneralEnvironment
from utility_functions import (seed_all,generate_synthetic_data_for_scaling,load_initial_state,setup_logger,plot_trajectories,
                               load_model,load_entire_model,get_model_paths,plot_and_log_metrics,visualize_and_log_simulated_trajectories,
                               get_policy_dist)

import joblib
import random
torch.autograd.set_detect_anomaly(True)
from mlflow_logger import MLflowLogger
from replay_buffer import *
from tqdm import trange
warnings.filterwarnings("ignore")

def setup_experiment(input_dim,hidden_dim,output_dim,num_layers,
                     activation = 'leaky_relu',dropout = 0,
                     use_batchnorm = False,use_layernorm = True,
                     optimizer_name = "sgd",
                     device = "cpu",
                     learning_rate_model = 1e-3,learning_rate_logZ = 1e-3,
                     logger=None,init_type = "xavier_uniform"):
    """Setup the forward and backward models, logZ, and optimizer."""
    assert output_dim != 2 * input_dim - 1, f"Output dim must be twice the input dim for mean and std predictions.Should be {int((input_dim - 1) * 2)} Instead got: {output_dim}"
    print(f"Init Type: {init_type}")
    # output_dim = int((input_dim - 1) * 2)  # Model outputs mean and std for (input_dim) features
    forward_model = MLPModel(input_dim=input_dim, hid_dim=hidden_dim,
                              output_dim=output_dim,num_layers=num_layers,
                              activation=activation, dropout=dropout,
                              use_batchnorm=use_batchnorm, use_layernorm=use_layernorm,
                              logger=logger, init_type=init_type).to(device)
    
    backward_model = MLPModel(input_dim=input_dim, hid_dim=hidden_dim,
                                 output_dim=output_dim,num_layers=num_layers,
                                 activation=activation, dropout=dropout,
                                 use_batchnorm=use_batchnorm, use_layernorm=use_layernorm,
                                 logger=logger, init_type=init_type).to(device)


    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    if optimizer_name == "adam":
        optimizer = optim.Adam([
            {'params': forward_model.parameters(), 'lr': learning_rate_model},
            {'params': backward_model.parameters(), 'lr': learning_rate_model},
            {'params': [logZ], 'lr': learning_rate_logZ},
        ])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD([
            {'params': forward_model.parameters(), 'lr': learning_rate_model},
            {'params': backward_model.parameters(), 'lr': learning_rate_model},
            {'params': [logZ], 'lr': learning_rate_logZ},
        ])
    else:
        logger.info("Currently only supporting Adam")
    if logger:
        logger.info("Finished setting up the forward and backward models, logZ, and optimizer.")
        logger.info(f"Forward Model: {forward_model}")
        logger.info(f"Backward Model: {backward_model}")
        logger.info(f"LogZ: {logZ}")
        logger.info(f"Optimizer: {optimizer}")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=100)
    return forward_model, backward_model, logZ, optimizer,None

class TrajectoryReplayBuffer:
    """Replay buffer storing entire trajectories for off-policy GFlowNet training."""

    def __init__(self, capacity: int = 15000):
        """
        Args:
            capacity (int): Maximum number of trajectories to store.
        """
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add(self, trajectories: List[Dict[str, Any]]) -> None:
        """
        Add a list of trajectory dictionaries to the buffer.

        Each trajectory dict can contain:
            {
                'states': np.array of shape (trajectory_length+1, state_dim),
                'actions': np.array of shape (trajectory_length, action_dim),
                'log_prob_f': np.array of shape (trajectory_length, ),
                'log_prob_b': np.array of shape (trajectory_length, ),  # optional
                'reward': float,
            }
        """
        self.buffer.extend(trajectories)
        if len(self.buffer) > self.capacity:
            excess = len(self.buffer) - self.capacity
            self.buffer = self.buffer[excess:]  # drop oldest

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Randomly sample a list of trajectory dictionaries.
        """
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)
    
def infer_sig(input_dim,forward_model,batch_norm = True):
    if batch_norm:
        input_example = torch.rand(2, input_dim)
        output = forward_model(input_example)
        predictions_mean = output[:, :output.shape[1] // 2]
        predictions_std = output[:, output.shape[1] // 2:]
        signature = infer_signature(input_example.numpy(), [predictions_mean.detach().numpy(),predictions_std.detach().numpy()])
    else:
        input_example = torch.rand(1, input_dim)  # Replace with appropriate dimensions
        output = forward_model(input_example)
        predictions_mean = output[:, :output.shape[1] // 2]
        predictions_std = output[:, output.shape[1] // 2:]
        signature = infer_signature(input_example.numpy(), [predictions_mean.detach().numpy(),predictions_std.detach().numpy()])
    return signature

def transform_log(x: torch.Tensor) -> torch.Tensor:
    """Transforms negative numbers to small positive and keeps positive numbers unchanged."""
    # Clip x to prevent overflow in exp
    diff_from_upper_bound = x - 50.0
    diff_from_lower_bound = x + 50.0
    x_clipped = torch.clamp(x, min=-50.0, max=50.0)
    return torch.log1p(torch.exp(x_clipped)) + torch.where(x > 50.0, diff_from_upper_bound, 0.0)

def train_off_policy_with_exploration(
    forward_model: torch.nn.Module,
    backward_model: torch.nn.Module,
    logZ: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    seed: int = 0,
    batch_size: int = 10,
    trajectory_length: int = 30,
    env=None,
    device: str = "cpu",
    init_exploration_noise: float = 0.1,
    n_iterations: int = 10_000,
    input_dim: int = 10,
    mlflow_logger: Any = None,
    n_simulations: int = 200,
    logger: Any = None,
    log_file_name: str = None,
    replay_capacity: int = 5000,
    warmup_steps: int = 100,
    distribution_type: str = "normal",
    n_runs: int = 1,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor, List[float], List[float]]:
    """
    Off-policy GFlowNet training loop with exploration and a replay buffer.
    
    Args:
        forward_model: Forward model to train.
        backward_model: Backward model to train.
        logZ: Log partition function parameter.
        optimizer: Torch optimizer for training.
        scheduler: Optional learning rate scheduler.
        seed (int): Random seed.
        batch_size (int): Number of trajectories to collect per iteration.
        trajectory_length (int): Number of steps in each trajectory.
        env: Environment object that provides get_state(), step(), reset(), etc.
        device (str): Device for Torch ("cpu" or "cuda").
        init_exploration_noise (float): Initial exploration noise for policy.
        n_iterations (int): Total training iterations.
        input_dim (int): Dimensionality of the state input (excluding time).
        mlflow_logger: Optional MLflow logger.
        config (Dict[str, Any]): Configuration dictionary with training hyperparams.
        run_name (str): Experiment/run name for logging.
        logger: Python logger object.
        log_file_name (str): Optional log file path.
        replay_capacity (int): Maximum capacity of the replay buffer.
        warmup_steps (int): Number of iterations to just fill replay buffer before training.
    
    Returns:
        (forward_model, backward_model, logZ, losses, rewards)
    """
    result_runs = {}
    for run_idx in range(n_runs):
        print(f"Run {run_idx + 1}/{n_runs}")
        random_seed = random.randint(0, 10000)
        seed_all(random_seed)
        # Initialize the model and other experiment components
        forward_model, backward_model, logZ, optimizer,scheduler = setup_experiment(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,num_layers=num_layers,
            activation=activation, dropout=dropout, use_batchnorm=use_batchnorm, use_layernorm=use_layernorm,
            optimizer_name=optimizer_name, device=device,
            learning_rate_model=learning_rate_model, learning_rate_logZ=learning_rate_logZ,
            logger=logger,init_type=init_type
        )
        # -----------------------------
        print(f"Chosen Distribution: {distribution_type}")
        # -----------------------------
        # Prepare containers
        losses = []
        rewards_log = []

        # Create replay buffer
        replay_buffer = TrajectoryReplayBuffer(capacity=replay_capacity)

        # Create multiple env copies for batch collection
        envs = [copy.deepcopy(env) for _ in range(batch_size)]

        # Create a schedule for decreasing noise
        exploration_schedule = torch.linspace(init_exploration_noise, 0, n_iterations)
        

        for iteration in trange(n_iterations):
            # -----------------------------
            # 1. Collect new trajectories
            # -----------------------------
            exploring = (iteration / n_iterations) < exploration_percentage
            noise_level = exploration_schedule[iteration].item()

            # We'll collect "batch_size" trajectories
            collected_trajectories = []
            for e_i, env_inst in enumerate(envs):
                env_inst.reset()
                # env_inst.reseed(seed + e_i)

            
            states = torch.tensor(
                [env_inst.get_state() for env_inst in envs],
                dtype=torch.float32,
                device=device
            )
            
            # We store per-trajectory data in lists
            # shape: (batch_size, trajectory_length + 1, state_dim) => for states
            batch_states = torch.zeros((batch_size, trajectory_length + 1, input_dim), device=device)
            batch_actions = torch.zeros((batch_size, trajectory_length, input_dim-1), device=device)
            batch_log_prob_f = torch.zeros(batch_size, trajectory_length, device=device)

            # Set initial states
            batch_states[:, 0, :] = states
            
            # Forward pass to build trajectories
            for t in range(trajectory_length):
                policy_dist, exploration_dist = get_policy_dist(
                    forward_model, states, off_policy_noise=noise_level,
                    dist_type=distribution_type, logger=logger,
                    extra_params={
                        'mixture_components': mixture_components,
                        'num_variables': input_dim - 1,
                    } if is_mixture else None
                )
                # Sample action from exploration_dist if exploring, else policy_dist
                if exploring:
                    action = exploration_dist.rsample()
                    logprob_f = exploration_dist.log_prob(action)
                else:
                    action = policy_dist.rsample()
                    logprob_f = policy_dist.log_prob(action)

                batch_actions[:, t, :] = action
                batch_log_prob_f[:, t] = logprob_f

                # Step each environment
                new_states_list = []
                for i, env_inst in enumerate(envs):
                    new_state, done_flag = env_inst.step(action[i].detach().cpu().numpy())
                    new_states_list.append(new_state)

                new_states = torch.tensor(new_states_list, dtype=torch.float32, device=device)
                batch_states[:, t + 1, :] = new_states
                states = new_states

            # For the backward model, we won't compute logPB yet. We'll do that during training from replay.

            # Compute final rewards
            final_features_list = []
            for i, env_inst in enumerate(envs):
                traj_data_np = batch_states[i].cpu().numpy()  # shape: (trajectory_length+1, input_dim+1)
                full_feats, feat_names = env_inst.generate_full_features(traj_data_np)
                final_features_list.append(full_feats.iloc[-1])


            df_final_features = pd.DataFrame(final_features_list)
            f_reward, f_names = env.calculate_final_reward_ml(df_final_features)
            #Clip rewards to 1
            # f_reward, f_names = env.custom_reward(df_final_features)
            # print(f"Reward is {f_reward}")
            if isinstance(f_reward, pd.Series):
                f_reward = f_reward.values

            if any(np.isnan(f_reward)) or any(np.isinf(f_reward)):
                # Handle NaN or inf values in rewards
                print(f"NaN or inf in rewards at iteration {iteration}. Skipping this batch.")
                continue

            else:
                # print(f"Final reward is {f_reward}")
                pass    
            # raw_rewards = torch.tensor(f_reward, dtype=torch.float32, device=device).clamp(min=1)
            raw_rewards = torch.tensor(f_reward, dtype=torch.float32, device=device)
            # Build each trajectory dict
            for i in range(batch_size):
                states_np = batch_states[i].detach().cpu().numpy()  # (trajectory_length+1, state_dim)
                actions_np = batch_actions[i].detach().cpu().numpy()  # (trajectory_length, action_dim)
                logprob_f_np = batch_log_prob_f[i].detach().cpu().numpy()  # (trajectory_length,)
                
                traj_dict = {
                    'states': states_np,
                    'actions': actions_np,
                    'log_prob_f': logprob_f_np,
                    'reward': raw_rewards[i].item()
                }
                collected_trajectories.append(traj_dict)

            # Add to replay buffer
            replay_buffer.add(collected_trajectories)

            # ----------------------------
            # 2. Sample from replay buffer
            # ----------------------------
            if len(replay_buffer) < batch_size or iteration < warmup_steps:
                # If replay buffer doesn't have enough data or we're still in warmup, skip training
                continue

            sample_batch = replay_buffer.sample(batch_size)
            

            # Convert the entire batch at once
            states_batch = torch.tensor(
                np.array([traj["states"] for traj in sample_batch]),
                dtype=torch.float32, device=device
            )  # Shape: (batch_size, trajectory_length+1, state_dim)

            actions_batch = torch.tensor(
                np.array([traj["actions"] for traj in sample_batch]),
                dtype=torch.float32, device=device
            )  # Shape: (batch_size, trajectory_length, action_dim)

            rewards_batch = torch.tensor(
                np.array([traj["reward"] for traj in sample_batch]),
                dtype=torch.float32, device=device
            )  # Shape: (batch_size,)
            
            # Check if any of the array of any batch include nan or inf
            # Initialize log probability storage
            logPF_tensor = torch.zeros(len(sample_batch), device=device)
            logPB_tensor = torch.zeros(len(sample_batch), device=device)

            # -----------------------------
            # 1. Forward Pass (logPF)
            # -----------------------------
            current_states = states_batch[:, :-1, :]  # Shape: (batch_size, trajectory_length, state_dim)
            actions = actions_batch                   # Shape: (batch_size, trajectory_length, action_dim)

            # Batch processing of get_policy_dist
            policy_dist, _ = get_policy_dist(
                forward_model, current_states , #.reshape(-1, current_states.shape[-1]), 
                off_policy_noise=0.0,
                dist_type=distribution_type,
                extra_params={
                    'mixture_components': mixture_components,
                    'num_variables': input_dim - 1,
                } if is_mixture else None
            )

            # Compute log probabilities for the entire batch
            step_logprob_f = policy_dist.log_prob(actions*0.999)  # Shape: (batch_size * trajectory_length,)

            # Reshape back to (batch_size, trajectory_length) and sum over time dimension
            logPF_tensor = step_logprob_f.sum(dim=1)

            # -----------------------------
            # 2. Backward Pass (logPB)
            # -----------------------------
            next_states = states_batch[:, 1:, :]  # Shape: (batch_size, trajectory_length, state_dim)
            prev_states = states_batch[:, :-1, :]  # Shape: (batch_size, trajectory_length, state_dim)

            # Compute backward actions (change in state)
            action_back = prev_states[:, :, 1:] - next_states[:, :, 1:]  # Ignore time dim for action
            action_back = action_back * 0.999 # Used to be 100 # Scale back to original scale
            # Batch processing of get_policy_dist for backward model
            policy_dist_back, _ = get_policy_dist(
                backward_model, next_states[:, 1:, :],  # Exclude initial state
                off_policy_noise=0.0,
                dist_type=distribution_type,
                extra_params={
                    'mixture_components': mixture_components,
                    'num_variables': input_dim - 1,
                } if is_mixture else None
            )

            # Compute log probabilities
            step_logprob_b = policy_dist_back.log_prob(action_back[:, 1:, :])  # Exclude initial state
            
            # Reshape back to (batch_size, trajectory_length-1) and sum over time dimension
            logPB_tensor = step_logprob_b.sum(dim=1)

            # ----------------------------
            # 3. Convert to tensors
            # ----------------------------
            logPF_tensor = logPF_tensor.squeeze()
            logPB_tensor = logPB_tensor.squeeze()
            # log_rewards_tensor = rewards_batch.log()
            log_rewards_tensor = transform_log(rewards_batch)
            logger.info(f"logPF_tensor: {logPF_tensor}")
            logger.info(f"logPB_tensor: {logPB_tensor}")
            logger.info(f"log_rewards_tensor: {log_rewards_tensor}")
            # ----------------------------
            # 4. Compute losses & update
            # ----------------------------
            optimizer.zero_grad()

            # logZ is a learnable parameter
            # reward_loss = (log_rewards_tensor - logZ).pow(2).mean()
            # trajectory_balance_loss = (logPF_tensor - logPB_tensor).pow(2).mean()
            # loss = reward_loss + trajectory_balance_loss
            loss = (logZ + logPF_tensor/100 - logPB_tensor/100 - log_rewards_tensor).pow(2).mean()

            loss.backward()
            
            # Update every N steps (e.g., every 4 iterations)
            if (iteration + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Store metrics
            losses.append(loss.item())
            rewards_log.append(log_rewards_tensor.mean().item())

            # Calculate gradient norms
            grad_norms = []
            for p in list(forward_model.parameters()) + list(backward_model.parameters()) + [logZ]:
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())

            # Logging
            if iteration % 10 == 0:
                mean_loss = np.mean(losses[-10:])
                mean_reward = np.mean(rewards_log[-10:])
                if logger:
                    logger.info(
                    f"[Off-Policy Iter {iteration}] Loss={mean_loss:.4f}, "
                    f"logZ={logZ.item():.4f}, Mean Log Reward={mean_reward:.4f}, "
                    f"Median Log Reward={np.median(rewards_log[-10:]):.4f}, "
                    f"Maximum Log Reward={np.max(rewards_log[-10:]):.4f}, "
                    f"Minimum Log Reward={np.min(rewards_log[-10:]):.4f}, "
                    f"STD Log Reward={np.std(rewards_log[-10:]):.4f}, "
                    f"Buffer Size={len(replay_buffer)}"
                    )

                
                print(
                f"[Off-Policy Iter {iteration}] Total Loss={mean_loss:.4f}, "
                f"logZ={logZ.item():.4f}, Mean Log Reward={mean_reward:.4f}, "
                f"Median Log Reward={np.median(rewards_log[-10:]):.4f}, "
                f"Maximum Log Reward={np.max(rewards_log[-10:]):.4f}, "
                f"Minimum Log Reward={np.min(rewards_log[-10:]):.4f}, "
                f"STD Log Reward={np.std(rewards_log[-10:]):.4f}, "
                f"Buffer Size={len(replay_buffer)}\n"
                f"Gradient Norms - Min: {min(grad_norms):.4f}, Mean: {np.mean(grad_norms):.4f}, "
                f"Median: {np.median(grad_norms):.4f}, Max: {max(grad_norms):.4f}"
                )
            
            # (Optional) scheduler step if you have one
            if scheduler is not None:
                scheduler.step()

            # (Optional) MLflow logging
            if mlflow_logger and iteration % mlflow_logger.log_every_n_steps == 0:
                # You can also log the models periodically:
                mlflow_logger.log_model(forward_model, artifact_path=f"forward_model_iteration_{iteration}")
                mlflow_logger.log_model(backward_model, artifact_path=f"backward_model_iteration_{iteration}")


        # Final logging and artifacts
        if mlflow_logger and n_runs == 1:
            plot_and_log_metrics(losses, rewards_log, logZ, mlflow_logger)
            visualize_and_log_simulated_trajectories(
                mlflow_logger=mlflow_logger,
                trajectory_length=trajectory_length,
                n_trajectories=n_simulations,
                plot_title="Simulated Trajectories",
                logger = logger,
                model_path = model_path,
                distribution_type=distribution,
                extra_parameters={
                    'mixture_components': mixture_components,
                    'num_variables': input_dim - 1,
                },
                config = config

            )
            mlflow_logger.log_artifact(log_file_name)
        if mlflow_logger and n_runs > 1:
            fwd_model_path, bwd_model_path = get_model_paths(mlflow_logger)
            result_runs[run_idx] = {
                'trained_agent': forward_model,
                'trained_backward_agent': backward_model,
                'fwd_model_path': fwd_model_path,
                'bwd_model_path': bwd_model_path,
                'seed': random_seed,
            }
            dir_path = os.path.dirname(os.path.dirname(fwd_model_path))
            os.makedirs(dir_path, exist_ok=True)
            joblib.dump(result_runs, os.path.join(dir_path, f"results_run.joblib"))


    return forward_model, backward_model, logZ, losses, rewards_log


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setup and run the training as before
    logger, log_file_name = setup_logger('experiment')
    config = yaml.safe_load(open('config/run_params_3.yaml', 'r'))
    training_parameters = config.get("model")['training_parameters']
    model_parameters = config.get("model")['model_parameters']
    mlflow_parameters = config.get("mlflow")
    oracle = config.get("oracle")

    # All training parameters
    seed = training_parameters['seed']
    seed_all(seed)
    torch.set_float32_matmul_precision('high')  # if available

    n_iterations = training_parameters['n_iterations']
    batch_size = training_parameters['batch_size']
    learning_rate_model = training_parameters['learning_rate']
    learning_rate_logZ = training_parameters['learning_rate_logZ']
    optimizer_name = training_parameters['optimizer']
    init_exploration_noise = training_parameters['init_exploration_noise']
    trajectory_length = training_parameters['trajectory_length']
    exploration_percentage = training_parameters.get("exploration_percentage", 0.25)
    replay_capacity = training_parameters.get("replay_buffer_size", 5000)
    warmup_steps = training_parameters.get("warmup_steps", 100)
    distribution = training_parameters.get("distribution_type", "normal")
    is_mixture = training_parameters['mixture']
    mixture_components = training_parameters['mixture_components']
    init_type = training_parameters.get("initialization_type", "xavier_uniform")

    # Model Parameters
    initial_state,feature_names = load_initial_state(config)
    input_dim = model_parameters['input_size']
    hidden_dim = model_parameters['hidden_size']
    output_dim = model_parameters['output_size']
    num_layers = model_parameters['num_layers']
    dropout = model_parameters['dropout']
    activation = model_parameters['activation']
    use_batchnorm = model_parameters['use_batchnorm']
    use_layernorm = model_parameters['use_layernorm']
    # Oracle Parameters
    model_path = oracle['model_path']

    if is_mixture:
            output_dim = 2 * (input_dim - 1) * mixture_components + mixture_components * (input_dim - 1)
            logger.info(f"Mixture model with {mixture_components} components. Output dim set to {output_dim}")
    # Results parameters
    n_simulations = config.get("results")['n_simulations']

    # Initialize the model and other experiment components
    forward_model, backward_model, logZ, optimizer,scheduler = setup_experiment(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,num_layers=num_layers,
        activation=activation, dropout=dropout, use_batchnorm=use_batchnorm, use_layernorm=use_layernorm,
        optimizer_name=optimizer_name, device=device,
        learning_rate_model=learning_rate_model, learning_rate_logZ=learning_rate_logZ,
        logger=logger,init_type=init_type
    )

    mlflow_logger = MLflowLogger(config=config,
                                 model_signature=infer_sig(input_dim,forward_model),
                                 logger=logger)


    env = GeneralEnvironment(initial_state=initial_state, config=config, model=forward_model,
                              input_dim=input_dim, max_steps=trajectory_length,model_path=model_path)

    

    forward_model, backward_model, logZ, losses, rewards = train_off_policy_with_exploration(
        forward_model=forward_model,
        backward_model=backward_model,
        logZ=logZ,
        optimizer=optimizer,
        scheduler=scheduler,
        seed=seed,
        n_iterations=n_iterations,
        batch_size=batch_size,
        trajectory_length=trajectory_length,
        env=env,
        device=device,
        init_exploration_noise=init_exploration_noise,
        input_dim=input_dim,  # Exclude timestep, usual use case is 10
        mlflow_logger=mlflow_logger,
        n_simulations=n_simulations,
        replay_capacity=replay_capacity,  # Default value in function
        warmup_steps=warmup_steps,  # Default value in function
        logger=logger,
        log_file_name=log_file_name,
        distribution_type=distribution,
        n_runs = training_parameters.get("n_runs", 1),
    )


    mlflow_logger.end_run()


