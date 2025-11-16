# GRID: GFlowNet-based Financial Risk Discovery

This repository contains the code for the ICLR 2026 paper on **GFlowNet-based discovery of financial stress trajectories**. The project trains forward and backward generative policies over macro/market variables, guided by a learned oracle of financial risk, and logs all runs and diagnostics with MLflow.

---

## 1. Project Overview

The main training entrypoint is `src/GFlownet.py`. At a high level:

- A **forward policy** and **backward policy** (`MLPModel` in `src/models.py`) are trained with a trajectory-balance style objective over a financial environment (`GeneralEnvironment` in `src/environments.py`).
- A pre-trained **oracle regressor** (in `oracles/*.joblib`) maps terminal states to a scalar reward (e.g., SPX drawdown proxy).
- Training is configured via YAML files in `config/` (e.g., `run_params_1.yaml`–`run_params_3.yaml`).
- Runs, models, plots, and SHAP explanations are logged with MLflow and can be browsed through the MLflow UI.

Typical workflow:

1. Start an MLflow tracking server (local).
2. Choose a config (oracle 1–3) in `config/run_params_*.yaml`.
3. Run `python src/GFlownet.py`.
4. Inspect the run in the MLflow UI (`http://localhost:5555`).

---

## 2. Installation

### 2.1. Clone and create a virtual environment

```bash
git clone https://github.com/anony-sub-papers/GRID.git
cd GRID

python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate  # on Windows (PowerShell / cmd)
```

### 2.2. Install dependencies via `pyproject.toml`

This project uses PEP 621/`pyproject.toml` with standard scientific and ML libraries (PyTorch, MLflow, scikit-learn, SHAP, etc.). Install them with:

```bash
pip install --upgrade pip
pip install .
```

or, equivalently:

```bash
pip install -e .
```

Key dependencies (see `pyproject.toml`):

- `torch`, `torchvision`
- `mlflow`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`, `plotly`
- `shap`, `tqdm`

---

## 3. Running the MLflow server

Training assumes an MLflow server is running locally and that artifacts are stored under `./mlruns`.

From the project root (`GRID/`), start MLflow as:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5555
```

Then open the MLflow UI at:

```text
http://localhost:5555
```

The YAML configs under `config/` expect `tracking_uri: "http://localhost:5555"` and will write runs into experiment IDs under `mlruns/`.

---

## 4. Running training

### 4.1. Select a configuration

The main configs are:

- `config/run_params_1.yaml` – Oracle 1 scenario (e.g., early 2000s stress case).
- `config/run_params_2.yaml` – Oracle 2 scenario (e.g., 2007 crisis-style setup).
- `config/run_params_3.yaml` – Oracle 3 scenario (e.g., 2022 inflation regime).

Each file controls:

- `oracle`: which oracle regressor to use (`oracle_1_MLPRegressor.joblib`, `oracle_2_MLPRegressor.joblib`, etc.).
- `mlflow`: experiment name, run name, logging frequency, and whether to save artifacts.
- `model.training_parameters`: GFlowNet hyperparameters.
- `model.model_parameters`: architecture of the forward / backward MLPs.
- `initial_state` and `variables`: the macro/market state and reward functional.

By default, `src/GFlownet.py` reads `config/run_params_3.yaml`. To switch scenarios, modify the path at the bottom of `src/GFlownet.py` or override via your own script.

### 4.2. Launch training

With the virtual environment active and MLflow server running:

```bash
cd GRID
source .venv/bin/activate
python src/GFlownet.py
```

You should see progress bars over iterations, logging of losses/rewards, and an MLflow link printed at the end of the run. In the MLflow UI, the run will appear under the experiment specified in the config (e.g., `GFlowNet_Financial_Risk_Discovery_Oracle1`).

---

## 5. Configuration and key parameters

All high-level experiment control is done through YAML files in `config/`.

### 5.1. Oracle block

```yaml
oracle:
  ML: true
  model_path: "oracles/oracle_1_MLPRegressor.joblib"
```

- `ML`: whether to use an ML-based oracle.
- `model_path`: path to a pre-trained scikit-learn regressor under `oracles/`.

### 5.2. MLflow block

```yaml
mlflow:
  tracking_uri: "http://localhost:5555"
  experiment_name: "GFlowNet_Financial_Risk_Discovery_Oracle1"
  run_name: "Oracle1Run_Final"
  save_artifacts: true
  log_every_n_steps: 200
  log_params: true
  log_metrics: true
  log_artifacts: true
  artifact_path: "artifacts"
```

- `tracking_uri`: where the MLflow server is running.
- `experiment_name`: logical grouping of related runs.
- `run_name`: human-friendly name visible in the UI.
- `save_artifacts`: whether to persist models/plots/CSVs.
- `log_every_n_steps`: frequency (in iterations) for logging models.
- `artifact_path`: top-level artifact folder within a run.

### 5.3. Training parameters

```yaml
model:
  training_parameters:
    n_runs: 1
    seed: 111
    n_iterations: 201
    batch_size: 16
    learning_rate: 0.0005
    learning_rate_logZ: 0.0005
    optimizer: "adam"
    init_exploration_noise: 20
    trajectory_length: 12
    exploration_percentage: 0.0
    replay_buffer_size: 5000
    warmup_steps: 0
    distribution_type: "mixture_beta"
    mixture: true
    mixture_components: 15
    initialization_type: "uniform"
```

- `n_runs`: number of independent random seeds for the same config.
- `seed`: base seed for reproducibility.
- `n_iterations`: GFlowNet training iterations.
- `batch_size`: number of trajectories per iteration.
- `learning_rate`, `learning_rate_logZ`: LR for network weights and log-partition parameter.
- `optimizer`: currently `"adam"` or `"sgd"`.
- `init_exploration_noise`: starting scale for off-policy exploration noise.
- `trajectory_length`: number of time steps per generated trajectory.
- `exploration_percentage`: fraction of training iterations using exploratory actions.
- `replay_buffer_size`, `warmup_steps`: off-policy replay buffer capacity and warmup.
- `distribution_type`: sampling distribution (e.g., `normal`, `mixture_beta`).
- `mixture`, `mixture_components`: whether to use mixture distributions and how many components.
- `initialization_type`: weight init scheme for MLPs.

### 5.4. Model architecture

```yaml
model_parameters:
  input_size: 7
  output_size: 12
  hidden_size: 256
  num_layers: 3
  dropout: 0.0
  activation: "leaky_relu"
  use_batchnorm: false
  use_layernorm: true
```

These map directly to the `MLPModel` constructor in `src/models.py` and control the shared architecture for the forward and backward policies.

### 5.5. Initial state, variables, and reward

```yaml
initial_state:
  Volume_spx: 77.03
  Close_ndx: -5.95
  Volume_ndx: 100.0
  Close_vix: -62.24
  IRLTCT01USM156N: -29.55
  BAMLH0A3HYCEY: 100.0

reward:
  formula: "SPX_X"

variables:
  SPX_X:
    formula: "906.83 + SPX * 418.76 + SPX_Volume * 130.88 + ..."
    components:
      Volume_spx: { transform: false, method: "C", lag: 0 }
      Close_ndx: { transform: false, method: "C", lag: 0 }
      # ... etc.
```

- `initial_state`: normalized macro/market variables at time 0 of the trajectory.
- `variables`: symbolic specifications of derived variables used in the reward.
- `reward.formula`: which variable (e.g. `SPX_X`) constitutes the scalar reward.

The environment (`GeneralEnvironment` in `src/environments.py`) uses these definitions to evolve the state and compute rewards via the oracle.

---

## 6. Code structure

Key modules under `src/`:

- `GFlownet.py`: main script. Parses config, constructs environment and models, sets up MLflow, and runs the off-policy GFlowNet training loop.
- `models.py`: defines `MLPModel`, the shared architecture for forward and backward policies.
- `environments.py`: defines `GeneralEnvironment`, which:
  - Maintains the current macro/market state.
  - Applies actions to transition to new states.
  - Generates full feature vectors for the oracle and reward.
- `reward_calculator.py` (if present) / reward logic: transforms environment variables into scalar rewards.
- `feature_creator.py`: builds feature matrices from raw financial data.
- `mlflow_logger.py`: thin wrapper around MLflow that:
  - Starts runs.
  - Logs params/metrics.
  - Logs simple `.pth` models as artifacts (`forward_model_iteration_k`, `backward_model_iteration_k`).
- `utility_functions.py`: assorted utilities:
  - Seeding (`seed_all`), loading initial state, and setting up loggers.
  - Model (de)serialization helpers (`load_entire_model`, `get_model_paths`).
  - Plotting and logging of losses/rewards (`plot_and_log_metrics`).
  - Trajectory simulation and visualization (`visualize_and_log_simulated_trajectories`).
  - SHAP-based interpretability helpers for oracle-based rewards.
- `replay_buffer.py`: generic replay buffer components; for the main script, an in-file `TrajectoryReplayBuffer` is used.

Additional utilities under `utilities/` support comparison baselines (BayesOpt, REINFORCE, SMCMC) and visualization for the paper.

---

## 7. Reproducing paper figures

The `Notebooks/` directory contains Jupyter notebooks used to generate most of the plots and tables in the paper:

- `Notebooks/building_oracle.ipynb`: oracle training / calibration.
- `Notebooks/evaluation.ipynb`: evaluation of trained GFlowNet policies and risk trajectories.
- `Notebooks/verifying_cases.ipynb`: detailed case studies for particular historical episodes.

Recommended workflow:

1. Run training with the desired config to populate `mlruns/` and `artifacts/`.
2. Start a Jupyter kernel in the same virtual environment:

   ```bash
   jupyter notebook
   ```

3. Open the relevant notebook and set the `root` and `run_id` to match the MLflow run you want to analyze.

---

## 8. Citation

This repository accompanies a paper submitted to  2026. If you use this code in academic work, please cite the paper once it is available.

> **BibTeX entry will be added after acceptance.** ✨

---

## 9. Troubleshooting

- **MLflow artifacts not visible:**
  - Ensure the server was started with `--default-artifact-root ./mlruns` from the project root.
  - Confirm `save_artifacts: true` and `log_artifacts: true` in the YAML config.
- **`FileNotFoundError` loading `model.pth`:**
  - Check that the run actually logged `forward_model_iteration_*` and `backward_model_iteration_*` folders under the run’s `artifacts/` directory.
- **CUDA / GPU issues:**
  - The code uses `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. On CPU-only machines, everything runs on CPU automatically, just more slowly.

If something is unclear or you need a minimal example config for a toy run, feel free to open an issue in the repository. 🙂
