# LaRes: Evolutionary Reinforcement Learning with LLM-based Adaptive Reward Search

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=jRjvcqtdtA)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/yeshenpy/LaRes)

**LaRes** is a novel hybrid framework that achieves efficient policy learning through reward function search. It leverages Large Language Models (LLMs) to generate and improve reward function populations, guiding Reinforcement Learning (RL) in policy learning.

**LaRes is currently the most sample-efficient reward generation method :trophy: and also the state-of-the-art approach in [Evolutionary Reinforcement Learning (ERL)](https://github.com/yeshenpy/Awesome-Evolutionary-Reinforcement-Learning)**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Adding New Tasks](#adding-new-tasks)
- [Citation](#citation)

## 🎯 Overview

LaRes integrates Evolutionary Algorithms (EAs) with Reinforcement Learning (RL) to improve policy learning through adaptive reward function search. The framework includes two main training scripts:

- **`LaRes_from_scratch.py`**: Training without human-designed reward initialization
- **`LaRes_with_init.py`**: Training with human-designed reward initialization

### Key Components

1. **LLM-based Reward Generation**: Uses LLMs to generate a population of candidate reward functions
2. **Shared Replay Buffer**: Maintains experiences from all policies with multiple rewards per experience
3. **Reward Relabeling**: Enables efficient reuse of historical data when reward functions are updated
4. **Thompson Sampling**: Prioritizes interactions with superior policies
5. **Reward Scaling & Parameter Constraints**: Ensures training stability when reward functions change

## ✨ Features

- **Sample Efficiency**: Shared replay buffer with reward relabeling mechanism
- **Stability**: Reward scaling and parameter constraint mechanisms
- **Exploration-Exploitation Balance**: Thompson sampling-based interaction mechanism
- **Flexible Initialization**: Supports both with and without human-designed reward functions

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yeshenpy/LaRes.git
cd LaRes
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate Metaworld-v2
```

### 3. Install MetaWorld

Follow the instructions from the [EvoRainbow MetaWorld repository](https://github.com/yeshenpy/EvoRainbow/tree/main/MetaWorld):

```bash
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld/
git checkout 2361d353d0895d5908156aec71341d4ad09dd3c2
pip install -e .
cd ..
```

**Important**: Make sure to use the specific commit version (`2361d353d0895d5908156aec71341d4ad09dd3c2`) as different versions of MetaWorld may have incompatible APIs.

### 4. Install Additional Dependencies

```bash
pip install openai wandb scipy
```

### 5. Configure API Keys

Edit the main training files (`LaRes_from_scratch.py` or `LaRes_with_init.py`) and set your OpenAI API key:

```python
client = OpenAI(api_key="your-api-key-here")
```

For WandB logging, you can optionally set:

```python
os.environ["WANDB_API_KEY"] = "your-wandb-key"
```

## ⚙️ Configuration

### Environment Variables

The code supports the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `WANDB_API_KEY`: Your Weights & Biases API key (optional)
- `WANDB_MODE`: Set to `"offline"` for offline logging (optional)
- `OPENAI_BASE_URL`: Custom OpenAI API base URL (optional)

### CUDA Configuration

By default, the code runs on CPU. 
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use GPU 0
```

## 🚀 Usage

### Training from Scratch (No Human Reward Initialization)

```bash
python LaRes_from_scratch.py \
    --env-name='window-close-v2' \
    --buffer_transfer=0 \
    --scale_type=2 \
    --Inter_Loop_freq=1000000 \
    --ucb_type='max' \
    --windows_length=500 \
    --elite_num=3 \
    --RL_ucb=1 \
    --c=0.0 \
    --EA_tau=0.1 \
    --damp=1e-1 \
    --model=gpt-4o-mini \
    --total-timesteps=1000000 \
    --seed=1 \
    --eval-episodes=20 \
    --LLM_freq=200000
```

### Training with Human Reward Initialization

```bash
python LaRes_with_init.py \
    --env-name='coffee-pull-v2' \
    --buffer_transfer=0 \
    --scale_type=2 \
    --Inter_Loop_freq=1000000 \
    --ucb_type='max' \
    --pop_size=5 \
    --windows_length=20 \
    --elite_num=3 \
    --RL_ucb=1 \
    --c=0.0 \
    --EA_tau=0.1 \
    --damp=1e-1 \
    --model=gpt-4o-mini \
    --total-timesteps=1000000 \
    --seed=1 \
    --eval-episodes=20 \
    --LLM_freq=200000
```

### Using the Provided Scripts

We provide example scripts in `run.sh` for both settings. You can uncomment and modify the commands as needed:

```bash
# Edit run.sh to uncomment desired commands
bash run.sh
```

### Key Parameters

- `--env-name`: MetaWorld environment name (e.g., `'window-close-v2'`, `'button-press-v2'`)
- `--model`: LLM model to use (e.g., `'gpt-4o-mini'`, `'gpt-4'`)
- `--total-timesteps`: Total training timesteps
- `--LLM_freq`: Frequency of LLM reward function updates (in timesteps)
- `--pop_size`: Population size for evolutionary algorithm
- `--elite_num`: Number of elite individuals to preserve
- `--windows_length`: Window length for Thompson sampling
- `--seed`: Random seed for reproducibility

### Monitoring Training

Training logs are saved to `./logs/` directory. Each run creates a subdirectory named with the experiment configuration. You can monitor training progress through:

1. **WandB Dashboard**: If configured, training metrics are logged to Weights & Biases
2. **Log Files**: Check the log files specified in `run.sh` (e.g., `./logs/1.log`)
3. **Model Checkpoints**: Saved in `./logs/{experiment_name}/` directory

### Output Files

- **Reward Functions**: Generated reward function code is saved in `./logs/{experiment_name}/Iter_{LLM_iter}_Reward_Code_{index}.py`
- **Responses**: LLM responses are saved in `./logs/{experiment_name}/Iter_{LLM_iter}_Response_{index}.txt`
- **Model Checkpoints**: Best models are saved as `best_actor_net.pth`, `best_qf1.pth`, etc.

## 📝 Adding New Tasks

To add a new MetaWorld task, you need to configure four dictionaries in the training script. The format differs slightly between `LaRes_from_scratch.py` and `LaRes_with_init.py`:

### For LaRes_from_scratch.py

Add entries to the dictionaries defined in the main function:

#### 1. `task_description_dict`

Provides a natural language description of the task:

```python
task_description_dict = {
    "your-task-v2": "Description of what the robotic arm should do",
    # ... other tasks
}
```

#### 2. `reward_function_format_dict`

Defines the expected reward function signature (used by LLM to generate reward functions):

```python
reward_function_format_dict = {
    "your-task-v2": """    def compute_reward(param1, param2, param3, actions):        
        ...
        return reward, reward_component_dict""",
    # ... other tasks
}
```

#### 3. `criteria_code_dict`

Specifies the success criteria description (can be same as task description for from_scratch):

```python
criteria_code_dict = {
    "your-task-v2": "Description of success criteria",
    # ... other tasks
}
```

#### 4. `input_dict`

Lists the available input variables and their descriptions in JSON format:

```python
input_dict = {
    "your-task-v2": """{"param1": "Description of param1",
        "param2": "Description of param2",
        "actions": "Actions taken"}""",
    # ... other tasks
}
```

### For LaRes_with_init.py

The task information is imported from `utils.py`. You need to add entries to the dictionaries in `utils.py`:

#### 1. `task_description_dict` (in utils.py)

```python
task_description_dict = {
    "your-task-v2": "Description of what the robotic arm should do",
    # ... other tasks
}
```

#### 2. `criteria_code_dict` (in utils.py)

Specifies the success criteria code (Python code that evaluates success):

```python
criteria_code_dict = {
    "your-task-v2": """success = float(obj_to_target <= 0.05)
        near_object = float(tcp_to_obj <= 0.03)""",
    # ... other tasks
}
```

#### 3. `input_dict` (in utils.py)

Lists available variables in list format:

```python
input_dict = {
    "your-task-v2": """List = ["tcp_center", "obj", "_target_pos", "obs", "action"]""",
    # ... other tasks
}
```

#### 4. `reward_function_dict` (in utils.py)

Contains the human-designed reward function template (optional, for initialization):

```python
reward_function_dict = {
    "your-task-v2": """def compute_reward(action, obs, tcp_center, _target_pos, ...):
        # Reward function code
        return (reward, ...)""",
    # ... other tasks
}
```

#### 5. `parents_function_dict` (in utils.py)

Contains the gripper caging reward function (if needed):

```python
parents_function_dict = {
    "your-task-v2": """def _gripper_caging_reward(...):
        # Gripper caging reward code
        return caging_and_gripping""",
    # ... other tasks
}
```

### Example: Adding a New Task to LaRes_from_scratch.py

```python
# In LaRes_from_scratch.py, add to the dictionaries:

task_description_dict = {
    # ... existing tasks
    "new-task-v2": "Control the robotic arm to perform the new task"
}

reward_function_format_dict = {
    # ... existing tasks
    "new-task-v2": """    def compute_reward(tcp, obj, target, actions):        
        ...
        return reward, reward_component_dict"""
}

criteria_code_dict = {
    # ... existing tasks
    "new-task-v2": "Control the robotic arm to perform the new task"
}

input_dict = {
    # ... existing tasks
    "new-task-v2": """{"tcp": "Position of the robotic arm",
        "obj": "Position of the object",
        "target": "Target position",
        "actions": "Actions taken"}"""
}
```

### Example: Adding a New Task to LaRes_with_init.py

```python
# In utils.py, add to the dictionaries:

criteria_code_dict = {
    # ... existing tasks
    "new-task-v2": """success = float(obj_to_target <= 0.05)
        near_object = float(tcp_to_obj <= 0.03)"""
}

task_description_dict = {
    # ... existing tasks
    "new-task-v2": "Description of the task"
}

input_dict = {
    # ... existing tasks
    "new-task-v2": """List = ["tcp_center", "obj", "_target_pos", "obs", "action"]"""
}

# Optional: Add reward function if you have a human-designed one
reward_function_dict = {
    # ... existing tasks
    "new-task-v2": """def compute_reward(action, obs, tcp_center, _target_pos, obj):
        # Your reward function code here
        return (reward, ...)"""
}
```

### Required Information for New Tasks

To add a new task, you typically need:

1. **Task Description**: A clear description of what the robot should accomplish
2. **Success Criteria**: 
   - For `LaRes_from_scratch.py`: A description string
   - For `LaRes_with_init.py`: Python code that evaluates success (e.g., `success = float(obj_to_target <= 0.05)`)
3. **Available Variables**: What information is available from the environment. Common variables include:
   - `tcp_center` or `tcp`: Position of the robotic arm end-effector
   - `obj`: Position of the object
   - `_target_pos` or `target`: Target position
   - `obs`: Observation array
   - `action`: Action taken
   - `init_tcp`: Initial TCP position
   - `left_pad`, `right_pad`: Gripper pad positions
   - Task-specific variables (check MetaWorld environment documentation)
4. **Reward Function Template**: The expected function signature based on available variables

### Finding Available Variables

To find what variables are available for a new task:

1. **Check MetaWorld Documentation**: Each environment exposes different variables
2. **Inspect Existing Tasks**: Look at similar tasks in `utils.py` to see what variables they use
3. **Use Environment's `get_dict()` Method**: The code uses `env._env.get_dict()` to get available variables. You can add a debug print to see what's available:

```python
org_info = env._env.get_dict()
print("Available variables:", org_info.keys())
```

### Tips

- Start with a similar existing task and modify it
- Check `utils.py` for comprehensive examples of all supported tasks
- The `input_dict` format differs: JSON string format for `LaRes_from_scratch.py`, list format for `LaRes_with_init.py`
- For `LaRes_with_init.py`, you can optionally provide a human-designed reward function in `reward_function_dict` to help with initialization

## 📁 Project Structure

```
LaRes/
├── LaRes_from_scratch.py      # Main training script (without human reward initialization)
├── LaRes_with_init.py          # Main training script (with human reward initialization)
├── run.sh                      # Example training commands
├── environment.yaml            # Conda environment configuration
├── utils.py                    # Utility functions and task dictionaries (for LaRes_with_init.py)
├── sac.py                      # SAC algorithm implementation
├── models.py                   # Neural network models
├── replay_buffer.py            # Experience replay buffer
├── reward_utils.py             # Reward utility functions
├── arguments.py                # Command-line argument parser
├── test_generate_code.py       # Code generation testing utility
├── utils/
│   ├── prompts/                # Prompt templates (for LaRes_with_init.py)
│   │   ├── initial_system.txt
│   │   ├── new_initial_user.txt
│   │   ├── code_feedback.txt
│   │   └── ...
│   └── no_init_prompts/        # Prompt templates (for LaRes_from_scratch.py)
│       ├── initial_system.txt
│       ├── new_initial_user.txt
│       ├── code_feedback.txt
│       └── ...
└── logs/                       # Training logs and outputs (created during training)
```

## 📊 Supported Tasks

### From Scratch (LaRes_from_scratch.py)

- `window-close-v2`
- `window-open-v2`
- `button-press-v2`
- `door-close-v2`
- `drawer-open-v2`

### With Initialization (LaRes_with_init.py)

All tasks from "from scratch" plus:
- `coffee-pull-v2`
- `coffee-push-v2`
- `hand-insert-v2`
- `basketball-v2`
- `dial-turn-v2`
- `soccer-v2`
- `push-back-v2`
- `pick-out-of-hole-v2`
- `hammer-v2`
- `peg-unplug-side-v2`
- `peg-insert-side-v2`
- `button-press-topdown-v2`

## 📚 Citation

If you use LaRes in your research, please cite:

```bibtex
@inproceedings{
li2025lares,
title={LaRes: Evolutionary Reinforcement Learning with {LLM}-based Adaptive Reward Search},
author={Pengyi Li and Hongyao Tang and Jinbin Qiao and YAN ZHENG and Jianye HAO},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=jRjvcqtdtA}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Troubleshooting

### Common Issues

#### 1. MetaWorld Import Errors

If you encounter import errors with MetaWorld:

```bash
# Make sure you're using the correct commit
cd metaworld/
git checkout 2361d353d0895d5908156aec71341d4ad09dd3c2
pip install -e .
```

#### 2. OpenAI API Errors

- **Rate Limiting**: If you hit rate limits, the code will automatically retry with exponential backoff
- **API Key**: Make sure your API key is correctly set in the training script
- **Model Availability**: Ensure the specified model (e.g., `gpt-4o-mini`) is available in your OpenAI account

#### 3. CUDA/GPU Issues

By default, the code runs on CPU. To use GPU:

1. Modify `CUDA_VISIBLE_DEVICES` in the training script
2. Ensure PyTorch with CUDA support is installed
3. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

#### 4. Task Configuration Errors

If you get `KeyError` when running a new task:

- Ensure the task name matches exactly (including `-v2` suffix)
- Check that all required dictionaries (`task_description_dict`, `input_dict`, etc.) contain the task
- Verify the task exists in MetaWorld: `python -c "import metaworld; print('window-close-v2' in metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS)"`

#### 5. Reward Function Generation Failures

If reward functions fail to generate:

- Check the LLM response files in `./logs/{experiment_name}/Iter_*_Response_*.txt`
- Verify the prompt templates are correctly formatted
- Ensure the `input_dict` contains all variables used in the reward function format

### Debugging Tips

1. **Check Logs**: Always check the log files first for error messages
2. **Test Environment**: Test the environment separately before training:
   ```python
   import metaworld
   env = metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['window-close-v2']()
   obs, _ = env.reset()
   print("Environment works!")
   ```
3. **Verify Variables**: Add debug prints to see available variables:
   ```python
   org_info = env._env.get_dict()
   print("Available variables:", list(org_info.keys()))
   ```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact me at [lipengyi@tju.edu.cn](...).

## 🙏 Acknowledgments

- **MetaWorld**: [https://github.com/rlworkgroup/metaworld](https://github.com/rlworkgroup/metaworld)
- **EvoRainbow**: [https://github.com/yeshenpy/EvoRainbow](https://github.com/yeshenpy/EvoRainbow)
- **OpenAI**: For providing the LLM API

---

**Note**: Make sure to configure your OpenAI API key before running the training scripts. The code will use LLM API calls to generate and improve reward functions during training. Monitor your API usage to avoid unexpected costs.
