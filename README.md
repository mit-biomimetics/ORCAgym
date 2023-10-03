# ORCAgym - based on gpuGym, legged_gym #
This repository is a bare-bones version of gpuGym for our ICRA 2024 submission, which is a port of legged_gym from the good folk over at RSL ([code](https://github.com/leggedrobotics/legged_gym), [website](https://leggedrobotics.github.io/legged_gym/), [paper](https://arxiv.org/abs/2109.11978)).  
Aside form the paper-specific robot implementations, this fork differs in some substantial reworking of how the environment and policy runner interact.  
Feel free to open issues both about the code and the paper.  

- [See Parameters](#reading-the-code)
- [Run Experiments](#reproducing-paper-results)
- [Install](#installation)

## Reading the Code ##
The overall code is organized in a similar fashion to legged_gym, in case you're familiar with it.
We recommend first reading the config file, `gym/envs/mini_cheetah/mini_cheetah_osc_config.py`.
In particular, search for:
- `control` to see/change PD gains and control frequency
- `osc` to see/change oscillator parameters
- `policy` to see
  - neural network details
  - actor and critic observations
  - actions
  - `weights` for reward weights
- `algorithm` for PPO hyperparameters
Next, see the file `gym/envs/mini_cheetah/mini_cheetah_osc.py` for implementation details on the reward (all starting with `def _reward_XXX()` where `XXX` matches the `weights` name in the config file).
Oscillator-related code is in the functions `compute_osc_slope()` (see equation 5 in paper) and `_step_oscillators()`. Gait-similarity rewards were used for internal evaluation but not used in the paper.
## Reproducing Paper Results ##
### Evaluated trained ORC Policies from the Paper ###
- All scripts are in "gym/scripts"
- You can either download the pre-trained policies [here](https://www.dropbox.com/sh/ohfojoek29efir6/AAB5pySq--r1Fwm90CUoB7o2a?dl=0), or train them locally by running `python train_ORC_all.py` (in "gym/scripts"), which will iterate over the training with all ORC combinations once. For the entire set of policies used in the paper, this was run 10 time (or rather, change l48 to `all_toggles = 10*['000', '010', '011', '100', '101', '110', '111']`), but this will take a long time, roughly overnight on a GTX4090...
  - If you download the policies, make a directory "logs" inside ORCAgym, and copy all subfolders ("ORC_xxx_FullSend") into it
- cd into ORCAgym/gym/scripts folder
- ```python play_ORC.py --task=mini_cheetah_osc --ORC_toggle=<xxx>```
   - By default the loaded policy is the last model of the last run of the experiment folder corresponding to the ORC_toggle
   - Other runs/model iteration can be selected by setting `--load_run` and `--checkpoint`.
### Running disturbance rejection experiments (Section IV.C)
- to generate the data used to evaluate disturbance rejection, run `python ORC_protocol_pushball.py`.
  - You may want to reduce the number of robots, by default we simulate 1800 robots (used in the paper) which is quite slow
- to evaluate the results, run `python ORC_pushball_analysis.py`

### Training ###  
```python gym/scripts/train.py --task=mini_cheetah_osc```
-  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
-  To run headless (no rendering) add `--headless`.
- **Important**: To improve performance, (if not used `headless`) once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
- The trained policy is saved in `<gpuGym>/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
-  The following command line arguments override the values set in the config files:
 - --task TASK: Task name, in our case `mini_cheetah_osc`.
 - --resume:   Resume training from a checkpoint
 - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
 - --run_name RUN_NAME:  Name of the run.
 - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
 - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
 - --num_envs NUM_ENVS:  Number of environments to create.
 - --seed SEED:  Random seed.

### Installation ###
1. Create a new python virtual env with python 3.8
2. Clone and initialize this repo
   - clone `gpu_gym`
3. Install GPU Gym Requirements:
```bash
pip install -r requirements.txt
```
1. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (Preview 3 should still work) from https://developer.nvidia.com/isaac-gym
     - Extract the zip package
     - Copy the `isaacgym` folder, and place it in a new location
   - Install `issacgym/python` requirements
   ```bash
   cd <issacgym_location>/python
   pip install -e .
   ```
2. Run an example to validate
    - Run the following command from within isaacgym
   ```bash
   cd <issacgym_location>/python/examples
   python 1080_balls_of_solitude.py
   ```
   - For troubleshooting check docs `isaacgym/docs/index.html`
3. Install gpuGym
    ```bash
    pip install -e .
    ```
4. Use WandB for experiment tracking - follow [this guide](https://docs.wandb.ai/quickstart)

---
#### CUDA Installation for Ubuntu 22.04 and above ####

Inspired by: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

1. Ensure Kernel Headers and Dev packages are installed
```bash
sudo apt-get install linux-headers-$(uname -r)
```****

2. Install nvidia-cuda-toolkit
```bash
sudo apt install nvidia-cuda-toolkit
```

3. Remove outdated Signing Key
```bash
sudo apt-key del 7fa2af80
```

4. Install CUDA
```bash
# ubuntu2004 or ubuntu2204 or newer.
DISTRO=ubuntu2204
# Likely what you want, but check if you need otherse
ARCH=x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
```

5. Reboot, and you're good.
```bash
sudo reboot
```

6. Use these commands to check your installation
```bash
nvidia-smi
nvcc --version
```

**Troubleshooting Docs**
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

---