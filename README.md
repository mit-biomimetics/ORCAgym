# gpuGym - a clone of legged_gym #
This repository is a port of legged_gym from the good folk over at RSL.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.

---
### Useful Links
Project website: https://leggedrobotics.github.io/legged_gym/
Paper: https://arxiv.org/abs/2109.11978

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Clone and initialize this repo
   - clone `gpu_gym`
3. Install GPU Gym Requirements:
```bash
pip install -r requirements.txt
```
4. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (Preview 3 should still work) from https://developer.nvidia.com/isaac-gym
     - Extract the zip package
     - Copy the `isaacgym` folder, and place it in a new location
   - Install `issacgym/python` requirements
   ```bash
   cd <issacgym_location>/python
   pip install -e .
   ```
5. Run an example to validate
    - Run the following command from within isaacgym
   ```bash
   cd <issacgym_location>/python/examples
   python 1080_balls_of_solitude.py
   ```
   - For troubleshooting check docs `isaacgym/docs/index.html`
6. Install gpuGym
    ```bash
    pip install -e .
    ```
7. Use WandB for experiment tracking - follow [this guide](https://docs.wandb.ai/quickstart)

---

### CUDA Installation for Ubuntu 22.04 and above
#### Inspired by: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

1. Ensure Kernel Headers and Dev packages are installed
```bash
sudo apt-get install linux-headers-$(uname -r)
```

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
## Use ##
### Train ###  
```python gym/scripts/train.py --task=mini_cheetah_ref```
-  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
-  To run headless (no rendering) add `--headless`.
- **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
- The trained policy is saved in `<gpuGym>/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
-  The following command line arguments override the values set in the config files:
 - --task TASK: Task name.
 - --resume:   Resume training from a checkpoint
 - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
 - --run_name RUN_NAME:  Name of the run.
 - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
 - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
 - --num_envs NUM_ENVS:  Number of environments to create.
 - --seed SEED:  Random seed.
 - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.


### Play (a trained policy) ###  
```python <gpuGym>/scripts/play.py --task=mini_cheetah_ref```
- By default the loaded policy is the last model of the last run of the experiment folder.
- Other runs/model iteration can be selected by setting `--load_run` and `--checkpoint`.



---
### Jenny's gpuGym weights crash course ###
https://hackmd.io/@yHrQmxajTZOYt87bbz6YRg/SJbscyN3t

----
### Adding a new environment ###
https://github.com/mit-biomimetics/gpuGym/wiki/Getting-Started#adding-a-new-environment

---
### WandB Overview ###
https://github.com/mit-biomimetics/gpuGym/wiki/WandB

---
### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`
2. If you get the following error: `RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error?`, try [restarting your computer](https://discuss.pytorch.org/t/solved-torch-cant-access-cuda-runtimeerror-unexpected-error-from-cudagetdevicecount-and-error-101-invalid-device-ordinal/115004).

