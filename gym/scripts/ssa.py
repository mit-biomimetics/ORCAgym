import numpy as np
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

import os

from gym.envs import __init__
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry

# torch needs to be imported after isaacgym imports in local source
import torch
import pandas as pd
import numpy as np
import pandas as pd


def setup(args):
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.commands.resampling_time = 1
    env_cfg.env.episode_length_s = 9999
    env_cfg.env.num_projectiles = 20
    task_registry.make_gym_and_sim()
    env = task_registry.make_env(args.task, env_cfg)
    env.cfg.init_state.reset_mode = "reset_to_range"

    return env
args = get_args()
env = setup(args)

xls = pd.ExcelFile("/home/aileen/ORCAgym/gym/scripts/mini_cheetah_logs.xlsx")
q = pd.read_excel(xls, 'q').to_numpy()
qd = pd.read_excel(xls, 'qd').to_numpy()
tau = pd.read_excel(xls, 'tau').to_numpy()

df_phase = pd.read_excel(xls, 'oscillators_phase').to_numpy() #only 1 column
#df_feet_contacts = pd.read_excel(xls,'feet_contact_forces').to_numpy()

keys = env.dof_names #for humanoid: ["right_hip_yaw", "right_hip_abad","right_hip_pitch","right_knee","right_ankle","left_hip_yaw","left_hip_abad","left_hip_pitch","left_knee","left_ankle"]
n= 50 #16 for humanoid
m=12 #10 for humanoid
data = dict.fromkeys(keys, np.empty((0,n)))

#format data as dict of actuators, each with samples (m) x envs (n) array
new_q = np.empty((0,n))
new_qd = np.empty((0,n))
new_tau = np.empty((0,n))

phase_env1 = np.empty((0,n))
phase_env2 = np.empty((0,n))
phase_env3 = np.empty((0,n))
phase_env4 = np.empty((0,n))
phases = [phase_env1, phase_env2, phase_env3, phase_env4]
for i in range(0,df_phase.shape[0], n):
    for p in range(len(phases)):
        phases[p] = np.vstack([phases[p],df_phase[i:i+n,p:p+1].T])
    new_q = np.vstack([new_q, q[i:i+n,:].T])
    new_qd = np.vstack([new_qd, qd[i:i+n,:].T])
    new_tau = np.vstack([new_tau, tau[i:i+n,:].T])

for i in range(0,new_q.shape[0],m):
    for k in range(len(keys)):
        #CHANGE THIS TO CHANGE PLOT
        data[keys[k]]= np.vstack([data[keys[k]], new_q[i+k,:]]) 

n_samples, n_timestamps = 100, 48
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)
print(X.shape)

X = data[keys[0]]
print(X.shape)
# We decompose the time series into three subseries
window_size = 15
groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

# Singular Spectrum Analysis
ssa = SingularSpectrumAnalysis(window_size=15, groups=groups)
X_ssa = ssa.fit_transform(X)

# Show the results for the first time series and its subseries
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(121)
ax1.plot(X[0], 'o-', label='Original')
ax1.legend(loc='best', fontsize=14)

ax2 = plt.subplot(122)
for i in range(len(groups)):
    ax2.plot(X_ssa[0, i], 'o--', label='SSA {0}'.format(i + 1))
ax2.legend(loc='best', fontsize=14)

plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.

