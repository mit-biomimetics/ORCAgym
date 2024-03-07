import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
df1 = pd.read_excel(xls, 'q').to_numpy()
df2 = pd.read_excel(xls, 'qd').to_numpy()
df_feet_contacts = pd.read_excel(xls,'grf').to_numpy()

print(env.dof_names)
#PCA
dataset = pd.DataFrame(df1)
features = env.dof_names
dataset.columns = features

features = np.char.mod('%s', features).tolist()
x = dataset.loc[:, features].values
print(dataset.loc[:, features].values.shape)
x = StandardScaler().fit_transform(x) # normalizing the features
print(x.shape)
print(np.mean(x),np.std(x))
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised = pd.DataFrame(x,columns=feat_cols)
print(normalised.tail())

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principal_Df = pd.DataFrame(data = principalComponents
            , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

print(principal_Df.tail())

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#print(swing_dataset['label'] )

plt.figure()
plt.figure(figsize=(10,10))
ax = plt.axes(projection ="3d")

plt.title(f"Principal Component Analysis of Dataset",fontsize=20)
ax.scatter3D(principal_Df.loc[:, 'principal component 1']
            , principal_Df.loc[:, 'principal component 2']
            ,principal_Df.loc[:, 'principal component 3'], s = 5)
ax.set_xlabel('Principal Component - 1',fontsize=20)
ax.set_ylabel('Principal Component - 2',fontsize=20)
ax.set_zlabel('Principal Component - 3',fontsize=20)
plt.show()
#plt.savefig(f'pca plot')