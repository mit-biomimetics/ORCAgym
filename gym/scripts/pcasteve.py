import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from gym.envs import __init__
from gym import LEGGED_GYM_ROOT_DIR
from gym.utils import get_args, task_registry


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

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

args = get_args()
env = setup(args)

#import data
set_num = 4 #from 1-4 for 4 legs 
xls = pd.ExcelFile("/home/aileen/ORCAgym/gym/scripts/mini_cheetah_logs.xlsx")
df1 = pd.read_excel(xls, 'q').to_numpy()[:,(set_num-1)*3:set_num*3]
#df2 = pd.read_excel(xls, 'qd').to_numpy()
#tau = pd.read_excel(xls, 'tau').to_numpy()
#df_feet_contacts = pd.read_excel(xls,'grf').to_numpy()

#randomize
df1 = shuffle_along_axis(df1,0)
# df1 = np.random.permutation(df1[:,0])
#df1 = df1[np.random.permutation(np.arange(df1.shape[0])), :]
print(df1.shape)

#PCA
dataset = pd.DataFrame(df1)
features = env.dof_names[(set_num-1)*3:set_num*3]
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

print(pca.components_)
print(sum(pca.components_[:,0]*pca.components_[:,1]))
print(sum(pca.components_[:,1]*pca.components_[:,2]))
print(sum(pca.components_[:,2]*pca.components_[:,0]))
principal_Df = pd.DataFrame(data = principalComponents
            , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


#checks on orthogonality
import math  
a = pca.components_[:,0]
b = pca.components_[:,1]
c= pca.components_[:,2]
print(np.dot(a,b))
print(math.acos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
print(math.pi/2)
print(math.acos(np.dot(c,b)/(np.linalg.norm(c)*np.linalg.norm(b))))
print(math.acos(np.dot(c,a)/(np.linalg.norm(c)*np.linalg.norm(a))))

#plotting
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(normalised.loc[:,"feature0"],normalised.loc[:,"feature1"],normalised.loc[:,"feature2"],s=5)
    # #principal_Df.loc[:, 'principal component 1']
    #         , principal_Df.loc[:, 'principal component 2']
    #         ,principal_Df.loc[:, 'principal component 3'], s = 5)
coeff = pca.components_.T
print(coeff)
for i in range(3):
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='r', shrinkA=0, shrinkB=0)

        a = Arrow3D([0,coeff[i,0]], [0,coeff[i,1]], [0,coeff[i,2]], **arrow_prop_dict)
        ax.add_artist(a)
        #plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, coeff[i,2]*1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')

plt.title(f"Principal Component Analysis of Dataset",fontsize=20)
ax.set_xlabel(features[0],fontsize=20)
ax.set_ylabel(features[1],fontsize=20)
ax.set_zlabel(features[2],fontsize=20)
plt.show()


#plt.savefig(f'pca plot')
