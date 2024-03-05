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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def create_plot(phases, variable_name,data, keys):
    #plot
    handles = []
    last = 0
    color_map = ['r','g','b']

    for k in range(0,4):
        color_map = ['r','g','b']
        for i in range(k*3,k*3+3):
            color = color_map.pop()
            #breaking up lines that loop over
            for j in range(45,50):
                for p in range(1,len(phases[k][:,j])):
                    if phases[k][p,j] < phases[k][p-1,j]:
                        line, = plt.plot(phases[k][last:p,j],data[keys[i]][last:p,j],"-o", markersize=2, color=color, label=keys[i])
                        last = p+1
            handles.append(line)

        plt.ylabel(f'{variable_name} (rad)',fontsize=20)
        plt.xlabel('Phase (rad)',fontsize=20)
        plt.title(f"{variable_name} vs Phase",fontsize=20)
        plt.legend(handles=handles)
        plt.savefig(f'plot_{variable_name}_last5_{k}')
        handles = []
        plt.cla()

def create_plot_small(phases, variable_name,data,keys):
    #plot
    last = 0
    for k in range(0,4):
        #breaking up lines that loop over
        for j in range(45,50):
            for p in range(1,len(phases[k][:,j])):
                if phases[k][p,j] < phases[k][p-1,j]:
                    line, = plt.plot(phases[k][last:p,j],data[keys[k]][last:p,j],"-o", markersize=2, color='k', label=keys[k])
                    last = p+1
        plt.ylabel(f'{variable_name}',fontsize=20)
        plt.xlabel('Phase (rad)',fontsize=20)
        plt.title(f"{variable_name} vs Phase",fontsize=20)
        plt.savefig(f'plot_{variable_name}_last5_{k}')
        plt.cla()


def plot_data(variable_name, full_data, df_phase):
    data, smalldata, phases, new_vardata, samples = process_data(variable_name, full_data, df_phase)
    keys = full_data['dof_names']
    if new_vardata.shape[0]/samples>4:
        create_plot(phases, variable_name,data, keys)
    else:
        create_plot_small(phases, variable_name,smalldata,['1','2','3','4'])

def process_data(variable_name, full_data, df_phase):
    vardata = full_data[variable_name]
    keys = full_data['dof_names'] #for humanoid: ["right_hip_yaw", "right_hip_abad","right_hip_pitch","right_knee","right_ankle","left_hip_yaw","left_hip_abad","left_hip_pitch","left_knee","left_ankle"]
    n= 50 #16 for humanoid
    m=vardata.shape[1] #10 for humanoid
    data = dict.fromkeys(keys, np.empty((0,n)))
    smalldata = dict.fromkeys(['1','2','3','4'], np.empty((0,n)))
    samples = vardata.shape[0]/n
    #format data as dict of actuators, each with samples (m) x envs (n) array
    new_vardata = np.empty((0,n))

    phase_env1 = np.empty((0,n))
    phase_env2 = np.empty((0,n))
    phase_env3 = np.empty((0,n))
    phase_env4 = np.empty((0,n))
    phases = [phase_env1, phase_env2, phase_env3, phase_env4]

    for i in range(0,df_phase.shape[0], n):
        for p in range(len(phases)):
            phases[p] = np.vstack([phases[p],df_phase[i:i+n,p:p+1].T])
        new_vardata = np.vstack([new_vardata, vardata[i:i+n,:].T])

    if new_vardata.shape[0]/samples>4:
        for i in range(0,new_vardata.shape[0],m):
            for k in range(len(keys)):
                #CHANGE THIS TO CHANGE PLOT
                data[keys[k]]= np.vstack([data[keys[k]], new_vardata[i+k,:]])
    else:
        for i in range(0,new_vardata.shape[0],m):
            for k in range(4):
                #CHANGE THIS TO CHANGE PLOT
                smalldata[str(k+1)]= np.vstack([smalldata[str(k+1)], new_vardata[i+k,:]])

    return data, smalldata, phases, new_vardata, samples

def phase_plot(full_data, df_phase):
    plotted_joint = "lh_hfe"
    q, _, phases, _,_ = process_data('dof_pos_obs', full_data, df_phase)
    qd,_,_,_,_= process_data('dof_vel', full_data, df_phase)
    color_map = ['r','g','b']
    command_data, command_new_vardata, comamnd_m = reorg_data('commands', full_data)
    command_indexes = [0]

    for i in range(1,command_data['x'].shape[0]):
        if command_data['x'][i-1,0] !=command_data['x'][i,0]:
            command_indexes.append(i)
    #print(command_indexes)

        #plot
    handles = []
    #int(len(keys)/4)
    color_map = ['r','g','b']
    for k in range(1,len(command_indexes)):
        color = color_map.pop()
        #print('here')
        for j in range(45,50):
            line, = plt.plot(q[plotted_joint][command_indexes[k-1]:command_indexes[k],j],qd[plotted_joint][command_indexes[k-1]:command_indexes[k],j],"-o", markersize=2, color=color, label=command_indexes[k])
        handles.append(line)

    plt.ylabel(f'qd',fontsize=20)
    plt.xlabel('q',fontsize=20)
    plt.title(f"Phase Plot",fontsize=20)
    plt.legend(handles=handles)
    plt.savefig(f'Phase Plot')
    #plt.show()
    handles = []
    plt.cla()

def time_plot(full_data, df_phase):
    q, _,phases,_,_ = process_data('dof_pos_obs', full_data, df_phase)
    qd,_,_,_,_ = process_data('dof_vel', full_data, df_phase)
    keys = full_data['dof_names'] #for humanoid: ["right_hip_yaw", "right_hip_abad","right_hip_pitch","right_knee","right_ankle","left_hip_yaw","left_hip_abad","left_hip_pitch","left_knee","left_ankle"]
    color_map = ['r','g','b']

    handles = []
    #int(len(keys)/4)
    for k in range(0,4):
        color_map = ['r','g','b']
        for i in range(k*3,k*3+3):
            color = color_map.pop()
            #breaking up lines that loop over
            for j in range(45,50):
                line, = plt.plot(q[keys[i]][:,j],"-o", markersize=2, color=color, label=keys[i])
            handles.append(line)
        plt.ylabel(f'q',fontsize=20)
        plt.xlabel('time',fontsize=20)
        plt.title(f"Time Plot",fontsize=20)
        plt.legend(handles=handles)
        #plt.show()
        plt.savefig(f'Time Plot_leg{k}')
        handles = []
        plt.cla()

def check_means(variable_name, full_data, df_phase):
    data, phases,_,_,_ = process_data(variable_name, full_data, df_phase)
    keys = full_data['dof_names']
    means = []
    std=[]
    for k in keys:
        d = np.reshape(data[k],(1,data[k].shape[0]*data[k].shape[1]))
        means.append(np.mean(d))
        std.append(np.std(d))
    means = np.array(means)
    std = np.array(std)

    #comparison
    means_comp = np.empty((4,4))
    std_comp=np.empty((4,4))
    for i in range(0,4):
        for j in range(0,4):
            means_comp[i,j] = np.allclose(means[i*3:i*3+3],means[j*3:j*3+3], atol=1e-1)
            std_comp[i,j] = np.allclose(std[i*3:i*3+3],std[j*3:j*3+3], atol=0.6)
    print(keys)
    print(means_comp)
    print(std_comp)
    print(means)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def reorg_data(variable_name, full_data):
    vardata = full_data[variable_name]
    n= 50 #16 for humanoid
    #format data as dict of actuators, each with samples (m) x envs (n) array
    new_vardata = np.empty((0,n))
    keys = ['x','y','z']
    data = dict.fromkeys(keys, np.empty((0,n)))
    m=vardata.shape[1] #10 for humanoid
    #(vardata.shape)
    for i in range(0,vardata.shape[0], n):
        new_vardata = np.vstack([new_vardata, vardata[i:i+n,:].T])

    if m >1:
        for i in range(0,new_vardata.shape[0],m):
            for k in range(3):
                data[keys[k]]= np.vstack([data[keys[k]], new_vardata[i+k,:]])
    return data, new_vardata, m

def reorg_plot_data(variable_name, full_data):
    data, new_vardata, m = reorg_data(variable_name, full_data)
    if m>1:
        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111, projection='3d')
        vals = np.hstack((data['x'][:,0:1],data['y'][:,0:1],data['z'][:,0:1]))
        #print(vals)
        ax.scatter3D(data['x'][:,0:1],data['y'][:,0:1],data['z'][:,0:1])
        a = Arrow3D([0,1], [0,0], [0,0], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
        ax.add_artist(a)
        a = Arrow3D([0,0], [0,1], [0,0], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
        ax.add_artist(a)
        a = Arrow3D([0,0], [0,0], [0,1], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
        ax.add_artist(a)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_aspect('equal')

        plt.savefig(f'plot_{variable_name}')
        #plt.show()
    else:
        fig = plt.figure(figsize=(10,10))
        x = np.arange(new_vardata[:,0:1].shape[0])
        plt.plot(x,new_vardata[:,0])
        plt.savefig(f'plot_{variable_name}')




data = dict(np.load("new_logs.npz"))
data_names = list(data.keys())
data['reward'] = np.reshape(data['reward'], (data['reward'].shape[0],1))
# for d in data_names[0:5]:
#     print(d)
#     plot_data(d, data, data['oscillators'])
#     #check_means(d, data, data['oscillators']) #ONLY WORKS FOR data with all 12 dof
# time_plot(data, data['oscillators'])
# phase_plot(data, data['oscillators'])
for d in data_names[5:-1]:
    print(d)
    reorg_plot_data(d, data)


