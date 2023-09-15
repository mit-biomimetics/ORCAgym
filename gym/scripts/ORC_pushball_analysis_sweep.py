import numpy as np
import matplotlib.pyplot as plt
import os
from entro import sliding_window_entropy_multi_trial
import osc_plotters as oscplt
import gc
import glob

def get_run_names(experiment_name, base_path):
    experiment_path = os.path.join(base_path, data_folder, experiment_name)
    return [os.path.basename(f).replace('_data.npz', '') for f in glob.glob(os.path.join(experiment_path, '*_data.npz'))]


# Setup
all_toggles = ["000", "001", "100", "101", "010", "110", "011", "111"]

all_pushmags = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5']

colors = ['lightsteelblue', 'mediumaquamarine', 'yellowgreen', 'forestgreen', 'sandybrown', 'peru', 'indianred', 'crimson']
toggle_settings = dict(zip(all_toggles, colors))
base_path = '/home/heim/Repos/trigym/gym/scripts/'
data_folder = 'FS_data'  # Adjust to your actual data folder name
obs2use = ['base_lin_vel', 'base_ang_vel', 'commands']

# Initialize data dictionary
data_dict = {}

fig, ax = plt.subplots()
# Load data

# set up a dict of dicts, containing fail rates for each toggle, and each pushmag
fail_rate = {toggle: {pushmag: [] for pushmag in all_pushmags}
                 for toggle in all_toggles}

for toggle in all_toggles:
    for pushmag in all_pushmags:
        subpath_name = f"ORC_{toggle}"
        experiment_name = pushmag
        save_name = subpath_name + "_" + experiment_name
        run_names = get_run_names(os.path.join(subpath_name, experiment_name),
                                  base_path)

        for run_name in run_names:
            data_path = os.path.join(base_path, data_folder,
                                     subpath_name, experiment_name,
                                     run_name+"_data.npz")
            with np.load(data_path, allow_pickle=True) as loaded_data:
                # data_dict = {key: loaded_data[key] for key in obs2use}
                fails = np.sum(loaded_data['commands'][:, -2, 0] < 0.1)
        # * plotting and analysis
            # print(f"Toggle: {toggle}, Run: {run_name}, Failed: {fails}")

                # ax.plot(loaded_data['base_lin_vel'][:, :, 0].T,
                #         linewidth=1,
                #         color=toggle_settings[toggle],
                #         alpha=0.5)

                # os.makedirs("FS_plots", exist_ok=True)
                # plt.savefig(os.path.join("FS_plots", f"{save_name}.png"))
                fail_rate[toggle][pushmag].append(fails)

            ax.clear()
        # print(' ')

num_envs = 1800
for dataset_name, fails_for_dataset in fail_rate.items():
    print(f"{dataset_name}")
    for pushmag, fails in fails_for_dataset.items():
        a = np.array(fails)
        mean_val = a.mean()/num_envs*100
        std_val = a.std()/num_envs*100
        print(f"{pushmag}: {mean_val:.4f}% +/- {std_val:.4f}%")
    print(' ')

# * let's also save the entire fail_rate dict, so it is easier to load later
np.savez_compressed(os.path.join("FS_plots", "fail_rate.npz"), **fail_rate)