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
all_toggles = ['000', '001', '010', '011', '100', '101', '110', '111']
colors = ['lightsteelblue', 'mediumaquamarine', 'yellowgreen', 'forestgreen', 'sandybrown', 'peru', 'indianred', 'crimson']
toggle_settings = dict(zip(all_toggles, colors))
base_path = '/home/heim/Repos/trigym/gym/scripts/'
data_folder = 'FS_data'  # Adjust to your actual data folder name
obs2use = ['base_lin_vel', 'base_ang_vel', 'commands']

# Initialize data dictionary
data_dict = {}

fig, ax = plt.subplots()
# Load data

fail_rate = {}

for toggle in all_toggles:
    experiment_name = "ORC_" + toggle
    run_names = get_run_names(experiment_name, base_path)

    fail_rate[toggle] = []

    for run_name in run_names:
        data_path = os.path.join(base_path, data_folder, 'ORC_'+toggle, f"{run_name}_data.npz")
        with np.load(data_path, allow_pickle=True) as loaded_data:
            data_dict = {key: loaded_data[key] for key in obs2use}
        fails = np.sum(data_dict['commands'][:, -2, 0] < 0.1)
    # * plotting and analysis
        # print(f"Toggle: {toggle}, Run: {run_name}, Failed: {fails}")

        ax.plot(data_dict['base_lin_vel'][:, :, 0].T,
                linewidth=1,
                color=toggle_settings[toggle],
                alpha=0.5)

        os.makedirs("FS_plots", exist_ok=True)
        plt.savefig(os.path.join("FS_plots", f"{run_name}.png"))
        fail_rate[toggle].append(fails)

        ax.clear()

        # # Explicitly delete the large objects to free memory
        # del data_dict
        # del loaded_data
        # gc.collect()  # Run the garbage collector
    print(' ')

for dataset_name, fails in fail_rate.items():
    a = np.array(fails)
    mean_val = a.mean()
    std_val = a.std()
    print(f"{dataset_name}: {mean_val:.4f} +/- {std_val:.4f}")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Plotting
# fig, ax = plt.subplots()
# legend_handles = []

# for (toggle, run_name), data in data_dict.items():
#     print(f"Toggle: {toggle}, Run: {run_name}, Failed: {np.sum(data['commands'][:, -2, 0] < 1)}")

#     ax.plot(data['base_lin_vel'][:, :, 0].T,
#             linewidth=1,
#             color=toggle_settings[toggle],
#             alpha=0.5)
#     plt.savefig(f"{run_name}.png")

