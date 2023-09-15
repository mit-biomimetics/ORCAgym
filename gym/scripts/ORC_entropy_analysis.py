import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
from entro import sliding_window_entropy_multi_trial
import osc_plotters as oscplt

def get_run_names(experiment_name, base_path, data_folder):
    experiment_path = os.path.join(base_path, data_folder, experiment_name)
    return [os.path.basename(f).replace('_data.npz', '') for f in glob.glob(os.path.join(experiment_path, '*_data.npz'))]

# Initialize variables to store global min and max for each variable
global_min = None
global_max = None

all_toggles = ['111', '110', '100', '000', '001', '010', '011', '101']
# colors = ['lightsteelblue', 'yellowgreen', 'indianred', 'crimson']
# linestyle_dict = {'0': (0, (3, 1, 1, 1, 1, 1)), '1': 'solid'}
# linewidth_dict = {'0': 2, '1': 3}
plotting_options = {

    '000': {'color': 'cornflowerblue',
            'linestyle': (0, (3, 1, 1, 1, 1, 1)),
            'linewidth': 2},
    '100': {'color': 'yellowgreen',
            'linestyle': 'solid',
            'linewidth': 3},
    '110': {'color': 'goldenrod',
            'linestyle': (0, (3, 1, 1, 1, 1, 1)),
            'linewidth': 2},
    '111': {'color': 'crimson',
            'linestyle': 'solid',
            'linewidth': 3},
}



base_path = '/home/heim/Repos/trigym/gym/scripts/'
data_folder = 'FS_data'
obs2use = ['base_lin_vel', 'base_ang_vel', 'base_height']  # , 'commands']

# Changed from a tuple key to a dictionary of dictionaries
data_dict = {toggle: {} for toggle in all_toggles}

# Load all data and find global min and max for each variable
for toggle in all_toggles:
    experiment_name = "ORC_" + toggle
    run_names = get_run_names(experiment_name, base_path, data_folder)

    for run_name in run_names:
        data_path = os.path.join(base_path, data_folder, 'ORC_'+toggle, f"{run_name}_data.npz")
        with np.load(data_path, allow_pickle=True) as loaded_data:
            if len(obs2use) == 1:
                data = loaded_data[obs2use[0]]
            else:
                data = np.concatenate(list(loaded_data[name] for name in obs2use), axis=2)

        # Store in a nested dictionary
        data_dict[toggle][run_name] = data

        if global_min is None:
            global_min = np.min(data, axis=(0, 1))
            global_max = np.max(data, axis=(0, 1))
        else:
            global_min = np.minimum(global_min, np.min(data, axis=(0, 1)))
            global_max = np.maximum(global_max, np.max(data, axis=(0, 1)))

# Remaining code stays the same, with adjustments to use data_dict
# Create common bin edges based on global min and max
num_bins = [10, 10, 10]  # Update as needed
num_bins = [10] * data.shape[2]
custom_bin_edges = [np.linspace(global_min[i], global_max[i], num_bins[i] + 1)
                    for i in range(len(global_min))]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
window_size = 10  # Update as needed
step_size = 4  # Update as needed
sample_rate = 100/step_size

all_entropy_values = {toggle: [] for toggle in all_toggles}

# Loop through the toggles again for plotting
idx = 0
for toggle in all_toggles:
    for run_name, data in data_dict[toggle].items():

        assert len(num_bins) == data.shape[2], "Check your bins"

        entropy_values_multi_trial = sliding_window_entropy_multi_trial(
            data, window_size, step_size, 'digitize', custom_bin_edges)

        options = plotting_options[toggle]
        oscplt.entropy(ax, entropy_values_multi_trial,
                       f'Entropy for pushball perturbation',
                       color=options['color'],
                       linestyle=options['linestyle'],
                       linewidth=options['linewidth'],
                       sample_rate=sample_rate)

        all_entropy_values[toggle].append(entropy_values_multi_trial)
        idx += 1
        plt.tight_layout(rect=[0, 0, 0.75, 1])

# save all entropies to disk
save_path = os.path.join(base_path, f"all_entropy_values_ORC.npz")
np.savez(save_path, **all_entropy_values)

# Manually set the position of the axes to make room for the legend
ax.set_position([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]

# Place legend outside the plot and make it 2 columns
legend_handles = [matplotlib.lines.Line2D([0], [0],
                                          color=opts['color'],
                                          linestyle=opts['linestyle'],
                                          linewidth=opts['linewidth'])
                  for opts in plotting_options.values()]
legend = ax.legend(handles=legend_handles,
                   labels=['ORC_'+toggle for toggle in plotting_options.keys()],
                   loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

observations_text = "Observations used:\n" + "\n".join(obs2use)
fig.text(0.73, 0.6, observations_text, ha='left', va='top',
         bbox={'facecolor': 'white', 'edgecolor': 'grey',
               'boxstyle': 'round, pad=0.5'})  # , fontsize=10)

# Manually set the position of the axes to make room for the legend
ax.set_position([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]

# Adjust layout to not overlap with the legend
# plt.subplots_adjust(right=0.7)  # leaves room for the legend
plt.subplots_adjust(right=0.6, left=0.1, top=0.9, bottom=0.1)

# Save and show the plot
save_path = os.path.join(base_path, f"entropy_plot_all_ORC.png")
# plt.savefig(save_path)
plt.savefig(save_path, bbox_extra_artists=(legend,), bbox_inches='tight')
print(f"Saved plot to {save_path}")
plt.show()