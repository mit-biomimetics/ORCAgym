import numpy as np

import matplotlib
# matplotlib.use("TkAgg")

import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

font = {'size': 24}
matplotlib.rc('font', **font)


# Function to plot phase vs phase velocity in a polar subplot with progressive colormap and legend
def plot_polar_phase(dataset, robot_idx, r_variable='grf', segment=None,
                              title=''):
    # n_points = len(ref_phase)

    def plot_individual_oscillator(ax, ref_phase, phase_velocity_data,
                                   oscillator_phase, title):
        n_points = len(ref_phase)
        # Create a progressive colormap for each cycle
        cmap = plt.cm.Set2
        num_cycles = np.sum(np.diff(oscillator_phase) < 0) + 1  # Count of phase wraps + 1
        colors = cmap(np.linspace(0, 1, num_cycles))

        start_idx = 0
        color_idx = 0

        # Loop through each point and plot with progressive color for each cycle
        for i in range(1, n_points):
            if oscillator_phase[i] < oscillator_phase[i-1]:  # Check for phase wrap to start new cycle
                ax.plot(ref_phase[start_idx:i+1], phase_velocity_data[start_idx:i+1], color=colors[color_idx % len(colors)], label=f"Cycle {color_idx+1}")
                color_idx += 1
                start_idx = i

        # Marking the zero-points and pi-crossings of the corresponding oscillator
        max_velocity = max(phase_velocity_data)
        zero_points = []
        pi_points = []
        for i in range(n_points):
            if oscillator_phase[i] < oscillator_phase[i-1]:
                zero_points.append(i)
            if oscillator_phase[i] > np.pi and oscillator_phase[i-1] < np.pi:
                pi_points.append(i)
        zero_points = np.array(zero_points)
        pi_points = np.array(pi_points)

        # * 
        # zero_points = np.where(np.diff(oscillator_phase) < 0)[0]
        # pi_points = np.where(np.abs(np.diff(oscillator_phase)) > np.pi)[0]
        for idx in zero_points:
            ax.plot([ref_phase[idx], ref_phase[idx]], [0, max_velocity],
                    color='red', alpha=0.5)
            ax.plot(ref_phase[idx], max_velocity, 'ro', markersize=5)
        for idx in pi_points:
            ax.plot([ref_phase[idx], ref_phase[idx]], [0, max_velocity],
                    color='blue', alpha=0.5)
            ax.plot(ref_phase[idx], max_velocity, 'bo', markersize=5)

        ax.set_title(title)
        ax.set_rlabel_position(-22.5)  # Move radial labels to avoid overlap with line
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # * ##################
    # unpack dataset
    ref_phase = dataset['oscillators'][robot_idx][:, 0]
    # phase_velocity = dataset['oscillators_vel'][robot_idx][:, 0]

    if segment is None:
        start_idx = 0
        end_idx = len(ref_phase)
    else:
        start_idx = int(segment[0] * fs)
        end_idx = int(segment[1] * fs)

    # Extracting the recorded phase velocity dataset for all oscillators during the transition to 3 m/s
    ref_phase = ref_phase[start_idx:end_idx]

    # Combined plots for all oscillators in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 16),
                            subplot_kw={'projection': 'polar'})
    axs = axs.ravel()
    plt.title(title)
    # Plotting for each oscillator
    for i in range(4):
        oscillator_phase_i = dataset['oscillators'][robot_idx, start_idx:end_idx, i]
        r_variable_i = dataset[r_variable][robot_idx, start_idx:end_idx, i]
        plot_individual_oscillator(axs[i], ref_phase,
                                   r_variable_i,
                                   oscillator_phase_i,
                                   f"Oscillator {i+1}")

    plt.tight_layout()
    return fig, axs
    # plt.show()
    from scipy.stats import entropy

def compute_multi_var_shannon_entropy(data, variable_names, num_bins=50):
    """
    Compute multi-variable Shannon entropy for specified variables in the dataset.
    Parameters:
    - data: np.ndarray, the dataset
    - variable_names: list, names of the variables to include in the multi-variable X
    - num_bins: int, number of bins for histogram (used in entropy calculation)
    Returns:
    - entropies: np.array, Shannon entropy for the multi-variable X at each time step across all bots
    """
    # Initialize entropy array
    num_time_steps = data[variable_names[0]].shape[1]
    entropies = np.zeros(num_time_steps)
    for t in range(num_time_steps):
        # Concatenate specified variables across bots to form multi-variable X
        multi_var_X = np.concatenate([data[var][:, t] for var in variable_names], axis=1)
        # Flatten the multi-variable data for this time step
        flat_data = multi_var_X.flatten()
        # Compute histogram
        hist, _ = np.histogram(flat_data, bins=num_bins, density=True)
        # Compute Shannon entropy for this time step
        entropies[t] = entropy(hist)

    return entropies

def entropy_(ax, entropy, title, color='b', linestyle='-', linewidth=1,
            sample_rate=100):
    time = np.linspace(0, len(entropy) / sample_rate, len(entropy))
    ax.plot(time, entropy,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color)

    ax.set_xlabel('Time (s)')  # Now in seconds
    ax.set_ylabel('Entropy (bits)')
    ax.set_title(title)

    ax.grid(True)  # Add grid lines for better readability

    # Optional: If you are plotting multiple lines, a legend can be helpful
    # ax.legend(loc='best')

def entropy(ax, entropy, title, label='', color='b', linestyle='-', linewidth=1, sample_rate=100):
    time = np.linspace(0, entropy.size / sample_rate, entropy.size)
    ax.plot(time, entropy,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            alpha=0.6,
            label=label)  # Add label to the plot for the legend

    ax.set_xlabel('Time (s)')  # Now in seconds
    ax.set_ylabel('Entropy (bits)')
    ax.set_title(title)
    ax.grid(True)  # Add grid lines for better readability

    # Place legend outside the plot and make it 2 columns
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
