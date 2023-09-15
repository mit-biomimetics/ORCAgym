
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def compute_entropy(counts, method='manual'):
    if method == 'manual':
        total = np.sum(counts)
        probabilities = counts / total
        # Add a small constant to avoid log(0)
        log_probabilities = np.log(probabilities + 1e-9)
        return -np.sum(probabilities * log_probabilities)
    elif method == 'scipy':
        return stats.entropy(counts)
    else:
        raise ValueError("Invalid method. Choose 'manual' or 'scipy'.")

def get_counts(data, bin_edges, method='digitize'):
    M, D = data.shape
    if method == 'digitize':
        digi_data = [np.digitize(data[:, i], bins=bin_edges[i])
                     for i in range(D)]
        counts = np.unique(np.column_stack(digi_data), return_counts=True)[1]
    elif method == 'histogramdd':
        counts = np.histogramdd(data, density=True, bins=bin_edges)[0]
    else:
        'No appropriate method found'
        return None
    # NOTE: could use raveled using np.ravel_multi_index. Should be the same.
    return counts.flatten()

def sliding_window_entropy_single_trial(data, window_size, step_size, method, bin_edges):
    M, D = data.shape
    entropy_values = []

    for start in range(0, M - window_size + 1, step_size):
        window_data = data[start:start + window_size, :]
        flat_counts = get_counts(window_data, bin_edges, method=method)
        # flat_counts = counts.flatten()
        flat_counts = flat_counts[flat_counts > 0]  # * needed?
        window_entropy = compute_entropy(flat_counts, method='manual')
        entropy_values.append(window_entropy)
    return entropy_values

def sliding_window_entropy_multi_trial(data, window_size, step_size, method, bin_edges):
    N, M, D = data.shape
    entropy_values = []

    for start in range(0, M - window_size + 1, step_size):
        window_data = data[:, start:start + window_size, :].reshape((-1, D))
        flat_counts = get_counts(window_data, bin_edges, method=method)
        # flat_counts = counts.flatten()
        flat_counts = flat_counts[flat_counts > 0]  # * needed?
        window_entropy = compute_entropy(flat_counts, method='manual')
        entropy_values.append(window_entropy)
    return np.array(entropy_values)

if __name__ == "__main__":
    data_path = '/home/heim/Repos/trigym/gym/scripts/osc_data/ORC_111/noise_step_1.0.npz'
    # dimensions = [0, 1, 2]
    window_size = 72
    step_size = 1
    num_bins = 10
    method = 'digitize'  # Choose between 'digitize' and 'histogramdd'
    plot_trials = True

    loaded_data = np.load(data_path, allow_pickle=True)
    # data = np.concatenate((loaded_data['base_lin_vel'],
    #                     loaded_data['base_ang_vel']
    #                     ),
    #                    axis=2)
    data = loaded_data['base_lin_vel']

    # number of bins for each dimension in data
    num_bins = [10, 10, 10]
    if len(num_bins) == 1:
        num_bins = num_bins * data.shape[2]

    assert len(num_bins) == data.shape[2], "check your bin number"

    # ! needs to be set the same across all experiments
    custom_bin_edges = [np.linspace(np.min(data[:, :, i]), np.max(data[:, :, i]), num_bins[i] + 1) for i in range(data.shape[2])]
    dimensions = data.shape[2]

    all_trials_entropy_values = []

    for trial in range(data.shape[0]):
        single_trial_data = data[trial, :, :]
        # reshaped_single_trial_data = single_trial_data.reshape(-1, dimensions)

        entropy_values_single_trial = sliding_window_entropy_single_trial(
            single_trial_data, window_size, step_size, method, custom_bin_edges
        )
        all_trials_entropy_values.append(entropy_values_single_trial)

# * now analyze all bots together
    multi_trial_values = []
    entropy_values_multi_trial = sliding_window_entropy_multi_trial(
        data, window_size, step_size, method, custom_bin_edges)
    multi_trial_values.append(entropy_values_multi_trial)

    if plot_trials:
        # plt.figure(figsize=(14, 8))
        # for i, entropy_values in enumerate(all_trials_entropy_values):
        #     plt.plot(entropy_values, label=f'Trial {i+1}', linewidth=1)

        # plt.title(f'Sliding Window Entropy Over Time for All Trials (Window Size = {window_size})')
        # plt.xlabel('Time Step')
        # plt.ylabel('Entropy')
        # plt.legend(loc='upper right', ncol=5)
        # plt.ylim([0, 4])
        # plt.show()

        entropy_values = np.array(all_trials_entropy_values)
        n_trials = entropy_values.shape[0]  # Number of trials
        n_timesteps = entropy_values.shape[1]  # Number of time steps
        mean_data = np.mean(entropy_values, axis=0)  # shape: (D, )
        # Compute the standard error of the mean (SEM) to get the confidence interval
        # Note: You can adjust the confidence level as needed.
        sem_data = np.std(entropy_values, axis=0) / np.sqrt(n_trials)  # shape: (D, )

        # Create 95% confidence interval bounds
        confidence_level = 0.95
        z_value = 1.96  # z-value for 95% confidence (from the standard normal distribution)
        lower_bound = mean_data - z_value * sem_data
        upper_bound = mean_data + z_value * sem_data

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot(mean_data, label='Mean', color='b')
        ax.fill_between(range(n_timesteps), lower_bound, upper_bound,
                        color='b', alpha=0.2, label='95% CI')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Mean Value')
        ax.set_title('Mean and 95% Confidence Interval Over Time Steps')
        ax.legend()

        fig2, ax2 = plt.subplots()
        ax2.plot(entropy_values_multi_trial, linewidth=1)
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.title('Entropy across multiple robots')
        # plt.ylim([0, 4])

        plt.show()
