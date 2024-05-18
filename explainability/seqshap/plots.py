import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_metric(values, y_label, save_dir, filename, y_threshold=None):
    iterations = range(1, len(values) + 1)  # Assuming iterations start from 

    plt.plot(iterations, values)
    plt.xlabel('Number of Subsequences')
    plt.ylabel(y_label)
    plt.title(f'Number of Subsequences vs. {y_label}')
    plt.grid(True)

    if y_threshold is not None:
        plt.axhline(y=y_threshold, color='r', linestyle='--', label=f'Y Threshold ({y_threshold})')
        plt.legend()
    
    # Set x-axis limits to start from 1
    plt.xlim(1, max(iterations))

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot as an image file
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()


def plot_subsequences(X, split_points, save_dir, filename):
    num_subsequences = len(split_points) - 1 # Number of subsequences
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Subsequences Visualization')
    ax.set_xlabel('Events')
    ax.set_ylabel('Feature Values')

    colors = plt.cm.get_cmap('tab10', X.shape[1])  # Generate colors for each feature
    
    for i in range(X.shape[1]):
        ax.plot(X[:, i], color=colors(i), label=f'Feature {i+1}')

    # Add vertical lines for split points
    for split_point in sorted(list(split_points))[1:-1]:
        ax.axvline(x=split_point, color='gray', linestyle='--')

    # Add legend
    ax.legend()

    # Add total number of subsequences to legend
    ax.text(0.95, 0.95, f'Total Subsequences: {num_subsequences}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot as an image file
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()


def visualize_phi_seq(phi_seq, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    num_feats, num_subseqs, num_events = phi_seq.shape

    # Iterate over each feature
    for feat_idx in range(num_feats):
        # Compute the mean across events for each subsequence
        mean_values = np.mean(phi_seq[feat_idx], axis=1)

        # Compute the deviation from the mean for each event
        phi_seq_feat = phi_seq[feat_idx] - mean_values[:, None]

        # Normalize phi_seq to range [0, 1] for visualization
        min_val = np.min(phi_seq_feat)
        max_val = np.max(phi_seq_feat)
        norm_values = (phi_seq_feat - min_val) / (max_val - min_val)

        # Define a diverging colormap
        cmap = plt.cm.RdBu_r

        # Create a custom colormap with a neutral color around 1/num_events
        mid_color = mcolors.TwoSlopeNorm(vmin=np.min(norm_values), vcenter=np.mean(norm_values), vmax=np.max(norm_values))(norm_values)
        adjusted_colormap = cmap(mid_color)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each box
        for subseq_idx in range(num_subseqs):
            for event_idx in range(num_events):
                ax.add_patch(plt.Rectangle((subseq_idx, event_idx), 1, 1, facecolor=adjusted_colormap[subseq_idx, event_idx]))

        # Set labels and title
        ax.set_xlabel('Subsequences')
        ax.set_ylabel('Events')
        ax.set_title(f'Visualization of Phi Sequence (Feature {feat_idx})')

        # Save the plot
        save_path = os.path.join(save_dir, f"{filename}_feat{feat_idx}.png")
        plt.savefig(save_path)
        plt.close()