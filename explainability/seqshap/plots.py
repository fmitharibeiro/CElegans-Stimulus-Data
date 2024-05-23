import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

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

    num_colors = X.shape[1] if len(X.shape) == 2 else 1
    colors = plt.cm.get_cmap('tab10', num_colors)  # Generate colors for each feature
    
    if num_colors > 1:
        for i in range(num_colors):
            ax.plot(X[:, i], color=colors(i), label=f'Feature {i+1}')
    else:
        ax.plot(X, color=colors(0))

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


def write_subsequence_ranges(X, save_dir, filename):
    # Check if X has only one feature
    if len(X.shape) == 2:
        num_subseqs, num_events = X.shape
        has_single_feature = True
    else:
        num_subseqs, num_events, num_feats = X.shape
        has_single_feature = False

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        for subseq_index in range(num_subseqs):
            # Initialize the start and end indices
            start_index = None
            end_index = None
            
            # Find the first and last non-NaN values in the subsequence
            for event_index in range(num_events):
                if has_single_feature:
                    if not np.isnan(X[subseq_index, event_index]):
                        if start_index is None:
                            start_index = event_index
                        end_index = event_index
                else:
                    if not np.isnan(X[subseq_index, event_index, :]).all():
                        if start_index is None:
                            start_index = event_index
                        end_index = event_index
            
            if start_index is not None and end_index is not None:
                f.write(f'Subsequence {subseq_index + 1}: Start Index = {start_index}, End Index = {end_index}\n')
            else:
                f.write(f'Subsequence {subseq_index + 1}: No valid range found (all NaNs)\n')

    print(f'Ranges of subsequences written to {filepath}')


def visualize_phi_seq(phi_seq, save_dir, filename, plot_title):
    num_feats, num_subseqs_input, num_subseqs_output = phi_seq.shape
    
    # Compute the average value of reshaped_phi_seq
    avg_value = np.nanmean(phi_seq)

    # Define a custom diverging color scale with a neutral color at the average value
    custom_colorscale = [
        [0.0, 'blue'],   # Low values
        [0.5, 'white'],  # Neutral at the average value
        [1.0, 'red']     # High values
    ]
    
    # Create a subplots figure with one row and num_feats columns
    fig = sp.make_subplots(rows=1, cols=num_feats, subplot_titles=[f'Feature {f+1}' for f in range(num_feats)])
    
    # Add a heatmap for each feature
    for f in range(num_feats):
        heatmap_data = phi_seq[f, :, :]
        fig.add_trace(go.Heatmap(
            z=heatmap_data, 
            coloraxis="coloraxis"
        ), row=1, col=f+1)

    # Update layout with the custom color scale
    fig.update_layout(
        title=plot_title,
        coloraxis={
            'colorscale': custom_colorscale,
            'cmin': np.nanmin(phi_seq),  # Minimum value for color scale
            'cmid': avg_value,  # Middle value for color scale
            'cmax': np.nanmax(phi_seq)   # Maximum value for color scale
        },
        height=600,  # Adjust height if necessary
        width=1500,  # Adjust width if necessary
    )

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the plot as an HTML file
    filepath = os.path.join(save_dir, filename)
    fig.write_html(filepath)
    print(f'Heatmap saved to {filepath}')