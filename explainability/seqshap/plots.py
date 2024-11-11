import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_metric(values, y_label, save_dir, filename, y_threshold=None):
    iterations = range(1, len(values) + 1)

    plt.plot(iterations, values)
    plt.xlabel('Number of Subsequences')
    plt.ylabel(y_label)
    plt.title(f'Number of Subsequences vs. {y_label}')
    plt.grid(True)

    if y_threshold is not None:
        plt.axhline(y=y_threshold, color='r', linestyle='--', label=f'Y Threshold ({y_threshold})')
        plt.legend()
    
    plt.xlim(1, max(iterations))

    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

def plot_derivatives_and_variances(derivatives, variances, save_dir, filename, split_points, num_events, y_threshold=None):
    """
    Plots the derivatives and variances in a (2x1) grid and saves the plot.

    Parameters
    ----------
    derivatives : np.ndarray
        The discrete derivatives, shaped (num_events-1, num_feats).
    variances : np.ndarray
        The variances of the derivatives, shaped (num_events-1,).
    save_dir : str
        The directory where the plot will be saved.
    filename : str
        The name of the file to save the plot.
    split_points : set
        The points where the derivative changes signal.
    num_events : int
        The total number of events.
    y_threshold : float, optional
        The y-threshold to draw in the variance plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a figure with 2 subplots (2x1 grid)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the derivatives in the top subplot
    if len(derivatives.shape) == 2:
        num_events_minus_1, num_feats = derivatives.shape
        x = np.arange(num_events_minus_1)
        for i in range(num_feats):
            axs[0].plot(x, derivatives[:, i], label=f'Feature {i+1}')
    else:
        num_events_minus_1 = derivatives.shape[0]
        x = np.arange(num_events_minus_1)
        axs[0].plot(x, derivatives, label='Derivative')
    axs[0].set_title('Derivatives')
    axs[0].set_xlabel('Event')
    axs[0].set_ylabel('Derivative')
    
    # Plot vertical lines at split points, excluding the first and last points
    for point in split_points:
        if point != 0 and point != num_events:
            axs[0].axvline(x=point, color='g', linestyle='--', label='Derivative Sign Change')

    # Avoid duplicate labels in the legend
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())
    
    # Plot the variances in the bottom subplot
    axs[1].plot(x, variances, color='r', label='Variance')
    if y_threshold is not None:
        axs[1].axhline(y=y_threshold, color='b', linestyle='--', label='Threshold')
    axs[1].set_title('Variances')
    axs[1].set_xlabel('Event')
    axs[1].set_ylabel('Variance')
    axs[1].legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

def plot_subsequences(X, split_points, save_dir, filename):
    num_subsequences = len(split_points) - 1
    
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
    first = True
    for split_point in sorted(list(split_points))[1:-1]:
        if first:
            ax.axvline(x=split_point, color='gray', linestyle='--', label='Subsequence Start')
            first = False
        ax.axvline(x=split_point, color='gray', linestyle='--')

    # Add total number of subsequences to legend
    ax.text(0.95, 0.95, f'Total Subsequences: {num_subsequences}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right')
    
    # Add legend outside the plot area
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()


def write_subsequence_ranges(X, save_dir, filename):
    # Check if X only has one feature
    if len(X.shape) == 2:
        num_subseqs, num_events = X.shape
        has_single_feature = True
    else:
        num_subseqs, num_events, num_feats = X.shape
        has_single_feature = False

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        for subseq_index in range(num_subseqs):
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


def visualize_phi_all(phi_cell, phi_seq, phi_feat, save_dir, filename, plot_title):
    num_feats, num_subseqs_input, num_subseqs_output = phi_cell.shape

    # Determine min and max values for color scaling
    min_val = min(np.nanmin(phi_cell), np.nanmin(phi_seq), np.nanmin(phi_feat))
    max_val = max(np.nanmax(phi_cell), np.nanmax(phi_seq), np.nanmax(phi_feat))
    abs_max_val = max(abs(min_val), abs(max_val))

    custom_colorscale = [
        [0.0, 'blue'],   # Low values (negative)
        [0.5, 'white'],  # Neutral at 0
        [1.0, 'red']     # High values (positive)
    ]

    fig = sp.make_subplots(
        rows=2, 
        cols=3, 
        subplot_titles=[f'Feature {1}', f'Feature {2}', 'Feature Importance', f'Feature {3}', f'Feature {4}', 'Subsequence Importance'],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}], 
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.2
    )

    # Add heatmaps for phi_cell (first two features in the first row)
    for i in range(2):
        fig.add_trace(go.Heatmap(
            z=phi_cell[i, :, :], 
            coloraxis="coloraxis",
            showscale=False,
            text=np.round(phi_cell[i, :, :], 2), 
            texttemplate="%{text}",
            zmin=-abs_max_val,
            zmax=abs_max_val,
            hoverongaps=False,
            xgap=1,
            ygap=1
        ), row=1, col=i+1)

    # Add heatmaps for phi_cell (last two features in the second row)
    for i in range(2, 4):
        fig.add_trace(go.Heatmap(
            z=phi_cell[i, :, :], 
            coloraxis="coloraxis",
            showscale=False,
            text=np.round(phi_cell[i, :, :], 2), 
            texttemplate="%{text}",
            zmin=-abs_max_val,
            zmax=abs_max_val,
            hoverongaps=False,
            xgap=1,
            ygap=1
        ), row=2, col=i-1)

    # Add heatmap for phi_feat in the first row, third column
    fig.add_trace(go.Heatmap(
        z=np.squeeze(phi_feat), 
        coloraxis="coloraxis",
        showscale=False,
        text=np.round(np.squeeze(phi_feat), 2), 
        texttemplate="%{text}",
        zmin=-abs_max_val,
        zmax=abs_max_val,
        hoverongaps=False,
        xgap=1,
        ygap=1
    ), row=1, col=3)

    # Add heatmap for phi_seq in the second row, third column
    fig.add_trace(go.Heatmap(
        z=phi_seq[0, :, :], 
        coloraxis="coloraxis",
        showscale=False,
        text=np.round(phi_seq[0, :, :], 2), 
        texttemplate="%{text}",
        zmin=-abs_max_val,
        zmax=abs_max_val,
        hoverongaps=False,
        xgap=1,
        ygap=1
    ), row=2, col=3)

    # Update layout with the custom color scale and axis titles
    fig.update_layout(
        title=plot_title,
        coloraxis={
            'colorscale': custom_colorscale,
            'cmin': -abs_max_val,  # Minimum value for color scale
            'cmid': 0,  # Middle value for color scale
            'cmax': abs_max_val   # Maximum value for color scale
        },
        height=800,
        width=1800,
        xaxis1_title="Output Subsequence",
        yaxis1_title="Input Subsequence",
        xaxis2_title="Output Subsequence",
        yaxis2_title="Input Subsequence",
        xaxis3_title="Output Subsequence",
        yaxis3_title="Feature",
        yaxis3={
            'tickvals': list(range(num_feats)),
            'ticktext': [str(i + 1) for i in range(num_feats)]
        },
        xaxis4_title="Output Subsequence",
        yaxis4_title="Input Subsequence",
        xaxis5_title="Output Subsequence",
        yaxis5_title="Input Subsequence",
        xaxis6_title="Output Subsequence",
        yaxis6_title="Input Subsequence"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)
    fig.write_html(filepath)

def plot_background(baseline, data_shape, filepath=None):
    """
    Plots the background array with each feature (dim 1) having a different color.
    
    Parameters:
    baseline (numpy array): A numpy array of shape (1000, 4).
    filepath (str): Optional. The file path to save the plot. If None, the plot will be displayed.
    """
    if os.path.exists(filepath):
        return

    if len(baseline.shape) == 1 and baseline.shape[0] == data_shape[1]:
        baseline = baseline.reshape(1, -1)
    if baseline.shape[1] != data_shape[1]:
        raise ValueError("The input array must have a shape of (1000, 4)")
    if baseline.shape[0] == 1:
        baseline = np.tile(baseline, (data_shape[0], 1))

    colors = ['r', 'g', 'b', 'c']
    labels = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
    
    plt.figure(figsize=(10, 6))
    
    for i in range(4):
        plt.plot(baseline[:, i], color=colors[i], label=labels[i])
    
    plt.title('Baseline Features Plot')
    plt.xlabel('Samples')
    plt.ylabel('Feature Value')
    plt.legend()
    
    if filepath:
        os.makedirs(filepath[:filepath.rfind("/")], exist_ok=True)
        plt.savefig(filepath)
    else:
        plt.show()