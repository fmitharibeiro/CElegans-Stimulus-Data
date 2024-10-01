import os
import numpy as np
import matplotlib.pyplot as plt

def plot_pruning_data(threshold, plot_pruning_out, plot_pruning_in, file_path, title='Pruning Data Visualization'):
    """
    Plots the pruning data for plot_pruning_out and plot_pruning_in and saves it to a file.

    Parameters
    ----------
    threshold : float
        The threshold value to be drawn as a horizontal line.
    
    plot_pruning_out : list
        Data for the pruning out plot.
    
    plot_pruning_in : list
        Data for the pruning in plot.
    
    file_path : str
        Path to save the plot image file.
    
    title : str, optional
        Title of the plot (default is 'Pruning Data Visualization').
    """
    x_range = range(-len(plot_pruning_out), 0)

    plt.figure(figsize=(10, 6))

    plt.plot(x_range, plot_pruning_out, label='Pruning Out', color='grey')
    plt.plot(x_range, plot_pruning_in, label='Pruning In', color='red')

    plt.axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')

    plt.xlabel('Sequence Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    plt.grid(True)

    # Save the plot to a file
    plt.savefig(file_path)

    plt.close()  # Close the plot to free up memory

def plot_background(baseline, data_shape, filepath=None):
    """
    Plots the background array with each feature (dim 1) having a different color.
    
    Parameters:
    baseline (numpy array): A numpy array of shape (1000, 4).
    filepath (str): Optional. The file path to save the plot. If None, the plot will be displayed.
    """
    if os.path.exists(filepath):
        return
    # Check the shape of the array
    if len(baseline.shape) == 1 and baseline.shape[0] == data_shape[1]:
        baseline = baseline.reshape(1, -1)
    if baseline.shape[1] != data_shape[1]:
        raise ValueError("The input array must have a shape of (1000, 4)")
    if baseline.shape[0] == 1:
        baseline = np.tile(baseline, (data_shape[0], 1))

    # Define colors for each feature
    colors = ['r', 'g', 'b', 'c']
    labels = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot each feature
    for i in range(4):
        plt.plot(baseline[:, i], color=colors[i], label=labels[i])
    
    # Add title and labels
    plt.title('Baseline Features Plot')
    plt.xlabel('Samples')
    plt.ylabel('Feature Value')
    plt.legend()
    
    # Save to file if filepath is provided, otherwise show the plot
    if filepath:
        os.makedirs(filepath[:filepath.rfind("/")], exist_ok=True)
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")
    else:
        plt.show()


def plot_sequences(directory, filename, *sequences_with_labels, title="Sequence Plot"):
    """
    Plots an arbitrary number of sequences, each of shape (1, n), with different colors and saves the plot.
    If the file already exists in the directory, the function does nothing.

    Parameters
    ----------
    directory : str
        The directory where the plot should be saved.
    filename : str
        The name of the file to save the plot.
    title : str, optional
        The title of the plot.
    *sequences_with_labels : tuple of (sequence, label)
        Each sequence must be of shape (1, n), where n is the number of time steps or values.
        Each sequence must have an associated label.
    """
    # Create the full file path
    file_path = os.path.join(directory, filename)
    
    # Check if the file already exists
    if os.path.exists(file_path):
        return
    
    # Ensure the directory exists, create it if not
    os.makedirs(directory, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Define a list of colors to cycle through for each sequence
    colors = plt.cm.get_cmap('tab10', len(sequences_with_labels))
    
    for idx, (sequence, label) in enumerate(sequences_with_labels):
        # Ensure the sequence is of shape (n,), flatten the (1, n) array
        sequence = np.squeeze(sequence)  
        
        # Plot each sequence with a unique color and label
        plt.plot(sequence, color=colors(idx), label=label)
    
    # Add plot title, labels, and grid
    plt.title(title)
    plt.xlabel("Event")  # Custom x-axis label
    plt.ylabel("Feature Value")  # Custom y-axis label
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the specified file
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory