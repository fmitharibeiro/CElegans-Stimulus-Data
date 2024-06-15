import os
import numpy as np
import matplotlib.pyplot as plt

def plot_pruning_data(plot_pruning_out, plot_pruning_in, file_path, title='Pruning Data Visualization'):
    """
    Plots the pruning data for plot_pruning_out and plot_pruning_in and saves it to a file.

    Parameters
    ----------
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

    plt.xlabel('Sequence Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    plt.grid(True)
    

    # Save the plot to a file
    plt.savefig(file_path)

    plt.close()  # Close the plot to free up memory

def plot_background(baseline, filepath=None):
    """
    Plots the background array with each feature (dim 1) having a different color.
    
    Parameters:
    baseline (numpy array): A numpy array of shape (1000, 4).
    filepath (str): Optional. The file path to save the plot. If None, the plot will be displayed.
    """
    # Check the shape of the array
    if baseline.shape[1] != 4:
        raise ValueError("The input array must have a shape of (1000, 4)")
    if baseline.shape[0] == 1:
        baseline = np.tile(baseline, (1000, 1))

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