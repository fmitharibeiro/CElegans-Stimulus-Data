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

    plt.plot(x_range, plot_pruning_out, label='Pruning Out', color='red')
    plt.plot(x_range, plot_pruning_in, label='Pruning In', color='blue')

    plt.xlabel('Sequence Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    plt.grid(True)
    

    # Save the plot to a file
    plt.savefig(file_path)

    plt.close()  # Close the plot to free up memory