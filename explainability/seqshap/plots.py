import os
import matplotlib.pyplot as plt

def plot_metric(values, y_label, save_dir, filename, y_threshold=None):
    iterations = range(1, len(values) + 1)  # Assuming iterations start from 

    plt.plot(iterations, values)
    plt.xlabel('Iterations')
    plt.ylabel(y_label)
    plt.title(f'Iterations vs. {y_label}')
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

