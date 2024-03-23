# import warnings
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
import statsmodels.api as sm
from scipy import stats
import os

# warnings.simplefilter('ignore')

# Define a function to save histograms for each time series in each sequence
def save_histograms(df_train, save_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for sequence_idx, sequence_df in enumerate(df_train, start=1):
        num_rows = 2
        num_cols = 2
        
        # Create a new figure for each sequence
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        plt.suptitle(f"Histograms of Time Series in Sequence {sequence_idx}")
        
        for i, col in enumerate(sequence_df.columns):
            plt.subplot(num_rows, num_cols, i+1)
            plt.hist(sequence_df[col], bins=20, alpha=0.5)
            plt.title(f"Time Series {i+1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
        
        # Adjust layout to prevent overlapping titles
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        plt.savefig(os.path.join(save_directory, f"sequence_{sequence_idx}_histograms.png"))
        plt.close()

# Define a function to save Q-Q plots for each time series in each sequence
def save_qq_plots(df_train, save_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for sequence_idx, sequence_df in enumerate(df_train, start=1):
        num_rows = 2
        num_cols = 2
        
        # Create a new figure for each sequence
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        plt.suptitle(f"Q-Q Plots of Time Series in Sequence {sequence_idx}")
        
        for i, col in enumerate(sequence_df.columns):
            plt.subplot(num_rows, num_cols, i+1)
            stats.probplot(sequence_df[col], dist="norm", plot=plt)
            plt.title(f"Time Series {i+1}")
            plt.xlabel("Theoretical quantiles")
            plt.ylabel("Sample quantiles")
        
        # Adjust layout to prevent overlapping titles
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        plt.savefig(os.path.join(save_directory, f"sequence_{sequence_idx}_qq_plots.png"))
        plt.close()

# Define a function to perform Augmented Dickey-Fuller tests for each time series in each sequence
def adf_test(df_train, save_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for sequence_idx, sequence_df in enumerate(df_train, start=1):
        adf_results = []
        
        print(f"Processing Sequence {sequence_idx}")
        
        if sequence_df.isnull().all().all():
            print("Skipping sequence due to all NaN values.")
            continue
        
        for col in sequence_df.columns:
            if sequence_df[col].isnull().all():
                print(f"Skipping Time Series {col} in Sequence {sequence_idx} due to all NaN values.")
                continue
            
            adf_result = sm.tsa.adfuller(sequence_df[col])
            adf_results.append({'Time Series': col,
                                'ADF Statistic': adf_result[0],
                                'p-value': adf_result[1],
                                'Critical Values': adf_result[4]})
        
        # Convert list of dictionaries to DataFrame
        adf_results_df = pd.DataFrame(adf_results)
        
        # Save ADF results to a text file
        adf_results_df.to_csv(os.path.join(save_directory, f"sequence_{sequence_idx}_adf_results.txt"), index=False)
        print(f"ADF results for Sequence {sequence_idx} saved successfully.")

# Common code to save result
def save_graph(df1, df2, title, save_directory):
    data = pd.concat([df1, df2])
    data.reset_index(inplace=True, drop=True)

    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for col in data.columns:
        if col.lower().startswith('pred'):
            data[col].plot(label=col, linestyle="dotted")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_directory, title + '.png'))
    plt.close()

def VARMA_model(train, test, seq):
    # fit model
    model = VARMAX(train, order=(1, 1))
    model_fit = model.fit(disp=True)
    # make prediction
    yhat = model_fit.forecast(steps=len(test))
    res = pd.DataFrame({"Pred1":yhat[f'TimeSeries_1_Sequence_{seq}'], "Pred2":yhat[f'TimeSeries_2_Sequence_{seq}'],
                      "Pred3":yhat[f'TimeSeries_3_Sequence_{seq}'], "Pred4":yhat[f'TimeSeries_4_Sequence_{seq}']})
    return res

# Load .mat file
mat_contents = scipy.io.loadmat('input/Sequences40.mat')

times = mat_contents['time']
inputs = mat_contents['ii']
# outputs = mat_contents['oo']

# Number of time series
num_time_series = inputs.shape[1]
num_sequences = inputs.shape[2]

# Create an empty DataFrame
df_train = [0] * num_sequences
df_ret = [0] * num_sequences

# Generate columns for each time series
for j in range(num_sequences):
    df_train[j] = pd.DataFrame()

    for i in range(num_time_series):
        col_name = f'TimeSeries_{i+1}_Sequence_{j+1}'
        df_train[j][col_name] = [inputs[k, i, j] for k in range(len(times))]
        # df_train[j][col_name] = [(inputs[k, i, j] - np.mean(inputs[:, i, j])) / np.std(inputs[:, i, j]) for k in range(len(times))]

    # df_ret[j] = VARMA_model(df_train[j], df_train[j], j)
    # save_graph(df_train[j], df_ret[j], "Vector Autoregression Moving-Average (VARMA)", "output")

print(df_train[0])

# Call the function to save histograms
if not os.path.exists("output/histograms"):
    save_histograms(df_train, "output/histograms")

# Call the function to save Q-Q plots
if not os.path.exists("output/qqplots"):
    save_qq_plots(df_train, "output/qqplots")

# Call the function to perform ADF tests
if not os.path.exists("output/adf_tests"):
    adf_test(df_train, "output/adf_tests")

# df_train = pd.DataFrame({'Act1':[x + random()*10 for x in range(0, 100)],
#                          'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})
# df_test = pd.DataFrame({'Act1':[x + random()*10 for x in range(101, 201)],
#                          'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})
# df_ret = VARMA_model(df_train, df_test)
# show_graph(df_train, df_ret, "Vector Autoregression Moving-Average (VARMA)")
