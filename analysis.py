import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy import stats
import os
import argparse


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

def shapiro_test(df_train, save_directory):
    """
    Perform Shapiro-Wilk test for normality for each time series in each sequence
    and save the results to text files.
    
    Parameters:
        df_train (list of pandas DataFrames): List containing DataFrames for each sequence.
        save_directory (str): Directory where the results will be saved.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for sequence_idx, sequence_df in enumerate(df_train, start=1):
        shapiro_results = []
        
        print(f"Processing Sequence {sequence_idx}")
        
        if sequence_df.isnull().all().all():
            print("Skipping sequence due to all NaN values.")
            continue
        
        for col in sequence_df.columns:
            if sequence_df[col].isnull().all():
                print(f"Skipping Time Series {col} in Sequence {sequence_idx} due to all NaN values.")
                continue
            
            shapiro_result = stats.shapiro(sequence_df[col])
            shapiro_results.append({'Time Series': col,
                                    'Test Statistic': shapiro_result[0],
                                    'p-value': shapiro_result[1]})
        
        # Convert list of dictionaries to DataFrame
        shapiro_results_df = pd.DataFrame(shapiro_results)
        
        # Save Shapiro-Wilk test results to a text file
        shapiro_results_df.to_csv(os.path.join(save_directory, f"sequence_{sequence_idx}_shapiro_results.txt"), index=False)
        print(f"Shapiro-Wilk test results for Sequence {sequence_idx} saved successfully.")

# Define a function to perform autocorrelation analysis for each time series in each sequence
def autocorrelation_analysis(df_train, save_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for sequence_idx, sequence_df in enumerate(df_train, start=1):
        print(f"Processing Sequence {sequence_idx}")
        
        if sequence_df.isnull().all().all():
            print("Skipping sequence due to all NaN values.")
            continue
        
        # Calculate number of rows and columns for subplots
        num_time_series = len(sequence_df.columns)
        num_rows = int(np.ceil(num_time_series / 2))
        num_cols = min(num_time_series, 2)
        
        # Create a new figure for each sequence
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        plt.suptitle(f"Autocorrelation Plots of Time Series in Sequence {sequence_idx}")
        
        for i, col in enumerate(sequence_df.columns):
            plt.subplot(num_rows, num_cols, i+1)
            
            # Plot autocorrelation function with significance limits
            plot_acf(sequence_df[col], lags=999, ax=plt.gca())
            plt.title(f"Time Series {col}")
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
        
        # Adjust layout to prevent overlapping titles
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        plt.savefig(os.path.join(save_directory, f"sequence_{sequence_idx}_autocorrelation_plots.png"))
        plt.close()
        print(f"Autocorrelation plots for Sequence {sequence_idx} saved successfully.")

# Define a function to perform partial autocorrelation analysis for each time series in each sequence
def partial_autocorrelation_analysis(df_train, save_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for sequence_idx, sequence_df in enumerate(df_train, start=1):
        print(f"Processing Sequence {sequence_idx}")
        
        if sequence_df.isnull().all().all():
            print("Skipping sequence due to all NaN values.")
            continue
        
        # Calculate number of rows and columns for subplots
        num_time_series = len(sequence_df.columns)
        num_rows = int(np.ceil(num_time_series / 2))
        num_cols = min(num_time_series, 2)
        
        # Create a new figure for each sequence
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        plt.suptitle(f"Partial Autocorrelation Plots of Time Series in Sequence {sequence_idx}")
        
        for i, col in enumerate(sequence_df.columns):
            plt.subplot(num_rows, num_cols, i+1)
            
            # Plot partial autocorrelation function with significance limits
            plot_pacf(sequence_df[col], lags=499, ax=plt.gca(), method='ols')
            plt.title(f"Time Series {col}")
            plt.xlabel("Lag")
            plt.ylabel("Partial Autocorrelation")
        
        # Adjust layout to prevent overlapping titles
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        plt.savefig(os.path.join(save_directory, f"sequence_{sequence_idx}_partial_autocorrelation_plots.png"))
        plt.close()
        print(f"Partial autocorrelation plots for Sequence {sequence_idx} saved successfully.")

# Common code to save result
def save_graph(df1, df2, title, save_directory):
    # Concatenate dataframes
    data = pd.concat([df1, df2], axis=1)
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot the data from both dataframes on the same axes
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_mapping = {}  # Dictionary to store color mapping
    for i, col in enumerate(data.columns):
        if 'var' in col.lower():
            data[col].plot(label=col, color=color_mapping.get(col, None))
        else:
            color = color_mapping.get(col, None)
            if color is None:
                color = color_cycle[i % len(color_cycle)]  # Cycle through default colors
                color_mapping[col] = color
            data[col].plot(label=col, linestyle="dotted", color=color)
    
    plt.title(title)
    plt.savefig(os.path.join(save_directory, title + '.png'))
    plt.close()

def VAR_model(train, test, seq):
    # Generate lagged values DataFrame
    def generate_lagged_values(coefficients, num_lags):
        num_time_series = coefficients.shape[1]
        lagged_values = np.zeros((num_lags, num_time_series))
        for lag in range(num_lags):
            for ts in range(num_time_series):
                col_name = f'TimeSeries_{ts+1}_Sequence_{seq+1}'
                lagged_values[lag, ts] = coefficients.loc[f'L{lag+1}.{col_name}', col_name]
        return pd.DataFrame(lagged_values, columns=[f"VAR_TimeSeries_{i+1}" for i in range(num_time_series)])
    # fit model
    num_lags = 500
    model = VAR(train)
    model_fit = model.fit(num_lags, trend='n')
    coefficients = model_fit.params
    print("Coeffs:")
    print(coefficients)
    res = generate_lagged_values(coefficients, num_lags)
    print("Res:")
    print(res)
    # print(np.array(train).shape)
    # make prediction
    # yhat = model_fit.forecast(np.array(train), steps=len(test))
    # res = pd.DataFrame({"Pred1":[x[0] for x in yhat], "Pred2":[x[1] for x in yhat],
    #                   "Pred3":[x[2] for x in yhat], "Pred4":[x[3] for x in yhat]})
    return res

def VARMA_model(train, test, seq):
    # Generate lagged values DataFrame
    def generate_lagged_values(coefficients, num_lags):
        num_time_series = coefficients.shape[1]
        lagged_values = np.zeros((num_lags, num_time_series))
        for lag in range(num_lags):
            for ts in range(num_time_series):
                col_name = f'TimeSeries_{ts+1}_Sequence_{seq+1}'
                lagged_values[lag, ts] = coefficients.loc[f'L{lag+1}.{col_name}', col_name]
        return pd.DataFrame(lagged_values, columns=[f"VARMA_TimeSeries_{i+1}" for i in range(num_time_series)])

    # Fit VARMA model
    num_lags = 500
    model = VARMAX(train, order=(num_lags, 0))
    model_fit = model.fit(maxiter=num_lags, disp=False)  # You can adjust maxiter and disp based on your needs
    coefficients = model_fit.params

    # Print coefficients for inspection
    print("Coefficients:")
    print(coefficients)

    # Generate lagged values DataFrame
    res = generate_lagged_values(coefficients, num_lags)
    
    # Print lagged values for inspection
    print("Lagged values:")
    print(res)
    
    return res

# Function to calculate the required values for each sequence and time series
def calculate_values(sequence_num, time_series):
    start_value = None
    duration = 0
    max_amplitude = 0
    amplitude_values = set()
    spike_count = 0

    for index, value in enumerate(time_series):
        if value != 0:
            if start_value is None:
                start_value = index
            duration += 1
            max_amplitude = max(max_amplitude, value)
            amplitude_values.add(value)
            if value == max_amplitude and (index == 0 or time_series[index - 1] != value) and (index == len(time_series) - 1 or time_series[index + 1] <= value):
                spike_count += 1  # Increment spike count

    if not amplitude_values:
        return [sequence_num, int(time_series.name.split('_')[1]), None, None, None, None, 0]

    return [
        sequence_num,
        int(time_series.name.split('_')[1]),
        start_value,
        duration,
        max_amplitude,
        len(amplitude_values) > 1,
        spike_count
    ]


def main(args):
    # Load .mat file
    mat_contents = scipy.io.loadmat('input/Sequences40.mat')

    times = mat_contents['time']

    if args.data_type == "INPUT":
        inputs = mat_contents['ii']
    elif args.data_type == 'OUTPUT':
        inputs = mat_contents['oo']
    else:
        raise ValueError("Invalid data type. Please choose either 'INPUT' or 'OUTPUT'.")

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

            time_series = [inputs[k, i, j] for k in range(len(times))]

            if args.transformation == 'Z':
                df_train[j][col_name] = [(inputs[k, i, j] - np.mean(inputs[:, i, j])) / np.std(inputs[:, i, j]) for k in range(len(times))]

            elif args.transformation == 'LOG':
                # Apply log transformation to values greater than 1e-10
                transformed_series = [np.log(x) if x > 1e-10 else x for x in time_series]
                df_train[j][col_name] = transformed_series
            
            elif args.transformation == 'SQRT':
                transformed_series = [np.sqrt(x) if x > 1e-10 else x for x in time_series]
                df_train[j][col_name] = transformed_series

            elif args.transformation == 'BOX-COX':
                # Apply Box-Cox transformation
                transformed_series, lambda_value = stats.boxcox(np.abs(time_series))
                df_train[j][col_name] = transformed_series

            else:
                df_train[j][col_name] = time_series
            # df_train[j][col_name] = [(inputs[k, i, j] - np.mean(inputs[:, i, j])) / np.std(inputs[:, i, j]) for k in range(len(times))]

        # Call the appropriate model function based on the model_type argument
        if args.model_type == 'NONE':
            continue
        elif args.model_type == 'VAR':
            df_ret[j] = VAR_model(df_train[j], df_train[j], j)
        elif args.model_type == 'VARMA':
            df_ret[j] = VARMA_model(df_train[j], df_train[j], j)
        else:
            raise ValueError("Invalid model type. Please choose either 'VAR' or 'VARMA'.")

        # Save graph
        save_graph(df_train[j], df_ret[j], f"{args.model_type}_Sequence_{j+1}", f"output/{args.data_type}_{args.transformation}/{args.model_type}s")

    print("Model training completed.")

    # Call the function to save histograms
    if not os.path.exists(f"output/{args.data_type}_{args.transformation}/histograms"):
        save_histograms(df_train, f"output/{args.data_type}_{args.transformation}/histograms")

    # Call the function to save Q-Q plots
    if not os.path.exists(f"output/{args.data_type}_{args.transformation}/qqplots"):
        save_qq_plots(df_train, f"output/{args.data_type}_{args.transformation}/qqplots")

    # Call the function to perform ADF tests
    if not os.path.exists(f"output/{args.data_type}_{args.transformation}/adf_tests"):
        adf_test(df_train, f"output/{args.data_type}_{args.transformation}/adf_tests")
    
    # Call the function to perform ADF tests
    if not os.path.exists(f"output/{args.data_type}_{args.transformation}/shapiro_tests"):
        shapiro_test(df_train, f"output/{args.data_type}_{args.transformation}/shapiro_tests")
    
    # Call the function to perform autocorrelation analysis
    if not os.path.exists(f"output/{args.data_type}_{args.transformation}/acf"):
        autocorrelation_analysis(df_train, f"output/{args.data_type}_{args.transformation}/acf")

    # Call the function to perform partial autocorrelation analysis
    if not os.path.exists(f"output/{args.data_type}_{args.transformation}/pacf"):
        partial_autocorrelation_analysis(df_train, f"output/{args.data_type}_{args.transformation}/pacf")
    
    if not os.path.exists(f'output/{args.data_type}_{args.transformation}/result.csv'):
        # Create an empty list to store the calculated values
        data = []

        # Iterate over each sequence and time series
        for sequence_num, sequence_df in enumerate(df_train, start=1):
            for series_num, series in sequence_df.items():
                data.append(calculate_values(sequence_num, series))

        # Create DataFrame from the list of calculated values
        columns = ["Sequence", "Series", "Start", "Duration", "Amplitude", "Smooth", "Spikes"]
        df_new = pd.DataFrame(data, columns=columns)

        # Display the resulting DataFrame
        print(df_new)

        # Save the resulting DataFrame to a CSV file
        df_new.to_csv(f'output/{args.data_type}_{args.transformation}/result.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['NONE', 'VAR', 'VARMA'], default='NONE', help='Type of model to run (VAR or VARMA)')
    parser.add_argument('--data_type', type=str, choices=['INPUT', 'OUTPUT'], default='OUTPUT', help='Type of data to analyse (INPUT or OUTPUT)')
    parser.add_argument('--transformation', type=str, choices=['NONE', 'Z', 'LOG', 'SQRT', 'BOX-COX'], default='NONE', help='Type of data transformation (INPUT or OUTPUT)')
    args = parser.parse_args()
    
    main(args)