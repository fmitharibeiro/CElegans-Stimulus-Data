# import warnings
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
import os

# warnings.simplefilter('ignore')

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
        df_train[j][col_name] = [(inputs[k, i, j] - np.mean(inputs[:, i, j])) / np.std(inputs[:, i, j]) for k in range(len(times))]

    # df_ret[j] = VARMA_model(df_train[j], df_train[j], j)
    # save_graph(df_train[j], df_ret[j], "Vector Autoregression Moving-Average (VARMA)", "output")

print(df_train[0])

# df_train = pd.DataFrame({'Act1':[x + random()*10 for x in range(0, 100)],
#                          'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})
# df_test = pd.DataFrame({'Act1':[x + random()*10 for x in range(101, 201)],
#                          'Act2':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})
# df_ret = VARMA_model(df_train, df_test)
# show_graph(df_train, df_ret, "Vector Autoregression Moving-Average (VARMA)")
