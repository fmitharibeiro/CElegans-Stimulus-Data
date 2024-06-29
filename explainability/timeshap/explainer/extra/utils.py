import ast, re, os, csv
import numpy as np, pandas as pd
from pathlib import Path

def correct_shap_vals_format(data):
    if not isinstance(data['Shapley Value'][0], str):
        return data['Shapley Value']
    # Function to correct the formatting of the list strings
    def correct_format(s):
        return re.sub(r'\s+', ',', s.replace('[ ', '[').replace(' ]', ']'))
    data['Shapley Value'] = data['Shapley Value'].apply(correct_format)
    data['Shapley Value'] = data['Shapley Value'].apply(ast.literal_eval)

    return data['Shapley Value']

def max_abs_value(lst):
    abs_lst = [abs(ele) for ele in lst]
    max_index = abs_lst.index(max(abs_lst))
    return lst[max_index]

def max_abs_preserve_sign(series):
    """
    For each position in the lists within the series, compute the maximum of the absolute values,
    preserving the sign of the maximum absolute value.
    
    Parameters:
    series (pd.Series): Pandas Series where each element is a list of numbers.
    
    Returns:
    list: A list of length equal to the length of the lists in the series, 
          containing the maximum absolute values preserving the sign.
    """
    # Convert the Series to a numpy array of shape (180, 50)
    array = np.array(series.tolist())
    
    # Calculate the maximum of the absolute values along the axis 0
    max_abs_indices = np.argmax(np.abs(array), axis=0)
    
    # Create the result list by selecting the maximum absolute values preserving the sign
    result = array[max_abs_indices, np.arange(array.shape[1])]
    
    return result.tolist()

def save_multiple_files(data, file_path, file_index, header):
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_path = os.path.join(file_dir, f"{base_name}_{file_index}{ext}")

    if '/' in new_file_path:
        Path(new_file_path.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
    
    with open(new_file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(header)
        writer.writerows(np.concatenate(data))

def read_multiple_files(file_path):
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.startswith(base_name) and f.endswith(ext)]
    dfs = [pd.read_csv(f) for f in sorted(files)]
    return pd.concat(dfs, ignore_index=True)

def file_exists(file_path):
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    files = [f for f in os.listdir(file_dir) if f.startswith(base_name) and f.endswith(ext)]
    return len(files) > 0