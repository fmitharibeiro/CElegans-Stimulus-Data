import ast, re, os, csv
import numpy as np, pandas as pd
from pathlib import Path

def correct_shap_vals_format(data):
    if not isinstance(data['Shapley Value'][0], str):
        return data['Shapley Value']
    
    # Function to correct the formatting of the list strings
    def correct_format(s):
        # Replace spaces with commas and fix brackets
        formatted_s = re.sub(r'\s+', ',', s.replace('[ ', '[').replace(' ]', ']'))
        # Replace any sequence of multiple commas with a single comma
        formatted_s = re.sub(r',+', ',', formatted_s)
        return formatted_s
    
    data['Shapley Value'] = data['Shapley Value'].apply(correct_format)
    data['Shapley Value'] = data['Shapley Value'].apply(ast.literal_eval)
    data['Shapley Value'] = data['Shapley Value'].apply(lambda x: [float(i) for i in x])

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
    array = np.array(series.tolist())

    max_abs_indices = np.argmax(np.abs(array), axis=0)

    result = array[max_abs_indices, np.arange(array.shape[1])]
    
    return result.tolist()

def save_multiple_files(data, file_path, file_index, header, num_digits):
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    file_index_str = str(file_index).rjust(num_digits, '0')
    new_file_path = os.path.join(file_dir, f"{base_name}_{file_index_str}{ext}")

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
    os.makedirs(file_dir, exist_ok=True)
    files = [f for f in os.listdir(file_dir) if f.startswith(base_name) and f.endswith(ext)]
    return len(files) > 0

def detect_last_saved_file_index(file_path):
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    try:
        files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.startswith(base_name) and f.endswith(ext)]
        if not files:
            return 0
    except FileNotFoundError:
        return 0
    last_file = max(files)
    match = re.search(r"_(\d+)\.csv$", last_file)
    if match:
        return int(match.group(1)) + 1
    return 0

def count_rows_in_last_file(file_path):
    file_dir, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    try:
        files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.startswith(base_name) and f.endswith(ext)]
        if not files:
            return 1
    except FileNotFoundError:
        return 1
    last_file = max(files)
    with open(last_file, 'r') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for the header row