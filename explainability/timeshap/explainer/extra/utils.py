import ast, re
import numpy as np

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