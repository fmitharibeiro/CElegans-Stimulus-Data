import ast, re

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
