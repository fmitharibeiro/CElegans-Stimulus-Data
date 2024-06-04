import ast, re

def correct_shap_vals_format(data):
    # Function to correct the formatting of the list strings
    def correct_format(s):
        return re.sub(r'\s+', ',', s.replace('[ ', '[').replace(' ]', ']'))
    data['Shapley Value'] = data['Shapley Value'].apply(correct_format)
    data['Shapley Value'] = data['Shapley Value'].apply(ast.literal_eval)

    return data['Shapley Value']