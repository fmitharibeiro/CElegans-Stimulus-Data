import scipy.io

# Load .mat file
mat_contents = scipy.io.loadmat('input/Sequences40.mat')

times = mat_contents['time']
outputs = mat_contents['oo']
inputs = mat_contents['ii']

# # Access variables in the .mat file
# # For example, if you have a variable named 'data' in the .mat file
# data = mat_contents['data']

# # Now you can work with the 'data' variable as you would with any Python variable
# # For example, print its shape
# print("Shape of 'data':", data.shape)
