import numpy as np
import yaml
from yaml.loader import SafeLoader

'''Reading the yaml file with the parameters.
Returning numpy arrays with either the baseline or random combinations.'''

# Opening the yaml file
with open('config.yaml', 'r') as stream:

    try:
        # Converting yaml document to python object
        parameters = yaml.load(stream, Loader=SafeLoader)
        # Create matrix from dictionary values
        para_np = np.array([[val for val in p['values']] for p in parameters['sweep_configuration'].values()])
        # Store values
        config_keys = parameters['sweep_configuration'].keys()

    except yaml.YAMLError as e:
        print(e)

#print(para_np)
#print(config_keys)


# Base line
# Slice matrix to get the first column
def base_line(keys=config_keys):
    base_line = para_np[:, 0]
    # create dictionary from keys and base_line
    base_line_dic = dict(zip(keys, base_line))

    return base_line_dic

# Random combinations
def random_combi(keys=config_keys):
    # Set the number of rows and columns in the matrix
    num_rows, num_cols = para_np.shape

    # Create an empty matrix with the desired shape
    bool_matrix = np.empty((num_rows, num_cols))

    # Iterate over the rows of the matrix
    for i in range(num_rows):
        # Generate a random integer between 0 and num_cols-1
        idx = np.random.randint(num_cols)
        # Set the element at the randomly generated index to 1
        bool_matrix[i, idx] = 1
        # Set all other elements in the row to 0
        bool_matrix[i, :idx] = 0
        bool_matrix[i, idx+1:] = 0

    # Create a boolean mask indicating which elements of parameters_np are equal to 1 in the matrix
    mask = bool_matrix == 1

    # Use the mask to select the corresponding elements of parameters_np
    filtered_parameters = para_np[mask]

    # Create dictionary from keys and filtered_parameters
    filtered_parameters_dic = dict(zip(keys, filtered_parameters))

    return filtered_parameters_dic

#print("base_line: {}".format(base_line()))
#print("random_combi: {}".format(random_combi()))