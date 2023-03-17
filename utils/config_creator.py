import numpy as np
import yaml
from yaml.loader import SafeLoader

# Opening the yaml file
with open('utils/config.yaml', 'r') as stream:

    try:
        # Converting yaml document to python object
        parameters = yaml.load(stream, Loader=SafeLoader)
        # Create matrix from dictionary values
        para_np = np.array([[val for val in p['values']] for p in parameters['configuration'].values()])
        # Store values
        config_keys = parameters['configuration'].keys()

    except yaml.YAMLError as e:
        print(e)


#####################################################################################
############################ Configuration ##########################################
#####################################################################################

def base_line(keys=config_keys):
    """
    Returns the base line configuration as a dictionary.
    """
    base_line = para_np[:, 0]
    base_line_dic = dict(zip(keys, base_line))

    return base_line_dic


def random_config(keys=config_keys):
    """
    Returns a random configuration as a dictionary.
    Creates a boolean matrix with the same shape as the matrix of parameters.
    The matrix is filled with 0s and 1s. The 1s are randomly distributed.
    """
    # Set the number of rows and columns in the matrix
    num_rows, num_cols = para_np.shape

    # Create an empty matrix with the desired shape
    bool_matrix = np.empty((num_rows, num_cols))

    # Randomly select one element in each row and set it to 1
    for i in range(num_rows):
        idx = np.random.randint(num_cols)
        bool_matrix[i, idx] = 1
        bool_matrix[i, :idx] = 0
        bool_matrix[i, idx+1:] = 0

    # Transfer configuration from matrix to dictionary
    mask = bool_matrix == 1
    filtered_parameters = para_np[mask]
    filtered_parameters_dic = dict(zip(keys, filtered_parameters))

    return filtered_parameters_dic


#####################################################################################

# To test the functions
if __name__ == '__main__':
    print("base_line: {}".format(base_line()))
    print("random_configuration: {}".format(random_config()))