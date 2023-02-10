import os
import numpy as np
import yaml
from yaml.loader import SafeLoader

# set parent directory
parent_dir = os.path.dirname(os.getcwd())

# function to get parameters from parameters.yaml files in selected run
def get_parameters(run):
    with open(os.path.join(parent_dir, 'runs', run, 'parameters.yaml'), 'r') as stream:
        try:
            parameters = yaml.load(stream, Loader=SafeLoader)
            print(parameters)
            return parameters

        except yaml.YAMLError as e:
            print(e)

# testing the function
get_parameters('0')

# combine this with training and gpu logs to run analysis
# store analysis results in a useful format to compare it to other runs
