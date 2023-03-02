import os
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

test_acc = []
parameters_list = []

# set parent directory
parent_dir = os.path.dirname(os.getcwd())

def read_logs_with_pd(csv_file):
    df = pd.read_csv(csv_file)

    return df

def get_parameters(run):
    with open(os.path.join(parent_dir, 'runs', run, 'parameters.yaml'), 'r') as stream:
        try:
            parameters = yaml.load(stream, Loader=SafeLoader)
            return parameters

        except yaml.YAMLError as e:
            print(e)


def get_5_best_acc():
    val_acc = []

    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        #parameters_list.append(get_parameters(run))
        #test_acc.append(parameters['test_accuracy'])
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        val_acc.append(logs['val_accuracy'].iloc[-2])

    val_acc = pd.Series(val_acc)
    best_5_runs = val_acc.nlargest(5).index
    # get best run and print the result
    best = val_acc.nlargest(5).iloc[0]
    print(best)
    #best_5_parameters = [parameters_list[i] for i in best_5_runs]

    return best_5_runs

def get_5_lowest_gpu():
    power_draw = []

    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        power_draw.append(logs['gpu_power_W'].sum())
    
    power_draw = pd.Series(power_draw)
    lowest_5_runs = power_draw.nsmallest(5).index

    return lowest_5_runs


if __name__ == '__main__':
    best_5_acc = get_5_best_acc()
    lowest_5_gpu = get_5_lowest_gpu()
    print(best_5_acc)
    print(lowest_5_gpu)
    

    