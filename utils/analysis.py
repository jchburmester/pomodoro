import os
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

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


def get_best_acc():
    val_acc = []

    # get validation accuracy for each run
    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        val_acc.append(logs['val_accuracy'].iloc[-2])

    # get best 5 runs, sorted automatically
    val_acc = pd.Series(val_acc)
    best_5_runs = val_acc.nlargest(5).index

    # for best runs, get index and paths to logs and parameters file
    best_5_runs_dict = {}
    
    for run in best_5_runs:
        best_5_runs_dict[run+1] = {'logs': os.path.join(parent_dir, 'runs', str(run+1).zfill(3), 'logs.csv'),
                                   'parameters': os.path.join(parent_dir, 'runs', str(run+1).zfill(3), 'parameters.yaml')}
    
    return best_5_runs_dict


def get_lowest_gpu():
    power_draw = []
    
    # get power draw for each run
    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        power_draw.append(logs['gpu_power_W'].mean())
    
    # get lowest 5 runs, sorted automatically
    power_draw = pd.Series(power_draw)
    lowest_5_runs = power_draw.nsmallest(5).index

    # for lowest runs, get index and paths to logs and parameters file
    lowest_5_runs_dict = {}
    for run in lowest_5_runs:
        lowest_5_runs_dict[run+1] = {'logs': os.path.join(parent_dir, 'runs', str(run+1).zfill(3), 'logs.csv'),
                                     'parameters': os.path.join(parent_dir, 'runs', str(run+1).zfill(3), 'parameters.yaml')}
  
    return lowest_5_runs_dict


def get_all_acc():
    all_val_acc = []

    # get validation accuracy for each run
    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        all_val_acc.append(logs['val_accuracy'].iloc[-2])

    return all_val_acc


def get_all_gpu():
    all_power_draw = []

    # get power draw for each run
    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        all_power_draw.append(logs['gpu_power_W'].mean())

    return all_power_draw


if __name__ == '__main__':
    best_5_acc = get_best_acc()
    lowest_5_gpu = get_lowest_gpu()
    

    