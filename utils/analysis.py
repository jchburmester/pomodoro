import os
import numpy as np
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader


# set parent directory
parent_dir = os.path.dirname(os.getcwd())

import yaml
from yaml.loader import SafeLoader

'''Reading the yaml file with all parameters.'''

# Opening the yaml file
with open('./config.yaml', 'r') as stream:

    try:
        # Converting yaml document to python object
        parameters = yaml.load(stream, Loader=SafeLoader)
        # Create matrix from dictionary values
        para_np = np.array([[val for val in p['values']] for p in parameters['configuration'].values()])
        # Store values
        config_keys = parameters['configuration'].keys()
        # create dictionary with all the parameters
        config_dict = {key: para_np[i] for i, key in enumerate(config_keys)}
    
    except yaml.YAMLError as e:
        print(e)


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


def get_all_runs():
    
    para_np_array = np.array(para_np).flatten()
    df = pd.DataFrame(columns=para_np_array)

    # get all runs and sort them by their best validation accuracy (second last row of each run)
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))
    # sort runs by validation accuracy
    all_runs.sort(key=lambda x: read_logs_with_pd(os.path.join(parent_dir, 'runs', x, 'logs.csv'))['val_accuracy'].iloc[-2], reverse=True)
    
    for run in all_runs:
        try:
            run_parameters = get_parameters(run)
            run_parameters_np = np.array([val for val in run_parameters.values()])
            run_parameters_np = run_parameters_np[1:-3]

            result = [any(x in s for x in run_parameters_np) for s in para_np_array]
            df = df.append(pd.Series(result, index=df.columns), ignore_index=True)
        except:
            print('Error in run: ', run)
       
    return df

def create_heatmap(df):

    df_array = df.to_numpy(dtype=float)
    para_np_list = np.array(para_np).flatten().tolist()

    # Create a heatmap with Seaborn
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(df_array, cmap='coolwarm', annot=False, fmt='.2f', ax=ax)

    # Set the axis labels
    ax.set_xlabel(para_np_list, fontsize=14)
    ax.set_ylabel('Run, from best to worst', fontsize=14)

    # Set the plot title
    ax.set_title('Heatmap Parameters On/Off for all Runs', fontsize=18)

    # Show the plot
    return plt.show()

if __name__ == '__main__':
    create_heatmap(get_all_runs())

    

    