import os
import numpy as np
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader
from matplotlib.colors import ListedColormap

from pandas_profiling import ProfileReport

# set parent directory
parent_dir = os.path.dirname(os.getcwd())

'''Reading the yaml file with all parameters.'''

# Opening the yaml file
with open('utils/config.yaml', 'r') as stream:

    try:
        # Converting yaml document to python object
        parameters = yaml.load(stream, Loader=SafeLoader)
        # Create matrix from dictionary values
        para_np = np.array([[val for val in p['values']] for p in parameters['configuration'].values()])
        # Store keys
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

def get_above_80():
    '''A function to get all runs with validation accuracy above 80%.
    It returns all runs above 80% accuracy and its parameters and power draw.'''
    para_np_array = np.array(para_np).flatten()
    df = pd.DataFrame(columns=['run', 'val_accuracy', 'parameters', 'power_draw'])

    # get all runs and sort them by their best validation accuracy (second last row of each run)
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))
    # sort runs by validation accuracy
    all_runs.sort(key=lambda x: read_logs_with_pd(os.path.join(parent_dir, 'runs', x, 'logs.csv'))['val_accuracy'].iloc[-2], reverse=True)
    # only take runs with validation accuracy above 80%
    all_runs = [run for run in all_runs if read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['val_accuracy'].iloc[-2] > 0.75]
    # store in the dataframe the parameters and power draw for each run
    for run in all_runs:
        try:
            run_parameters = get_parameters(run)
            run_parameters_np = np.array([val for val in run_parameters.values()])
            run_parameters_np = run_parameters_np[1:-3]

            result = [any(x in s for x in run_parameters_np) for s in para_np_array]
            power_draw = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['gpu_power_W'].mean()
            val_accuracy = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['val_accuracy'].iloc[-2]
            df = df.append({'run': run, 'val_accuracy': val_accuracy, 'parameters': result, 'power_draw': power_draw}, ignore_index=True)
        except:
            print('Error in run: ', run)

    # get rid of column runs
    df = df.drop(columns=['run'])
    
    return df


def create_heatmap(df, config_dict, para_np):

    para_np_list = np.array(para_np).flatten().tolist()

    # access only the parameters column in the dataframe
    df_new = df['parameters'].apply(pd.Series)

    print(df_new)
    df_array = df_new.to_numpy(dtype=float)
    
    #config_keys = list(config_dict.keys())
    #config_keys_list = [config_keys[i//4] if i%4==0 else '' for i in range(len(para_np_list))]

    # Create a heatmap with Seaborn
    colors = ['#e6eaeb', '#3b0e1a'] # define your own color scheme
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(df_array, cmap=cmap, aspect='auto')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9, wspace=0.2, hspace=0.6)

    ax.set_ylabel('') # Remove y-axis label

    # Set the tick labels
    ax.set_xticks(np.arange(len(para_np_list)))
    ax.set_xticklabels(para_np_list, rotation=90)

    # Set the top labels
    #ax2 = ax.twiny()
    #ax2.set_xticks(np.arange(len(para_np_list)))
    #ax2.set_xticks(ax.get_xticks() + 0.5)
    #ax2.set_xticklabels(config_keys_list, rotation=45)
    #ax2.tick_params(axis='x')

    # Set the plot title
    ax.set_title('Parameter heatmap for all runs above 75% accuracy', fontsize=16)
    
    # have as y label on the left the accuracy values from the original dataframe column val_accuracy
    ax.set_yticks(np.arange(len(df['val_accuracy'])))
    ax.set_yticklabels(df['val_accuracy'], rotation=0)

    # have as y label on the right the power draw values from the original dataframe column power_draw
    ax2 = ax.twinx()
    ax2.set_yticks(np.arange(len(df['power_draw'])))
    ax2.set_yticks(ax.get_yticks())
    # round the power draw values to 2 decimals
    ax2.set_yticklabels([round(x, 2) for x in df['power_draw']], rotation=0)
    ax2.tick_params(axis='y')


    plt.show()

def create_summary_csv():
    # Define the path to the runs directory
    runs_dir = "runs/"

    # Create an empty list to hold the data for each run
    data = []

    # Loop through each subdirectory in the runs directory
    for subdir in os.listdir(runs_dir):
        # Check if the subdirectory is a valid run (has a parameters.yaml file)
        params_file = os.path.join(runs_dir, subdir, "parameters.yaml")
        if os.path.exists(params_file):
            # Load the parameters from the YAML file
            with open(params_file, "r") as f:
                params = yaml.safe_load(f)

            # Remove seed
            params.pop("seed", None)
            
            # Read the logs.csv file into a DataFrame
            logs_file = os.path.join(runs_dir, subdir, "logs.csv")
            logs_df = pd.read_csv(logs_file)
            
            # Get the last accuracy and val_accuracy values from the logs
            last_acc = logs_df["accuracy"].iloc[-2]
            last_val_acc = logs_df["val_accuracy"].iloc[-2]
            
            # Calculate the average GPU power consumption
            avg_gpu_watts = logs_df["gpu_power_W"].mean()
            
            # Add the data to the list
            data.append({
                "run": subdir,
                **params,
                "last_accuracy": last_acc,
                "last_val_accuracy": last_val_acc,
                "avg_gpu_power_watts": avg_gpu_watts
            })

    # Convert the data to a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv("runs_summary.csv", index=False)

    # Generate summary statistics using pandas_profiling
    profile = ProfileReport(df, title="Runs Summary Report")
    profile.to_file(output_file="runs_summary_report.html")

if __name__ == '__main__':
    

    create_heatmap(get_above_80(), config_dict, para_np)



