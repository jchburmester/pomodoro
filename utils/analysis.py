import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from yaml.loader import SafeLoader
from matplotlib.colors import ListedColormap
#from pandas_profiling import ProfileReport

# Set parent directory
parent_dir = os.path.dirname(os.getcwd())


#####################################################################################
############################ Loading data ###########################################
#####################################################################################

# Opening the yaml file
with open('./utils/config.yaml', 'r') as stream:

    try:
        # Converting yaml document to python object
        parameters = yaml.load(stream, Loader=SafeLoader)
        # Create matrix from dictionary values
        para_np = np.array([[val for val in p['values']] for p in parameters['configuration'].values()])
        # Store keys
        config_keys = parameters['configuration'].keys()
        # Store keys and values
        config_dict = {key: para_np[i] for i, key in enumerate(config_keys)}
    
    except yaml.YAMLError as e:
        print(e)


#####################################################################################
############################ Helper functions #######################################
#####################################################################################

def read_logs_with_pd(csv_file):
    """
    Read the logs.csv file with pandas
    """
    df = pd.read_csv(csv_file)

    return df


def get_parameters(run):
    """
    Get the parameters of a run
    """
    with open(os.path.join(parent_dir, 'runs', run, 'parameters.yaml'), 'r') as stream:
        try:
            parameters = yaml.load(stream, Loader=SafeLoader)
            return parameters

        except yaml.YAMLError as e:
            print(e)


#####################################################################################
############################ Analysis ###############################################
#####################################################################################

# To get the best 5 runs based on validation accuracy, gpu power draw, or efficiency
def get_best_5_runs(mode):
    """
    Get the best 5 runs based on validation accuracy, gpu power draw, or efficiency
    
        Parameter:
        ----------
            mode : string
                options: 'acc', 'gpu', 'eff'
    """

    metrics = []
    top_5_runs_dict = {}

    for run in os.listdir(os.path.join(parent_dir, 'runs')):
        logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))
        if mode == 'acc':
            metrics.append(logs['val_accuracy'].iloc[-2])
        elif mode == 'gpu':
            metrics.append(logs['gpu_power_W'].mean())
        elif mode == 'eff':
            metrics.append((logs['val_accuracy'].iloc[-2])/(logs['gpu_power_W'].mean()))
        else:
            print('Invalid mode. Please choose from acc, gpu, or eff.')

    metrics = pd.Series(metrics)

    if mode == 'acc':
        top_5 = metrics.nlargest(5).index
    elif mode == 'gpu':
        top_5 = metrics.nsmallest(5).index
    elif mode == 'eff':
        top_5 = metrics.nlargest(5).index

    # Store best 5 runs in dictionary
    for run in top_5:
        top_5_runs_dict[run+1] = {'logs': os.path.join(parent_dir, 'runs', str(run+1).zfill(3), 'logs.csv'),
                                  'parameters': os.path.join(parent_dir, 'runs', str(run+1).zfill(3), 'parameters.yaml')}
    
    return top_5_runs_dict


def get_all_runs():
    """
    Get all runs and their respective parameter values
    """
    
    # Get all parameter titles
    para_np_array = np.array(para_np).flatten()
    df = pd.DataFrame(columns=para_np_array)

    # Get all runs and sort them by their best validation accuracy
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))
    all_runs.sort(key=lambda x: read_logs_with_pd(os.path.join(parent_dir, 'runs', x, 'logs.csv'))['val_accuracy'].iloc[-2], reverse=True)
    
    # Store in the dataframe the parameters for each run
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


def get_baseline():
    """
    Get baseline run for later comparison
    """

    # Get baseline logs and parameters
    baseline_logs = read_logs_with_pd(os.path.join(parent_dir, 'runs', '001', 'logs.csv'))
    baseline_parameters = get_parameters('001')

    # Store in dictionary
    baseline_dict = {'logs': baseline_logs,
                        'parameters': baseline_parameters}
    
    return baseline_dict


def get_above_80():
    """
    Get all runs with validation accuracy above 80%
    """

    # Get all parameter titles
    para_np_array = np.array(para_np).flatten()
    df = pd.DataFrame(columns=['run', 'val_accuracy', 'parameters', 'power_draw'])

    # Get all runs and sort them by their best validation accuracy
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))
    all_runs.sort(key=lambda x: read_logs_with_pd(os.path.join(parent_dir, 'runs', x, 'logs.csv'))['val_accuracy'].iloc[-2], reverse=True)
    all_runs = [run for run in all_runs if read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['val_accuracy'].iloc[-2] > 0.80]

    # Store in the dataframe the parameters and power draw for each run
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

    # Drop column runs
    df = df.drop(columns=['run'])
    
    return df


def create_summary_csv():
    """
    Extract the parameters and logs for each run and stores them in a CSV file
    """
    
    runs_dir = "./runs/"
    data = []

    # Extract data for each run
    for subdir in sorted(os.listdir(runs_dir)):
        params_file = os.path.join(runs_dir, subdir, "parameters.yaml")
        
        if os.path.exists(params_file):

            with open(params_file, "r") as f:
                params = yaml.safe_load(f)

            params.pop("seed", None)
            logs_file = os.path.join(runs_dir, subdir, "logs.csv")
            logs_df = pd.read_csv(logs_file)
            
            # Get the last accuracy and val_accuracy values from the logs
            last_acc = logs_df["accuracy"].iloc[-2]
            last_val_acc = logs_df["val_accuracy"].iloc[-2]
            
            # Calculate the average GPU power consumption
            avg_gpu_watts = logs_df["gpu_power_W"].mean()

            # Calculate the average time per epoch
            logs_df["time"] = pd.to_datetime(logs_df["time"])
            avg_time_per_epoch = logs_df["time"].diff().mean().total_seconds()
            avg_time_per_epoch = round(avg_time_per_epoch, 2)
            
            # Add the data to the list
            data.append({
                "run": subdir,
                **params,
                "last_accuracy": last_acc,
                "last_val_accuracy": last_val_acc,
                "avg_gpu_power_watts": avg_gpu_watts,
                "avg_seconds_per_epoch": avg_time_per_epoch
            })
    
    # Convert the data to a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv("runs_summary.csv", index=False)

    # Generate summary statistics using pandas_profiling
    profile = ProfileReport(df, title="Runs Summary Report")
    profile.to_file(output_file="runs_summary_report.html")


#####################################################################################
############################ Statistics for Report ##################################
#####################################################################################

def corr():
    """
    Calculate the correlation between accuracy and power draw
    """

    # Get all runs and store in a dataframe the power draw and accuracy
    df = pd.DataFrame(columns=['power_draw', 'accuracy'])
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))

    for run in all_runs:
        try:
            power_draw = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['gpu_power_W'].mean()
            accuracy = get_parameters(run)['test_accuracy']
            df = df.append({'power_draw': power_draw, 'accuracy': accuracy}, ignore_index=True)

        except:
            print('Error in run: ', run)
    
    # Calculate the correlation
    corr = df['power_draw'].corr(df['accuracy'])
    print('Correlation between accuracy and power draw: ', corr)


def gpu_per_parameter():
    """
    Store in a new dataframe the power draws for each parameter
    """

    # Extract power draw and parameters for each run
    df = pd.DataFrame(columns=['parameters', 'power_draw'])
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))

    for run in all_runs:
        try:
            power_draw = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['gpu_power_W'].mean()
            parameters = get_parameters(run)
            del parameters['model']
            del parameters['seed']
            del parameters['n_parameters']
            del parameters['test_accuracy']

            df = df.append({'parameters': parameters, 'power_draw': power_draw}, ignore_index=True)

        except:
            print('Error in run: ', run)


    triplets_list = []

    # For every combination of key value pairs in the dataframe, store the power draw in the dictionary
    for i in range(len(df)):
        for key, value in df['parameters'][i].items():
            # store triplets of parameter, value and power draw and append it to a list
            triplet = (key, value, df['power_draw'][i])
            triplets_list.append(triplet)

    # Combine all triplets that share the first two elements (parameter and value) and store the power draw in a list for each parameter value combination
    triplets_dict = {}
    for triplet in triplets_list:
        if triplet[0:2] in triplets_dict:
            triplets_dict[triplet[0:2]].append(triplet[2])
        else:
            triplets_dict[triplet[0:2]] = [triplet[2]]
    
    triplets_dict = dict(sorted(triplets_dict.items(), key=lambda item: item[0]))
    
    # Drop batch size 1
    triplets_dict.pop(('batch_size', 1))
    return triplets_dict, df


def stats(df):
    """
    Inference statistics to see how the parameters affect the power draw
    """

    # For every run, check for the parameter value and store it under the respective column together with the power draw
    df_new = pd.DataFrame(columns=['run_id', 'power_draw','preprocessing', 'augmentation', 'batch_size', 'lr', 'lr_schedule', 'partitioning', 'optimizer', 'optimizer_momentum', 'internal', 'precision'])
    
    for i in range(len(df)):
        for key, value in df['parameters'][i].items():
            df_new.loc[i, key] = value
        df_new.loc[i, 'power_draw'] = df['power_draw'][i]
        df_new.loc[i, 'run_id'] = i
    
    # Drop baseline run and run_id's
    df_new = df_new.drop(0)
    df_new = df_new.drop(['run_id'], axis=1)

    # One-hot encode the categorical variables
    df_sm = pd.get_dummies(df_new, columns=['preprocessing', 'augmentation', 'batch_size', 'lr', 'lr_schedule', 'partitioning', 'optimizer', 'optimizer_momentum', 'internal', 'precision'])

    # Split the data
    X_train, _, y_train, _ = train_test_split(df_sm.drop(['power_draw'], axis=1), df_sm['power_draw'].astype(float), test_size=0.2)

    # Fit a multiple linear regression model using statsmodels
    model = sm.OLS(y_train, X_train).fit()

    # Get coefficients and p-values to see which parameters are significant for the power draw
    alpha = 0.1
    coef = model.params
    p_values = model.pvalues

    batches = [coef[i:i+4] for i in range(0, len(coef), 4)]
    batches = [[(coef.index[i], coef[i], p_values[i]) for i in range(len(coef)) if coef.index[i] in batch] for batch in batches]

    winners = []
    for batch in batches:
        # sort the batch by the coef
        batch = sorted(batch, key=lambda x: x[1])
        # for the lowest coef, check if the p-value is below the alpha value
        for i in range(len(batch)):
            if batch[i][2] < alpha:
                winners.append(batch[i][0])
                break
            else:
                continue    
        
    # Create a dictionary with the winning parameters
    result_dict = {}
    for winner in winners:
        for key in config_dict:
            for value in config_dict[key]:
                column_name = key + "_" + value
                if winner == column_name:
                    result_dict[key] = value
    
    # Print the winning dictionary
    print('The 10 parameters that have the most significant effect on the power draw are:')
    print(result_dict)

    # Print the model summary if required
    # print(model.summary())

    # Store the model summary in a csv file
    summary = model.summary()
    summary_as_csv = summary.as_csv()
    with open("statistics_summary.csv", "w") as text_file:
        text_file.write(summary_as_csv)

    return result_dict


#####################################################################################
############################ Visualization ##########################################
#####################################################################################

def plot_distributions():
    """
    Plotting the distributions of accuracy and power draw
    """

    # Get all runs and store in a dataframe the power draw and accuracy
    df = pd.DataFrame(columns=['power_draw', 'accuracy'])
    all_runs = os.listdir(os.path.join(parent_dir, 'runs'))

    for run in all_runs:
        try:
            power_draw = read_logs_with_pd(os.path.join(parent_dir, 'runs', run, 'logs.csv'))['gpu_power_W'].mean()
            accuracy = get_parameters(run)['test_accuracy']
            df = df.append({'power_draw': power_draw, 'accuracy': accuracy}, ignore_index=True)

        except:
            print('Error in run: ', run)

    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    sns.violinplot(y='power_draw', data=df, ax=ax1, color='lightgrey', inner='point')
    sns.violinplot(y='accuracy', data=df, ax=ax2, color='lightgrey', inner='point')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9, wspace=0.2, hspace=0.6)
    
    # Labels
    ax1.set_xticklabels(['Power draw'])
    ax2.set_xticklabels(['Accuracy'])
    
    # Titles
    ax1.set_title('Power draw distribution')
    ax2.set_title('Accuracy distribution')
    
    # Show means and medians
    ax1.axhline(df['power_draw'].mean(), color='red', linestyle='dashed', linewidth=1)
    ax1.axhline(df['power_draw'].median(), color='black', linestyle='dashed', linewidth=1)
    ax2.axhline(df['accuracy'].mean(), color='red', linestyle='dashed', linewidth=1)
    ax2.axhline(df['accuracy'].median(), color='black', linestyle='dashed', linewidth=1)
    
    # Save the plot
    plt.savefig('distributions.png', dpi=300)


def create_heatmap(df, para_np):
    """
    Creating a heatmap of the selected runs
    """

    # Get all parameter titles
    para_np_list = np.array(para_np).flatten().tolist()

    # access only the parameters column in the dataframe
    df_new = df['parameters'].apply(pd.Series)

    df_array = df_new.to_numpy(dtype=float)
    
    # Create a heatmap
    colors = ['#e6eaeb', '#3b0e1a'] # define your own color scheme
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(df_array, cmap=cmap, aspect='auto')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9, wspace=0.2, hspace=0.6)

    # Title and labels
    ax.set_ylabel('') # Remove y-axis label
    ax.set_xticks(np.arange(len(para_np_list)))
    ax.set_xticklabels(para_np_list, rotation=90)
    ax.set_title('Parameter heatmap for all runs above 75% accuracy', fontsize=16)
    ax.set_yticks(np.arange(len(df['val_accuracy'])))
    ax.set_yticklabels(df['val_accuracy'], rotation=0)
    ax2 = ax.twinx()
    ax2.set_yticks(np.arange(len(df['power_draw'])))
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([round(x, 2) for x in df['power_draw']], rotation=0)
    ax2.tick_params(axis='y')

    plt.show()


def plot_triplets(dic, _):
    """
    Plot the power draw per parameter value
    """

    fig, ax = plt.subplots(figsize=(30, 15))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)

    # Title and labels
    ax.set_title('Power draw per parameter value', fontsize=16)
    ax.set_xlabel('Parameter value', fontsize=14)
    ax.set_ylabel('Power draw (W)', fontsize=14)
    ax.set_xticks(np.arange(len(dic.keys())))
    ax.set_xticklabels([key[0] + ' ' + str(key[1]) for key in dic.keys()], rotation=90)
    ax.set_yticks(np.arange(0, 100, 10))
    ax.set_yticklabels(np.arange(0, 100, 10), rotation=0)
    ax.grid(True)
    ax.hlines([np.mean(dic[key]) for key in dic.keys()], xmin=np.arange(len(dic.keys())) - 0.4, xmax=np.arange(len(dic.keys())) + 0.4, color='red', label='Mean')
    ax.legend()

    # Plot the data
    for i, key in enumerate(dic.keys()):
        ax.scatter([i] * len(dic[key]), dic[key], s=50, alpha=0.5)

    plt.show()


#####################################################################################
############################ Main function ##########################################
#####################################################################################

def analysis():
    _, df = gpu_per_parameter()
    return stats(df)


#####################################################################################

if __name__ == '__main__':
    # create_summary_csv()
    # create_heatmap(get_above_80(), para_np)
    #plot_triplets(gpu_p_p)
    #_, df = gpu_per_parameter()
    #print(stats(df))
    #corr()
    plot_distributions()
    #pass