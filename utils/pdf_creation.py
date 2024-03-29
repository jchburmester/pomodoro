from borb.pdf import SingleColumnLayout
from borb.pdf import FixedColumnWidthTable
from borb.pdf import Paragraph
from borb.pdf import TableCell
from borb.pdf import Document
from borb.pdf import Page
from borb.pdf import PDF
from borb.pdf import Chart
import yaml
import csv
import os
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal

from utils.analysis import get_best_5_runs, get_baseline
from utils.analysis import get_best_5_runs, get_baseline


"""
- heatmap function
- test execution from main.py
- statt number of parameters, plot seconds per epoch
"""

def create_heatmap(df):
    
    # create heatmap
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(df, cmap="hot", interpolation="nearest")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Runs")
    ax.set_title("Heatmap")
    
    return plt.gcf()


def create_gpu_plot(logs, key, title):
    """
    Create GPU plot for single run.

    Parameters:
    -----------
        logs : 
            logs for respective run
        key : int
            number of run

    Returns: 
    --------
        plot : matplotlib.figure.Figure
            GPU plot    
    """
    # convert values to lists for plotting
    all_epochs = logs["epoch"].tolist()
    gpu = logs["gpu_power_W"].tolist()

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(all_epochs, gpu, label="GPU Power Consumption")
    ax.set_xlabel("Epochs")
    ax.set_title(title + " (run number {})".format(key))
    ax.set_xticks(all_epochs[::2])
    # plot only every second xtick for better visibility
    ax.set_xticklabels(all_epochs[::2], visible=True)
    ax.legend()

    plot = plt.gcf()

    return plot


def create_evaluation_plot(logs, key, title):
    """
    For single run, visualization of loss, accuracy per epoch.

    Parameters:
    ----------
        logs :
            logs for respective run
        key : int
            number of run
        title : str
            plot title

    Returns:
    -------
        plot : matplotlib.figure.Figure
            the final plot for respective category
    
    """
    # convert values to lists for plotting
    train_epochs = logs["epoch"][1:-1].tolist()

    # exclude pretraining and posttraining values for loss and accuracy plots
    train_loss = logs["loss"][1:-1].tolist()
    val_loss = logs["val_loss"][1:-1].tolist()
    train_acc = logs["accuracy"][1:-1].tolist()
    val_acc = logs["val_accuracy"][1:-1].tolist()

    fig, ax1 = plt.subplots()

    # plot loss
    ax1.set_ylabel('Loss') 
    ax1.plot(train_epochs, train_loss, label="Training Loss", color="lightcoral")
    ax1.plot(train_epochs, val_loss, label="Validation Loss", color="darkred")
    ax1.tick_params(axis ='y')

    # plot accuracy on different y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy') 
    ax2.plot(train_epochs, train_acc, label="Training Accuracy", color="darkturquoise")
    ax2.plot(train_epochs, val_acc, label="Validation Accuracy", color="midnightblue")
    ax2.tick_params(axis ='y')

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right')

    ax1.set_xlabel('Epochs')
    ax1.set_title(title + " (run number {})".format(key))
    # plot only every second xtick for better visibility
    ax1.set_xticks(train_epochs[::2])
    ax1.set_xticklabels(train_epochs[::2], visible=True)

    plot = plt.gcf()

    return plot


def read_params_yaml(yaml_file):
    """
    Reads parameters from .yaml file
    
    Parameters:
    -----------
        yaml_file : str
            path to .yaml file

    Returns:
    --------
        params_keys : list
            parameter keys
        params_values : list
            parameter values
    """
    # Opening the yaml file
    with open(yaml_file, 'r') as stream:

        try:
            # Converting yaml document to python object
            parameters = yaml.load(stream, Loader=SafeLoader)
            # Create list from dictionary values
            params_values = [str(elem) for elem in parameters.values()]
            params_keys = [str(elem) for elem in parameters.keys()]

        except yaml.YAMLError as e:
            print(e)

    return params_keys, params_values


def translate_gpu(logs):
        """
        Translates GPU power consumption from W to kWh.
        """

        power_draw = logs['gpu_power_W']
        time = logs['time']
        time = pd.to_datetime(time)
        time = time.diff().dt.total_seconds()
        time = time[1:].reset_index(drop=True)
        power_draw = power_draw[1:].reset_index(drop=True)
        total_power_draw = (power_draw * time)/3600
        logs['gpu_power_W'] = total_power_draw

        return logs


def read_logs_with_pd(path_to_csv_file):
    """
    Reads logs from .csv file. Path can be either single path or list of paths.

    Parameters:
    ----------
        path_to_csv_file : str or list of strings
            path to the .csv file to be read
    
    Returns:
    --------
        df : pandas.core.frame.DataFrame
            dataframe with logs
    or
        logs : list
            list of logs
    """
    if type(path_to_csv_file) == list:
        logs = []
        # Opening the csv file, convert into dictionary, and append to list
        for path in path_to_csv_file:
            df = pd.read_csv(path)
            # interpolate missing values
            df = df.interpolate()
            logs.append(df)
            df = translate_gpu(df)
        
        return logs

    else:
        df = pd.read_csv(path_to_csv_file)
        df = translate_gpu(df)
        df = df.interpolate()

        return df
    

def create_table_fivebest(layout, keys, time, first, second, third, fourth, fifth, mode="acc"):
    """
    Creates table that shows the 5 best runs for the specified mode.

    Parameters:
    ----------
        layout : borb.pdf.canvas.layout.page_layout.multi_column_layout.SingleColumnLayout
            Layout of current pdf
        keys : list
            stores the run number
        time : list
            average time per epoch
        first : pandas.core.frame.DataFrame
            best run according to mode
        second : pandas.core.frame.DataFrame
            second best run according to mode
        third : pandas.core.frame.DataFrame
        fourth : pandas.core.frame.DataFrame
        fifth : pandas.core.frame.DataFrame
        mode : str
            Either accuracy, gpu or efficiency. Will be the parameter that the winner table is based on.
    """

    # make sure that pdf is only created if a valid mode is given
    if mode != "acc" and mode != "gpu" and mode != "eff":
        print("Mode must be either 'acc', 'gpu' or 'eff'.")
        return
    
    # table for five best runs according to accuracy
    elif mode == "acc":
        # table heading
        layout.add(Paragraph("Table 1: The five best runs according to accuracy.",
                             font="Helvetica-Bold", font_size=10))

        layout.add(
            FixedColumnWidthTable(number_of_columns=6, number_of_rows=6,
                                # first column should be smaller than remaining
                                column_widths=[Decimal(0.2), Decimal(1), Decimal(1), Decimal(1), Decimal(1), Decimal(1)])
            .add(
                TableCell(
                    Paragraph(" "),
                    border_top=False,
                    border_left=False,
                )
            )
            .add(Paragraph("Run Number", font="Helvetica-Bold"))
            .add(Paragraph("GPU Power Draw (in W/hr)", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Average Time per Epoch", font="Helvetica-Bold"))
            .add(Paragraph("Efficiency (acc/gpu)", font="Helvetica-Bold"))
            
            # best run
            .add(Paragraph("1."))
            .add(Paragraph(str(keys[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((first['val_accuracy'].iloc[-2])/(first['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph(str(keys[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((second['val_accuracy'].iloc[-2])/(second['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph(str(keys[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((third['val_accuracy'].iloc[-2])/(third['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph(str(keys[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fourth['val_accuracy'].iloc[-2])/(fourth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph(str(keys[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))            
            .add(Paragraph(str(np.round(fifth['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fifth['val_accuracy'].iloc[-2])/(fifth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

    # table for five best runs according to gpu
    elif mode == "gpu":       
        # table heading
        layout.add(Paragraph("Table 2: The five best runs according to GPU.",
                             font="Helvetica-Bold", font_size=10))

        layout.add(
            FixedColumnWidthTable(number_of_columns=6, number_of_rows=6, 
                                # first column should be smaller than remaining
                                column_widths=[Decimal(0.2), Decimal(1), Decimal(1), Decimal(1), Decimal(1), Decimal(1)])
            .add(
                TableCell(
                    Paragraph(" "),
                    border_top=False,
                    border_left=False,
                )
            )
            .add(Paragraph("Run Number", font="Helvetica-Bold"))
            .add(Paragraph("GPU Power Draw (in W/hr)", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Efficiency (acc/gpu)", font="Helvetica-Bold"))

            # best run according to gpu
            .add(Paragraph("1."))
            .add(Paragraph(str(keys[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((first['val_accuracy'].iloc[-2])/(first['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph(str(keys[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((second['val_accuracy'].iloc[-2])/(second['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph(str(keys[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((third['val_accuracy'].iloc[-2])/(third['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph(str(keys[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fourth['val_accuracy'].iloc[-2])/(fourth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph(str(keys[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fifth['val_accuracy'].iloc[-2])/(fifth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

    # table for five best runs according to efficiency
    elif mode == "eff":
        # table heading
        layout.add(Paragraph("Table 3: The five best runs according to efficiency (acc/gpu).",
                             font="Helvetica-Bold", font_size=10))
        
        layout.add(
            FixedColumnWidthTable(number_of_columns=6, number_of_rows=6, 
                                # first column should be smaller than remaining
                                column_widths=[Decimal(0.2), Decimal(1), Decimal(1), Decimal(1), Decimal(1), Decimal(1)])
            .add(
                TableCell(
                    Paragraph(" "),
                    border_top=False,
                    border_left=False,
                )
            )
            .add(Paragraph("Run Number", font="Helvetica-Bold"))
            .add(Paragraph("GPU Power Draw (in W/hr)", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Efficiency (acc/gpu)", font="Helvetica-Bold"))

            # best run according to eff
            .add(Paragraph("1."))
            .add(Paragraph(str(keys[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((first['val_accuracy'].iloc[-2])/(first['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph(str(keys[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((second['val_accuracy'].iloc[-2])/(second['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph(str(keys[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((third['val_accuracy'].iloc[-2])/(third['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph(str(keys[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fourth['val_accuracy'].iloc[-2])/(fourth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph(str(keys[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['val_accuracy'].iloc[-2], 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(time[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fifth['val_accuracy'].iloc[-2])/(fifth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )


def create_table_params(layout, params, vals, key=None, mode="acc"):
    """
    Creates a table that displays the parameters of the winning run, or the baseline run.

    Parameters:
    -----------
        layout : borb.pdf.canvas.layout.page_layout.multi_column_layout.SingleColumnLayout
            pdf layout
        params : list 
            parameter names of winning run
        vals : list
            parameter values of winning run
        key : int
            run number of winning run
        mode : str
            either 'acc', 'gpu', 'eff' or 'base'
    """    

    # makes sure that right mode is given
    if mode != "acc" and mode != "gpu" and mode != "eff" and mode != "base":
        print("Mode must be either 'acc', 'gpu' or 'eff'.")
        return
    
    # table heading according to mode    
    elif(mode=="acc"):
        layout.add(Paragraph("Table 4: Parameter values for the winning run in accuracy (run number {}).".format(key),
                             font="Helvetica-Bold", font_size=10))
    elif(mode=="gpu"):
        layout.add(Paragraph("Table 5: Parameter values for the winning run in GPU (run number {}).".format(key),
                             font="Helvetica-Bold", font_size=10))
    elif(mode=="eff"):
        layout.add(Paragraph("Table 6: Parameter values for the winning run in efficiency (run number {}).".format(key),
                             font="Helvetica-Bold", font_size=10))
    elif(mode=="base"):
        layout.add(Paragraph("Table 7: Parameter values for the baseline run.",
                             font="Helvetica-Bold", font_size=10))

    layout.add(
        FixedColumnWidthTable(number_of_columns=2, number_of_rows=14)

        .add(Paragraph("Parameter", font="Helvetica-Bold"))
        .add(Paragraph("Value", font="Helvetica-Bold"))
        .add(Paragraph(params[0]))
        .add(Paragraph(vals[0]))
        .add(Paragraph(params[1]))
        .add(Paragraph(vals[1]))
        .add(Paragraph(params[2]))
        .add(Paragraph(vals[2]))
        .add(Paragraph(params[3]))
        .add(Paragraph(vals[3]))
        .add(Paragraph(params[4]))
        .add(Paragraph(vals[4]))
        .add(Paragraph(params[5]))
        .add(Paragraph(vals[5]))
        .add(Paragraph(params[6]))
        .add(Paragraph(vals[6]))
        .add(Paragraph(params[7]))
        .add(Paragraph(vals[7]))
        .add(Paragraph(params[8]))
        .add(Paragraph(vals[8]))
        .add(Paragraph(params[9]))
        .add(Paragraph(vals[9]))
        .add(Paragraph(params[10]))
        .add(Paragraph(vals[10]))
        .add(Paragraph(params[11]))
        .add(Paragraph(vals[11]))
        .add(Paragraph(params[12]))
        .add(Paragraph(vals[12]))
        .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
    )


def main(acc, gpu, eff, base):
    """
    Creates PDF file. The PDF will contain:
    1) Tables with measures for the five best runs in each category
    2) Tables with the parameters of the best run in each category
    3) Acc, loss and gpu plots for the best efficiency run

    Parameters:
    ----------
        acc : dict
            logs and parameters of five best runs according to accuracy
        gpu : dict
            logs and parameters of five best runs according to gpu
        eff : dict
            logs and parameters of five best runs according to efficiency
        base : dict
            logs and parameters of baseline model
    """

    # first extract all keys from files
    acc_keys = list(acc.keys())
    gpu_keys = list(gpu.keys())
    eff_keys = list(eff.keys())

    # store log paths of all runs with list comprehension
    acc_logs_paths = [acc[key]['logs'] for key in acc_keys]
    gpu_logs_paths = [gpu[key]['logs'] for key in gpu_keys]
    eff_logs_paths = [eff[key]['logs'] for key in eff_keys]

    # store parameter paths of all runs
    acc_parameters_paths = [acc[key]['parameters'] for key in acc_keys]
    gpu_parameters_paths = [gpu[key]['parameters'] for key in gpu_keys]
    eff_parameters_paths = [eff[key]['parameters'] for key in eff_keys]

    # extract average time per epoch
    time_acc = [acc[key]['time'] for key in acc_keys]
    time_gpu = [gpu[key]['time'] for key in gpu_keys]
    time_eff = [eff[key]['time'] for key in eff_keys]

    # extract parameters of the top run for each category
    first_acc_params, first_acc_values = read_params_yaml(acc_parameters_paths[0])
    first_gpu_params, first_gpu_values = read_params_yaml(gpu_parameters_paths[0])
    first_eff_params, first_eff_values = read_params_yaml(eff_parameters_paths[0])

    # extract parameters of all runs for each category
    acc_params = [read_params_yaml(path)[0] for path in acc_parameters_paths]
    acc_values = [read_params_yaml(path)[1] for path in acc_parameters_paths]
    gpu_params = [read_params_yaml(path)[0] for path in gpu_parameters_paths]
    gpu_values = [read_params_yaml(path)[1] for path in gpu_parameters_paths]
    eff_params = [read_params_yaml(path)[0] for path in eff_parameters_paths]
    eff_values = [read_params_yaml(path)[1] for path in eff_parameters_paths]

    # get top 5 logs for each category
    logs_acc1, logs_acc2, logs_acc3, logs_acc4, logs_acc5 = read_logs_with_pd(acc_logs_paths)
    logs_gpu1, logs_gpu2, logs_gpu3, logs_gpu4, logs_gpu5 = read_logs_with_pd(gpu_logs_paths)
    logs_eff1, logs_eff2, logs_eff3, logs_eff4, logs_eff5 = read_logs_with_pd(eff_logs_paths)

    # pdf setup
    document = Document()
    page = Page()
    document.add_page(page)
    layout = SingleColumnLayout(page)

    # display credits
    layout.add(Paragraph("PDF creation inspired by: https://github.com/jorisschellekens/borb-examples#321-fixedcolumnwidthtable", font="Helvetica", font_size=Decimal(8)))

    # displays information about the five best runs according to respective category
    create_table_fivebest(layout, acc_keys, time_acc, logs_acc1, logs_acc2, logs_acc3, logs_acc4, logs_acc5, mode="acc")
    create_table_fivebest(layout, gpu_keys, time_gpu, logs_gpu1, logs_gpu2, logs_gpu3, logs_gpu4, logs_gpu5, mode="gpu")
    create_table_fivebest(layout, eff_keys, time_eff, logs_eff1, logs_eff2, logs_eff3, logs_eff4, logs_eff5, mode="eff")
    
    # for the best run of each category, display parameters in table
    create_table_params(layout, first_acc_params, first_acc_values, acc_keys[0], mode="acc")
    create_table_params(layout, first_gpu_params, first_gpu_values, gpu_keys[0], mode="gpu")
    create_table_params(layout, first_eff_params, first_eff_values, eff_keys[0], mode="eff")

    # for the best efficiency run, visualization of acc, loss and gpu per epoch. Plot GPU in extra plot because of different x- and y-value range
    layout.add(Chart(create_evaluation_plot(logs_eff1, eff_keys[0], title="Best efficiency run"), width=Decimal(400), height=Decimal(256)))
    layout.add(Chart(create_gpu_plot(logs_eff1, eff_keys[0], title="Best efficiency run, GPU values"), width=Decimal(400), height=Decimal(256)))

    # store
    with open("result.pdf", "wb") as out_file_handle:
        PDF.dumps(out_file_handle, document)


def create_pdf():
        """
        Initiates PDF creation with all results (called in main.py)
        """
        best_acc = get_best_5_runs('acc')
        lowest_gpu = get_best_5_runs('gpu')
        best_eff = get_best_5_runs('eff')
        baseline = get_baseline()
        main(best_acc, lowest_gpu, best_eff, baseline)


if __name__ == "__main__":
    create_pdf()
