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

from analysis import get_best_5_runs, get_baseline


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TO-DO: 
- (in the end) transfer path from main and delete path in main call (bottom line of this file)
- get run number of head folder
- heatmap
- 70/80/90 accuracy und Auto/Mikrowellen GPU
- Include this in credits: https://github.com/jorisschellekens/borb-examples#321-fixedcolumnwidthtable
- Loss plots: Title, for which runs
- GPU: Title, for which runs, which GPU variables
- Accuracy plots: Title, for which runs
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

def create_plot_loss(logs):

    # exclude the pretraining and posttraining values for the loss plot
    epochs = logs["epoch"][1:-1].tolist()
    train_loss = logs["loss"][1:-1].tolist()
    val_loss = logs["val_loss"][1:-1].tolist()

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epochs, train_loss, label="Training Loss")
    ax.plot(epochs, val_loss, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss plot")
    ax.legend()

    return plt.gcf()


def create_plot_acc(logs):

    # exclude the pretraining and posttraining values for the accuracy plot
    epochs = logs["epoch"][1:-1].tolist()
    train_acc = logs["accuracy"][1:-1].tolist()
    val_acc = logs["val_accuracy"][1:-1].tolist()
    
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epochs, train_acc, label="Training Accuracy")
    ax.plot(epochs, val_acc, label="Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy in %")
    ax.set_title("Accuracy plot")
    ax.legend()

    return plt.gcf()


def create_plot_gpu(logs):

    # include the pretraining and posttraining values for gpu plot
    epochs = logs["epoch"].tolist()
    gpu = logs["gpu_power_W"].tolist()
    
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epochs, gpu, label="GPU Power Consumption")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Consumption in kWh")
    ax.set_title("GPU plot")
    ax.legend()

    return plt.gcf()

def create_heatmap(df):
    
    # create heatmap
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(df, cmap="hot", interpolation="nearest")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Runs")
    ax.set_title("Heatmap")
    
    return plt.gcf()


def create_evaluation_plot(logs, title):

    all_epochs = logs["epoch"].tolist()
    # exclude pretraining and posttraining values for loss and accuracy plots
    train_epochs = logs["epoch"][1:-1].tolist()

    train_loss = logs["loss"][1:-1].tolist()
    val_loss = logs["val_loss"][1:-1].tolist()
    train_acc = logs["accuracy"][1:-1].tolist()
    val_acc = logs["val_accuracy"][1:-1].tolist()
    gpu = logs["gpu_power_W"].tolist()

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(train_epochs, train_loss, label="Training Loss")
    ax.plot(train_epochs, val_loss, label="Validation Loss")
    ax.plot(train_epochs, train_acc, label="Training Accuracy")
    ax.plot(train_epochs, val_acc, label="Validation Accuracy")
    ax.plot(all_epochs, gpu, label="GPU Power Consumption")
    ax.set_xlabel("Epochs")
    #ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    return plt.gcf()


def read_params_yaml(yaml_file):
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


def read_logs_with_pd(path_to_csv_file):

    if type(path_to_csv_file) == list:
        logs = []
        # Opening the csv file, convert into dictionary, and append to list
        for path in path_to_csv_file:
            df = pd.read_csv(path)
            logs.append(df)
        
        return logs

    else:
        df = pd.read_csv(path_to_csv_file)

        return df


def read_logs(path_to_csv_file):

    if type(path_to_csv_file) == list:
        logs = []
        # Opening the csv file, convert into dictionary, and append to list
        for path in path_to_csv_file:
            csv_file = open(path)
            csv_reader = csv.DictReader(csv_file)
            data = {}
            for row in csv_reader:
                data.update(row)
            csv_file.close()
            logs.append(data)
        
        return logs

    
    else:
        # Opening the csv file and convert into dictionary
        csv_file = open(path_to_csv_file)
        csv_reader = csv.DictReader(csv_file)
        data = {}
        for row in csv_reader:
            data.update(row)
        csv_file.close()

        return data



def create_table_fivebest(layout, keys, n_params, n_params_base, base, first, second, third, fourth, fifth, mode="acc"):
    """
    Creates a table that shows the 5 best runs for the specified mode.
    Parameters:
    ----------
        layout : 
            Layout of current pdf
        mode : str
            Either accuracy, gpu or efficiency. Will be the parameter that the winner table is based on.
    """

    # makes sure that pdf is only created if a valid mode is given
    if mode != "acc" and mode != "gpu" and mode != "eff":
        print("Mode must be either 'acc', 'gpu' or 'eff'.")
        return
    
    elif mode == "acc":

        # table heading
        layout.add(Paragraph("Table 1: The five best runs according to accuracy.",
                             font="Helvetica-Bold", font_size=10))

        # acc table
        layout.add(
            FixedColumnWidthTable(number_of_columns=6, number_of_rows=7,
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
            .add(Paragraph("Mean GPU Power Draw (in W) per epoch", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Efficiency (acc/gpu)", font="Helvetica-Bold"))
            
            # baseline run
            .add(Paragraph("B", font="Helvetica-Bold"))
            .add(Paragraph("1", font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(base['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(base['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params_base), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((base['val_accuracy'].iloc[-2])*100/(base['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # best run according to accuracy
            .add(Paragraph("1."))
            .add(Paragraph(str(keys[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((first['val_accuracy'].iloc[-2])*100/(first['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph(str(keys[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((second['val_accuracy'].iloc[-2])*100/(second['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph(str(keys[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((third['val_accuracy'].iloc[-2])*100/(third['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph(str(keys[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fourth['val_accuracy'].iloc[-2])*100/(fourth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph(str(keys[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))            
            .add(Paragraph(str(np.round(fifth['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fifth['val_accuracy'].iloc[-2])*100/(fifth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

    elif mode == "gpu":       
        # table heading
        layout.add(Paragraph("Table 2: The five best runs according to GPU.",
                             font="Helvetica-Bold", font_size=10))
        
        # gpu table
        layout.add(
            FixedColumnWidthTable(number_of_columns=6, number_of_rows=7, 
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
            .add(Paragraph("Mean GPU Power Draw (in W) per epoch", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Efficiency (acc/gpu)", font="Helvetica-Bold"))

            # baseline run
            .add(Paragraph("B", font="Helvetica-Bold"))
            .add(Paragraph("1", font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(base['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(base['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params_base), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((base['val_accuracy'].iloc[-2])*100/(base['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # best run according to gpu
            .add(Paragraph("1."))
            .add(Paragraph(str(keys[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((first['val_accuracy'].iloc[-2])*100/(first['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph(str(keys[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((second['val_accuracy'].iloc[-2])*100/(second['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph(str(keys[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((third['val_accuracy'].iloc[-2])*100/(third['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph(str(keys[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fourth['val_accuracy'].iloc[-2])*100/(fourth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph(str(keys[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fifth['val_accuracy'].iloc[-2])*100/(fifth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

    elif mode == "eff":
        # table heading
        layout.add(Paragraph("Table 3: The five best runs according to efficiency (acc/gpu).",
                             font="Helvetica-Bold", font_size=10))
        
        # gpu table
        layout.add(
            FixedColumnWidthTable(number_of_columns=6, number_of_rows=7, 
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
            .add(Paragraph("Mean GPU Power Draw (in W) per epoch", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Efficiency (acc/gpu)", font="Helvetica-Bold"))

            # baseline run
            .add(Paragraph("B", font="Helvetica-Bold"))
            .add(Paragraph("1", font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(base['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(base['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params_base), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((base['val_accuracy'].iloc[-2])*100/(base['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # best run according to gpu
            .add(Paragraph("1."))
            .add(Paragraph(str(keys[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(first['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[0]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((first['val_accuracy'].iloc[-2])*100/(first['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph(str(keys[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(second['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[1]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((second['val_accuracy'].iloc[-2])*100/(second['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph(str(keys[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(third['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[2]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((third['val_accuracy'].iloc[-2])*100/(third['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph(str(keys[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fourth['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[3]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fourth['val_accuracy'].iloc[-2])*100/(fourth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph(str(keys[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['gpu_power_W'].mean(), 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round(fifth['val_accuracy'].iloc[-2]*100, 3)), font="Helvetica-oblique"))
            .add(Paragraph(str(n_params[4]), font="Helvetica-oblique"))
            .add(Paragraph(str(np.round((fifth['val_accuracy'].iloc[-2])*100/(fifth['gpu_power_W'].mean()), 3)), font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )


def create_table_params(layout, params, vals, mode="acc"):
    """
    Creates a table that displays the parameters of the winning run.
    """    

    # table heading according to mode
    if mode != "acc" and mode != "gpu" and mode != "eff" and mode != "base":
        print("Mode must be either 'acc', 'gpu' or 'eff'.")
        return
    elif(mode=="acc"):
        layout.add(Paragraph("Table 4: Parameter values for the winning run in accuracy.",
                             font="Helvetica-Bold", font_size=10))
    elif(mode=="gpu"):
        layout.add(Paragraph("Table 5: Parameter values for the winning run in GPU.",
                             font="Helvetica-Bold", font_size=10))
    elif(mode=="eff"):
        layout.add(Paragraph("Table 6: Parameter values for the winning run in efficiency.",
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

    # first extract all keys from acc file
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

    # extract parameters of the baseline run
    base_params, base_values = [str(elem) for elem in base['parameters'].keys()], [str(elem) for elem in base['parameters'].values()]

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

    #logs_full = read_logs_with_pd(logs_path)

    # extract number of parameters from parameter files
    num_params_acc = [int(value[-2]) for value in acc_values]
    num_params_gpu = [int(value[-2]) for value in gpu_values]
    num_params_eff = [int(value[-2]) for value in eff_values]
    num_params_base = int(base_values[-2])

    # pdf setup
    document = Document()
    page = Page()
    document.add_page(page)
    layout = SingleColumnLayout(page)
    #layout.vertical_margin = page.get_page_info().get_height() * Decimal(0.01)

    create_table_fivebest(layout, acc_keys, num_params_acc, num_params_base, base['logs'], logs_acc1, logs_acc2, logs_acc3, logs_acc4, logs_acc5, mode="acc")
    create_table_fivebest(layout, gpu_keys, num_params_gpu, num_params_base, base['logs'], logs_gpu1, logs_gpu2, logs_gpu3, logs_gpu4, logs_gpu5, mode="gpu")
    create_table_fivebest(layout, eff_keys, num_params_eff, num_params_base, base['logs'], logs_eff1, logs_eff2, logs_eff3, logs_eff4, logs_eff5, mode="eff")
    create_table_params(layout, first_acc_params, first_acc_values, mode="acc")
    create_table_params(layout, first_gpu_params, first_gpu_values, mode="gpu")
    create_table_params(layout, first_eff_params, first_eff_values, mode="eff")
    create_table_params(layout, base_params, base_values, mode="base")

    layout.add(Chart(create_evaluation_plot(logs_acc1, title="Best accuracy run"), width=Decimal(400), height=Decimal(256)))
    layout.add(Chart(create_evaluation_plot(logs_gpu1, title="Best GPU run"), width=Decimal(400), height=Decimal(256)))
    layout.add(Chart(create_evaluation_plot(logs_eff1, title="Best efficiency run"), width=Decimal(400), height=Decimal(256)))

    # store
    with open("result.pdf", "wb") as out_file_handle:
        PDF.dumps(out_file_handle, document)


# Function to create a pdf file with all results, called in main.py
def create_pdf():
        best_acc = get_best_5_runs('acc')
        lowest_gpu = get_best_5_runs('gpu')
        best_eff = get_best_5_runs('eff')
        baseline = get_baseline()
        main(best_acc, lowest_gpu, best_eff, baseline)