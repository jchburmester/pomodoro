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
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal

from analysis import get_5_best_acc, get_5_lowest_gpu

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


def read_logs_with_pd(csv_file):
    df = pd.read_csv(csv_file)

    return df


def read_logs(csv_file):
    # Opening the csv file and convert into dictionary
    csv_file = open(csv_file)
    csv_reader = csv.DictReader(csv_file)
    data = {}
    for row in csv_reader:
        data.update(row)
    csv_file.close()

    return data


def create_table_fivebest(layout, first, second, third, fourth, fifth, mode="acc"):
    """
    Creates a table that shows the 5 best runs for the specified mode.
    Parameters:
    ----------
        layout : 
            Layout of current pdf
        mode : str
            Either accuracy or gpu. Will be the parameter that the winner list is based on.
    """
    # makes sure that pdf is only created if a valid mode is given
    if mode != "acc" and mode != "gpu":
        print("Mode must be either 'acc' or 'gpu'")
        return
    
    elif mode == "acc" :

        # table heading
        layout.add(Paragraph("Table 1: The five best runs according to accuracy.",
                             font="Helvetica-Bold", font_size=10))

        # acc table
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
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("GPU (in kWh)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Energy Quotient", font="Helvetica-Bold"))
            
            # best run according to accuracy
            .add(Paragraph("1"))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(first.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph(first.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(second.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph(second.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(third.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph(third.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(fourth.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph(fourth.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(fifth.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph(fifth.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

    else:       
        # table heading
        layout.add(Paragraph("Table 2: The five best runs according to GPU.",
                             font="Helvetica-Bold", font_size=10))
        
        # gpu table
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
            .add(Paragraph("GPU (in kWh)", font="Helvetica-Bold"))
            .add(Paragraph("Accuracy (in %)", font="Helvetica-Bold"))
            .add(Paragraph("Number of Parameters", font="Helvetica-Bold"))
            .add(Paragraph("Energy Quotient", font="Helvetica-Bold"))

            # best run according to gpu
            .add(Paragraph("1"))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(first.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph(first.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # second best run
            .add(Paragraph("2."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(second.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph(second.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # third best run
            .add(Paragraph("3."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(third.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph(third.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # fourth best run
            .add(Paragraph("4."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(fourth.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph(fourth.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))

            # fifth best run
            .add(Paragraph("5."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph(fifth.get('gpu_power_W'), font="Helvetica-oblique"))
            .add(Paragraph(fifth.get('accuracy'), font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

def create_table_params(layout, params, vals, mode="acc"):
    """
    Creates a table that displays the parameters of the winning run.
    """    

    # table heading according to mode
    if mode != "acc" and mode != "gpu":
        print("Mode must be either 'acc' or 'gpu'")
        return
    elif(mode=="acc"):
        layout.add(Paragraph("Table 3: Parameter values for the winning run in accuracy.",
                             font="Helvetica-Bold", font_size=10))
    else:
        layout.add(Paragraph("Table 3: Parameter values for the winning run in GPU.",
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


def main(paths):

    acc_winners = paths[0]
    gpu_winners = paths[1]

    first_acc_params, first_acc_values = read_params_yaml(params_path)
    first_gpu_params, first_gpu_values = read_params_yaml(params_path)

    logs_first = read_logs(logs_path)
    logs_second = read_logs(logs_path)
    logs_third = read_logs(logs_path)
    logs_fourth = read_logs(logs_path)
    logs_fifth = read_logs(logs_path)

    logs_full = read_logs_with_pd(logs_path)

    # pdf setup
    document = Document()
    page = Page()
    document.add_page(page)
    layout = SingleColumnLayout(page)
    #layout.vertical_margin = page.get_page_info().get_height() * Decimal(0.01)

    create_table_fivebest(layout, logs_first, logs_second, logs_third, logs_fourth, logs_fifth, mode="acc")
    create_table_fivebest(layout, logs_first, logs_second, logs_third, logs_fourth, logs_fifth, mode="gpu")
    create_table_params(layout, first_acc_params, first_acc_values, mode="acc")
    create_table_params(layout, first_gpu_params, first_gpu_values, mode="gpu")

    layout.add(Chart(create_plot_loss(logs_full), width=Decimal(300), height=Decimal(256)))
    layout.add(Chart(create_plot_acc(logs_full), width=Decimal(300), height=Decimal(256)))
    layout.add(Chart(create_plot_gpu(logs_full), width=Decimal(300), height=Decimal(256)))

    # store
    with open("result.pdf", "wb") as out_file_handle:
        PDF.dumps(out_file_handle, document)


if __name__ == "__main__":
    
    ### !!! delete following line later and instead transfer path from main
    paths_to_winners = [get_5_best_acc, get_5_lowest_gpu]
    main(paths_to_winners)