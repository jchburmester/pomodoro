from borb.pdf import SingleColumnLayout
from borb.pdf import FixedColumnWidthTable
from borb.pdf import Paragraph
from borb.pdf import TableCell
from borb.pdf import Document
from borb.pdf import Page
from borb.pdf import PDF
import yaml
from yaml.loader import SafeLoader
import numpy as np
from decimal import Decimal

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TO-DO: 
- (in the end) transfer path from main and delete path in main call (bottom line of this file)

- Tabellenüberschriften
- replace acc and gpu tables with real values
- heatmap
- loss & accuracy plots
- (plot w&b)
- 70/80/90 accuracy und Auto/Mikrowellen GPU

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""

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


def create_table_fivebest(layout, mode="acc"):
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
        # create accuracy table
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
            .add(Paragraph("1"))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("2."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("3."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("4."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("5."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

    else:        
        # create gpu table
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
            .add(Paragraph("1"))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("2."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("3."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("4."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("5."))
            .add(Paragraph("run#", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            .add(Paragraph("acc", font="Helvetica-oblique"))
            .add(Paragraph("gpu", font="Helvetica-oblique"))
            # set padding on all cells
            .set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        )

def create_table_params(layout, params, vals):
    """
    Creates a table that displays the parameters of the winning run.
    """    
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


def main(params_path):

    first_params, first_values = read_params_yaml(params_path)

    # pdf setup
    document = Document()
    page = Page()
    document.add_page(page)
    layout = SingleColumnLayout(page)

    create_table_fivebest(layout, mode="acc")
    create_table_fivebest(layout, mode="gpu")
    create_table_params(layout, first_params, first_values)

    # store
    with open("result.pdf", "wb") as out_file_handle:
        PDF.dumps(out_file_handle, document)


if __name__ == "__main__":
    
    ### !!! delete following line later and instead transfer path from main
    path = "C:/Users/marle/Documents/Studium/Kurse/IANNwTF/pomodoro/pomodoro/runs/001/parameters.yaml"
    
    main(path)