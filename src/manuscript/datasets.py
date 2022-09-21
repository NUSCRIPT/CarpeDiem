import os
import pandas as pd


from manuscript import inout

def primary_patient_data(version='latest'):
    """
    Primary dataset with information about patients.

    Input:
        version   str, optional, e.g.: '220825_1544'

    Output

        primary_patient_data   dataframe

    """

    if version=='latest':
        version = '220830_1307'


    p = inout.get_material_path(
        f'general/03_overwrite_PF_Cr/03data-external_{version}.csv.gz'
        )


    df = pd.read_csv(p, index_col=0).reset_index(drop=True)
    return df



def secondary_patient_data(version):
    """
    Patient data with additional information, such as UMAP data and
    cluster assignment.

    """


    p = inout.get_material_path(
        f'{version}.csv.gz'
        )

    df = pd.read_csv(p)

    return df