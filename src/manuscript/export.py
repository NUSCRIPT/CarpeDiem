import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  # edit-able in illustrator
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = 'sans-serif'


import datetime
import os
import numpy as np

from manuscript import inout


def _extend_material_path(user, p, date=False):
    """
    Takes extension path p, and makes it as a subfile
    within material folder
    """

    p_b = os.path.join(
        inout.get_path_from_settings('materials_path'),
        user
        )

    inout.ensure_presence_of_directory(p_b)


    p = os.path.join(
        p_b,
        p)

    if date:
        [fo, fn] = os.path.split(p)
        [fb, ext] = os.path.splitext(fn)
        dt = datetime.datetime.today().strftime('%y%m%d_%H%M')
        p = os.path.join(fo, fb + '_' + dt + ext)

    return p


def image(user, p, date=False, **kwargs):
    """
    Will export a current figure;
        - makes font edit-able in illustrator
    Input:
        date    optional; default True: will insert
                                date, hour, minute before file
                                extension
    """

    mpl.rcParams['pdf.fonttype'] = 42  # edit-able in illustrator

    p = _extend_material_path(user, p)

    if date:
        [fo, fn] = os.path.split(p)
        [fb, ext] = os.path.splitext(fn)
        dt = datetime.datetime.today().strftime('%y%m%d_%H%M')
        p = os.path.join(fo, fb + '_' + dt + ext)

    inout.ensure_presence_of_directory(p)
    call_kwargs = {
        "bbox_inches": "tight"
    }
    call_kwargs.update(kwargs)

    mpl.pyplot.savefig(p, **call_kwargs)


def raster_image(user, p, dpi, date=False):
    """
    Will export a current figure with 600 dpi
    Input:
        p       str path to file
        dpi     int dots per inch
        date    optional; default True: will insert
                                date, hour, minute before file
                                extension
    """

    mpl.rcParams['pdf.fonttype'] = 42  # edit-able in illustrator

    p = _extend_material_path(user, p, date)
    inout.ensure_presence_of_directory(p)

    mpl.pyplot.savefig(p, dpi=dpi, bbox_inches='tight')


def full_frame(user, p, df, date=False, index=False):
    """
    Will export a dataframe to materials
        - makes font edit-able in illustrator
    Input:
        p                   subpath within material folder
        df                  dataframe to export
        date                optional; default True: will insert
                                date, hour, minute before file
                                extension
        index          optional; default True: will also
                                export the index
    """

    p = _extend_material_path(user, p)

    if p.endswith('.csv.gz'):
        p = p[:-3]
        compress = True
        file_format = 'csv'
    elif p.endswith('.csv'):
        compress = False
        file_format = 'csv'
    elif p.endswith('.xlsx'):
        file_format = 'xlsx'
    elif p.endswith('.parquet'):
        file_format = 'parquet'
    else:
        raise EnvironmentError(
            'No support for preseent file type.')

    if date:
        [fo, fn] = os.path.split(p)
        [fb, ext] = os.path.splitext(fn)
        dt = datetime.datetime.today().strftime('%y%m%d_%H%M')
        p = os.path.join(fo, fb + '_' + dt + ext)

    inout.ensure_presence_of_directory(p)

    if file_format == 'csv':
        if compress:
            p = p + '.gz'
            df.to_csv(p, compression='gzip', index=index)
        else:
            df.to_csv(p, index=index)
    elif file_format == 'xlsx':
        df.to_excel(p, index=index)
        
    elif file_format == 'parquet':
        if index==False:
            df.index = np.arange(0, len(df.index))
        df.to_parquet(p)