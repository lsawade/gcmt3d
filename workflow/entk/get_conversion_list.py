import os.path as p
from glob import glob


def get_conversion_list(conversion_path_dir: str):
    """Takes in conversion path directory and spits out list of conversion
    path files.

    Args:
        window_path_dir (str): directory with window path files

    Returns:
        list of paths to window path files

    Last modified: Lucas Sawade, 2020.09.25 12.00 (lsawade@princeton.edu)
    """

    # Create empty dictionary
    conversion_list = glob(p.join(conversion_path_dir, '*'))

    return conversion_list
