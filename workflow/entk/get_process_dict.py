import os.path as p
from glob import glob
from copy import deepcopy


def get_process_dict(process_path_dir: str):
    """Takes in process path directory and spits out dictionary of process
    path files.

    .. code::

        {
            'body': List of process paths,
            'surface': List of process paths,
            'mantle': List of process paths
         }

    The reason for the split is simply that we can only process one dataset
    at a time, otherwise the sane file is accessed by two different processes.

    Args:
        process_path_dir (str): [description]

    Returns:
        dictionary of processpaths.
    """

    # Create empty dictionary
    wave_list = ["body", "surface", "mantle"]
    process_dict = {}

    for _wave in wave_list:
        process_list = glob(p.join(process_path_dir, '*' + _wave + '*'))

        if len(process_list) != 0:
            process_dict[_wave] = deepcopy(process_list)

    return process_dict
