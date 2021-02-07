import os.path as p


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
    conversion_list = []
    for _file in ['observed', 'CMT',
                  'CMT_rr', 'CMT_tt', 'CMT_pp', 'CMT_rt', 'CMT_rp', 'CMT_tp',
                  'CMT_lat', 'CMT_lon', 'CMT_depth']:
        conversion_list.append(p.join(conversion_path_dir, _file + ".yml"))

    return conversion_list
