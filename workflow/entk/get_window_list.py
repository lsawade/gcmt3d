import os.path as p


def get_window_list(window_path_dir: str):
    """Takes in window path directory and spits out list of window
    path files.

    Args:
        window_path_dir (str): directory with window path files

    Returns:
        list of paths to window path files

    Last modified: Lucas Sawade, 2020.09.25 12.00 (lsawade@princeton.edu)
    """

    # Create empty dictionary
    window_list = []

    for _wave in ["body", "surface", "mantle"]:
        window_list.append(p.join(window_path_dir, f"{_wave}.window.yml"))

    return window_list
