import os
from glob import glob
from ...window.load_winfile_json import load_winfile_json


def check_number_of_windows(eventdir: str) -> int:
    """Checks the number of windows computted for a given event.

    Parameters
    ----------
    eventdir : str
        Directory of GCMT3D event in database.

    Returns
    -------
    int
        number of windows in total

    Raises
    ------
    ValueError
        Raised if something else than a directory is given.
    """

    eventdir = os.path.abspath(eventdir)

    if not os.path.isdir(eventdir):
        raise ValueError("Location is not a directory. "
                         "Event directory required.")
    windowdir = os.path.join(eventdir, "window_data")
    windowglob = os.path.join(windowdir, "*.windows.json")

    # Get all wavetype files.
    windowfiles = glob(windowglob)

    # Number of
    number_of_windows = 0

    for _file in windowfiles:

        print(_file)
        measurement_dict = load_winfile_json(_file, simple=True, v=True)

        for _key, _measurements in measurement_dict.items():
            number_of_windows += len(_measurements["wins"])

    return number_of_windows


def bin():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("eventdir", type=str,
                        help="filename for either a measurement or histogram file")
    args = parser.parse_args()
    print(check_number_of_windows(args.eventdir))
