"""

This script contains functions to create pycmt3d inversion dictionaries. They
have to have following setup:

:param asdf_file_dict: the dictionary which provides the path
                       information of asdf file, for example:
                       {"obsd":"/path/obsd/asdf", "synt":"/path/synt/asdf",
                       "Mrr": "/path/Mrr/synt/asdf", ...}

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

"""

import os
import glob
import sys

PARMAP = {"Mrr": "CMT_rr",
          "Mtt": "CMT_tt",
          "Mpp": "CMT_pp",
          "Mrt": "CMT_rt",
          "Mrp": "CMT_rp",
          "Mtp": "CMT_tp",
          "dep": "CMT_depth",
          "lon": "CMT_lon",
          "lat": "CMT_lat",
          "ctm": "CMT_ctm",
          "hdr": "CMT_hdr"}


def create_inversion_dict(processed_dir, bandstring):
    """ This creates a dictionary with all files necessary simulation files for
    the inversion.

    :param npar: Number of parameters to be inverted.
    :param processed_dir: directory with the processed data.
    :param bandstring: string that is contained in the file name and checks
                       for the bandpass filtering.
    :return: dictionary with data files

    """

    # Get absolute path of process dir in case it wasn't given
    processed_dir = os.path.abspath(processed_dir)

    # Find files in processed seismos directory

    obsd_file = glob.glob(os.path.join(processed_dir,
                                       "*observed." + bandstring + ".h5"))[0]
    synt_file = glob.glob(os.path.join(processed_dir,
                                       "*synthetic_CMT."
                                       + bandstring + ".h5"))[0]

    # Initialize filedict
    filedict = dict()

    # Set observed filename
    filedict["obsd"] = obsd_file

    # Set observed filename
    filedict["synt"] = synt_file

    # Creating a dictionary from synthetic files and their names
    for key, value in PARMAP.items():

        # Get perturbed file with value
        pert_file = glob.glob(os.path.join(processed_dir,
                                           "*synthetic*" + value + "*"
                                           + bandstring + "*.h5"))

        if len(pert_file) != 0:
            # Write into file dict
            filedict[key] = pert_file[0]

    return filedict


if __name__ == "__main__":
    for key, value in create_inversion_dict(sys.argv[1], "017_040").items():
        print(key, value)
