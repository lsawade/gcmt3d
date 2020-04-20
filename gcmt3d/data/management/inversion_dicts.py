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

from .create_process_paths import get_processing_list
from .create_process_paths import get_windowing_list
from gcmt3d.asdf.utils import write_yaml_file

import logging
from ...log_util import modify_logger

# Create logger
logger = logging.getLogger(__name__)
modify_logger(logger)

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


def create_full_inversion_dict_list(cmt_file_db,  process_obs_dir,
                                    process_syn_dir, window_process_dir,
                                    npar=9):
    """
    :param cmt_file_db: the cmtsolution file in the database.
    :type cmt_file_db: str
    :param process_obs_dir: directory with the process parameter files for
                            the observed data
    :type cmt_file_db: str
    :param process_syn_dir: directory with the process parameter files for
                            the synthetic data
    :type process_syn_dir: str
    :param window_process_dir: path to the process directory
    :type window_process_dir: str
    :param npar: Number of parameters to invert for
    :type npar: int
    :param verbose: if True verbose output
    :type verbose: bool

    :return: list of full inversion dictionaries


    The inversion dictionaries contain all necessary info to start the
    inversion. Each Inversion dictionary has then a respective band (and
    possible wavetype).

    **Dictionary structure:**
    .. code-block:: yaml

        Window_file: "path/to/window/file"
        ASDF_dict:
            Mrr: "path/to/CMT_rr.h5"
            Mtt: "path/to/CMT_tt.h5"
            Mpp: "path/to/CMT_pp.h5"
            Mrt: "path/to/CMT_rt.h5"
            Mrp: "path/to/CMT_rp.h5"
            Mtp: "path/to/CMT_tp.h5"
            dep: "path/to/CMT_dep.h5"
            lon: "path/to/CMT_lon.h5"
            lat: "path/to/CMT_lat.h5"
            ctm: "path/to/CMT_ctm.h5"
            hdr: "path/to/CMT_hdr.h5"

    """

    # Get CMT dir
    cmt_dir = os.path.dirname(cmt_file_db)

    # inversion dictionary
    output_dir = os.path.join(cmt_dir, "inversion", "inversion_dicts")

    # Get the window file list
    window_path_list, win_list = get_windowing_list(cmt_file_db,
                                                    window_process_dir)

    # Get the processing list
    processing_list, obs_list, syn_list = get_processing_list(cmt_file_db,
                                                              process_obs_dir,
                                                              process_syn_dir)

    # Create empty list of dictionaries
    inv_dict_list = []
    outfile_list = []

    for window_file in win_list:

        # Get Basename
        window_file_name = os.path.basename(window_file)

        # extract info
        band_and_wave_type = (window_file_name.split(".")[1]).split("#")

        # Get Passband
        band = band_and_wave_type[0]

        # Create empty dictionary
        inv_dict = dict()

        # Create Inversion Dictionary
        inv_dict["window_file"] = window_file

        # Add ASDF dictionary
        # Initialize filedict
        asdf_dict = dict()

        # Set observed filename
        asdf_dict["obsd"] = [s for s in obs_list
                             if band in os.path.basename(s)][0]

        # Set observed filename
        asdf_dict["synt"] = [s for s in syn_list
                             if (band in os.path.basename(s))
                             and ("synthetic_CMT." in os.path.basename(s))][0]

        # Creating a dictionary from synthetic files and their names
        for key, value in PARMAP.items():

            # Get perturbed file with value
            pert_file = [s for s in syn_list
                         if (band in os.path.basename(s))
                         and ("synthetic_%s." % value
                              in os.path.basename(s))]

            if len(pert_file) > 1:
                logger.error("Found %d synthetic perturbation files."
                             % len(pert_file))
                raise ValueError

            elif len(pert_file) != 0:
                # Write into file dict
                asdf_dict[key] = pert_file[0]

        inv_dict["asdf_dict"] = asdf_dict

        # Append dictionary to inversion dictionary list.
        inv_dict_list.append(inv_dict)

        output_file_name = os.path.join(output_dir,
                                        "inversion" + window_file_name[7:])

        outfile_list.append(output_file_name)

    return inv_dict_list, outfile_list


def create_g3d_inversion_dict_list(cmt_file_db,  process_obs_dir,
                                   process_syn_dir, window_process_dir):
    """
    :param cmt_file_db: the cmtsolution file in the database.
    :type cmt_file_db: str
    :param process_obs_dir: directory with the process parameter files for
                            the observed data
    :type cmt_file_db: str
    :param process_syn_dir: directory with the process parameter files for
                            the synthetic data
    :type process_syn_dir: str
    :param window_process_dir: path to the process directory
    :type window_process_dir: str
    :param npar: Number of parameters to invert for
    :type npar: int
    :param verbose: if True verbose output
    :type verbose: bool

    :return: list of full inversion dictionaries


    The inversion dictionaries contain all necessary info to start the
    inversion. Each Inversion dictionary has then a respective band (and
    possible wavetype).

    **Dictionary structure:**
    .. code-block:: yaml

        Window_file: "path/to/window/file"
        ASDF_dict:
            Mrr: "path/to/observed.h5"
            Mtt: "path/to/synthetic.h5"


    """

    # Get CMT dir
    cmt_dir = os.path.dirname(cmt_file_db)

    # New synt dir =
    new_synt_dir = os.path.join(cmt_dir, "inversion", "inversion_output",
                                "cmt3d", "new_synt")


    # inversion dictionary
    output_dir = os.path.join(cmt_dir, "inversion", "inversion_dicts")

    # Get the window file list
    window_path_list, win_list = get_windowing_list(cmt_file_db,
                                                    window_process_dir)

    # Get the processing list
    processing_list, obs_list, syn_list = get_processing_list(cmt_file_db,
                                                              process_obs_dir,
                                                              process_syn_dir)

    syn_list = glob.glob(os.path.join(new_synt_dir, "*synt*h5"))

    # Create empty list of dictionaries
    inv_dict_list = []
    outfile_list = []

    for window_file in win_list:

        # Get Basename
        window_file_name = os.path.basename(window_file)

        # extract info
        band_and_wave_type = (window_file_name.split(".")[1]).split("#")

        # Get Passband
        band = band_and_wave_type[0]

        # Create empty dictionary
        inv_dict = dict()

        # Create Inversion Dictionary
        inv_dict["window_file"] = window_file

        # Add ASDF dictionary
        # Initialize filedict
        asdf_dict = dict()

        # Set observed filename
        asdf_dict["obsd"] = [s for s in obs_list
                             if band in os.path.basename(s)][0]

        "200709121110A.9p_ZT.040_100_synt.h5"
        "200709121110A.9p_ZT.090_250_synt.h5"

        # Set observed filename
        asdf_dict["synt"] = [s for s in syn_list
                             if (band in os.path.basename(s))
                             and ("synt." in os.path.basename(s))][0]

        inv_dict["asdf_dict"] = asdf_dict

        # Append dictionary to inversion dictionary list.
        inv_dict_list.append(inv_dict)

        output_file_name = os.path.join(output_dir,
                                        "grid" + window_file_name[7:])

        outfile_list.append(output_file_name)

    return inv_dict_list, outfile_list


def write_inversion_dicts(dict_list, filename_list):
    """ Writes inversion dictionaries to the given output directory

    :param dict_list: List of inversion dictionaries

    :param filename_list: list of paths in string format corresponding to
                          writing location of the dictionaries in the list

    """

    # Loop over lists.
    for inv_dict, filename in zip(dict_list, filename_list):

        # Write file
        write_yaml_file(inv_dict, filename)


if __name__ == "__main__":
    for key, value in create_inversion_dict(sys.argv[1], "017_040").items():
        print(key, value)
