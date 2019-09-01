"""

This script contains functions to create path files for the processing of the
observed and synthetic data.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

"""

import os
import glob

from ...asdf.utils import smart_read_yaml
from ...asdf.utils import is_mpi_env
from ...asdf.utils import write_yaml_file

attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
        "CMT_depth", "CMT_lat", "CMT_lon"]


def create_process_path_obs(cmt_filename, process_dir, verbose=True):
    """ This function writes a yaml conversion path file for 1 simulation
    file. This file is later on need for the creation of ASDF files and the
    processing involved ASDF files.

    :param cmt_filename: cmt_filename in the database (Important this only
                         works if the directory structure exists already)
    :param process_dir: path to the directory containing all processing files.
    :param verbose: boolean on whether process info should be written

    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Output directory
    process_path_dir = os.path.join(cmt_dir, "seismograms",
                                    "process_paths")

    # Get all process possibilities
    process_param_files = glob.glob(os.path.join(process_dir, "*"))

    if verbose:
        print("Processing parameter files to be used:")
        print(process_param_files)

    for _i, process_param_file in enumerate(process_param_files):

        # Get band
        process_params = smart_read_yaml(process_param_file,
                                         mpi_mode=is_mpi_env())
        band = process_params["pre_filt"][1:-1]
        lP = 1 / band[1]  # Get low period bound from hz filtervalue
        hP = 1 / band[0]  # Get high period bound from hz filtervalue

        # Define input ASDF
        input_asdf = os.path.join(cmt_dir, "seismograms", "obs",
                                  "raw_observed.h5")
        input_tag = "obs"

        # Output file parameters
        output_asdf = os.path.join(cmt_dir, "seismograms",
                                   "processed_seismograms",
                                   "processed_observed.%03.0f_%03.0f.h5"
                                   % (lP, hP))
        output_tag = "processed_observed"

        # Pathfile directory
        yaml_file_path = os.path.join(process_path_dir,
                                      "process_observed.%03.0f_%03.0f.yml"
                                      % (lP, hP))

        # Create dictionary
        if verbose:
            print("Writing path file %s." % yaml_file_path)

        d = {"input_asdf": input_asdf,
             "input_tag": input_tag,
             "output_asdf": output_asdf,
             "output_tag": output_tag,
             "process_param_file": process_param_file}

        # Writing the directory to file
        write_yaml_file(d, yaml_file_path)


def create_process_path_syn(cmt_filename, process_dir, npar, verbose=True):
    """ This function writes a yaml processing path file all simulations
    file. This is needed for the processing of the ASDF files.


    :param cmt_filename: cmt_filename in the database (Important this only
        works if the directory structure exists already)
    :param process_dir: path to the directory containing all processing files.
    :param npar: number of parameters to invert for
    :param verbose: boolean on whether process info should be written
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Output directory
    process_path_dir = os.path.join(cmt_dir, "seismograms",
                                    "process_paths")

    # Get all process possibilities
    process_param_files = glob.glob(os.path.join(process_dir, "*"))

    if verbose:
        print("Processing parameter files to be used:")
        print(process_param_files)

    for _i, process_param_file in enumerate(process_param_files):

        for _j, at in enumerate(attr[:npar + 1]):

            # Get band
            process_params = smart_read_yaml(process_param_file,
                                             mpi_mode=is_mpi_env())
            band = process_params["pre_filt"][1:-1]
            lP = 1 / band[1]  # Get low period bound from hz filtervalue
            hP = 1 / band[0]  # Get high period bound from hz filtervalue

            # Define input ASDF
            input_asdf = os.path.join(cmt_dir, "seismograms", "syn",
                                      "%s.h5" % at)
            input_tag = "syn"

            # Output file parameters
            output_asdf = os.path.join(cmt_dir, "seismograms",
                                       "processed_seismograms",
                                       'processed_synthetic_'
                                       '%s.%03.0f_%03.0f.h5'
                                       % (at, lP, hP))
            output_tag = "processed_synthetic"

            # Pathfile directory
            yaml_file_path = os.path.join(process_path_dir,
                                          "process_synthetic_"
                                          "%s.%03.0f_%03.0f.yml"
                                          % (at, lP, hP))

            # Create dictionary
            if verbose:
                print("Writing path file %s." % yaml_file_path)

            d = {"input_asdf": input_asdf,
                 "input_tag": input_tag,
                 "output_asdf": output_asdf,
                 "output_tag": output_tag,
                 "process_param_file": process_param_file}

            # Writing the directory to file
            write_yaml_file(d, yaml_file_path)


def create_window_path(cmt_filename, window_process_dir,
                       figure_mode=False, verbose=True):
    """ This function writes a yaml processing path file all simulations
    file. This is needed for the processing of the ASDF files.

    :param cmt_filename: cmt_filename in the database (Important this only
                        works if the directory structure exists already)
    :param process_dir: path to the directory containing all processing files.
    :param npar: number of parameters to invert for
    :param verbose: boolean on whether process info should be written
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Output directory
    window_path_dir = os.path.join(cmt_dir, "window_data", "window_paths")

    # Get all process possibilities
    window_param_files = glob.glob(os.path.join(window_process_dir, "*"))

    if verbose:
        print("Processing parameter files to be used:")
        print(window_param_files)

    for _i, window_param_file in enumerate(window_param_files):

        # Get band
        window_process_params = smart_read_yaml(window_param_file,
                                                mpi_mode=is_mpi_env())
        # Get lower and upper period bound
        lP = window_process_params["default"]["min_period"]
        hP = window_process_params["default"]["max_period"]

        # Check BodyWave/SurfaceWave flag
        if "body" in os.path.basename(window_param_file):
            wave_type = "#body_wave"
        elif "surface" in os.path.basename(window_param_file):
            wave_type = "#surface_wave"
        else:
            wave_type = ""

        # Define observed ASDF
        obsd_asdf = os.path.join(cmt_dir, "seismograms",
                                 "processed_seismograms",
                                 "processed_observed.%03.0f_%03.0f.h5"
                                 % (lP, hP))
        obsd_tag = "processed_observed"

        # Synthetic ASDF
        synt_asdf = os.path.join(cmt_dir, "seismograms",
                                 "processed_seismograms",
                                 "processed_synthetic_CMT.%03.0f_%03.0f.h5"
                                 % (lP, hP))
        synt_tag = "processed_synthetic"

        # Output file parameters
        output_file = os.path.join(cmt_dir, "window_data",
                                   'windows.%03.0f_%03.0f%s.json'
                                   % (lP, hP, wave_type))

        # Pathfile directory
        yaml_file_path = os.path.join(window_path_dir,
                                      "windows.%03.0f_%03.0f%s.yml"
                                      % (lP, hP, wave_type))

        # Create dictionary
        if verbose:
            print("Writing path file %s." % yaml_file_path)

        d = {"obsd_asdf": obsd_asdf,
             "obsd_tag": obsd_tag,
             "synt_asdf": synt_asdf,
             "synt_tag": synt_tag,
             "output_file": output_file,
             "figure_mode": figure_mode,
             "window_param_file": window_param_file}

        # Writing the directory to file
        write_yaml_file(d, yaml_file_path)


def get_processing_list(cmt_file_db, process_obs_dir, process_syn_dir, npar=9,
                        verbose=False):
    """This function returns a list of all process path files. It is needed
    for the EnTK workflow. This way the processing of each ASDF file can be
    assigned to one task.

    :param cmt_file_db: the cmtsolution file in the database.
    :type cmt_file_db: str
    :param process_obs_dir: directory with the process parameter files for
                            the observed data
    :type cmt_file_db: str
    :param process_syn_dir: directory with the process parameter files for
                            the synthetic data
    :type process_syn_dir: str
    :param npar: Number of parameters to invert for
    :type npar: int
    :param verbose: verbose output if true
    :type verbose: bool
    :return: tuple of 3 lists - 1 all processing path files;
                                2 the observed output files;
                                3 the synthetic output files
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_file_db))

    # Output directory
    process_path_dir = os.path.join(cmt_dir, "seismograms",
                                    "process_paths")

    # Get all process possibilities
    process_obs_param_files = glob.glob(os.path.join(process_obs_dir, "*"))
    process_syn_param_files = glob.glob(os.path.join(process_syn_dir, "*"))

    # If verbose print all used observed and synthetic files
    if verbose:
        print("Processing parameter files to be used:\n")
        print("    Observed:\n")
        for process_file in process_obs_param_files:
            print("    " + process_file)
        print("\n    Synthetic:\n")
        for process_file in process_syn_param_files:
            print("    " + process_file)
        print(" ")

    # Create empty process_path file list
    process_path_file_list = []
    obs_output_file_list = []
    syn_output_file_list = []

    # Get observed path files
    for _i, process_param_file in enumerate(process_obs_param_files):
        # Get band
        process_params = smart_read_yaml(process_param_file,
                                         mpi_mode=is_mpi_env())
        band = process_params["pre_filt"][1:-1]
        lP = 1 / band[1]  # Get low period bound from hz filtervalue
        hP = 1 / band[0]  # Get high period bound from hz filtervalue

        # Output file parameters
        obs_output_file_list.append(os.path.join(cmt_dir, "seismograms",
                                                 "processed_seismograms",
                                                 "processed_observed"
                                                 ".%03.0f_%03.0f.h5"
                                                 % (lP, hP)))

        # Pathfile directory
        yaml_file_path = os.path.join(process_path_dir,
                                      "process_observed.%03.0f_%03.0f.yml"
                                      % (lP, hP))
        # Add the path file to list
        process_path_file_list.append(yaml_file_path)

    # Get synthetic path files
    for _i, process_param_file in enumerate(process_syn_param_files):
        for _j, at in enumerate(attr[:npar + 1]):
            # Get band
            process_params = smart_read_yaml(process_param_file,
                                             mpi_mode=is_mpi_env())
            band = process_params["pre_filt"][1:-1]
            lP = 1 / band[1]  # Get low period bound from hz filtervalue
            hP = 1 / band[0]  # Get high period bound from hz filtervalue

            # Output file parameters
            syn_output_file_list.append(os.path.join(cmt_dir, "seismograms",
                                                     "processed_seismograms",
                                                     'processed_synthetic_'
                                                     '%s.%03.0f_%03.0f.h5'
                                                     % (at, lP, hP)))

            # Pathfile directory
            yaml_file_path = os.path.join(process_path_dir,
                                          "process_synthetic_"
                                          "%s.%03.0f_%03.0f.yml"
                                          % (at, lP, hP))

            # Add the path file to list
            process_path_file_list.append(yaml_file_path)

    if verbose:
        print("Resulting Processing Path files:")
        print("--------------------------------\n")
        for process_file in process_path_file_list:
            print("    " + process_file)
        print(" ")

    return process_path_file_list, obs_output_file_list, syn_output_file_list


def get_windowing_list(cmt_file_db, window_process_dir, verbose=False):
    """This function returns a list of all windowing path files. It is needed
    for the EnTK workflow. This way the windowing of each passband ASDF file
    can be assigned to one task.

    :param cmt_file_db: the cmtsolution file in the database.
    :type cmt_file_db: str
    :param window_process_dir: path to the process directory
    :type window_process_dir: str
    :param verbose: if True verbose output
    :type verbose: bool

    :return:  tuple of 2 lists - 1 all windowing path files; 2 the output files
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_file_db))

    # Output directory
    window_path_dir = os.path.join(cmt_dir, "window_data", "window_paths")

    # Get all process possibilities
    window_param_files = glob.glob(os.path.join(window_process_dir, "*"))

    if verbose:
        print("Processing parameter files to be used:")
        print("--------------------------------------\n")
        for param_file in window_param_files:
            print(param_file)
        print(" ")

    window_processing_list = []
    output_file_list = []

    for _i, window_param_file in enumerate(window_param_files):

        # Get band
        window_process_params = smart_read_yaml(window_param_file,
                                                mpi_mode=is_mpi_env())
        # Get lower and upper period bound
        lP = window_process_params["default"]["min_period"]
        hP = window_process_params["default"]["max_period"]

        # Check BodyWave/SurfaceWave flag
        if "body" in os.path.basename(window_param_file):
            wave_type = "#body_wave"
        elif "surface" in os.path.basename(window_param_file):
            wave_type = "#surface_wave"
        else:
            wave_type = ""

        # Output file parameters
        output_file_list.append(os.path.join(cmt_dir, "window_data",
                                             'windows.%03.0f_%03.0f%s.json'
                                             % (lP, hP, wave_type)))

        # Pathfile directory
        yaml_file_path = os.path.join(window_path_dir,
                                      "windows.%03.0f_%03.0f%s.yml"
                                      % (lP, hP, wave_type))

        window_processing_list.append(yaml_file_path)

    # Create dictionary
    if verbose:
        print("Resulting Windowing Path files:")
        print("--------------------------------\n")
        for process_file in window_processing_list:
            print("    " + process_file)
        print(" ")

    return window_processing_list, output_file_list
