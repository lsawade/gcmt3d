"""

This script contains functions to create path files for the processing of the
observed and synthetic data.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2019

"""

import os
import glob
from ...asdf.utils import smart_read_yaml
from ...asdf.utils import write_yaml_file

attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
        "CMT_depth", "CMT_lat", "CMT_lon"]


def create_process_path_obs(cmt_filename, process_dir, verbose=True):
    """ This function writes a yaml conversion path file for 1 simulation
    file. This file is later on need for the creation of ASDF files and the
    processing involved ASDF files.

    Args:
        cmt_filename: cmt_filename in the database (Important this only works
                      if the directory structure exists already)
        process_dir: path to the directory containing all processing files.
        verbose: boolean on whether process info should be written
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Output directory
    process_path_dir = os.path.join(cmt_dir, "seismograms",
                                    "process_paths")

    # Get all process possibilities
    process_param_files = glob.glob(os.path.join(process_dir, "*"))

    for _i, process_param_file in enumerate(process_param_files):

        # Get band
        process_params = smart_read_yaml(process_param_file)
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
                                   "processed_observed_%03.0f_%03.0f.h5"
                                   % (lP, hP))
        output_tag = "proc_obs_%03.0f_%03.0f" % (lP, hP)

        # Pathfile directory
        yaml_file_path = os.path.join(process_path_dir,
                                      "process_observed_%03.0f_%03.0f.yml"
                                      % (lP, hP))

        # Create dictionary
        if verbose:
            print("Writing path file %s." % yaml_file_path)

        d = {"input_asdf": input_asdf,
             "input_tag": input_tag,
             "output_asdf": output_asdf,
             "output_tag": output_tag}

        # Writing the directory to file
        write_yaml_file(d, yaml_file_path)


def create_process_path_syn(cmt_filename, process_dir, npar, verbose=True):
    """ This function writes a yaml processing path file all simulations
    file. This is needed for the processing of the ASDF files.

    Args:
        cmt_filename: cmt_filename in the database (Important this only works
                      if the directory structure exists already)
        process_dir: path to the directory containing all processing files.
        npar: number of parameters to invert for
        verbose: boolean on whether process info should be written
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Output directory
    process_path_dir = os.path.join(cmt_dir, "seismograms",
                                    "process_paths")

    # Get all process possibilities
    process_param_files = glob.glob(os.path.join(process_dir, "*"))

    for _i, process_param_file in enumerate(process_param_files):

        for _j, at in enumerate(attr[:npar + 1]):

            # Get band
            process_params = smart_read_yaml(process_param_file)
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
                                       '%s_%03.0f_%03.0f.h5'
                                       % (at, lP, hP))
            output_tag = "proc_syn_%s_%03.0f_%03.0f" % (at, lP, hP)

            # Pathfile directory
            yaml_file_path = os.path.join(process_path_dir,
                                          "process_synthetic_"
                                          "%s_%03.0f_%03.0f.yml"
                                          % (at, lP, hP))

            # Create dictionary
            if verbose:
                print("Writing path file %s." % yaml_file_path)

            d = {"input_asdf": input_asdf,
                 "input_tag": input_tag,
                 "output_asdf": output_asdf,
                 "output_tag": output_tag}

            # Writing the directory to file
            write_yaml_file(d, yaml_file_path)


def create_window_path(cmt_filename, window_process_dir, npar,
                       figure_mode=False, verbose=True):
    """ This function writes a yaml processing path file all simulations
    file. This is needed for the processing of the ASDF files.

    Args:
        cmt_filename: cmt_filename in the database (Important this only works
                      if the directory structure exists already)
        process_dir: path to the directory containing all processing files.
        npar: number of parameters to invert for
        verbose: boolean on whether process info should be written
    """

    # CMT directory Name
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Output directory
    window_path_dir = os.path.join(cmt_dir, "window_data", "window_paths")

    # Get all process possibilities
    window_param_files = glob.glob(os.path.join(window_process_dir, "*"))

    for _i, window_param_file in enumerate(window_param_files):

        # Get band
        window_process_params = smart_read_yaml(window_param_file)
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
                                 "processed_observed_%03.0f_%03.0f.h5"
                                 % (lP, hP))
        obsd_tag = "proc_obs_%03.0f_%03.0f" % (lP, hP)

        # Synthetic ASDF
        synt_asdf = os.path.join(cmt_dir, "seismograms", "syn",
                                 "processed_seismograms",
                                 "processed_synthetic_CMT_%03.0f_%03.0f.h5"
                                 % (lP, hP))
        synt_tag = "proc_syn_CMT_%03.0f_%03.0f" % (lP, hP)

        # Output file parameters
        output_file = os.path.join(cmt_dir, "window_data",
                                   'windows_%03.0f_%03.0f%s.json'
                                   % (lP, hP, wave_type))

        # Pathfile directory
        yaml_file_path = os.path.join(window_path_dir,
                                      "windows_%03.0f_%03.0f%s.yml"
                                      % (lP, hP, wave_type))

        # Create dictionary
        if verbose:
            print("Writing path file %s." % yaml_file_path)

        d = {"obsd_asdf": obsd_asdf,
             "obsd_tag": obsd_tag,
             "synt_asdf": synt_asdf,
             "synt_tag": synt_tag,
             "output_file": output_file,
             "figure_mode": figure_mode}

        # Writing the directory to file
        write_yaml_file(d, yaml_file_path)
