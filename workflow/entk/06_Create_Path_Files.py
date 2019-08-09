#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script writes specfem sources into the respective simulation directories.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

from gcmt3d.asdf.utils import smart_read_yaml
from gcmt3d.asdf.utils import is_mpi_env
from gcmt3d.data.management.create_process_paths import create_process_path_obs
from gcmt3d.data.management.create_process_paths import create_process_path_syn
from gcmt3d.data.management.create_process_paths import create_window_path
import os
import argparse


def main(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = smart_read_yaml(databaseparam_path,
                                mpi_mode=is_mpi_env())

    if DB_params["verbose"]:
        print("Creating processing path files for the observed data...")

    # Create Processing path files observed
    process_obs_dir = os.path.join(param_path, "ProcessObserved")
    create_process_path_obs(cmt_filename, process_obs_dir, verbose=True)

    if DB_params["verbose"]:
        print("Creating processing path files for the synthetic data...")

    # Create Processing path files synthetics
    process_syn_dir = os.path.join(param_path, "ProcessSynthetic")

    create_process_path_syn(cmt_filename, process_syn_dir, DB_params["npar"],
                            verbose=True)

    if DB_params["verbose"]:
        print("Creating processing path files for windowing the data...")

    # Create Window Path Files:
    window_dir = os.path.join(param_path, "CreateWindows")
    create_window_path(cmt_filename, window_dir,
                       figure_mode=DB_params["figure_mode"], verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)
    args = parser.parse_args()

    # Run
    main(args.filename)
