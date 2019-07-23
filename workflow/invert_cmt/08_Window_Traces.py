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
from gcmt3d.asdf.window import WindowASDF
import os
import glob
import argparse

def main(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Database Parameters
    DB_params = smart_read_yaml(databaseparam_path, mpi_mode=is_mpi_env())

    # Get processing path from cmt_filename in database
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))
    window_path_dir = os.path.join(cmt_dir, "window_data", "window_paths")

    # Get all files to be processed
    window_pathfiles = glob.glob(os.path.join(window_path_dir, "*"))

    if DB_params["verbose"]:
        print("\nStart windowing all trace pairs ...\n")

    for _i, path_file in enumerate(window_pathfiles):

        if DB_params["verbose"]:
            print("\nWindowing path file:\n")
            print(path_file + "\n")
            print("Start windowing traces from path file ...\n")

        # Load process path file to get parameter file location
        params = smart_read_yaml(path_file, mpi_mode=is_mpi_env())\
            ["window_param_file"]

        # Create Smart Process class
        proc = WindowASDF(path_file, params, verbose=DB_params["verbose"],
                          debug=False)
        proc.smart_run()

        if DB_params["verbose"]:
            print("\nDONE windowing traces file.\n")

    if DB_params["verbose"]:
        print("\nDONE windowed all data!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)
    args = parser.parse_args()

    # Run
    main(args.filename)
