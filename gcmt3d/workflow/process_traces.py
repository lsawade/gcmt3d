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
from gcmt3d.asdf.process import ProcASDF
import os
import glob
import argparse
import warnings

# Get logger to log progress
from gcmt3d import logger

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r'*.numerictypes')
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r'*.asdf_data_set')
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r'*.numerictypes')


def process(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Database Parameters
    DB_params = smart_read_yaml(databaseparam_path,
                                mpi_mode=is_mpi_env())


    # Get processing path from cmt_filename in database
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))
    process_path_dir = os.path.join(cmt_dir, "seismograms", "process_paths")

    # Get all files to be processed
    process_pathfiles = glob.glob(os.path.join(process_path_dir, "*"))

    logger.info("\nStart processing all data ...\n")

    for _i, path_file in enumerate(process_pathfiles):

        logger.info("\nProcessing path file:\n")
        logger.info(path_file + "\n")
        logger.info("Start processing traces from path file ...\n")

        # Load process path file to get parameter file location
        params = smart_read_yaml(path_file, mpi_mode=is_mpi_env())\
            ["process_param_file"]

        # Create Smart Process class
        proc = ProcASDF(path_file, params, verbose=DB_params["verbose"])
        proc.smart_run()

        logger.info("\nDONE processing path file.\n")

    logger.info("\nDONE processing all data!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    process(args.filename)
