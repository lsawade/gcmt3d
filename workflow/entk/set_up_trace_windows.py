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
import warnings
import logging

# Get logger to log progress
from gcmt3d import logger

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r'*.numerictypes')
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r'*.asdf_data_set')
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r'*.numerictypes')

def windowing(cmt_filename):

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

    logger.info("\nStart windowing all trace pairs ...\n")

    for _i, path_file in enumerate(window_pathfiles):

        logger.info("\nWindowing path file:\n")
        logger.info(path_file + "\n")
        logger.info("Start windowing traces from path file ...\n")

        # Load process path file to get parameter file location
        params = smart_read_yaml(path_file, mpi_mode=is_mpi_env())\
            ["window_param_file"]

        pyflex_logger = logging.getLogger("pyflex")
        pyflex_logger.setLevel(logging.DEBUG)

        # Create Smart Process class
        proc = WindowASDF(path_file, params, verbose=DB_params["verbose"],
                          debug=False)
        proc.smart_run()

        logger.info("\nDONE windowing traces file.\n")

    logger.info("\nDONE windowed all data!\n")
