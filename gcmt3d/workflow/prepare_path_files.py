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

from gcmt3d.asdf.utils import read_yaml_file
from gcmt3d.data.management.create_process_paths import PathCreator
import os

# Get logger to log progress
from gcmt3d import logger


def make_paths(cmt_filename, param_path, conversion, ddeg=None, dz=None):

    # Load database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    dbparams = read_yaml_file(databaseparam_path)

    logger.info("Creating dynamic parameter and path files . . .")

    # Create Processing path files observed
    process_dir = os.path.join(param_path, "Process")
    window_dir = os.path.join(param_path, "Window")
    inv_dir = os.path.join(param_path, "CMTInversion")
    P = PathCreator(cmt_filename, window_dir, process_dir, inv_dir,
                    conversion=conversion, npar=dbparams["npar"],
                    dlocation=ddeg, ddepth=dz,
                    figure_mode=dbparams["figure_mode"])
    P.write_all()
