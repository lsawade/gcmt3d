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

from gcmt3d.source import CMTSource
from gcmt3d.data import SpecfemSources
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env
import os

# Get logger to log progress
from gcmt3d import logger


def write_sources(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    inversionparam_path = os.path.join(param_path,
                                       "CMTInversion/InversionParams.yml")

    # Load Parameters
    DB_params = smart_read_yaml(databaseparam_path,
                                mpi_mode=is_mpi_env())

    # Inversion Params
    INV_params = smart_read_yaml(inversionparam_path, mpi_mode=is_mpi_env())

    # File and directory
    cmt_dir = os.path.dirname(cmt_filename)
    cmt = CMTSource.from_CMTSOLUTION_file(cmt_filename)
    outdir = os.path.join(cmt_dir, "CMT_SIMs")

    # Basic parameters
    dm = float(INV_params["config"]["dmoment"])      # 10**22 dyne*cm
    dz = float(INV_params["config"]["ddepth"])       # 1000 m
    ddeg = float(INV_params["config"]["dlocation"])  # 0.001 deg

    logger.info(" ")
    logger.info("  Perturbation parameters")
    logger.info("  " + 50 * "*")
    logger.info("  𝚫M: %g" % dm)
    logger.info("  𝚫z: %g" % dz)
    logger.info("  𝚫deg: %g" % ddeg)
    logger.info("  " + 50 * "*" + "\n")

    # Create source creation class
    sfsource = SpecfemSources(cmt, cmt_dir, npar=DB_params['npar'],
                              dm=dm, dx=dz, ddeg=ddeg,
                              verbose=DB_params['verbose'], outdir=outdir)

    # Write sources
    sfsource.write_sources()
