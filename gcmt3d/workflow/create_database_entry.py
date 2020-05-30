#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a script that will create a database entry given a cmt solution in the


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

import os
from gcmt3d.data.management.skeleton import DataBaseSkeleton
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env

def create_entry(cmt_filename, databasedir, param_path, specfem=True):
    # Define parameter directory
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml")
    stations_path = os.path.join(param_path,
                                 "RequestParams/STATIONS")

    # Load Parameters
    dbparams = smart_read_yaml(databaseparam_path, mpi_mode=is_mpi_env())
    specfemspecs = smart_read_yaml(specfemspec_path, mpi_mode=is_mpi_env())

    # Check whether stationsfile in Parampath
    if os.path.exists(stations_path):
        stations_file = stations_path
    else:
        # if no stations file in the parameter directory,
        # the standard stations file is going to be used
        stations_file = None

    # Database Setup.
    if specfem is False:
        specfem_dir = None
    else:
        specfem_dir = specfemspecs["SPECFEM_DIR"]

    DB = DataBaseSkeleton(basedir=databasedir,
                          cmt_fn=cmt_filename,
                          specfem_dir=specfem_dir,
                          stations_file=stations_file,
                          overwrite=dbparams['overwrite'])

    # Database Create entry
    DB.create_all()

    # Return new earthquake location.
    cmt_in_database = os.path.join(DB.Cdirs[0], "C" + DB.Cids[0])

    return cmt_in_database
