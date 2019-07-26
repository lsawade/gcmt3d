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
import sys
import argparse

def Create_Entry(cmt_filename):
    """ This function creates a database entry for the inversion.
    :param cmt_filename: path to cmt file that an Entry is supposed to be
                         created for
    :return:
        no returns since it is simply a copy and create dirs function

    """

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                    "Database/DatabaseParameters.yml")
    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml")

    # Load Parameters
    DB_params = smart_read_yaml(databaseparam_path, mpi_mode=is_mpi_env())
    specfemspecs = smart_read_yaml(specfemspec_path, mpi_mode=is_mpi_env())

    # Database Setup.
    DB = DataBaseSkeleton(basedir=DB_params["databasedir"],
                          cmt_fn=cmt_filename,
                          specfem_dir=specfemspecs["SPECFEM_DIR"],
                          verbose=DB_params['verbose'],
                          overwrite=DB_params['overwrite'])

    # Database Create entry
    DB.create_all()

    # Return new earthquake location.
    cmt_in_database = os.path.join(DB.eq_dirs[0], "eq_" + DB.eq_ids[0])

    return cmt_in_database

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    Create_Entry(args.filename)
