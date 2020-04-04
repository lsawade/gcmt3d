#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script cleans up the database post-specfem-run to delete unnecessary
files. Meaning, it will move the STATIONS, Par_file, and CMTSOLUTION to the
OUTPUT_FILES directory, before it removes both the DATA and the DATABASES_MPI
directories.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""


from gcmt3d.runSF3D.runSF3D import RunSimulation
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env
import argparse
import os

# Get logger to log progress
from gcmt3d import logger

def clean_up(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml")
    comp_and_modules_path = os.path.join(param_path,
                                         "SpecfemParams/CompilersAndModules.yml")

    # Load Parameters
    specfemspecs = smart_read_yaml(specfemspec_path, mpi_mode=is_mpi_env())
    cm_dict = smart_read_yaml(comp_and_modules_path, mpi_mode=is_mpi_env())

    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Set up RunSimulation class with parameters from the files.
    RD = RunSimulation(cmt_dir, N=specfemspecs['nodes'],
                       n=specfemspecs['tasks'],
                       npn=specfemspecs['tasks_per_node'],
                       memory_req=specfemspecs['memory_req'],
                       modules=cm_dict['modulelist'],
                       gpu_module=cm_dict['gpu_module'],
                       GPU_MODE=specfemspecs["GPU_MODE"],
                       walltime=specfemspecs['walltime_solver'])

    if specfemspecs["verbose"]:
        print("Deleting unnecessary stuff ...")

    # Clean up Simulation directory
    RD.clean_up()

    if specfemspecs["verbose"]:
        print("Deleting DONE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    clean_up(args.filename)
