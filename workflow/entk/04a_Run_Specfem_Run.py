#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script runs specfem on the cluster.

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


def main(cmt_filename):

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

    RD = RunSimulation(cmt_dir, N=specfemspecs['nodes'],
                       n=specfemspecs['tasks'],
                       npn=specfemspecs['tasks_per_node'],
                       memory_req=specfemspecs['memory_req'],
                       modules=cm_dict['modulelist'],
                       gpu_module=cm_dict['gpu_module'],
                       GPU_MODE=specfemspecs["GPU_MODE"],
                       walltime=specfemspecs['walltime_solver'],
                       verbose=specfemspecs['verbose'])

    # Print Run specifications of verbose is True
    if specfemspecs['verbose']:
        print(RD)

    # Run Simulation by calling the class
    RD()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    main(args.filename)
