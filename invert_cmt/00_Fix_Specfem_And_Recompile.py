#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This is a script that will fix specfem given the parameters set in the
`params/SpecfemParams/SpecfemParams.yml` file.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

import os
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env


def main():

    # Define parameter directory
    param_path = os.path.dirname(os.path.dirname(__file__))
    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml.")

    # Load parameters
    smart_read_yaml(specfemspec_path, mpi_mode=is_mpi_env())

    # Define the specfemdatafixer
    DF = DATAFixer(SPECFEM_DIR,
                   NEX_XI=NEX_XI, NEX_ETA=NEX_ETA,
                   NPROC_XI=NPROC_XI, NPROC_ETA=NPROC_ETA,
                   ROTATE_SEISMOGRAMS_RT=ROTATE_SEISMOGRAMS_RT,
                   RECORD_LENGTH=RECORD_LENGTH, MODEL=MODEL,
                   WRITE_SEISMOGRAMS_BY_MASTER=WRITE_SEISMOGRAMS_BY_MASTER,
                   OUTPUT_SEISMOS_ASCII_TEXT=OUTPUT_SEISMOS_ASCII_TEXT,
                   OUTPUT_SEISMOS_SAC_ALPHANUM=OUTPUT_SEISMOS_SAC_ALPHANUM,
                   OUTPUT_SEISMOS_SAC_BINARY=OUTPUT_SEISMOS_SAC_BINARY,
                   OUTPUT_SEISMOS_ASDF=OUTPUT_SEISMOS_ASDF,
                   MOVIE_SURFACE=MOVIE_SURFACE,
                   MOVIE_VOLUME=MOVIE_VOLUME,
                   MOVIE_COARSE=MOVIE_COARSE,
                   nodes=nodes, tasks=tasks, walltime=walltime,
                   nodes_solver=nodes_solver, tasks_solver=tasks_solver,
                   walltime_solver=walltime_solver,
                   verbose=verbose)

    # Run `Par_file` fixer.
    DF.fix_parfiles()

    # Run the mesher to a slurm scheduler.
    DF.run_mesher()


if __name__ == '__main__':
    main()
