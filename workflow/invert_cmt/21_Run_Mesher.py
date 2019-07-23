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
from gcmt3d.runSF3D.runSF3D import DATAFixer


def main():
    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml")
    comp_and_modules_path = os.path.join(param_path,
                                    "SpecfemParams/CompilersAndModules.yml")

    # Load parameters
    sf_dict = smart_read_yaml(specfemspec_path, mpi_mode=is_mpi_env())
    cm_dict = smart_read_yaml(comp_and_modules_path, mpi_mode=is_mpi_env())

    # Define the specfemdatafixer
    DF = DATAFixer(sf_dict["SPECFEM_DIR"],
                   NEX_XI=sf_dict["NEX_XI"], NEX_ETA=sf_dict["NEX_ETA"],
                   NPROC_XI=sf_dict["NPROC_XI"], NPROC_ETA=sf_dict["NPROC_ETA"],
                   ROTATE_SEISMOGRAMS_RT=sf_dict["ROTATE_SEISMOGRAMS_RT"],
                   RECORD_LENGTH=sf_dict["RECORD_LENGTH"],
                   MODEL=sf_dict["MODEL"],
                   WRITE_SEISMOGRAMS_BY_MASTER=sf_dict[
                       "WRITE_SEISMOGRAMS_BY_MASTER"],
                   OUTPUT_SEISMOS_ASCII_TEXT=sf_dict[
                       "OUTPUT_SEISMOS_ASCII_TEXT"],
                   OUTPUT_SEISMOS_SAC_ALPHANUM=sf_dict[
                       "OUTPUT_SEISMOS_SAC_ALPHANUM"],
                   OUTPUT_SEISMOS_SAC_BINARY=sf_dict[
                       "OUTPUT_SEISMOS_SAC_BINARY"],
                   OUTPUT_SEISMOS_ASDF=sf_dict["OUTPUT_SEISMOS_ASDF"],
                   MOVIE_SURFACE=sf_dict["MOVIE_SURFACE"],
                   MOVIE_VOLUME=sf_dict["MOVIE_VOLUME"],
                   MOVIE_COARSE=sf_dict["MOVIE_COARSE"],
                   GPU_MODE=sf_dict["GPU_MODE"],
                   GPU_RUNTIME=sf_dict["GPU_RUNTIME"],
                   GPU_PLATFORM=sf_dict["GPU_PLATFORM"],
                   GPU_DEVICE=sf_dict["GPU_DEVICE"],
                   ADIOS_ENABLED=sf_dict["ADIOS_ENABLED"],
                   ADIOS_FOR_FORWARD_ARRAYS=sf_dict["ADIOS_FOR_FORWARD_ARRAYS"],
                   ADIOS_FOR_MPI_ARRAYS=sf_dict["ADIOS_FOR_MPI_ARRAYS"],
                   ADIOS_FOR_ARRAYS_SOLVER=sf_dict["ADIOS_FOR_ARRAYS_SOLVER"],
                   ADIOS_FOR_SOLVER_MESHFILES=sf_dict[
                       "ADIOS_FOR_SOLVER_MESHFILES"],
                   ADIOS_FOR_AVS_DX=sf_dict["ADIOS_FOR_AVS_DX"],
                   ADIOS_FOR_KERNELS=sf_dict["ADIOS_FOR_KERNELS"],
                   ADIOS_FOR_MODELS=sf_dict["ADIOS_FOR_MODELS"],
                   ADIOS_FOR_UNDO_ATTENUATION=sf_dict[
                       "ADIOS_FOR_UNDO_ATTENUATION"],
                   modules=cm_dict["modulelist"],
                   gpu_module=cm_dict["gpu_module"],
                   gpu_version=cm_dict["gpu_version"],
                   cc=cm_dict["cc"],
                   cpp=cm_dict["cpp"],
                   mpicc=cm_dict["mpicc"],
                   f90=cm_dict["f90"],
                   mpif90=cm_dict["mpif90"],
                   nodes=sf_dict["nodes"],
                   tasks=sf_dict["tasks"],
                   tasks_per_node=sf_dict["tasks_per_node"],
                   walltime=sf_dict["walltime"],
                   walltime_solver=sf_dict["walltime_solver"],
                   memory_req=sf_dict["memory_req"],
                   verbose=sf_dict["verbose"])

    # Run the mesher to a slurm scheduler.
    DF.run_mesher()


if __name__ == '__main__':
    main()
