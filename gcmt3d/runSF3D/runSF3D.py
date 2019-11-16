"""
This file contains the class to run specfem for the database.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: May 2019


"""
import os
import re
import subprocess
import shutil
from shlex import split


class RunSimulation(object):
    """Class handles the running of specfem after its directory has been
    added to the database. """

    def __init__(self, earthquake_dir, N=1, npn=24, n=24, memory_req=None,
                 npar=9, walltime="00:30:00", modules=None,
                 gpu_module=None, GPU_MODE=False, verbose=False):
        """
        Initializes Run parameters

        Args:
            database_eq_dir: string with directory name
            N: integer with number of Nodes
            n: integer with number of tasks
            npn: integer with tasks per node
            npar: integer number of parameters
            walltime: string with max time "hh:mm:ss"
            modules: list of modules to be loaded for the run (for comilers
                     etc.) Default ['intel', 'openmpi']
            gpu_module: GPU module to be loaded. Default 'cudatoolkit'.
            verbose: boolean deciding on whether to print stuff

        Returns: Nothing really, it just runs specfem with the above options

        """

        self.earthquake = earthquake_dir
        self.simdir = os.path.join(self.earthquake, "CMT_SIMs")
        self.N = N
        self.n = n
        self.npn = npn
        self.memory_req = memory_req
        self.walltime = walltime
        self.modules = modules
        self.gpu_module = gpu_module
        self.GPU_MODE = GPU_MODE
        self.attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                     "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]
        if npar in [6, 7, 9]:
            self.npar = npar
        else:
            raise ValueError("Wrong number. must be 6, 7, or 9.")
        self.batchdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "batch")
        self.v = verbose

    def __call__(self):
        """Runs the Simulation using the shell and batch files in the batch
        subdirectory."""

        if self.GPU_MODE is True and type(self.GPU_MODE) is bool:

            if self.v:
                print("GPU mode!")

            # batch driver wrapper
            batchwrapper = os.path.join(self.batchdir, "drive_GPU.sh")

            # batch driver script
            batchscript = os.path.join(self.batchdir, "drive_GPU.sbatch")

            # Create command
            bashCommand = "%s %s %s %s %s %s %s %s '%s' %s %s %s" \
                          % (batchwrapper,
                             self.N,
                             self.n,
                             self.npn,
                             self.memory_req,
                             self.npar,
                             self.simdir,
                             self.walltime,
                             " ".join(self.modules),
                             self.gpu_module,
                             int(self.v),
                             batchscript)

            print(bashCommand)

        else:

            if self.v:
                print("CPU mode!")

            # batch driver wrapper
            batchwrapper = os.path.join(self.batchdir, "drive.sh")

            # batch driver script
            batchscript = os.path.join(self.batchdir, "drive.sbatch")

            # Create command
            bashCommand = "%s %s %s %s %s %s '%s' %s %s" % (batchwrapper,
                                                            self.N,
                                                            self.n,
                                                            self.npar,
                                                            self.simdir,
                                                            self.walltime,
                                                            " ".join(
                                                                self.modules),
                                                            int(self.v),
                                                            batchscript)

        # Send command
        process = subprocess.run(split(bashCommand), check=True, text=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

        # catch outputs
        if self.v:
            print(bashCommand)
            print("Command has been sent.")
            print("Output:\n", process.stdout)
            print("Errors:\n", process.stderr)

    def replace_STATIONS(self, statfile):
        """This function handles the replacement of the STATION file in the
        database directory."""

        if (self.npar is None) or (self.simdir is None):
            raise ValueError("No number of parameters or Sim dir given")
        else:
            for at in self.attr[:self.npar+1]:
                newstatfile = os.path.join(self.simdir, at, "DATA", "STATIONS")
                self._replace_file(statfile, newstatfile)

    def clean_up(self):
        """This function cleans out the simulation directory, getting rid of
        unnecessary MPI and database files. The parameter files are first
        copied to the directory OUTPUT_FILES and then `bin`, `DATA` and
        `DATABASES_MPI` are deleted since they are irrelevant for future
        reproducibility."""

        for at in self.attr[:self.npar+1]:
            # Copy Par_file CMTSOLUTION and STATIONS files to the OUTPUTFILES
            # directory
            atsimdir = os.path.join(self.simdir, at)
            datadir = os.path.join(atsimdir, "DATA")
            outdir = os.path.join(atsimdir, "OUTPUT_FILES")

            # Stations
            self._replace_file(os.path.join(datadir, "STATIONS"),
                               os.path.join(outdir, "STATIONS"))
            # Par_file
            self._replace_file(os.path.join(datadir, "Par_file"),
                               os.path.join(outdir, "Par_file"))

            # CMTSOLUTION
            self._replace_file(os.path.join(datadir, "CMTSOLUTION"),
                               os.path.join(outdir, "CMTSOLUTION"))

            # Remove bin symlink
            self._remove_file(os.path.join(atsimdir, "bin"))

            # remove unnecessary
            self._remove_dir(os.path.join(atsimdir, "DATA"))
            self._remove_dir(os.path.join(atsimdir, "DATABASES_MPI"))

    def _replace_file(self, source, destination):
        """Mini function that replaces a directory"""
        if os.path.exists(source):
            if os.path.exists(destination) and os.path.isfile(destination):
                if self.v:
                    print("File already here. Removing %s ..." % destination)
                os.remove(destination)

            if self.v:
                print("Copying %s to %s ..." % (source, destination))
            shutil.copyfile(source, destination)
        else:
            print("%s does not exist. Not copied to %s."
                  % (source, destination))

    def _remove_file(self, file):
        """ Mini function to remove file."""
        if os.path.exists(file) and os.path.isfile(file):
            if self.v:
                print("Removing %s ..." % file)
            os.remove(file)
        else:
            print("%s does not exist. Hence not deleted." % file)

    def _remove_dir(self, direc):
        """ Mini function to remove directory."""
        if os.path.exists(direc) and os.path.isdir(direc):
            if self.v:
                print("Removing %s ..." % direc)
            shutil.rmtree(direc)
        else:
            print("%s does not exist. Hence not deleted." % direc)

    def __str__(self):
        """string return"""
        string = ""
        string += "Earthquake directory: %s\n" % self.earthquake
        string += "Simulation directory: %s\n" % self.simdir
        string += "Number of Nodes: %d\n" % self.N
        string += "Number of Tasks: %d\n" % self.n
        string += "Number of Parameters: %d\n\n" % self.npar
        return string


class DATAFixer(object):
    """Not necessary but it handles the fixing of the parfile"""

    def __init__(self, specfemdir,
                 nodes=1,
                 tasks=24,
                 tasks_per_node=24,
                 memory_req=40000,
                 walltime="00:30:00",
                 walltime_solver=None,
                 cc="icc",
                 cpp="icpc",
                 mpicc="mpicc",
                 f90="ifort",
                 mpif90="mpifort",
                 modules=['intel', 'openmpi'],
                 gpu_module=None,
                 gpu_version=None,
                 NEX_XI=128, NEX_ETA=128,
                 NPROC_XI=1, NPROC_ETA=1,
                 GPU_MODE=False,
                 GPU_RUNTIME=1,
                 GPU_PLATFORM='NVIDIA',
                 GPU_DEVICE='Tesla',
                 ADIOS_ENABLED=False,
                 ADIOS_FOR_FORWARD_ARRAYS=True,
                 ADIOS_FOR_MPI_ARRAYS=True,
                 ADIOS_FOR_ARRAYS_SOLVER=True,
                 ADIOS_FOR_SOLVER_MESHFILES=True,
                 ADIOS_FOR_AVS_DX=True,
                 ADIOS_FOR_KERNELS=True,
                 ADIOS_FOR_MODELS=True,
                 ADIOS_FOR_UNDO_ATTENUATION=True,
                 ROTATE_SEISMOGRAMS_RT=True,
                 RECORD_LENGTH=10,
                 MODEL="s40rts",
                 WRITE_SEISMOGRAMS_BY_MASTER=True,
                 OUTPUT_SEISMOS_ASCII_TEXT=False,
                 OUTPUT_SEISMOS_SAC_ALPHANUM=False,
                 OUTPUT_SEISMOS_SAC_BINARY=False,
                 OUTPUT_SEISMOS_ASDF=False,
                 MOVIE_SURFACE=False,
                 MOVIE_VOLUME=False,
                 MOVIE_COARSE=False,
                 verbose=False):
        """
        Initializes Run parameters in the Parfile

        :param specfemdir: string with directory name
        :param nodes: number of computational nodes for the mesher. Default `1`
        :param tasks: number of tasks per node for the mesher. Default `24`
        :param tasks_per_node: maximum number of tasks per node. Default `24`.
        :param walltime: Requested server time for the nodes.
                         Default `00:30:00`
        :param walltime_solver: walltime of mesher and solver can be different.
        f90: string with fortran compiler
        :param mpif90: string with fortran mpi compiler
        :param cc: string with c compiler
        :param cpp: string with c++ compiler
        :param mpicc: string with c mpi compiler
        :param modules: list of modules to be loaded for the run (for comilers
                 etc.) Default ['intel', 'openmpi']
        :param gpu_module: GPU module to be loaded. Default 'None'.
        :param gpu_version: version that is used to compile specfem.
        :param walltime: Requested server time for the nodes. Default `None`.
                  `None` since this class is used for fixing the `Par_file`
        :param NEX_XI: Number of elements along the chunk (s. Specfem Manual).
                Default `128`.
        :param NEX_ETA: Number of elements along the first chunk (s. Specfem
                 Manual). Default `128`.
        :param NPROC_XI: Number of MPI processors (s. Specfem Manual).
                  Default `1`.
        :param NPROC_ETA: Number of MPI processors (s. Specfem Manual).
                   Default `1`.
        :param GPU_MODE: Set whether specfem should be GPU enabled (not yet
                supported). Default `False`.
        :param ADIOS_ENABLED: Set whether ADIOS should be turned on (not yet
                       supported). Default `False`
        :param RECORD_LENGTH: length of the final seismic record in minutes.
                       Default `1`.
        :param MODEL: velocity model, right now only 3D models are supported.
               Default `'s40rts'`
        :param ROTATE_SEISMOGRAMS_RT: This sets whether the seismograms are
                               output as as NEZ or RTZ if True RTZ is
                               chosen. Default `True`.
        :param WRITE_SEISMOGRAMS_BY_MASTER: Write seismograms by master job.
                                     Default `False`.
        :param OUTPUT_SEISMOS_ASCII_TEXT: Output seismograms in `ASCII` format.
                                   Default `False`.
        :param OUTPUT_SEISMOS_SAC_ALPHANUM: Output seismograms in `ASCII`
                                            format. Default `False`.
        :param OUTPUT_SEISMOS_SAC_BINARY: Output seismograms in `ASCII` format.
                                   Default `False`.
        :param OUTPUT_SEISMOS_ASDF: Output seismograms in `ASCII` format.
                             Default `False`.
        :param MOVIE_SURFACE: Write movie on the surface. Default `False`
        :param MOVIE_VOLUME: Write movie within the volume. Default `False`
        :param MOVIE_COARSE: Write coarse movie. Default `False`
        :param verbose: boolean deciding on whether to print stuff.
                Default `False`.

        Returns: Nothing really it just runs specfem with the above options
        """

        # Specfem parameters
        self.specfemdir = specfemdir

        # Slurm Resources for mesher
        self.nodes = nodes
        self.tasks = tasks
        self.tasks_per_node = tasks_per_node
        self.memory_req = memory_req
        self.walltime = walltime  # walltime for mesher

        # The data fixer can also run specfem for that a specific walltime
        # has to be set
        self.walltime_solver = walltime_solver

        # Compilers
        self.mpif90 = mpif90
        self.f90 = f90
        self.cc = cc
        self.cpp = cpp
        self.mpicc = mpicc

        # Modules
        self.modules = modules
        self.gpu_module = gpu_module

        # MPI STUFF
        self.NPROC_XI = NPROC_XI
        self.NPROC_ETA = NPROC_ETA
        self.NEX_XI = NEX_XI
        self.NEX_ETA = NEX_ETA

        # GPU acceleration parameters
        self.GPU_MODE = GPU_MODE
        self.GPU_RUNTIME = GPU_RUNTIME
        self.GPU_PLATFORM = GPU_PLATFORM
        self.GPU_DEVICE = GPU_DEVICE
        self.gpu_version = gpu_version

        # ADIOS parameters
        self.ADIOS_ENABLED = ADIOS_ENABLED
        self.ADIOS_FOR_FORWARD_ARRAYS = ADIOS_FOR_FORWARD_ARRAYS
        self.ADIOS_FOR_MPI_ARRAYS = ADIOS_FOR_MPI_ARRAYS
        self.ADIOS_FOR_ARRAYS_SOLVER = ADIOS_FOR_ARRAYS_SOLVER
        self.ADIOS_FOR_SOLVER_MESHFILES = ADIOS_FOR_SOLVER_MESHFILES
        self.ADIOS_FOR_AVS_DX = ADIOS_FOR_AVS_DX
        self.ADIOS_FOR_KERNELS = ADIOS_FOR_KERNELS
        self.ADIOS_FOR_MODELS = ADIOS_FOR_MODELS
        self.ADIOS_FOR_UNDO_ATTENUATION = ADIOS_FOR_UNDO_ATTENUATION

        # PARAMETERS
        self.vmodel = MODEL
        self.RECORD_LENGTH = RECORD_LENGTH

        # Processing
        self.ROTATE_SEISMOGRAMS_RT = ROTATE_SEISMOGRAMS_RT

        # OUTPUT seismograms as:
        self.WRITE_SEISMOGRAMS_BY_MASTER = WRITE_SEISMOGRAMS_BY_MASTER
        self.OUTPUT_SEISMOS_ASCII_TEXT = OUTPUT_SEISMOS_ASCII_TEXT
        self.OUTPUT_SEISMOS_SAC_ALPHANUM = OUTPUT_SEISMOS_SAC_ALPHANUM
        self.OUTPUT_SEISMOS_SAC_BINARY = OUTPUT_SEISMOS_SAC_BINARY
        self.OUTPUT_SEISMOS_ASDF = OUTPUT_SEISMOS_ASDF

        # Visualization
        self.MOVIE_SURFACE = MOVIE_SURFACE
        self.MOVIE_VOLUME = MOVIE_VOLUME
        self.MOVIE_COARSE = MOVIE_COARSE

        self.v = verbose

        # Set the directory for the batch script
        self.batchdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "batch")

    def fix_parfiles(self):
        """This function changes the number of nodes within the parfile
        in each subdirectory of the CMT_SIMs directory."""

        parfile = os.path.join(self.specfemdir, "DATA/Par_file")

        # Make sure it's a global simulation:
        self.replace_varval(parfile, "NCHUNKS", str(6))

        # Replace elements along surface of the two sides of first chunk
        self.replace_varval(parfile, "NEX_XI", self.NEX_XI)
        self.replace_varval(parfile, "NEX_ETA", self.NEX_ETA)

        # Replace number of MPI processors along the two sides of the first
        # chunk
        self.replace_varval(parfile, "NPROC_XI", self.NPROC_XI)
        self.replace_varval(parfile, "NPROC_ETA", self.NPROC_ETA)

        # Check whether velocity model has to change
        if self.vmodel not in [
                "transversely_isotropic_prem_plus_3D_crust_2.0",
                "3D_anisotropic", "3D_attenuation", "s20rts",
                "s40rts", "s362ani", "s362iso", "s362wmani",
                "s362ani_prem", "s362ani_3DQ", "s362iso_3DQ",
                "s29ea", "sea99_jp3d1994", "sea99", "jp3d1994",
                "heterogen", "full_sh", "sgloberani_aniso",
                "sgloberani_iso"]:
            raise ValueError("Wrong velocity model name. Existing model is"
                             " used.")
        else:
            self.replace_varval(parfile, "MODEL", self.vmodel)

        # Replace RECORD_LENGTH_IN_MINUTES
        self.replace_varval(parfile, "RECORD_LENGTH_IN_MINUTES", "%s.0d0" %
                            self.RECORD_LENGTH)

        # Processing
        self.replace_varval(parfile, "ROTATE_SEISMOGRAMS_RT",
                            self.ROTATE_SEISMOGRAMS_RT)

        # OUTPUT seismograms as:
        self.replace_varval(parfile, "WRITE_SEISMOGRAMS_BY_MASTER",
                            self.WRITE_SEISMOGRAMS_BY_MASTER)
        self.replace_varval(parfile, "OUTPUT_SEISMOS_ASCII_TEXT",
                            self.OUTPUT_SEISMOS_ASCII_TEXT)
        self.replace_varval(parfile, "OUTPUT_SEISMOS_SAC_ALPHANUM",
                            self.OUTPUT_SEISMOS_SAC_ALPHANUM)
        self.replace_varval(parfile, "OUTPUT_SEISMOS_SAC_BINARY",
                            self.OUTPUT_SEISMOS_SAC_BINARY)
        self.replace_varval(parfile, "OUTPUT_SEISMOS_ASDF",
                            self.OUTPUT_SEISMOS_ASDF)

        # Visualization
        self.replace_varval(parfile, "MOVIE_SURFACE", self.MOVIE_SURFACE)
        self.replace_varval(parfile, "MOVIE_VOLUME", self.MOVIE_VOLUME)
        self.replace_varval(parfile, "MOVIE_COARSE", self.MOVIE_COARSE)

        # Acceleration parameters supported
        self.replace_varval(parfile, "GPU_MODE", self.GPU_MODE)
        self.replace_varval(parfile, "GPU_RUNTIME", self.GPU_RUNTIME)
        self.replace_varval(parfile, "GPU_PLATFORM", self.GPU_PLATFORM)
        self.replace_varval(parfile, "GPU_DEVICE", self.GPU_DEVICE)

        # ADIOS
        self.replace_varval(parfile, "ADIOS_ENABLED", self.ADIOS_ENABLED)
        self.replace_varval(parfile, "ADIOS_FOR_MPI_ARRAYS",
                            self.ADIOS_FOR_MPI_ARRAYS)
        self.replace_varval(parfile, "ADIOS_FOR_ARRAYS_SOLVER",
                            self.ADIOS_FOR_ARRAYS_SOLVER)
        self.replace_varval(parfile, "ADIOS_FOR_SOLVER_MESHFILES",
                            self.ADIOS_FOR_SOLVER_MESHFILES)
        self.replace_varval(parfile, "ADIOS_FOR_AVS_DX", self.ADIOS_FOR_AVS_DX)
        self.replace_varval(parfile, "ADIOS_FOR_KERNELS",
                            self.ADIOS_FOR_KERNELS)
        self.replace_varval(parfile, "ADIOS_FOR_MODELS", self.ADIOS_FOR_MODELS)
        self.replace_varval(parfile, "ADIOS_FOR_UNDO_ATTENUATION",
                            self.ADIOS_FOR_UNDO_ATTENUATION)

    def configure_and_make(self):
        """ Configures specfem and runs 'make all' in the specfemdirectory."""

        if self.GPU_MODE and type(self.GPU_MODE) is bool:

            if self.v:
                print("GPU mode!")

            # Create configuration command
            configure_script = os.path.join(self.batchdir, "configure_GPU.sh")

            # Create command compilation command
            bashCommand = "%s %s %s %s %s %s %s '%s' %s %s" \
                          % (configure_script,
                             self.specfemdir,
                             self.mpif90,
                             self.f90,
                             self.cc,
                             self.cpp,
                             self.mpicc,
                             " ".join(self.modules),
                             self.gpu_module,
                             self.gpu_version)

        else:
            if self.v:
                print("CPU mode!")

            # Create configuration command
            configure_script = os.path.join(self.batchdir, "configure.sh")

            # Create command compilation command
            bashCommand = "%s %s %s %s %s %s %s '%s'" \
                          % (configure_script,
                             self.specfemdir,
                             self.mpif90,
                             self.f90,
                             self.cc,
                             self.cpp,
                             self.mpicc,
                             " ".join(self.modules))

        if self.v:
            # Send command
            process = subprocess.run(split(bashCommand), check=True, text=True)
        else:
            # Send command
            process = subprocess.run(split(bashCommand), check=True, text=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

        # catch outputs
        if self.v:
            print(bashCommand)
            print("Command has been sent.")
            print("Output:\n", process.stdout)
            print("Errors:\n", process.stderr)
            print("Compilation DONE!\n")

    def run_mesher(self):
        """This function runs the mesher after """

        if self.GPU_MODE and type(self.GPU_MODE) is bool:

            if self.v:
                print("GPU mode!")

            # batch driver script
            batchscript = os.path.join(self.batchdir, "mesh_GPU.sbatch")

            # Create command -N nodes, -n tasks, -D change directory
            bashCommand = "sbatch -N %s -n %s --ntasks-per-node %s -D %s -t " \
                          "%s --gres=gpu:%s -p pReserved %s" \
                          % (self.nodes,
                             self.tasks,
                             self.tasks_per_node,
                             self.specfemdir,
                             self.walltime,
                             self.tasks_per_node,
                             batchscript)

        else:

            if self.v:
                print("CPU mode!")

            # batch driver script
            batchscript = os.path.join(self.batchdir, "mesh.sbatch")

            # Create command -N nodes, -n tasks, -D change directory
            bashCommand = "sbatch -N %s -n %s -D %s -t %s %s" \
                          % (self.nodes,
                             self.tasks,
                             self.specfemdir,
                             self.walltime,
                             batchscript)

        if self.v:
            # Send command
            process = subprocess.run(split(bashCommand), check=True, text=True)
        else:
            # Send command
            process = subprocess.run(split(bashCommand), check=True, text=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

        # catch outputs
        if self.v:
            print(bashCommand)
            print("Command has been sent.")
            print("Output:\n", process.stdout)
            print("Errors:\n", process.stderr)
            print("Meshing SUBMITTED!\n")

    def run_solver(self):
        """This function runs the solver within the"""

        # Check of solver runtime is set
        if self.GPU_MODE and type(self.GPU_MODE) is bool:

            if self.v:
                print("GPU mode!")

            # batch driver script
            batchscript = os.path.join(self.batchdir, "solver_GPU.sbatch")

            # Create command -N nodes, -n tasks, -D change directory
            bashCommand = "sbatch -N %s -n %s --ntasks-per-node %s -D %s " \
                          "-t %s --mem %s --gres=gpu:%s -p pReserved %s" \
                          % (self.nodes,
                             self.tasks,
                             self.tasks_per_node,
                             self.specfemdir,
                             self.walltime_solver,
                             self.memory_req,
                             self.tasks_per_node,
                             batchscript)

        else:
            # batch driver script
            batchscript = os.path.join(self.batchdir, "solver.sbatch")

            # Create command -N nodes, -n tasks, -D change directory
            bashCommand = "sbatch -N %s -n %s -D %s -t %s %s" \
                          % (self.nodes,
                             self.tasks,
                             self.specfemdir,
                             self.walltime_solver,
                             batchscript)

        if self.v:
            # Send command
            process = subprocess.run(split(bashCommand), check=True, text=True)
        else:
            # Send command
            process = subprocess.run(split(bashCommand), check=True, text=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

        # catch outputs
        if self.v:
            print(bashCommand)
            print("Command has been sent.")
            print("Output:\n", process.stdout)
            print("Errors:\n", process.stderr)

    @staticmethod
    def replace_varval(filename, var, newval):
        """ This function updates the value of a function within a text file

        !Note! That this file is customized to change the specfem `Par_file`,
               Meaning, if newval is `True` or `False` it will be replaced with
               `.true.` or `.false` respectively.

        Args:
            var: variable name -- string
            newval: new variable value, string, number or list

        Throws and error if variable doesnt exist or has multiple definitions.
        """

        file = open(filename, 'r+')
        content_lines = []
        counter = 0

        # Specfem Par_file modification:
        if type(newval) == bool:
            if newval is True:
                newval = '.true.'
            elif newval is False:
                newval = '.false.'

        for line in file:
            # more robust than simple string comparison
            if re.match(" *" + var + " *=", line):
                counter += 1
                # Split the line into to at the equal sign
                line_components = line.split('=')

                # set the value of the line again
                line_components[1] = str(newval) + "\n"
                updated_line = "= ".join(line_components)
                content_lines.append(updated_line)
            else:
                content_lines.append(line)

        # Check whether variable is in file or has multiple definitions
        if counter == 0:
            raise ValueError("Variable not in file.")
        elif counter > 1:
            raise ValueError("Variable is defined in multiple places. Cannot "
                             "overwrite.")
        else:
            file.seek(0)
            file.truncate()
            file.writelines(content_lines)

        file.close()

    @staticmethod
    def get_val(filename, var):
        """ Function searches file for variable and returns that value as a
        string.

        Args:
            filename: string
            var: string

        Returns:
            val
        """

        file = open(filename, 'r+')
        counter = 0

        for line in file:
            # more robust than simple string comparison
            if re.match(" *" + var + ' *=', line):
                counter += 1
                # Split the line into to at the equal sign
                line_components = line.split('=')

                # set the value of the line again
                val = line_components[1]

        # Check whether variable is in file or has multiple definitions
        if counter == 0:
            raise ValueError("Variable not in file.")
        elif counter > 1:
            raise ValueError("Variable is defined in multiple places. Cannot "
                             "overwrite.")
        else:
            return val


sys.addpath(os.path.join(__file__, ".."," ..")