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
import warnings


class RunSimulation(object):
    """Class handles the running of specfem after its directory has been
    added to the database. """

    def __init__(self, earthquake_dir, N=1, npar=9, n=24,
                 walltime="00:30:00", verbose=False):
        """
        Initializes Run parameters

        Args:
            database_eq_dir: string with directory name
            N: integer with number of Nodes
            n: integer with number of tasks
            npar: integer number of parameters
            walltime: string with max time "hh:mm:ss"
            verbose: boolean deciding on whether to print stuff

        Returns: Nothing really, it just runs specfem with the above options

        """

        self.earthquake = earthquake_dir
        self.simdir = os.path.join(self.earthquake, "CMT_SIMs")
        self.N = N
        self.n = n
        self.walltime = walltime
        self.attr = ["CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
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

        # batch driver wrapper
        batchwrapper = os.path.join(self.batchdir, "drive.sh")

        # batch driver script
        batchscript = os.path.join(self.batchdir, "drive.sbatch")

        # Create command
        bashCommand = "%s %s %s %s %s %s %s %s" % (batchwrapper, self.N,
                                                   self.n, self.npar,
                                                   self.simdir, self.walltime,
                                                   int(self.v), batchscript)

        # Send command
        process = subprocess.run(bashCommand.split(), check=True, text=True)
        print(process)
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
            for at in self.attr:
                newstatfile = os.path.join(self.simdir, at, "DATA", "STATIONS")
                self._replace_file(statfile, newstatfile)

    @staticmethod
    def _replace_file(source, destination):
        """Mini function that replaces a directory"""
        if os.path.exists(destination) and os.path.isfile(destination):
            os.remove(destination)
        shutil.copyfile(source, destination)

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
                 NEX_XI=128, NEX_ETA=128,
                 NPROC_XI=1, NPROC_ETA=1,
                 RECORD_LENGTH=10,
                 MODEL=None,
                 nodes=1,
                 tasks=24,
                 walltime="00:30:00",
                 verbose=False):
        """
        Initializes Run parameters in the Parfile

        Args:
            specfemdir: string with directory name
            NEX_XI: Number of elements along the chunk (s. Specfem Manual)
            NEX_ETA: Number of elements along the first chunk (s. Specfem
            Manual)
            NPROC_XI: Number of MPI processors (s. Specfem Manual)
            NPROC_ETA: Number of MPI processors (s. Specfem Manual)
            RECORD_LENGTH: length of the final seismic record in minutes
            MODEL: velocity model, right now only 3D models are supported
            verbose: boolean deciding on whether to print stuff

        Returns: Nothing really it just runs specfem with the above options

        """

        self.specfemdir = specfemdir
        self.NPROC_XI = NPROC_XI
        self.NPROC_ETA = NPROC_ETA
        self.NEX_XI = NEX_XI
        self.NEX_ETA = NEX_ETA
        self.RECORD_LENGTH = RECORD_LENGTH
        self.vmodel = MODEL
        self.nodes = nodes
        self.tasks = tasks
        self.walltime = walltime

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
        if self.vmodel is not None:
            if self.vmodel not in [
                    "transversely_isotropic_prem_plus_3D_crust_2.0",
                    "3D_anisotropic", "3D_attenuation", "s20rts",
                    "s40rts", "s362ani", "s362iso", "s362wmani",
                    "s362ani_prem", "s362ani_3DQ", "s362iso_3DQ",
                    "s29ea", "sea99_jp3d1994", "sea99", "jp3d1994",
                    "heterogen", "full_sh", "sgloberani_aniso",
                    "sgloberani_iso"]:
                warnings.warn("Wrong velocity model name. Existing model is "
                              "used.")
            else:
                self.replace_varval(parfile, "MODEL", self.vmodel)

        # Replace RECORD_LENGTH_IN_MINUTES
        self.replace_varval(parfile, "RECORD_LENGTH_IN_MINUTES", "%s.0d0" %
                            self.RECORD_LENGTH)

    def run_mesher(self):
        """This function runs the mesher after """

        # batch driver script
        batchscript = os.path.join(self.batchdir, "mesh.sbatch")

        # Create command -N nodes, -n tasks, -D change directory
        bashCommand = "sbatch -N %s -n %s -D %s -t %s %s" % (self.N,
                                                             self.n,
                                                             self.specfemdir,
                                                             self.walltime,
                                                             batchscript)

        # Send command
        process = subprocess.run(bashCommand.split(), check=True, text=True)
        print(process)
        # catch outputs
        if self.v:
            print(bashCommand)
            print("Command has been sent.")
            print("Output:\n", process.stdout)
            print("Errors:\n", process.stderr)

    @staticmethod
    def replace_varval(filename, var, newval):
        """ This function updates the value of a function within a text file

        Args:
            var: variable name -- string
            newval: new variable value, string, number or list

        Throws and error if variable doesnt exist or has multiple definitions.
        """

        file = open(filename, 'r+')
        content_lines = []
        counter = 0

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
