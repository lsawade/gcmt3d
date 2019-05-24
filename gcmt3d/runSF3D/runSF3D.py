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

    def __init__(self, specfemdir, simdir, NEX=128, NPROC=1, npar=None,
                 verbose=False):
        """
        Initializes Run parameters

        Args:
            specfemdir: string with directory name
            NEX: Number of elements along the first chunk (s. Specfem Manual)
            NPROC: Number of MPI processors (s. Specfem Manual)
            verbose: boolean deciding on whether to print stuff

        Returns: Nothing really it just runs specfem with the above options

        """

        self.specfemdir = specfemdir
        self.NPROC = NPROC
        self.NEX = NEX
        self.simdir = simdir
        self.attr = ["CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                     "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]
        if npar in [6, 7, 9]:
            self.npar = npar
        elif npar is None:
            pass
        else:
            raise ValueError("Wrong number. must be 6, 7, or 9.")
        self.v = verbose

    def fix_parfiles(self):
        """This function changes the number of nodes within the parfile
        in each subdirectory of the CMT_SIMs directory."""

        parfile = os.path.join(self.specfemdir, "DATA/Par_file")

        # Make sure it's a global simulation:
        self.replace_varval(parfile, "NCHUNKS", self.NEX)

        # Replace elements along surface of the two sides of first chunk
        self.replace_varval(parfile, "NEX_XI", self.NEX)
        self.replace_varval(parfile, "NEX_ETA", self.NEX)

        # Replace number of MPI processors along the two sides of the first
        # chunk
        self.replace_varval(parfile, "NPROC_XI", self.NPROC)
        self.replace_varval(parfile, "NPROC_ETA", self.NPROC)

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
                line_components[1] = str(newval)
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
            if re.match(" *" + var + " *=", line):
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
