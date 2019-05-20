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
import subprocess
from warnings import warn


class RunSimulation(object):
    """Class handles the running of specfem after its directory has been
    added to the database. """

    def __init__(self, earthquake_dir, N=1, npar=9, n=1, verbose=False):
        """
        Initializes Run parameters

        Args:
            database_eq_dir: string with directory name
            N: integer with number of Nodes
            n: integer with number of tasks
            npar: integer number of parameters
            verbose: boolean deciding on whether to print stuff

        Returns: Nothing really it just runs specfem with the above options

        """

        self.earthquake = earthquake_dir
        self.simdir = os.path.join(self.earthquake, "CMT_SIMs")
        self.N = N
        self.n = n
        self.npar = npar
        self.batchdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "batch")
        self.v = verbose

    def __call__(self):
        """Runs the Simulation using the shell and batch files in the batch
        subdirectory."""

        # batch driver script
        batchscript = os.path.join(self.batchdir, "drive.sbatch")

        bashCommand = "drive.sh %s %s %s %s %s" % (self.N, self.n,
                                                   self.npar, self.simdir,
                                                   batchscript)

        # Start process
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

        # catch outputs
        output, error = process.communicate()

        if self.v:
            print("Output:\n", output)
            warn("Errors:\n", error)

    def __str__(self):
        """string return"""

        string = ""
        string += "Earthquake directory: %d\n" % self.earthquake
        string += "Simulation directory: %d\n" % self.simdir
        string += "Number of Nodes: %d\n" % self.N
        string += "Number of Tasks: %d\n" % self.n
        string += "Number of Parameters: %d\n\n" % self.npar
