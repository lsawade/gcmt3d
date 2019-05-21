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

        # batch driver wrapper
        batchwrapper = os.path.join(self.batchdir, "drive.sh")

        # batch driver script
        batchscript = os.path.join(self.batchdir, "drive.sbatch")

        # Create command
        bashCommand = "%s %s %s %s %s %s %s" % (batchwrapper,self.N, 
                                                self.n, self.npar,
                                                self.simdir, batchscript,
                                                int(self.v))
        
        # Send command
        process = subprocess.run(bashCommand.split(), check=True,
                                 text=True)

        # catch outputs
        if self.v:
            print(bashCommand)
            print("Command has been sent.")
            print("Output:\n", process.stdout)
            print("Errors:\n", process.stderr)

    def __str__(self):
        """string return"""

        string = ""
        string += "Earthquake directory: %s\n" % self.earthquake
        string += "Simulation directory: %s\n" % self.simdir
        string += "Number of Nodes: %d\n" % self.N
        string += "Number of Tasks: %d\n" % self.n
        string += "Number of Parameters: %d\n\n" % self.npar
        return string

