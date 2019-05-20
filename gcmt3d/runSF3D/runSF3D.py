"""
This file contains the class to run specfem for the database.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: May 2019


"""


class RunSimulation(object):
    """Class handles the running of specfem after its directory has been
    added to the database. """

    def __init__(self, database_dir, npar=9, n=1):
        """
        Initializes Run parameters

        Args:
            database_eq_dir: string with directory name
            N: integer with number of Nodes
            n: integer with number of tasks
            npar: integer number of parameters .

        Returns: Nothing really it just runs specfem with the above options

        """


        self.N = N
        self.n = n
        self.npar = npar

    def __call__(self):
        """Runs the Simulation using the shell and batch files in the batch
        subdirectory."""



