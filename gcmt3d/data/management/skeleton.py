"""

This script contains functions to create the skeleton structure for the GCMT3D
database of Earthquakes. If wanted.
"""


from ...source import CMTSource
import glob, os



class DataBaseSkeleton(object):
    """Class to handle data skeleton creation. If specfem directory is given,
    the class copies the necessary data from the specfem folder."""

    def __init__(self, basedir=None, cmt_fn=None, specfem_dir=None,
                 verbose=False):
        """
        Args:
            basedir: str with path to database directory, e.g
                            ".../path/to/database/"
            cmt_fn: path to cmt solution, e.g ".../path/to/CMTSOLUTION", or,
                            e.g ".../path/to/cmtsolutionfile with list."
            specfem_dir: str with path to specfem directory, e.g
                            ".../path/to/specfem3d_globe"

        """

        # Check if things exists
        self.basedir = basedir
        self.cmt_fn = cmt_fn
        self.specfem_dir = specfem_dir
        self.v = verbose

    def create_all(self):
        """ Writes complete database structure."""

        # Create earthquake directory
        self._create_eq_dir()

        # Create
        self._create_CMT_SIM_dir()

    def _create_eq_dir(self):
        """Creates response subdirectory"""

        # throw error

        # Consider list of earthquakes

        # copy earthquake into CMT directory


    def _create_CMT_SIM_dir(self):
        """
        Creates CMT simulation directory and copies necessary files from given
        specfem directory. Important here is the fact that specfem has to be
        compiled already and there is no way of testing that
        """

        # Throw error if specfem path is none existent

    def _create_response_dir(self):
        """Creates response subdirectory"""

    def _create_seismogram_dir(self):
        """Creates response subdirectory"""


    def __str__(self):
        """This is what is printed when the initialized class is called in the
        print function. the function returns a string that contains the paths
        to the different directories specified."""

        string = "Database directory:\n"
        string += self.basedir + "\n\n"
        string +="CMTSOLUTION path:\n"
        string += self.cmt_fn + "\n\n"
        string +="Specfem directory"

        return string